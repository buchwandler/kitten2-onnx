from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.external_data import (
    logical_tensor_bytes,
    map_external_float32_tensor,
    resolve_external_data_range,
)
from kitten2_onnx.ternary_tritpack_storage import (
    pack_weight_reference,
    rewrite_matmul_to_tritpack_storage,
)

_RECONSTRUCTION_OPS = {"Cast", "Gather", "Mul", "Reshape", "Slice", "Transpose"}


def _load_weight(source_path: Path, tensor: onnx.TensorProto) -> np.ndarray:
    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        tensor_slice = resolve_external_data_range(
            source_path.parent,
            tensor,
            logical_bytes=logical_tensor_bytes(tensor),
            require_exact_length=True,
        )
        mapped = map_external_float32_tensor(
            tensor_slice, tuple(int(dimension) for dimension in tensor.dims)
        )
        try:
            return np.array(mapped, copy=True)
        finally:
            mapped._mmap.close()
    return np.asarray(numpy_helper.to_array(tensor)).copy()


def _select_weight(
    source_path: Path, source_model: onnx.ModelProto, requested_node: str | None
) -> tuple[str, np.ndarray]:
    initializers = {tensor.name: tensor for tensor in source_model.graph.initializer}
    candidates = [
        (node.name, initializers[node.input[1]])
        for node in source_model.graph.node
        if node.op_type == "MatMul"
        and len(node.input) == 2
        and node.name
        and node.input[1] in initializers
        and initializers[node.input[1]].data_type == TensorProto.FLOAT
        and len(initializers[node.input[1]].dims) == 2
        and initializers[node.input[1]].dims[0] > 0
        and initializers[node.input[1]].dims[0] % 128 == 0
        and (requested_node is None or node.name == requested_node)
    ]
    candidates.sort(key=lambda row: int(np.prod(row[1].dims)))

    if requested_node is not None and not candidates:
        raise ValueError(f"no eligible initializer MatMul named {requested_node!r}")
    for node_name, tensor in candidates:
        weight = _load_weight(source_path, tensor)
        try:
            pack_weight_reference(weight)
        except ValueError:
            if requested_node is not None:
                raise
            continue
        return node_name, weight
    raise ValueError("source model has no exact group-128 ternary initializer MatMul")


def _matmul_model(weight: np.ndarray, source_model: onnx.ModelProto, node_name: str):
    k, n = weight.shape
    source_opset = next(
        (entry.version for entry in source_model.opset_import if entry.domain == ""), 18
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name=node_name)],
        "single-tritpack-matmul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, n])],
        [numpy_helper.from_array(weight, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", source_opset)])
    model.ir_version = min(model.ir_version, source_model.ir_version, 10)
    return model


def _peak_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _profile_session(
    model: onnx.ModelProto,
    activation: np.ndarray,
    *,
    warm_runs: int,
    optimized_model_path: Path | None = None,
) -> dict:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if optimized_model_path is not None:
        options.optimized_model_filepath = str(optimized_model_path)

    started = time.perf_counter()
    session = ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    session_creation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    output = session.run(None, {"A": activation})[0]
    first_run_seconds = time.perf_counter() - started

    repeated_seconds = []
    for _ in range(warm_runs):
        started = time.perf_counter()
        session.run(None, {"A": activation})
        repeated_seconds.append(time.perf_counter() - started)

    peak_rss_bytes = _peak_rss_bytes()
    del session
    gc.collect()
    return {
        "session_creation_seconds": session_creation_seconds,
        "first_run_seconds": first_run_seconds,
        "warm_run_count": warm_runs,
        "warm_run_median_seconds": statistics.median(repeated_seconds),
        "process_peak_rss_bytes": peak_rss_bytes,
        "output": output,
    }


def benchmark(source_path: Path, report_path: Path, node_name: str | None, warm_runs: int) -> dict:
    source_path = source_path.resolve()
    report_path = report_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if report_path.exists():
        raise FileExistsError(f"report path already exists: {report_path}")
    try:
        report_path.relative_to(source_path.parent)
    except ValueError:
        pass
    else:
        raise ValueError("profile report must be outside the source model directory")
    if warm_runs <= 0:
        raise ValueError("warm_runs must be positive")

    source_model = onnx.load(str(source_path), load_external_data=False)
    selected_node, weight = _select_weight(source_path, source_model, node_name)
    source = _matmul_model(weight, source_model, selected_node)
    candidate = rewrite_matmul_to_tritpack_storage(source, node_name=selected_node)
    onnx.checker.check_model(source)
    onnx.checker.check_model(candidate)

    activation = np.random.default_rng(2028).normal(size=(2, weight.shape[0])).astype(np.float32)
    baseline_profile = _profile_session(source, activation, warm_runs=warm_runs)
    with tempfile.TemporaryDirectory(prefix="tritpack-profile-") as temporary_dir:
        optimized_path = Path(temporary_dir) / "optimized.onnx"
        candidate_profile = _profile_session(
            candidate,
            activation,
            warm_runs=warm_runs,
            optimized_model_path=optimized_path,
        )
        optimized_model = onnx.load(str(optimized_path), load_external_data=False)
        optimized_nodes = list(optimized_model.graph.node)
        optimized_matmuls = [node for node in optimized_nodes if node.op_type == "MatMul"]
        optimized_initializers = {tensor.name for tensor in optimized_model.graph.initializer}
        optimized_matmul_weight_is_initializer = bool(
            optimized_matmuls and optimized_matmuls[0].input[1] in optimized_initializers
        )
        remaining_reconstruction_nodes = [
            node.op_type for node in optimized_nodes if node.op_type in _RECONSTRUCTION_OPS
        ]
        optimized_graph_written = optimized_path.is_file()

    baseline_output = baseline_profile.pop("output")
    candidate_output = candidate_profile.pop("output")
    max_absolute_error = float(np.max(np.abs(candidate_output - baseline_output)))
    parity_passed = bool(np.allclose(candidate_output, baseline_output, rtol=1e-4, atol=2e-4))
    if not parity_passed:
        raise RuntimeError(f"one-MatMul parity failed; max absolute error={max_absolute_error}")

    report = {
        "format": "kitten2-onnx-tritpack-one-matmul-profile-v1",
        "source_model": str(source_path),
        "source_model_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "selected_node": selected_node,
        "weight_shape": list(weight.shape),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "provider": "CPUExecutionProvider",
        "checker_passed": True,
        "standard_matmul_count": sum(node.op_type == "MatMul" for node in candidate.graph.node),
        "matmul_nbits_count": sum(node.op_type == "MatMulNBits" for node in candidate.graph.node),
        "parity": {
            "passed": parity_passed,
            "rtol": 1e-4,
            "atol": 2e-4,
            "max_absolute_error": max_absolute_error,
        },
        "baseline": baseline_profile,
        "candidate": candidate_profile,
        "optimized_graph": {
            "written_to_disposable_temporary_file": optimized_graph_written,
            "node_types": [node.op_type for node in optimized_nodes],
            "matmul_weight_is_initializer": optimized_matmul_weight_is_initializer,
            "remaining_reconstruction_nodes": remaining_reconstruction_nodes,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_model", type=Path)
    parser.add_argument("--node-name")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warm-runs", type=int, default=20)
    args = parser.parse_args()
    report = benchmark(args.source_model, args.report, args.node_name, args.warm_runs)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
