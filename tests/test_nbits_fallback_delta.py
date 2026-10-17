from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.external_data import make_external_tensor
from kitten2_onnx.nbits_fallback_delta import build_fallback_delta
from kitten2_onnx.quantize_ternary_nbits import rewrite_matmul_to_nbits4


def _weight() -> np.ndarray:
    rng = np.random.default_rng(73)
    signs = rng.choice(np.asarray([-1, 0, 1], dtype=np.float32), size=(128, 3))
    scales = np.asarray([0.25, 0.5, 1.75], dtype=np.float32)
    return np.ascontiguousarray(signs * scales)


def _model(weight: np.ndarray, *, shared: bool, unused: bool) -> onnx.ModelProto:
    nodes = [helper.make_node("MatMul", ["A", "W"], ["Y1"], name="first")]
    outputs = [helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [2, 3])]
    if shared:
        nodes.append(helper.make_node("MatMul", ["A", "W"], ["Y2"], name="second"))
        outputs.append(helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [2, 3]))
    initializers = [numpy_helper.from_array(weight, name="W")]
    if unused:
        initializers.append(numpy_helper.from_array(np.asarray([2.0], dtype=np.float32), name="unused"))
    graph = helper.make_graph(
        nodes,
        "fallback-delta-test",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 128])],
        outputs,
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model


def _save_external(model: onnx.ModelProto, path: Path, location: str) -> None:
    payload = bytearray()
    for tensor in model.graph.initializer:
        array = np.ascontiguousarray(numpy_helper.to_array(tensor))
        data = memoryview(array).cast("B")
        offset = len(payload)
        payload.extend(data)
        tensor.CopyFrom(
            make_external_tensor(
                tensor.name,
                tensor.data_type,
                tuple(tensor.dims),
                location,
                offset,
                len(data),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / location).write_bytes(payload)
    onnx.save_model(model, str(path))


def _inventory_row(node: str, output: str, weight: np.ndarray) -> dict[str, object]:
    return {
        "node": node,
        "initializer": "W",
        "activation": "A",
        "output": output,
        "shape": list(weight.shape),
        "source_sha256": hashlib.sha256(weight.tobytes(order="C")).hexdigest(),
        "eligible_exact_ternary": True,
    }


def _build_case(tmp_path: Path, *, shared: bool, unused: bool):
    weight = _weight()
    baseline_dir = tmp_path / "baseline"
    parent_dir = tmp_path / "parent"
    baseline_path = baseline_dir / "lm.onnx"
    parent_path = parent_dir / "lm.onnx"
    baseline = _model(weight, shared=shared, unused=unused)
    _save_external(baseline, baseline_path, "baseline.data")

    parent = rewrite_matmul_to_nbits4(_model(weight, shared=False, unused=unused))
    if shared:
        second = copy.deepcopy(parent.graph.node[0])
        second.name = "second_nbits4"
        second.output[0] = "Y2"
        parent.graph.node.append(second)
        parent.graph.output.append(helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [2, 3]))
    _save_external(parent, parent_path, "parent.data")
    rows = [_inventory_row("first", "Y1", weight)]
    if shared:
        rows.append(_inventory_row("second", "Y2", weight))
    inventory = {
        "source_graph_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "inventory_sha256": "test-inventory",
        "nodes": rows,
    }
    return weight, baseline_path, parent_path, inventory


def _run(path: Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, inputs)


def test_delta_restores_selected_node_streams_data_and_prunes_only_unused_nbits_initializers(
    tmp_path: Path,
):
    weight, baseline_path, parent_path, inventory = _build_case(
        tmp_path, shared=False, unused=True
    )
    protected_before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            baseline_path,
            baseline_path.parent / "baseline.data",
            parent_path,
            parent_path.parent / "parent.data",
        )
    }
    candidate_path = tmp_path / "candidate" / "lm.onnx"
    fallback_path = candidate_path.parent / "fallback.data"

    delta = build_fallback_delta(
        parent_path,
        baseline_path,
        inventory,
        {"first"},
        candidate_path,
        fallback_path,
    )

    candidate = onnx.load(str(candidate_path), load_external_data=False)
    assert [node.op_type for node in candidate.graph.node] == ["MatMul"]
    initializer_names = {tensor.name for tensor in candidate.graph.initializer}
    assert "W" in initializer_names
    assert "unused" in initializer_names
    assert not {"W_nbits4", "W_nbits4_scales", "W_nbits4_zero_points"} & initializer_names
    assert (candidate_path.parent / "parent.data").samefile(parent_path.parent / "parent.data")
    assert delta.fallback_storage["fallback_initializer_count"] == 1
    assert delta.fallback_storage["fallback_fp32_bytes"] == weight.nbytes
    assert delta.fallback_storage["initializers"][0]["nbits_bytes_if_quantized"] == 207
    assert delta.fallback_storage["nbits_bytes_removed"] == 207
    assert delta.fallback_storage["net_bundle_byte_cost"] == weight.nbytes - 207
    assert delta.source_hashes["fallback_initializers"]["W"] == inventory["nodes"][0]["source_sha256"]

    activation = np.random.default_rng(91).normal(size=(2, 128)).astype(np.float32)
    expected = _run(baseline_path, {"A": activation})
    actual = _run(candidate_path, {"A": activation})
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_before
    } == protected_before


def test_delta_keeps_shared_nbits_initializers_until_last_consumer_is_restored(tmp_path: Path):
    weight, baseline_path, parent_path, inventory = _build_case(
        tmp_path, shared=True, unused=False
    )
    activation = np.random.default_rng(92).normal(size=(2, 128)).astype(np.float32)
    expected = _run(baseline_path, {"A": activation})

    partial_path = tmp_path / "partial" / "lm.onnx"
    partial = build_fallback_delta(
        parent_path,
        baseline_path,
        inventory,
        {"first"},
        partial_path,
        partial_path.parent / "fallback.data",
    )
    partial_graph = onnx.load(str(partial_path), load_external_data=False)
    assert [node.op_type for node in partial_graph.graph.node] == ["MatMul", "MatMulNBits"]
    assert partial.fallback_storage["nbits_bytes_removed"] == 0
    actual_partial = _run(partial_path, {"A": activation})
    for actual, reference in zip(actual_partial, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)

    full_path = tmp_path / "full" / "lm.onnx"
    full = build_fallback_delta(
        parent_path,
        baseline_path,
        inventory,
        {"first", "second"},
        full_path,
        full_path.parent / "fallback.data",
    )
    full_graph = onnx.load(str(full_path), load_external_data=False)
    assert [node.op_type for node in full_graph.graph.node] == ["MatMul", "MatMul"]
    assert sum(tensor.name == "W" for tensor in full_graph.graph.initializer) == 1
    assert full.fallback_storage["fallback_initializer_count"] == 1
    assert full.fallback_storage["fallback_fp32_bytes"] == weight.nbytes
    assert full.fallback_storage["nbits_bytes_removed"] == 207
    actual_full = _run(full_path, {"A": activation})
    for actual, reference in zip(actual_full, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)


def test_delta_rejects_unknown_nodes_and_baseline_hash_mismatch(tmp_path: Path):
    _, baseline_path, parent_path, inventory = _build_case(
        tmp_path, shared=False, unused=False
    )
    with pytest.raises(ValueError, match="Unknown eligible inventory nodes"):
        build_fallback_delta(
            parent_path,
            baseline_path,
            inventory,
            {"missing"},
            tmp_path / "unknown" / "lm.onnx",
            tmp_path / "unknown" / "fallback.data",
        )

    inventory["source_graph_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="source graph hash"):
        build_fallback_delta(
            parent_path,
            baseline_path,
            inventory,
            {"first"},
            tmp_path / "hash" / "lm.onnx",
            tmp_path / "hash" / "fallback.data",
        )
