"""Measure same-activation error for every exact NBits projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from kitten2_onnx.external_data import logical_tensor_bytes, resolve_external_data_range
from kitten2_onnx.lm_parity_reference import (
    CACHE_ATOL,
    CACHE_RTOL,
    compare_tensors,
    load_reference_pack,
)
from kitten2_onnx.quantize_ternary_nbits import (
    _array_sha256,
    _close_memmap,
    _load_float32_weight,
)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _node_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {
        output: node
        for node in model.graph.node
        if node.op_type == "MatMulNBits" and node.domain == "com.microsoft"
        for output in node.output
    }


def _link_parent_data(parent_path: Path, model: onnx.ModelProto, destination: Path) -> list[Path]:
    locations = set()
    for tensor in model.graph.initializer:
        if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
            locations.add(
                resolve_external_data_range(
                    parent_path.parent,
                    tensor,
                    logical_bytes=logical_tensor_bytes(tensor),
                ).location
            )
    linked = []
    try:
        for location in sorted(locations):
            source = (parent_path.parent / location).resolve()
            target = destination / location
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if not target.is_file() or not os.path.samefile(target, source):
                    raise FileExistsError(f"Sensitivity external-data path conflicts with parent data: {target}")
                continue
            os.link(source, target)
            linked.append(target)
    except BaseException:
        for path in linked:
            path.unlink(missing_ok=True)
        raise
    return linked


def _prefill_feed(reference) -> dict[str, np.ndarray]:
    metadata = reference.metadata
    lm_metadata = metadata["lm_metadata"]
    input_mode = lm_metadata.get("input_mode", "inputs_embeds")
    past_names = [name for name in lm_metadata["input_names"] if name.startswith("past_")]
    kv_heads = int(metadata["num_key_value_heads"])
    head_dim = int(metadata["head_dim"])
    return {
        "attention_mask": np.ones((1, reference.prompt_length), dtype=np.int64),
        "position_ids": np.arange(reference.prompt_length, dtype=np.int64)[None],
        **{
            name: np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32)
            for name in past_names
        },
        **reference.prefill_inputs(input_mode),
    }


def measure_sensitivity(
    parent_model: str | Path,
    baseline_model: str | Path,
    inventory_path: str | Path,
    reference_path: str | Path,
    output_model: str | Path,
) -> dict[str, Any]:
    parent_path = Path(parent_model).resolve()
    baseline_path = Path(baseline_model).resolve()
    inventory_file = Path(inventory_path).resolve()
    instrumented_path = Path(output_model).resolve()
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    if inventory.get("source_graph_sha256") != hashlib.sha256(baseline_path.read_bytes()).hexdigest():
        raise ValueError("Inventory source graph hash does not match baseline model")

    parent = onnx.load(str(parent_path), load_external_data=False)
    baseline = onnx.load(str(baseline_path), load_external_data=False)
    parent_nodes = _node_map(parent)
    baseline_initializers = {tensor.name: tensor for tensor in baseline.graph.initializer}
    parent_initializers = {tensor.name: tensor for tensor in parent.graph.initializer}
    values = {
        value.name: value
        for value in (*parent.graph.input, *parent.graph.output, *parent.graph.value_info)
    }
    output_names = sorted(
        {
            name
            for row in inventory["nodes"]
            if row.get("eligible_exact_ternary")
            for name in (row["activation"], row["output"])
        }
    )
    missing = [name for name in output_names if name not in values]
    if missing:
        raise ValueError(f"NBits sensitivity values have no type information: {missing[:5]}")

    instrumented = onnx.ModelProto()
    instrumented.CopyFrom(parent)
    declared_outputs = {value.name for value in instrumented.graph.output}
    for name in output_names:
        if name not in declared_outputs:
            instrumented.graph.output.add().CopyFrom(values[name])
            declared_outputs.add(name)

    instrumented_path.parent.mkdir(parents=True, exist_ok=True)
    linked_data = _link_parent_data(parent_path, instrumented, instrumented_path.parent)
    temporary_path = instrumented_path.with_name(f".{instrumented_path.name}.tmp")
    try:
        onnx.save_model(instrumented, str(temporary_path), save_as_external_data=False)
        os.replace(temporary_path, instrumented_path)
        import onnxruntime as ort

        reference = load_reference_pack(reference_path)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(instrumented_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        outputs = session.run(output_names, _prefill_feed(reference))
        values_by_name = dict(zip(output_names, outputs, strict=True))
        sensitivity_rows = []
        for row in inventory["nodes"]:
            if not row.get("eligible_exact_ternary"):
                continue
            node_name = row["node"]
            node = parent_nodes.get(row["output"])
            if node is None or node.input[0] != row["activation"]:
                raise ValueError(f"Parent NBits node does not match inventory row: {node_name}")
            initializer_name = row["initializer"]
            source_tensor = baseline_initializers.get(initializer_name)
            if source_tensor is None:
                raise ValueError(f"Baseline initializer is missing: {initializer_name}")
            weight, is_memmap = _load_float32_weight(baseline_path, source_tensor)
            try:
                source_hash = _array_sha256(weight)
                if source_hash != row["source_sha256"]:
                    raise ValueError(f"Baseline initializer hash mismatch: {initializer_name}")
                expected = np.matmul(values_by_name[row["activation"]], weight)
                actual = values_by_name[row["output"]]
                comparison = compare_tensors(actual, expected, atol=CACHE_ATOL, rtol=CACHE_RTOL)
            finally:
                _close_memmap(weight, is_memmap)
            auxiliary_bytes = sum(
                logical_tensor_bytes(parent_initializers[name])
                for name in node.input[1:4]
                if name in parent_initializers
            )
            fallback_bytes = logical_tensor_bytes(source_tensor)
            local_max = float(comparison["max_normalized_tolerance_violation"])
            outside_count = int(comparison["outside_tolerance_count"])
            sensitivity_rows.append(
                {
                    "node": node_name,
                    "layer": row.get("layer"),
                    "module": row.get("module"),
                    "initializer": initializer_name,
                    "shape": list(source_tensor.dims),
                    "local_max_abs_error": float(comparison["max_absolute_error"]),
                    "local_normalized_error": local_max,
                    "output_error_count": outside_count,
                    "fallback_byte_cost": fallback_bytes,
                    "nbits_bytes_if_quantized": auxiliary_bytes,
                    "priority": local_max * max(outside_count, 1) / max(fallback_bytes, 1),
                }
            )
        sensitivity_rows.sort(
            key=lambda item: (
                -item["priority"],
                -item["local_normalized_error"],
                item["fallback_byte_cost"],
                item["node"],
            )
        )
        return {
            "format": "kitten2-onnx-nbits-sensitivity-v1",
            "status": "sensitivity_complete",
            "parent_graph_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            "baseline_graph_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "reference_pack_sha256": reference.metadata["pack_sha256"],
            "tolerance": {"atol": CACHE_ATOL, "rtol": CACHE_RTOL},
            "measured_node_count": len(sensitivity_rows),
            "ranked_nodes": sensitivity_rows,
        }
    finally:
        temporary_path.unlink(missing_ok=True)
        instrumented_path.unlink(missing_ok=True)
        for path in linked_data:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--instrumented-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure_sensitivity(
        args.parent_model,
        args.baseline_model,
        args.inventory,
        args.reference,
        args.instrumented_model,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "ranked_nodes"}, indent=2))


if __name__ == "__main__":
    main()
