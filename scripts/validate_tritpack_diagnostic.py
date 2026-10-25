"""Structurally validate a packed tritpack diagnostic bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from kitten2_onnx.external_data import logical_tensor_bytes, resolve_external_data_range


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"external-data range ended early: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _tensor_sha256(bundle_lm: Path, tensor: onnx.TensorProto) -> str:
    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        external = resolve_external_data_range(
            bundle_lm,
            tensor,
            logical_bytes=logical_tensor_bytes(tensor),
            require_exact_length=True,
        )
        return _sha256_range(external.path, external.offset, external.length)
    values = np.ascontiguousarray(numpy_helper.to_array(tensor))
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def validate(bundle_path: Path, inventory_path: Path, report_path: Path) -> dict[str, Any]:
    bundle_path = bundle_path.resolve()
    manifest = json.loads((bundle_path / "bundle.json").read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    file_hashes = manifest["files"]
    actual_files = {
        path.relative_to(bundle_path).as_posix()
        for path in bundle_path.rglob("*")
        if path.is_file() and path.name != "bundle.json"
    }
    if actual_files != set(file_hashes):
        raise ValueError(
            f"bundle file list differs: missing={sorted(set(file_hashes) - actual_files)}, "
            f"unlisted={sorted(actual_files - set(file_hashes))}"
        )
    for relative_path, expected_hash in file_hashes.items():
        if _sha256_file(bundle_path / relative_path) != expected_hash:
            raise ValueError(f"bundle file hash mismatch: {relative_path}")

    storage = manifest["lm"]["storage"]["transformer"]
    if (
        storage.get("method") != "ternary-tritpack-exact"
        or storage.get("format_version") != report["format_version"]
        or storage.get("transformed_matmul_count") != 196
        or storage.get("transformed_initializer_count") != 196
        or storage.get("graph_sha256") != report["graph_sha256"]
        or len(storage.get("weights", [])) != 196
        or storage.get("payload_sha256") != report["external_data_sha256"]
        or storage.get("source_group_size") != 128
        or storage.get("trits_per_byte") != 5
        or storage.get("packed_bytes_per_group") != 26
        or storage.get("runtime_materialization_required") is not True
        or storage.get("runtime_memory_profile") != "low-memory-exact-ternary"
        or storage.get("logical_fp32_operand_bytes") != report["logical_fp32_operand_bytes"]
        or storage.get("materialized_external_bytes") != report["logical_fp32_operand_bytes"]
    ):
        raise ValueError("bundle tritpack metadata does not match the conversion report")
    if (
        report["source_graph_sha256"] != inventory["source_graph_sha256"]
        or report["source_external_data_sha256"] != inventory["source_external_data_sha256"]
        or report["inventory_sha256"] != inventory["inventory_sha256"]
        or report["transformed_matmul_count"] != 196
        or report["storage_reconstruction_verified"] is not True
    ):
        raise ValueError("conversion report does not match the 196-MatMul source inventory")

    source_graph = Path(inventory["source_graph"])
    if _sha256_file(source_graph) != inventory["source_graph_sha256"]:
        raise ValueError("source graph hash differs from the exact ternary inventory")
    model_path = bundle_path / manifest["lm"]["model"]
    model = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(str(model_path))
    nodes_by_name = {node.name: node for node in model.graph.node if node.name}
    if any(node.op_type == "MatMulNBits" for node in model.graph.node):
        raise ValueError("tritpack diagnostic contains MatMulNBits")

    eligible_rows = [row for row in inventory["nodes"] if row["eligible_exact_ternary"]]
    eligible_names = {row["node"] for row in eligible_rows}
    if len(eligible_rows) != 196 or set(report["transformed_nodes"]) != eligible_names:
        raise ValueError("diagnostic does not account for all 196 eligible MatMuls")
    storage_weights = {
        entry["source_initializer"]: entry for entry in storage.get("weights", [])
    }
    if len(storage_weights) != len(report["initializer_reports"]):
        raise ValueError("tritpack weight metadata contains duplicate or missing initializers")
    for entry in report["initializer_reports"]:
        weight = storage_weights.get(entry["source_initializer"])
        if weight is None or any(
            weight.get(key) != entry[key]
            for key in (
                "source_shape",
                "source_sha256",
                "code_initializer",
                "scale_initializer",
                "transformed_nodes",
            )
        ):
            raise ValueError(f"tritpack weight metadata differs: {entry['source_initializer']}")
        first_node = nodes_by_name.get(entry["transformed_nodes"][0])
        if first_node is None or weight.get("weight_value") != first_node.input[1]:
            raise ValueError(f"tritpack weight value metadata differs: {entry['source_initializer']}")
        if any(
            nodes_by_name[node_name].input[1] != weight["weight_value"]
            for node_name in entry["transformed_nodes"]
        ):
            raise ValueError(f"tritpack MatMul weight value differs: {entry['source_initializer']}")
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    target_initializers = {row["initializer"] for row in eligible_rows}
    if target_initializers & initializer_names:
        raise ValueError("diagnostic retains a source FP32 ternary initializer")
    for row in eligible_rows:
        node = nodes_by_name.get(row["node"])
        if node is None or node.op_type != "MatMul" or len(node.input) != 2:
            raise ValueError(f"transformed standard MatMul is missing: {row['node']}")
        if node.input[1] == row["initializer"]:
            raise ValueError(f"standard MatMul still uses its source initializer: {row['node']}")

    source_hashes = {row["source_sha256"] for row in eligible_rows}
    payload_relative = Path(storage["payload_file"])
    if payload_relative.is_absolute() or ".." in payload_relative.parts:
        raise ValueError("tritpack payload path must remain relative to the LM graph")
    payload_file = (model_path.parent / payload_relative).resolve()
    if not payload_file.is_relative_to(bundle_path):
        raise ValueError("tritpack payload path escapes the diagnostic bundle")
    if _sha256_file(payload_file) != report["external_data_sha256"]:
        raise ValueError("tritpack payload hash differs from the conversion report")
    packed_code_bytes = 0
    scale_bytes = 0
    for entry in report["initializer_reports"]:
        codes = next(tensor for tensor in model.graph.initializer if tensor.name == entry["code_initializer"])
        scales = next(tensor for tensor in model.graph.initializer if tensor.name == entry["scale_initializer"])
        code_slice = resolve_external_data_range(
            model_path.parent,
            codes,
            logical_bytes=logical_tensor_bytes(codes),
            require_exact_length=True,
        )
        scale_slice = resolve_external_data_range(
            model_path.parent,
            scales,
            logical_bytes=logical_tensor_bytes(scales),
            require_exact_length=True,
        )
        if (
            codes.data_type != TensorProto.UINT8
            or scales.data_type != TensorProto.FLOAT
            or code_slice.path != payload_file.resolve()
            or scale_slice.path != payload_file.resolve()
            or code_slice.length != entry["code_bytes"]
            or scale_slice.length != entry["scale_bytes"]
            or _sha256_range(code_slice.path, code_slice.offset, code_slice.length)
            != entry["code_sha256"]
            or _sha256_range(scale_slice.path, scale_slice.offset, scale_slice.length)
            != entry["scale_sha256"]
        ):
            raise ValueError(f"tritpack payload tensor verification failed: {entry['source_initializer']}")
        packed_code_bytes += code_slice.length
        scale_bytes += scale_slice.length
    if packed_code_bytes != report["packed_code_bytes"] or scale_bytes != report["scale_bytes"]:
        raise ValueError("tritpack payload byte totals differ from the conversion report")

    candidate_float_hashes = {
        _tensor_sha256(model_path.parent, tensor)
        for tensor in model.graph.initializer
        if tensor.data_type == TensorProto.FLOAT
    }
    if source_hashes & candidate_float_hashes:
        raise ValueError("diagnostic contains a duplicate full FP32 source weight")

    return {
        "status": "passed",
        "bundle": str(bundle_path),
        "graph_sha256": _sha256_file(model_path),
        "external_data_sha256": _sha256_file(payload_file),
        "transformed_matmul_count": len(eligible_rows),
        "transformed_initializer_count": len(report["initializer_reports"]),
        "packed_code_bytes": packed_code_bytes,
        "scale_bytes": scale_bytes,
        "logical_fp32_operand_bytes": report["logical_fp32_operand_bytes"],
        "bundle_file_count": len(file_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("conversion_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.bundle, args.inventory, args.conversion_report)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
