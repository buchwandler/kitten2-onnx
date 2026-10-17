"""Map ONNX MatMul weights to checkpoint modules by exact tensor identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto
from safetensors import safe_open

from kitten2_onnx.quantize_ternary_nbits import (
    SOURCE_GROUP_SIZE,
    _array_sha256,
    _close_memmap,
    _is_exact_ternary_weight,
    _load_float32_weight,
)

_MODULE_ORDER = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_LAYER_WEIGHT = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")
_LINEAR_NODE = re.compile(r"^node_linear(?:_(\d+))?$")


def _checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    plain = path / "model.safetensors"
    if plain.is_file():
        return [plain]
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"No SafeTensors checkpoint found under {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_identity(key: str) -> tuple[int | None, str]:
    match = _LAYER_WEIGHT.fullmatch(key)
    if match:
        return int(match.group(1)), match.group(2)
    return None, key.removeprefix("model.").removesuffix(".weight")


def _checkpoint_hashes(checkpoint: Path) -> dict[str, list[dict[str, Any]]]:
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[str] = set()
    for shard in _checkpoint_files(checkpoint):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                if not key.endswith(".weight"):
                    continue
                if key in seen_keys:
                    raise ValueError(f"Duplicate checkpoint tensor across shards: {key}")
                seen_keys.add(key)
                tensor = handle.get_tensor(key)
                if tensor.ndim != 2:
                    continue
                fp32_weight = tensor.to(dtype=torch.float32).numpy()
                onnx_weight = np.ascontiguousarray(fp32_weight.T)
                digest = _array_sha256(onnx_weight)
                layer, module = _checkpoint_identity(key)
                hashes[digest].append(
                    {
                        "checkpoint_key": key,
                        "shape": list(tensor.shape),
                        "layer": layer,
                        "module": module,
                    }
                )
                del onnx_weight, fp32_weight, tensor
    return hashes


def _linear_neighborhood(node_name: str) -> tuple[int, str] | None:
    match = _LINEAR_NODE.fullmatch(node_name)
    if not match:
        return None
    ordinal = int(match.group(1) or 0)
    if ordinal >= 196:
        return None
    return ordinal // len(_MODULE_ORDER), _MODULE_ORDER[ordinal % len(_MODULE_ORDER)]


def _resolve_checkpoint_match(
    node_name: str,
    source_shape: tuple[int, int],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    expected_checkpoint_shape = [source_shape[1], source_shape[0]]
    matching_shapes = [row for row in candidates if row["shape"] == expected_checkpoint_shape]
    if len(matching_shapes) == 1:
        return matching_shapes[0], "sha256+shape"
    if len(matching_shapes) <= 1:
        return None, None

    neighborhood = _linear_neighborhood(node_name)
    if neighborhood is not None:
        layer, module = neighborhood
        contextual = [
            row
            for row in matching_shapes
            if row["layer"] == layer and row["module"] == module
        ]
        if len(contextual) == 1:
            return contextual[0], "sha256+shape+layer-neighborhood"
    return None, None


def _source_external_hashes(model_path: Path, model: onnx.ModelProto) -> dict[str, str]:
    locations = {
        entry.value
        for tensor in model.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    hashes = {}
    for location in sorted(locations):
        relative = Path(location)
        if relative.is_absolute():
            raise ValueError(f"ONNX external-data locations must be relative: {location}")
        path = (model_path.parent / relative).resolve()
        if not path.is_relative_to(model_path.parent.resolve()):
            raise ValueError(f"ONNX external-data path escapes graph directory: {location}")
        hashes[location] = _sha256_file(path)
    return hashes


def build_inventory(
    source_graph: str | Path,
    checkpoint: str | Path,
    *,
    expected_eligible_matmul_count: int = 196,
) -> dict[str, Any]:
    graph_path = Path(source_graph).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    model = onnx.load(str(graph_path), load_external_data=False)
    initializer_by_name = {tensor.name: tensor for tensor in model.graph.initializer}
    checkpoint_hashes = _checkpoint_hashes(checkpoint_path)
    rows = []
    eligible_rows = []
    unresolved = []
    ambiguous = []

    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "MatMul":
            continue
        node_name = node.name or f"MatMul#{node_index}"
        initializer_name = node.input[1] if len(node.input) > 1 else ""
        tensor = initializer_by_name.get(initializer_name)
        source_shape: tuple[int, int] | None = None
        source_sha256 = None
        eligible = False
        checkpoint_match = None
        mapping_method = None
        if tensor is not None and tensor.data_type == TensorProto.FLOAT and len(tensor.dims) == 2:
            weight, is_memmap = _load_float32_weight(graph_path, tensor)
            try:
                source_shape = tuple(int(dimension) for dimension in tensor.dims)
                eligible, exact_result = _is_exact_ternary_weight(weight)
                source_sha256 = exact_result if eligible else _array_sha256(weight)
            finally:
                _close_memmap(weight, is_memmap)
            candidates = checkpoint_hashes.get(source_sha256, [])
            checkpoint_match, mapping_method = _resolve_checkpoint_match(
                node_name, source_shape, candidates
            )
            if eligible and checkpoint_match is None:
                if candidates:
                    ambiguous.append(
                        {
                            "node": node_name,
                            "initializer": initializer_name,
                            "candidate_keys": [row["checkpoint_key"] for row in candidates],
                        }
                    )
                else:
                    unresolved.append(
                        {
                            "node": node_name,
                            "initializer": initializer_name,
                            "source_sha256": source_sha256,
                        }
                    )

        row = {
            "node": node_name,
            "initializer": initializer_name or None,
            "shape": list(source_shape) if source_shape is not None else None,
            "source_sha256": source_sha256,
            "eligible_exact_ternary": eligible,
            "layer": checkpoint_match["layer"] if checkpoint_match else None,
            "module": checkpoint_match["module"] if checkpoint_match else None,
            "checkpoint_key": checkpoint_match["checkpoint_key"] if checkpoint_match else None,
            "mapping_method": mapping_method,
            "activation": node.input[0] if node.input else None,
            "output": node.output[0] if node.output else None,
        }
        rows.append(row)
        if eligible:
            eligible_rows.append(row)

    if unresolved or ambiguous:
        raise ValueError(
            "Could not map all exact ternary MatMuls to checkpoint weights: "
            f"unresolved={unresolved}, ambiguous={ambiguous}"
        )
    if len(eligible_rows) != expected_eligible_matmul_count:
        raise ValueError(
            f"Expected {expected_eligible_matmul_count} exact ternary MatMuls, "
            f"found {len(eligible_rows)}"
        )

    modules_by_layer: dict[int, list[str]] = defaultdict(list)
    for row in eligible_rows:
        if row["layer"] is not None and row["module"] is not None:
            modules_by_layer[row["layer"]].append(row["module"])
    layer_rows = [
        {"layer": layer, "modules": sorted(modules)}
        for layer, modules in sorted(modules_by_layer.items())
    ]
    repeated_seven_matrix_structure = (
        len(modules_by_layer) == 28
        and all(
            len(modules) == 7 and set(modules) == set(_MODULE_ORDER)
            for modules in modules_by_layer.values()
        )
    )
    if expected_eligible_matmul_count == 196 and not repeated_seven_matrix_structure:
        raise ValueError("Exact ternary inventory does not confirm seven mapped matrices per layer")

    source_graph_sha256 = _sha256_file(graph_path)
    source_external_data_sha256 = _source_external_hashes(graph_path, model)
    checkpoint_files = [
        {"path": str(path), "sha256": _sha256_file(path)}
        for path in _checkpoint_files(checkpoint_path)
    ]
    inventory_identity = {
        "format": "kitten2-onnx-ternary-nbits-inventory-v1",
        "source_group_size": SOURCE_GROUP_SIZE,
        "source_graph_sha256": source_graph_sha256,
        "source_external_data_sha256": source_external_data_sha256,
        "checkpoint_sha256s": [row["sha256"] for row in checkpoint_files],
        "expected_eligible_matmul_count": expected_eligible_matmul_count,
        "nodes": rows,
    }
    inventory_json = json.dumps(
        inventory_identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "format": "kitten2-onnx-ternary-nbits-inventory-v1",
        "source_group_size": SOURCE_GROUP_SIZE,
        "source_graph": str(graph_path),
        "source_graph_sha256": source_graph_sha256,
        "source_external_data_sha256": source_external_data_sha256,
        "checkpoint_files": checkpoint_files,
        "matmul_count": len(rows),
        "eligible_exact_ternary_matmul_count": len(eligible_rows),
        "mapped_eligible_matmul_count": sum(row["checkpoint_key"] is not None for row in eligible_rows),
        "ambiguous_mapping_count": len(ambiguous),
        "repeated_seven_matrix_structure": repeated_seven_matrix_structure,
        "layers": layer_rows,
        "inventory_sha256": hashlib.sha256(inventory_json).hexdigest(),
        "nodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_graph", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-eligible-matmul-count", type=int, default=196)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/reports/ternary-nbits-inventory.json"),
    )
    args = parser.parse_args()
    report = build_inventory(
        args.source_graph,
        args.checkpoint,
        expected_eligible_matmul_count=args.expected_eligible_matmul_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: {report['eligible_exact_ternary_matmul_count']} eligible MatMuls, "
        f"inventory SHA-256 {report['inventory_sha256']}"
    )


if __name__ == "__main__":
    main()
