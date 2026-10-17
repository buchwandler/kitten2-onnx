"""Build disposable ONNX candidates that restore selected NBits MatMuls."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

from kitten2_onnx.external_data import (
    ExternalTensorSlice,
    logical_tensor_bytes,
    make_external_tensor,
    resolve_external_data_range,
)
from kitten2_onnx.quantize_int8_streaming import ExternalDataWriter

_COPY_BUFFER_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class FallbackDelta:
    """Paths and evidence for one disposable fallback candidate."""

    graph_path: Path
    fallback_data_path: Path
    manifest_path: Path
    excluded_nodes: tuple[str, ...]
    fallback_storage: dict[str, Any]
    source_hashes: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_COPY_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_range(tensor_slice: ExternalTensorSlice) -> str:
    digest = hashlib.sha256()
    remaining = tensor_slice.length
    with tensor_slice.path.open("rb") as stream:
        stream.seek(tensor_slice.offset)
        while remaining:
            block = stream.read(min(_COPY_BUFFER_SIZE, remaining))
            if not block:
                raise ValueError(f"External-data file ended early: {tensor_slice.path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _copy_range(
    tensor_slice: ExternalTensorSlice,
    writer: ExternalDataWriter,
) -> tuple[int, int, str]:
    offset = writer.start_tensor()
    digest = hashlib.sha256()
    copied = 0
    remaining = tensor_slice.length
    with tensor_slice.path.open("rb") as stream:
        stream.seek(tensor_slice.offset)
        while remaining:
            block = stream.read(min(_COPY_BUFFER_SIZE, remaining))
            if not block:
                raise ValueError(f"External-data file ended early: {tensor_slice.path}")
            writer.write(block)
            digest.update(block)
            copied += len(block)
            remaining -= len(block)
    actual_offset, length = writer.finish_tensor(offset)
    if copied != length:
        raise ValueError("Copied fallback initializer has an unexpected byte count")
    return actual_offset, length, digest.hexdigest()


def _inventory_nodes(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in inventory.get("nodes", []):
        if row.get("eligible_exact_ternary"):
            name = row.get("node")
            if not name or name in rows:
                raise ValueError(f"Inventory has a missing or duplicate eligible node name: {name!r}")
            rows[name] = row
    return rows


def _baseline_nodes(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    rows = {}
    for index, node in enumerate(model.graph.node):
        if node.op_type == "MatMul":
            name = node.name or f"MatMul#{index}"
            if name in rows:
                raise ValueError(f"Baseline graph has duplicate MatMul node name: {name}")
            rows[name] = node
    return rows


def _parent_node_for(
    parent_nodes: list[onnx.NodeProto],
    baseline_node: onnx.NodeProto,
    inventory_row: dict[str, Any],
) -> tuple[int, onnx.NodeProto]:
    expected_output = tuple(baseline_node.output)
    expected_activation = baseline_node.input[0]
    matches = [
        (index, node)
        for index, node in enumerate(parent_nodes)
        if node.op_type == "MatMulNBits"
        and node.domain == "com.microsoft"
        and tuple(node.output) == expected_output
        and node.input
        and node.input[0] == expected_activation
    ]
    if len(matches) != 1:
        node_name = inventory_row.get("node")
        raise ValueError(
            f"Expected one parent MatMulNBits node for {node_name!r}, found {len(matches)}"
        )
    return matches[0]


def _ensure_parent_external_files(
    parent_path: Path,
    parent: onnx.ModelProto,
    candidate_dir: Path,
) -> list[Path]:
    locations: set[str] = set()
    for tensor in parent.graph.initializer:
        if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
            tensor_slice = resolve_external_data_range(
                parent_path.parent,
                tensor,
                logical_bytes=logical_tensor_bytes(tensor),
            )
            locations.add(tensor_slice.location)

    if candidate_dir.resolve() == parent_path.parent.resolve():
        return []

    created: list[Path] = []
    try:
        for location in sorted(locations):
            destination = candidate_dir / location
            source = (parent_path.parent / location).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                if not destination.is_file() or not os.path.samefile(destination, source):
                    raise FileExistsError(
                        f"Candidate external-data path already exists and is not the parent data file: {destination}"
                    )
                continue
            os.link(source, destination)
            created.append(destination)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return created


def _external_tensor_slice(model_path: Path, tensor: onnx.TensorProto) -> ExternalTensorSlice:
    if tensor.data_location != TensorProto.EXTERNAL and not tensor.external_data:
        raise ValueError(f"Fallback source initializer is not external: {tensor.name}")
    return resolve_external_data_range(
        model_path.parent,
        tensor,
        logical_bytes=logical_tensor_bytes(tensor),
        require_exact_length=True,
    )


def _nbits_auxiliaries(node: onnx.NodeProto) -> tuple[str, ...]:
    if len(node.input) < 4:
        raise ValueError(f"MatMulNBits node has fewer than four inputs: {node.name}")
    return tuple(name for name in node.input[1:4] if name)


def _fallback_storage(
    selected_initializers: dict[str, dict[str, Any]],
    auxiliary_names: dict[str, set[str]],
    parent_initializers: dict[str, onnx.TensorProto],
    removed_names: set[str],
    copied_bytes: dict[str, int],
) -> dict[str, Any]:
    entries = []
    counted_removed: set[str] = set()
    for name in sorted(selected_initializers):
        row = selected_initializers[name]
        auxiliaries = auxiliary_names.get(name, set())
        nbits_bytes_if_quantized = sum(
            logical_tensor_bytes(parent_initializers[auxiliary])
            for auxiliary in auxiliaries
            if auxiliary in parent_initializers
        )
        removed = sorted(auxiliaries & removed_names)
        nbits_bytes_removed = sum(
            logical_tensor_bytes(parent_initializers[auxiliary])
            for auxiliary in removed
            if auxiliary not in counted_removed
        )
        counted_removed.update(removed)
        fallback_fp32_bytes = copied_bytes.get(name, 0)
        entries.append(
            {
                "initializer": name,
                "source_shape": list(row["shape"]),
                "source_sha256": row["source_sha256"],
                "fallback_fp32_bytes": fallback_fp32_bytes,
                "nbits_bytes_if_quantized": nbits_bytes_if_quantized,
                "nbits_bytes_removed": nbits_bytes_removed,
                "net_bundle_byte_cost": fallback_fp32_bytes - nbits_bytes_removed,
            }
        )
    return {
        "fallback_initializer_count": len(entries),
        "fallback_fp32_bytes": sum(row["fallback_fp32_bytes"] for row in entries),
        "nbits_bytes_removed": sum(row["nbits_bytes_removed"] for row in entries),
        "net_bundle_byte_cost": sum(row["net_bundle_byte_cost"] for row in entries),
        "initializers": entries,
    }


def build_fallback_delta(
    parent_model: str | Path,
    baseline_model: str | Path,
    inventory: dict[str, Any],
    excluded_nodes: set[str] | list[str] | tuple[str, ...],
    output_model: str | Path,
    output_fallback_data: str | Path,
) -> FallbackDelta:
    """Restore selected baseline MatMuls in a copy of an immutable NBits parent.

    The candidate graph reuses the parent NBits external data through diagnostic-only
    hard links when it is written in a different directory. Restored FP32 tensors are
    streamed into a separate small external-data file.
    """
    parent_path = Path(parent_model).resolve()
    baseline_path = Path(baseline_model).resolve()
    graph_path = Path(output_model).resolve()
    fallback_data_path = Path(output_fallback_data).resolve()
    manifest_path = graph_path.with_suffix(graph_path.suffix + ".manifest.json")
    if graph_path in {parent_path, baseline_path} or fallback_data_path in {
        parent_path,
        baseline_path,
    }:
        raise ValueError("Candidate outputs must not overwrite either source graph")
    if graph_path in {fallback_data_path, manifest_path} or fallback_data_path == manifest_path:
        raise ValueError("Candidate graph, fallback data, and manifest paths must be distinct")
    if graph_path.parent != fallback_data_path.parent:
        raise ValueError("Candidate graph and fallback external data must share a directory")
    if graph_path.exists() or fallback_data_path.exists() or manifest_path.exists():
        raise FileExistsError("Candidate output path already exists")

    selected = tuple(sorted(set(excluded_nodes)))
    inventory_by_node = _inventory_nodes(inventory)
    unknown = set(selected) - inventory_by_node.keys()
    if unknown:
        raise ValueError(f"Unknown eligible inventory nodes: {sorted(unknown)}")

    parent = onnx.load(str(parent_path), load_external_data=False)
    baseline = onnx.load(str(baseline_path), load_external_data=False)
    expected_baseline_hash = inventory.get("source_graph_sha256")
    baseline_graph_hash = _sha256_file(baseline_path)
    if expected_baseline_hash and expected_baseline_hash != baseline_graph_hash:
        raise ValueError("Inventory source graph hash does not match the baseline graph")

    baseline_by_name = _baseline_nodes(baseline)
    source_initializers = {tensor.name: tensor for tensor in baseline.graph.initializer}
    parent_initializers = {tensor.name: tensor for tensor in parent.graph.initializer}
    selected_rows: dict[str, dict[str, Any]] = {}
    parent_node_indices: dict[int, str] = {}
    auxiliary_names: dict[str, set[str]] = defaultdict(set)

    for name in selected:
        row = inventory_by_node[name]
        baseline_node = baseline_by_name.get(name)
        if baseline_node is None or len(baseline_node.input) != 2:
            raise ValueError(f"Baseline MatMul node is missing or unsupported: {name}")
        initializer_name = row.get("initializer")
        if baseline_node.input[1] != initializer_name:
            raise ValueError(f"Inventory initializer does not match baseline node: {name}")
        source_tensor = source_initializers.get(initializer_name)
        if source_tensor is None or source_tensor.data_type != TensorProto.FLOAT or len(source_tensor.dims) != 2:
            raise ValueError(f"Baseline fallback initializer is not rank-2 float32: {initializer_name}")
        if list(source_tensor.dims) != row.get("shape"):
            raise ValueError(f"Inventory shape does not match baseline initializer: {initializer_name}")
        parent_index, parent_node = _parent_node_for(list(parent.graph.node), baseline_node, row)
        if parent_index in parent_node_indices:
            raise ValueError("Multiple inventory nodes resolve to one parent MatMulNBits node")
        parent_node_indices[parent_index] = name
        selected_rows[initializer_name] = row
        auxiliary_names[initializer_name].update(_nbits_auxiliaries(parent_node))

    output_dir = graph_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    created_symlinks: list[Path] = []
    wrote_fallback_data = False
    wrote_graph = False
    try:
        created_symlinks = _ensure_parent_external_files(parent_path, parent, output_dir)
        copied_hashes: dict[str, str] = {}
        copied_bytes: dict[str, int] = {}
        fallback_initializers: list[onnx.TensorProto] = []
        wrote_fallback_data = True
        with ExternalDataWriter(fallback_data_path) as writer:
            for name in sorted(selected_rows):
                row = selected_rows[name]
                source_tensor = source_initializers[name]
                existing = parent_initializers.get(name)
                if existing is not None:
                    if (
                        existing.data_type != TensorProto.FLOAT
                        or list(existing.dims) != list(source_tensor.dims)
                    ):
                        raise ValueError(f"Parent initializer conflicts with baseline fallback: {name}")
                    existing_slice = _external_tensor_slice(parent_path, existing)
                    existing_hash = _sha256_range(existing_slice)
                    if existing_hash != row["source_sha256"]:
                        raise ValueError(f"Parent initializer hash does not match baseline: {name}")
                    copied_hashes[name] = existing_hash
                    copied_bytes[name] = 0
                    continue

                source_slice = _external_tensor_slice(baseline_path, source_tensor)
                offset, length, source_hash = _copy_range(source_slice, writer)
                if source_hash != row["source_sha256"]:
                    raise ValueError(f"Inventory source hash does not match baseline initializer: {name}")
                fallback_initializers.append(
                    make_external_tensor(
                        name,
                        source_tensor.data_type,
                        tuple(source_tensor.dims),
                        fallback_data_path.name,
                        offset,
                        length,
                    )
                )
                copied_hashes[name] = source_hash
                copied_bytes[name] = length

        candidate = copy.deepcopy(parent)
        candidate.graph.initializer.extend(fallback_initializers)
        rewritten_nodes = []
        replaced_names = set(parent_node_indices.values())
        for index, node in enumerate(candidate.graph.node):
            inventory_name = parent_node_indices.get(index)
            if inventory_name is None:
                rewritten_nodes.append(node)
                continue
            original = copy.deepcopy(baseline_by_name[inventory_name])
            if tuple(original.output) != tuple(node.output) or original.input[0] != node.input[0]:
                raise ValueError(f"Baseline and parent MatMul signatures differ: {inventory_name}")
            rewritten_nodes.append(original)
        del candidate.graph.node[:]
        candidate.graph.node.extend(rewritten_nodes)

        consumers: dict[str, int] = defaultdict(int)
        for node in candidate.graph.node:
            for name in node.input:
                if name:
                    consumers[name] += 1
        graph_values = {
            value.name
            for value in (*candidate.graph.input, *candidate.graph.output, *candidate.graph.value_info)
        }
        removable_auxiliaries = set().union(*auxiliary_names.values()) if auxiliary_names else set()
        removed_names = set()
        retained_initializers = []
        for tensor in candidate.graph.initializer:
            if (
                tensor.name in removable_auxiliaries
                and tensor.name not in graph_values
                and not consumers[tensor.name]
            ):
                removed_names.add(tensor.name)
                continue
            retained_initializers.append(tensor)
        del candidate.graph.initializer[:]
        candidate.graph.initializer.extend(retained_initializers)

        source_hashes = {
            "parent_graph_sha256": _sha256_file(parent_path),
            "baseline_graph_sha256": baseline_graph_hash,
            "inventory_sha256": inventory.get("inventory_sha256"),
            "fallback_initializers": copied_hashes,
        }
        fallback_storage = _fallback_storage(
            selected_rows,
            auxiliary_names,
            parent_initializers,
            removed_names,
            copied_bytes,
        )
        temporary_graph = graph_path.with_name(f".{graph_path.name}.tmp")
        onnx.save_model(candidate, str(temporary_graph), save_as_external_data=False)
        os.replace(temporary_graph, graph_path)
        wrote_graph = True

        manifest = {
            "format": "kitten2-onnx-nbits-fallback-delta-v1",
            "parent_graph": str(parent_path),
            "parent_graph_sha256": source_hashes["parent_graph_sha256"],
            "baseline_graph": str(baseline_path),
            "baseline_graph_sha256": baseline_graph_hash,
            "inventory_sha256": source_hashes["inventory_sha256"],
            "excluded_nodes": list(selected),
            "replaced_nodes": sorted(replaced_names),
            "reused_parent_external_data": sorted(
                str(path.relative_to(output_dir)) for path in created_symlinks
            ),
            "fallback_data_file": fallback_data_path.name,
            "fallback_data_bytes": fallback_data_path.stat().st_size,
            "fallback_storage": fallback_storage,
            "source_hashes": source_hashes,
            "candidate_graph_sha256": _sha256_file(graph_path),
            "fallback_data_sha256": _sha256_file(fallback_data_path),
        }
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_manifest, manifest_path)
        return FallbackDelta(
            graph_path=graph_path,
            fallback_data_path=fallback_data_path,
            manifest_path=manifest_path,
            excluded_nodes=selected,
            fallback_storage=fallback_storage,
            source_hashes=source_hashes,
        )
    except BaseException:
        if wrote_graph:
            graph_path.unlink(missing_ok=True)
        if wrote_fallback_data:
            fallback_data_path.unlink(missing_ok=True)
        for path in created_symlinks:
            path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_graph = graph_path.with_name(f".{graph_path.name}.tmp")
        temporary_graph.unlink(missing_ok=True)
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.unlink(missing_ok=True)
