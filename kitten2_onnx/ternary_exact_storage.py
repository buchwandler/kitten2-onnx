"""Lossless INT8 storage for exact group-128 ternary MatMul weights."""

from __future__ import annotations

import copy
import hashlib
import os
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.external_data import (
    logical_tensor_bytes,
    make_external_tensor,
    resolve_external_data_range,
)
from kitten2_onnx.quantize_int8_streaming import ExternalDataWriter
from kitten2_onnx.quantize_ternary_nbits import (
    SOURCE_GROUP_SIZE,
    ZERO_POINTS,
    TernaryNBitsConfig,
    _close_memmap,
    _graph_names,
    _is_exact_ternary_weight,
    _load_float32_weight,
    _unique_graph_name,
    pack_codes,
    pack_zero_points,
)

TERNARY_NEGATIVE_ZERO_CODE = np.int8(-2)
TERNARY_NEGATIVE_CODE = np.int8(-1)
TERNARY_POSITIVE_ZERO_CODE = np.int8(0)
TERNARY_POSITIVE_CODE = np.int8(1)
TERNARY_CODE_VALUES = (-2, -1, 0, 1)
TERNARY_ENCODING_VERSION = 2

def _reconstruction_constants(
    used_names: set[str],
) -> tuple[str, str, str, list[onnx.TensorProto]]:
    marker_name = _unique_graph_name("ternary_exact_negative_zero_marker", used_names)
    one_name = _unique_graph_name("ternary_exact_one", used_names)
    two_name = _unique_graph_name("ternary_exact_two", used_names)
    return marker_name, one_name, two_name, [
        numpy_helper.from_array(
            np.asarray(TERNARY_NEGATIVE_ZERO_CODE, dtype=np.int8), name=marker_name
        ),
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name=one_name),
        numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), name=two_name),
    ]


@dataclass(frozen=True)
class TernaryExactStorageConfig:
    group_size: int = 128
    code_dtype: str = "int8"
    scale_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.group_size != 128:
            raise ValueError(f"group_size must be 128, found {self.group_size}")
        if self.code_dtype != "int8":
            raise ValueError(f"code_dtype must be 'int8', found {self.code_dtype!r}")
        if self.scale_dtype != "float32":
            raise ValueError(f"scale_dtype must be 'float32', found {self.scale_dtype!r}")


def _encode_group(
    group: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group.ndim != 2 or group.shape[0] != 128:
        raise ValueError(f"ternary group must have shape [128, N], found {group.shape}")

    negative_zero = (group == 0) & np.signbit(group)
    scales = np.abs(group).max(axis=0).astype(np.float32, copy=False)
    codes = np.where(
        negative_zero,
        TERNARY_NEGATIVE_ZERO_CODE,
        np.where(
            group < 0,
            TERNARY_NEGATIVE_CODE,
            np.where(
                group > 0,
                TERNARY_POSITIVE_CODE,
                TERNARY_POSITIVE_ZERO_CODE,
            ),
        ),
    ).astype(np.int8)
    reconstructed = codes.astype(np.float32) * scales[None, :]
    reconstructed = np.where(
        codes == TERNARY_NEGATIVE_ZERO_CODE,
        np.float32(-0.0),
        reconstructed,
    ).astype(np.float32, copy=False)
    if not np.array_equal(
        np.ascontiguousarray(reconstructed).view(np.uint32),
        np.ascontiguousarray(group).view(np.uint32),
    ):
        raise ValueError("ternary INT8 reconstruction is not bit-identical to the source group")
    return codes, scales, reconstructed



def _encode_weight(weight: np.ndarray, config: TernaryExactStorageConfig) -> tuple[np.ndarray, np.ndarray]:
    if weight.dtype != np.float32:
        raise TypeError(f"weight must have dtype float32, found {weight.dtype}")
    if weight.ndim != 2 or any(dimension <= 0 for dimension in weight.shape):
        raise ValueError(f"weight must have two positive dimensions [K, N], found {weight.shape}")
    if weight.shape[0] % config.group_size:
        raise ValueError(f"weight K dimension must be divisible by {config.group_size}")
    is_ternary, _ = _is_exact_ternary_weight(weight)
    if not is_ternary:
        raise ValueError("weight is not exact group-128 ternary")

    group_count, n = weight.shape[0] // config.group_size, weight.shape[1]
    codes = np.empty((group_count, config.group_size, n), dtype=np.int8)
    scales = np.empty((group_count, 1, n), dtype=np.float32)
    for group_index in range(group_count):
        start = group_index * config.group_size
        codes[group_index], scales[group_index, 0], _ = _encode_group(
            weight[start : start + config.group_size]
        )
    return codes, scales


def _reconstruction_nodes(
    weight_name: str,
    node_name: str,
    code_name: str,
    scale_name: str,
    shape_name: str,
    reconstructed_name: str,
    marker_name: str,
    one_name: str,
    two_name: str,
    used_names: set[str],
) -> list[onnx.NodeProto]:
    mask_name = _unique_graph_name(f"{weight_name}_ternary_exact_negative_zero_mask", used_names)
    code_float_name = _unique_graph_name(f"{weight_name}_ternary_exact_float", used_names)
    mask_float_name = _unique_graph_name(f"{weight_name}_ternary_exact_mask_float", used_names)
    offset_name = _unique_graph_name(f"{weight_name}_ternary_exact_marker_offset", used_names)
    adjusted_code_name = _unique_graph_name(f"{weight_name}_ternary_exact_adjusted_code", used_names)
    scaled_name = _unique_graph_name(f"{weight_name}_ternary_exact_scaled", used_names)
    sign_name = _unique_graph_name(f"{weight_name}_ternary_exact_sign", used_names)
    value_name = _unique_graph_name(f"{weight_name}_ternary_exact_value", used_names)
    return [
        helper.make_node(
            "Equal",
            [code_name, marker_name],
            [mask_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_equal", used_names),
        ),
        helper.make_node(
            "Cast",
            [code_name],
            [code_float_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_code_cast", used_names),
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Cast",
            [mask_name],
            [mask_float_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_mask_cast", used_names),
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Mul",
            [mask_float_name, two_name],
            [offset_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_marker_offset", used_names),
        ),
        helper.make_node(
            "Add",
            [code_float_name, offset_name],
            [adjusted_code_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_adjust_code", used_names),
        ),
        helper.make_node(
            "Mul",
            [adjusted_code_name, scale_name],
            [scaled_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_scale", used_names),
        ),
        helper.make_node(
            "Sub",
            [one_name, offset_name],
            [sign_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_sign", used_names),
        ),
        helper.make_node(
            "Mul",
            [scaled_name, sign_name],
            [value_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_signed_zero", used_names),
        ),
        helper.make_node(
            "Reshape",
            [value_name, shape_name],
            [reconstructed_name],
            name=_unique_graph_name(f"{node_name}_ternary_exact_reshape", used_names),
        ),
    ]



def rewrite_matmul_to_exact_ternary_storage(
    model: onnx.ModelProto,
    *,
    node_name: str,
    config: TernaryExactStorageConfig | None = None,
) -> onnx.ModelProto:
    """Rewrite one inline FP32 MatMul initializer using exact INT8 storage."""
    config = config or TernaryExactStorageConfig()
    rewritten = copy.deepcopy(model)
    matches = [
        node
        for node in rewritten.graph.node
        if node.op_type == "MatMul" and node.name == node_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one MatMul named {node_name!r}, found {len(matches)}")
    node = matches[0]
    if len(node.input) != 2:
        raise ValueError("target MatMul must have exactly two inputs")

    initializers = {tensor.name: tensor for tensor in rewritten.graph.initializer}
    weight_name = node.input[1]
    tensor = initializers.get(weight_name)
    if tensor is None:
        raise ValueError("target MatMul weight must be a graph initializer")
    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        raise ValueError("external-data weights require compress_ternary_matmuls_streaming")
    if tensor.data_type != TensorProto.FLOAT:
        raise TypeError("target MatMul weight initializer must be float32")

    weight = np.asarray(numpy_helper.to_array(tensor))
    codes, scales = _encode_weight(weight, config)
    k, n = weight.shape
    used_names = _graph_names(rewritten.graph)
    marker_name, one_name, two_name, reconstruction_initializers = _reconstruction_constants(
        used_names
    )
    code_name = _unique_graph_name(f"{weight_name}_ternary_exact_codes", used_names)
    scale_name = _unique_graph_name(f"{weight_name}_ternary_exact_scales", used_names)
    shape_name = _unique_graph_name(f"{weight_name}_ternary_exact_shape", used_names)
    reconstructed_name = _unique_graph_name(
        f"{weight_name}_ternary_exact_weight", used_names
    )
    rewritten.graph.initializer.extend(
        [
            *reconstruction_initializers,
            numpy_helper.from_array(codes, name=code_name),
            numpy_helper.from_array(scales, name=scale_name),
            numpy_helper.from_array(np.asarray([k, n], dtype=np.int64), name=shape_name),
        ]
    )

    node_index = next(index for index, candidate in enumerate(rewritten.graph.node) if candidate is node)
    node.input[1] = reconstructed_name
    additions = _reconstruction_nodes(
        weight_name,
        node_name,
        code_name,
        scale_name,
        shape_name,
        reconstructed_name,
        marker_name,
        one_name,
        two_name,
        used_names,
    )
    for offset, added_node in enumerate(additions):
        rewritten.graph.node.insert(node_index + offset, added_node)

    consumers = [candidate for candidate in rewritten.graph.node if weight_name in candidate.input]
    if not consumers and weight_name not in {
        value.name for value in (*rewritten.graph.input, *rewritten.graph.output)
    }:
        rewritten.graph.initializer.remove(tensor)
    return rewritten


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_external_reference(
    tensor: onnx.TensorProto,
    location: str,
    offset: int,
    length: int,
) -> None:
    entries = {
        item.key: item.value
        for item in tensor.external_data
        if item.key not in {"location", "offset", "length", "basepath"}
    }
    entries.update({"location": location, "offset": str(offset), "length": str(length)})
    del tensor.external_data[:]
    for key, value in entries.items():
        item = tensor.external_data.add()
        item.key = key
        item.value = value
    tensor.data_location = TensorProto.EXTERNAL


def _inventory_rows(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in inventory.get("nodes", []):
        if not row.get("eligible_exact_ternary"):
            continue
        name = row.get("node")
        if not name or name in rows:
            raise ValueError(f"inventory has a missing or duplicate eligible node name: {name!r}")
        rows[name] = row
    return rows


def _external_source_hashes(model_path: Path, model: onnx.ModelProto) -> dict[str, str]:
    paths: dict[str, Path] = {}
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL and not tensor.external_data:
            continue
        tensor_slice = resolve_external_data_range(
            model_path.parent,
            tensor,
            logical_bytes=logical_tensor_bytes(tensor),
        )
        paths[tensor_slice.location] = tensor_slice.path
    return {location: _sha256_file(path) for location, path in sorted(paths.items())}


def _weight_info(
    model_path: Path,
    tensor: onnx.TensorProto,
    inventory_rows: list[dict[str, Any]],
    config: TernaryExactStorageConfig,
) -> tuple[int, int, str]:
    if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) != 2:
        raise ValueError(f"MatMul weight must be a rank-2 float32 initializer: {tensor.name}")
    k, n = (int(value) for value in tensor.dims)
    if k <= 0 or n <= 0 or k % config.group_size:
        raise ValueError(f"MatMul weight has unsupported group-128 shape: {tensor.name}")
    expected_shape = [k, n]
    if any(row.get("shape") != expected_shape for row in inventory_rows):
        raise ValueError(f"inventory shape does not match initializer {tensor.name!r}")

    weight, is_memmap = _load_float32_weight(model_path, tensor)
    try:
        is_ternary, source_sha256 = _is_exact_ternary_weight(weight)
    finally:
        _close_memmap(weight, is_memmap)
    if not is_ternary:
        raise ValueError(f"initializer is not exact group-128 ternary: {tensor.name}")
    expected_hashes = {row.get("source_sha256") for row in inventory_rows}
    expected_hashes.discard(None)
    if expected_hashes and expected_hashes != {source_sha256}:
        raise ValueError(f"inventory source hash does not match initializer {tensor.name!r}")
    return k, n, source_sha256


def _write_exact_weight(
    model_path: Path,
    tensor: onnx.TensorProto,
    writer: ExternalDataWriter,
    *,
    code_name: str,
    scale_name: str,
    external_location: str,
    config: TernaryExactStorageConfig,
    source_sha256: str,
) -> tuple[list[onnx.TensorProto], dict[str, Any]]:
    k, n = (int(value) for value in tensor.dims)
    weight, is_memmap = _load_float32_weight(model_path, tensor)
    code_offset = writer.start_tensor()
    code_digest = hashlib.sha256()
    code_bytes = 0
    negative_zero_count = 0
    try:
        for start in range(0, k, config.group_size):
            group = np.asarray(weight[start : start + config.group_size])
            codes, _, _ = _encode_group(group)
            negative_zero_count += int(np.count_nonzero(codes == TERNARY_NEGATIVE_ZERO_CODE))
            code_block = np.ascontiguousarray(codes)
            code_payload = memoryview(code_block).cast("B")
            writer.write(code_payload)
            code_digest.update(code_payload)
            code_bytes += code_block.nbytes
    finally:
        _close_memmap(weight, is_memmap)
    code_offset, actual_code_bytes = writer.finish_tensor(code_offset)
    if actual_code_bytes != code_bytes:
        raise ValueError(f"written ternary code data has unexpected size: {tensor.name}")

    weight, is_memmap = _load_float32_weight(model_path, tensor)
    scale_offset = writer.start_tensor()
    scale_digest = hashlib.sha256()
    scale_bytes = 0
    try:
        for start in range(0, k, config.group_size):
            group = np.asarray(weight[start : start + config.group_size])
            _, scales, _ = _encode_group(group)
            scale_block = np.ascontiguousarray(scales).reshape(1, n)
            scale_payload = memoryview(scale_block).cast("B")
            writer.write(scale_payload)
            scale_digest.update(scale_payload)
            scale_bytes += scale_block.nbytes
    finally:
        _close_memmap(weight, is_memmap)
    scale_offset, actual_scale_bytes = writer.finish_tensor(scale_offset)
    if actual_scale_bytes != scale_bytes:
        raise ValueError(f"written ternary scale data has unexpected size: {tensor.name}")

    groups = k // config.group_size
    tensors = [
        make_external_tensor(
            code_name,
            TensorProto.INT8,
            (groups, config.group_size, n),
            external_location,
            code_offset,
            code_bytes,
        ),
        make_external_tensor(
            scale_name,
            TensorProto.FLOAT,
            (groups, 1, n),
            external_location,
            scale_offset,
            scale_bytes,
        ),
    ]
    return tensors, {
        "source_initializer": tensor.name,
        "source_shape": [k, n],
        "source_sha256": source_sha256,
        "code_initializer": code_name,
        "code_bytes": code_bytes,
        "code_sha256": code_digest.hexdigest(),
        "scale_initializer": scale_name,
        "scale_bytes": scale_bytes,
        "scale_sha256": scale_digest.hexdigest(),
        "encoding_version": TERNARY_ENCODING_VERSION,
        "code_values": list(TERNARY_CODE_VALUES),
        "negative_zero_code": int(TERNARY_NEGATIVE_ZERO_CODE),
        "negative_zero_count": negative_zero_count,
        "negative_zero_preserved": True,
        "exact_reconstruction": True,
    }


def _report_stage(
    callback: Callable[[str, dict[str, Any]], None] | None,
    stage: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback(stage, details)


def compress_ternary_matmuls_streaming(
    source_model: str | Path,
    output_model: str | Path,
    output_data: str | Path,
    *,
    inventory: dict[str, Any],
    exclude_nodes: set[str] | None = None,
    config: TernaryExactStorageConfig | None = None,
    stage_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Rewrite eligible MatMuls while streaming external initializers and output data."""
    config = config or TernaryExactStorageConfig()
    source_path = Path(source_model).resolve()
    output_path = Path(output_model).resolve()
    data_path = Path(output_data).resolve()
    if output_path == source_path or data_path in {source_path, output_path}:
        raise ValueError("candidate outputs must not overwrite the source model or each other")
    if output_path.parent != data_path.parent:
        raise ValueError("candidate graph and external data must share a directory")
    if output_path.exists() or data_path.exists():
        raise FileExistsError("candidate output path already exists")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_name(f".{data_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_graph = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    published_data = False
    try:
        _report_stage(stage_callback, "load_graph")
        model = onnx.load(str(source_path), load_external_data=False)
        _report_stage(stage_callback, "hash_source_graph")
        source_graph_sha256 = _sha256_file(source_path)
        inventory_graph_sha256 = inventory.get("source_graph_sha256")
        if inventory_graph_sha256 and inventory_graph_sha256 != source_graph_sha256:
            raise ValueError("inventory source graph hash does not match the source model")
        source_external_paths = {
            resolve_external_data_range(
                source_path.parent,
                tensor,
                logical_bytes=logical_tensor_bytes(tensor),
            ).path.resolve()
            for tensor in model.graph.initializer
            if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data
        }
        if output_path in source_external_paths or data_path in source_external_paths:
            raise ValueError("candidate outputs must not overwrite source external data")
        _report_stage(stage_callback, "hash_source_external_data")
        source_external_data_sha256 = _external_source_hashes(source_path, model)

        _report_stage(stage_callback, "validate_inventory")
        rows_by_node = _inventory_rows(inventory)
        excluded = set(exclude_nodes or ())
        unknown_exclusions = excluded - rows_by_node.keys()
        if unknown_exclusions:
            raise ValueError(f"unknown eligible MatMul exclusions: {sorted(unknown_exclusions)}")
        selected_rows = {
            name: row for name, row in rows_by_node.items() if name not in excluded
        }
        if not selected_rows:
            raise ValueError("no exact group-128 ternary MatMuls selected")
        expected_count = inventory.get("eligible_exact_ternary_matmul_count")
        if expected_count is not None and int(expected_count) != len(rows_by_node):
            raise ValueError("inventory eligible MatMul count does not match its node rows")

        matmuls_by_name: dict[str, tuple[int, onnx.NodeProto]] = {}
        for index, node in enumerate(model.graph.node):
            if node.op_type != "MatMul":
                continue
            name = node.name or f"MatMul#{index}"
            if name in matmuls_by_name:
                raise ValueError(f"source graph has duplicate MatMul name: {name}")
            matmuls_by_name[name] = (index, node)

        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        targets_by_weight: dict[str, list[tuple[str, int, onnx.NodeProto]]] = defaultdict(list)
        rows_by_weight: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node_name, row in selected_rows.items():
            found = matmuls_by_name.get(node_name)
            if found is None:
                raise ValueError(f"inventory MatMul is missing from source graph: {node_name}")
            node_index, node = found
            initializer_name = row.get("initializer")
            if len(node.input) != 2 or node.input[1] != initializer_name:
                raise ValueError(f"inventory initializer does not match MatMul {node_name!r}")
            tensor = initializers.get(initializer_name)
            if tensor is None:
                raise ValueError(f"MatMul weight initializer is missing: {initializer_name}")
            targets_by_weight[initializer_name].append((node_name, node_index, node))
            rows_by_weight[initializer_name].append(row)

        source_hashes: dict[str, str] = {}
        source_shapes: dict[str, tuple[int, int]] = {}
        for index, (initializer_name, rows) in enumerate(rows_by_weight.items(), start=1):
            _report_stage(
                stage_callback,
                f"scan_weight_{index:03d}_of_{len(rows_by_weight):03d}",
                initializer=initializer_name,
                nodes=[row.get("node") for row in rows],
            )
            tensor = initializers[initializer_name]
            k, n, source_hash = _weight_info(source_path, tensor, rows, config)
            source_hashes[initializer_name] = source_hash
            source_shapes[initializer_name] = (k, n)

        used_names = _graph_names(model.graph)
        marker_name, one_name, two_name, reconstruction_initializers = _reconstruction_constants(
            used_names
        )
        names_by_weight: dict[str, tuple[str, str, str, str]] = {}
        for initializer_name in targets_by_weight:
            names_by_weight[initializer_name] = (
                _unique_graph_name(f"{initializer_name}_ternary_exact_codes", used_names),
                _unique_graph_name(f"{initializer_name}_ternary_exact_scales", used_names),
                _unique_graph_name(f"{initializer_name}_ternary_exact_shape", used_names),
                _unique_graph_name(f"{initializer_name}_ternary_exact_weight", used_names),
            )

        consumers: dict[str, set[int]] = defaultdict(set)
        for index, node in enumerate(model.graph.node):
            for name in node.input:
                if name:
                    consumers[name].add(index)
        graph_values = {
            value.name for value in (*model.graph.input, *model.graph.output)
        }
        removed_weights = {
            name
            for name, rows in targets_by_weight.items()
            if name not in graph_values
            and consumers[name]
            and consumers[name] <= {index for _, index, _ in rows}
        }

        external_location = temporary_data.name
        new_initializers: list[onnx.TensorProto] = []
        new_nodes_by_index: dict[int, list[onnx.NodeProto]] = defaultdict(list)
        new_initializers.extend(reconstruction_initializers)
        initializer_reports: list[dict[str, Any]] = []
        source_fp32_bytes_replaced = 0

        _report_stage(stage_callback, "copy_unmodified_initializers")
        with ExternalDataWriter(temporary_data) as writer:
            for tensor in model.graph.initializer:
                if tensor.name in removed_weights:
                    continue
                if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
                    tensor_slice = resolve_external_data_range(
                        source_path.parent,
                        tensor,
                        logical_bytes=logical_tensor_bytes(tensor),
                    )
                    offset, length = writer.copy_range(tensor_slice)
                    copied = copy.deepcopy(tensor)
                    _set_external_reference(copied, external_location, offset, length)
                    new_initializers.append(copied)
                else:
                    new_initializers.append(copy.deepcopy(tensor))

            for index, (initializer_name, rows) in enumerate(targets_by_weight.items(), start=1):
                _report_stage(
                    stage_callback,
                    f"encode_weight_{index:03d}_of_{len(targets_by_weight):03d}",
                    initializer=initializer_name,
                    nodes=[name for name, _, _ in rows],
                )
                tensor = initializers[initializer_name]
                code_name, scale_name, shape_name, reconstructed_name = names_by_weight[
                    initializer_name
                ]
                encoded_tensors, report = _write_exact_weight(
                    source_path,
                    tensor,
                    writer,
                    code_name=code_name,
                    scale_name=scale_name,
                    external_location=external_location,
                    config=config,
                    source_sha256=source_hashes[initializer_name],
                )
                new_initializers.extend(encoded_tensors)
                k, n = source_shapes[initializer_name]
                new_initializers.append(
                    numpy_helper.from_array(
                        np.asarray([k, n], dtype=np.int64), name=shape_name
                    )
                )
                first_node_name, first_index, _ = min(
                    rows, key=lambda target: target[1]
                )
                new_nodes_by_index[first_index].extend(
                    _reconstruction_nodes(
                        initializer_name,
                        first_node_name,
                        code_name,
                        scale_name,
                        shape_name,
                        reconstructed_name,
                        marker_name,
                        one_name,
                        two_name,
                        used_names,
                    )
)
                for _, _, node in rows:
                    node.input[1] = reconstructed_name
                if initializer_name in removed_weights:
                    source_fp32_bytes_replaced += int(np.prod(tensor.dims)) * 4
                report["transformed_nodes"] = [name for name, _, _ in rows]
                initializer_reports.append(report)

        model.graph.ClearField("initializer")
        model.graph.initializer.extend(new_initializers)
        rewritten_nodes: list[onnx.NodeProto] = []
        for index, node in enumerate(model.graph.node):
            rewritten_nodes.extend(new_nodes_by_index.get(index, ()))
            rewritten_nodes.append(node)
        model.graph.ClearField("node")
        model.graph.node.extend(rewritten_nodes)

        temporary_relative_location = temporary_data.relative_to(output_path.parent).as_posix()
        final_relative_location = data_path.relative_to(output_path.parent).as_posix()
        _report_stage(stage_callback, "write_candidate_graph")
        onnx.save_model(model, str(temporary_graph), save_as_external_data=False)
        _report_stage(stage_callback, "onnx_checker")
        onnx.checker.check_model(str(temporary_graph))

        for tensor in model.graph.initializer:
            if tensor.data_location != TensorProto.EXTERNAL and not tensor.external_data:
                continue
            external = {item.key: item.value for item in tensor.external_data}
            if external.get("location") == temporary_relative_location:
                _set_external_reference(
                    tensor,
                    final_relative_location,
                    int(external.get("offset", "0")),
                    int(external.get("length", str(logical_tensor_bytes(tensor)))),
                )
        onnx.save_model(model, str(temporary_graph), save_as_external_data=False)

        transformed_nodes = [
            node_name
            for rows in targets_by_weight.values()
            for node_name, _, _ in rows
        ]
        code_bytes = sum(row["code_bytes"] for row in initializer_reports)
        scale_bytes = sum(row["scale_bytes"] for row in initializer_reports)
        negative_zero_count = sum(row["negative_zero_count"] for row in initializer_reports)
        report = {
            "format": "kitten2-onnx-ternary-exact-storage-v2",
            "encoding_version": TERNARY_ENCODING_VERSION,
            "code_values": list(TERNARY_CODE_VALUES),
            "negative_zero_code": int(TERNARY_NEGATIVE_ZERO_CODE),
            "negative_zero_count": negative_zero_count,
            "negative_zero_preserved": True,
            "source_graph_sha256": source_graph_sha256,
            "source_external_data_sha256": source_external_data_sha256,
            "output_graph_sha256": _sha256_file(temporary_graph),
            "compact_external_data_sha256": _sha256_file(temporary_data),
            "inventory_sha256": inventory.get("inventory_sha256"),
            "transformed_matmul_count": len(transformed_nodes),
            "transformed_nodes": transformed_nodes,
            "initializer_reports": initializer_reports,
            "code_bytes": code_bytes,
            "scale_bytes": scale_bytes,
            "source_fp32_bytes_replaced": source_fp32_bytes_replaced,
            "exact_reconstruction": all(
                row["exact_reconstruction"] for row in initializer_reports
            ),
            "excluded_nodes": sorted(excluded),
            "source_hashes": source_hashes,
        }

        _report_stage(stage_callback, "publish_external_data")
        os.replace(temporary_data, data_path)
        published_data = True
        _report_stage(stage_callback, "publish_graph")
        os.replace(temporary_graph, output_path)
        _report_stage(stage_callback, "complete")
        return report
    except BaseException:
        temporary_graph.unlink(missing_ok=True)
        temporary_data.unlink(missing_ok=True)
        if published_data and not output_path.exists():
            data_path.unlink(missing_ok=True)
        raise


def _hash_external_initializer(model_dir: Path, tensor: onnx.TensorProto) -> str:
    tensor_slice = resolve_external_data_range(
        model_dir,
        tensor,
        logical_bytes=logical_tensor_bytes(tensor),
        require_exact_length=True,
    )
    digest = hashlib.sha256()
    with tensor_slice.path.open("rb") as stream:
        stream.seek(tensor_slice.offset)
        remaining = tensor_slice.length
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"External initializer ended early: {tensor.name}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _map_external_initializer(
    model_dir: Path, tensor: onnx.TensorProto, dtype: np.dtype, shape: tuple[int, ...]
) -> np.memmap:
    if tensor.data_location != TensorProto.EXTERNAL or tuple(tensor.dims) != shape:
        raise ValueError(f"External initializer has unexpected storage or shape: {tensor.name}")
    tensor_slice = resolve_external_data_range(
        model_dir,
        tensor,
        logical_bytes=logical_tensor_bytes(tensor),
        require_exact_length=True,
    )
    return np.memmap(
        tensor_slice.path,
        mode="r",
        dtype=dtype,
        offset=tensor_slice.offset,
        shape=shape,
        order="C",
    )


def _prune_graph_to_outputs(graph: onnx.GraphProto) -> set[str]:
    needed = {value.name for value in graph.output}
    retained_nodes = []
    for node in reversed(graph.node):
        if any(name in needed for name in node.output):
            retained_nodes.append(node)
            needed.update(name for name in node.input if name)
    retained_nodes.reverse()
    del graph.node[:]
    graph.node.extend(retained_nodes)
    del graph.initializer[:]
    # The caller restores the live initializers after this helper returns.
    live_values = needed | {value.name for value in graph.input}
    del graph.value_info[:]
    return live_values


def transcode_exact_ternary_matmuls_to_nbits4_streaming(
    input_model: str | Path,
    output_model: str | Path,
    output_external_data: str | Path,
    *,
    exact_storage_metadata: dict[str, Any],
    config: TernaryNBitsConfig | None = None,
    exclude_nodes: set[str] | None = None,
    exclude_initializers: set[str] | None = None,
    column_chunk: int = 64,
) -> dict[str, Any]:
    """Transcode exact-int8 ternary MatMuls to q4 without materializing FP32 weights."""
    config = config or TernaryNBitsConfig(bits=4)
    if config.bits != 4:
        raise ValueError("exact ternary INT8 transcoding requires bits=4")
    if column_chunk <= 0:
        raise ValueError("column_chunk must be positive")

    source_path = Path(input_model).resolve()
    output_path = Path(output_model).resolve()
    data_path = Path(output_external_data).resolve()
    if source_path == output_path or source_path == data_path or output_path == data_path:
        raise ValueError("source model and candidate outputs must be distinct")
    if output_path.parent != data_path.parent:
        raise ValueError("candidate graph and external data must share a directory")
    if output_path.exists() or data_path.exists():
        raise FileExistsError("candidate output path already exists")

    if exact_storage_metadata.get("method") != "ternary-exact-int8":
        raise ValueError("source LM metadata is not ternary-exact-int8")
    if (
        exact_storage_metadata.get("encoding_version") != TERNARY_ENCODING_VERSION
        or exact_storage_metadata.get("code_values") != list(TERNARY_CODE_VALUES)
        or exact_storage_metadata.get("negative_zero_code") != int(TERNARY_NEGATIVE_ZERO_CODE)
        or exact_storage_metadata.get("negative_zero_preserved") is not True
    ):
        raise ValueError("source exact ternary storage metadata is unsupported")
    source_graph_sha256 = _sha256_file(source_path)
    if exact_storage_metadata.get("graph_sha256") != source_graph_sha256:
        raise ValueError("source exact ternary graph hash does not match metadata")

    model = onnx.load(str(source_path), load_external_data=False)
    source_external_hashes = _external_source_hashes(source_path, model)
    if exact_storage_metadata.get("external_data_sha256") != source_external_hashes:
        raise ValueError("source exact ternary external-data hashes do not match metadata")
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    initializer_reports = exact_storage_metadata.get("initializer_reports")
    if not isinstance(initializer_reports, list) or not initializer_reports:
        raise ValueError("source exact ternary metadata has no initializer reports")

    reports_by_source: dict[str, dict[str, Any]] = {}
    reports_by_node: dict[str, dict[str, Any]] = {}
    graph_nodes: dict[str, tuple[int, onnx.NodeProto]] = {}
    for index, node in enumerate(model.graph.node):
        label = node.name or f"MatMul#{index}"
        if label in graph_nodes:
            raise ValueError(f"source graph has duplicate node label: {label}")
        graph_nodes[label] = (index, node)
    for report in initializer_reports:
        source_name = report.get("source_initializer")
        if not isinstance(source_name, str) or source_name in reports_by_source:
            raise ValueError("source exact ternary metadata has a missing or duplicate initializer")
        reports_by_source[source_name] = report
        nodes = report.get("transformed_nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(f"source initializer has no transformed nodes: {source_name}")
        for node_name in nodes:
            if node_name in reports_by_node:
                raise ValueError(f"source exact ternary node is listed more than once: {node_name}")
            reports_by_node[node_name] = report

    excluded_node_names = set(exclude_nodes or ())
    excluded_initializer_names = set(exclude_initializers or ())
    unknown_nodes = excluded_node_names - reports_by_node.keys()
    unknown_initializers = excluded_initializer_names - reports_by_source.keys()
    if unknown_nodes:
        raise ValueError(f"Unknown exact ternary MatMul node exclusions: {sorted(unknown_nodes)}")
    if unknown_initializers:
        raise ValueError(
            f"Unknown exact ternary initializer exclusions: {sorted(unknown_initializers)}"
        )

    selected_nodes: dict[str, tuple[int, onnx.NodeProto, dict[str, Any]]] = {}
    for node_name, report in reports_by_node.items():
        if (
            node_name in excluded_node_names
            or report["source_initializer"] in excluded_initializer_names
        ):
            continue
        found = graph_nodes.get(node_name)
        if found is None:
            raise ValueError(f"source exact ternary MatMul is missing from graph: {node_name}")
        index, node = found
        if node.op_type != "MatMul" or len(node.input) != 2:
            raise ValueError(f"source exact ternary node is not a two-input MatMul: {node_name}")
        selected_nodes[node_name] = (index, node, report)
    if not selected_nodes:
        raise ValueError("no exact ternary MatMuls were selected for q4 transcoding")

    producer_by_value = {
        output: node for node in model.graph.node for output in node.output if output
    }
    initializer_names = set(initializers)

    def ancestor_initializers(value_name: str) -> set[str]:
        pending = [value_name]
        visited_values: set[str] = set()
        ancestors: set[str] = set()
        while pending:
            value = pending.pop()
            if value in visited_values:
                continue
            visited_values.add(value)
            if value in initializer_names:
                ancestors.add(value)
                continue
            producer = producer_by_value.get(value)
            if producer is not None:
                pending.extend(name for name in producer.input if name)
        return ancestors

    targets_by_source: dict[str, list[tuple[str, int, onnx.NodeProto]]] = defaultdict(list)
    for node_name, (index, node, report) in selected_nodes.items():
        code_name = report.get("code_initializer")
        scale_name = report.get("scale_initializer")
        ancestors = ancestor_initializers(node.input[1])
        if code_name not in ancestors or scale_name not in ancestors:
            raise ValueError(
                f"source MatMul does not reconstruct its reported codes/scales: {node_name}"
            )
        targets_by_source[report["source_initializer"]].append((node_name, index, node))

    for source_name in targets_by_source:
        report = reports_by_source[source_name]
        shape = report.get("source_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"source exact ternary shape is invalid: {source_name}")
        k, n = (int(value) for value in shape)
        if k <= 0 or n <= 0 or k % SOURCE_GROUP_SIZE:
            raise ValueError(f"source exact ternary shape is unsupported: {source_name}")
        group_count = k // SOURCE_GROUP_SIZE
        code_tensor = initializers.get(report.get("code_initializer"))
        scale_tensor = initializers.get(report.get("scale_initializer"))
        if code_tensor is None or scale_tensor is None:
            raise ValueError(
                f"source exact ternary code/scale initializer is missing: {source_name}"
            )
        if code_tensor.data_type != TensorProto.INT8 or tuple(code_tensor.dims) != (
            group_count,
            SOURCE_GROUP_SIZE,
            n,
        ):
            raise ValueError(f"source exact ternary code shape is invalid: {source_name}")
        if scale_tensor.data_type != TensorProto.FLOAT or tuple(scale_tensor.dims) != (
            group_count,
            1,
            n,
        ):
            raise ValueError(f"source exact ternary scale shape is invalid: {source_name}")
        for tensor, expected_hash, role in (
            (code_tensor, report.get("code_sha256"), "code"),
            (scale_tensor, report.get("scale_sha256"), "scale"),
        ):
            if tensor.data_location != TensorProto.EXTERNAL or not expected_hash:
                raise ValueError(
                    f"source exact ternary {role} storage is not external: {source_name}"
                )
            if _hash_external_initializer(source_path.parent, tensor) != expected_hash:
                raise ValueError(
                    f"source exact ternary {role} hash does not match metadata: {source_name}"
                )

    used_names = _graph_names(model.graph)
    names_by_source: dict[str, tuple[str, str, str]] = {}
    q4_placeholders: list[onnx.TensorProto] = []
    source_shapes: dict[str, tuple[int, int]] = {}
    blocks_per_source_group = SOURCE_GROUP_SIZE // config.block_size
    for source_name in targets_by_source:
        report = reports_by_source[source_name]
        k, n = (int(value) for value in report["source_shape"])
        group_count = k // SOURCE_GROUP_SIZE
        block_count = group_count * blocks_per_source_group
        packed_name = _unique_graph_name(f"{source_name}_nbits4", used_names)
        scales_name = _unique_graph_name(f"{source_name}_nbits4_scales", used_names)
        zero_points_name = _unique_graph_name(f"{source_name}_nbits4_zero_points", used_names)
        names_by_source[source_name] = (packed_name, scales_name, zero_points_name)
        source_shapes[source_name] = (k, n)
        q4_placeholders.extend(
            [
                make_external_tensor(
                    packed_name,
                    TensorProto.UINT8,
                    (n, block_count, config.block_size // 2),
                    data_path.name,
                    0,
                    n * block_count * (config.block_size // 2),
                ),
                make_external_tensor(
                    scales_name,
                    TensorProto.FLOAT,
                    (n * block_count,),
                    data_path.name,
                    0,
                    n * block_count * np.dtype(np.float32).itemsize,
                ),
                make_external_tensor(
                    zero_points_name,
                    TensorProto.UINT8,
                    (n * ((block_count + 1) // 2),),
                    data_path.name,
                    0,
                    n * ((block_count + 1) // 2),
                ),
            ]
        )

    rewritten = copy.deepcopy(model)
    for node_name, (index, node, report) in selected_nodes.items():
        packed_name, scales_name, zero_points_name = names_by_source[report["source_initializer"]]
        replacement = helper.make_node(
            "MatMulNBits",
            [node.input[0], packed_name, scales_name, zero_points_name],
            list(node.output),
            name=_unique_graph_name(f"{node.name or node_name}_nbits4", used_names),
            domain="com.microsoft",
            K=source_shapes[report["source_initializer"]][0],
            N=source_shapes[report["source_initializer"]][1],
            bits=4,
            block_size=config.block_size,
            accuracy_level=config.accuracy_level,
        )
        rewritten.graph.node[index].CopyFrom(replacement)
    rewritten.graph.initializer.extend(q4_placeholders)
    if not any(opset.domain == "com.microsoft" for opset in rewritten.opset_import):
        rewritten.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    source_initializer_names = set(initializers)
    q4_names = {tensor.name for tensor in q4_placeholders}
    live_values = _prune_graph_to_outputs(rewritten.graph)
    kept_source_names = (source_initializer_names & live_values) - q4_names
    kept_q4_names = q4_names & live_values
    if kept_q4_names != q4_names:
        raise ValueError("a transcoded q4 initializer is not used by a graph output")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_name(f".{data_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_graph = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    temp_location = temporary_data.name
    output_initializers: list[onnx.TensorProto] = []
    tensor_reports: list[dict[str, Any]] = []
    signed_zero_differences = 0
    packed_code_bytes = 0
    scale_bytes = 0
    zero_point_bytes = 0
    published_data = False
    try:
        with ExternalDataWriter(temporary_data) as writer:
            for source_name, (
                packed_name,
                scales_name,
                zero_points_name,
            ) in names_by_source.items():
                k, n = source_shapes[source_name]
                group_count = k // SOURCE_GROUP_SIZE
                block_count = group_count * blocks_per_source_group
                report = reports_by_source[source_name]
                code_tensor = initializers[report["code_initializer"]]
                scale_tensor = initializers[report["scale_initializer"]]
                code_values = _map_external_initializer(
                    source_path.parent,
                    code_tensor,
                    np.dtype(np.int8),
                    (group_count, SOURCE_GROUP_SIZE, n),
                )
                scale_values = _map_external_initializer(
                    source_path.parent,
                    scale_tensor,
                    np.dtype("<f4"),
                    (group_count, 1, n),
                )
                try:
                    code_offset = writer.start_tensor()
                    packed_digest = hashlib.sha256()
                    negative_zero_count = 0
                    packed_block_bytes = config.block_size // 2
                    for n_start in range(0, n, column_chunk):
                        n_end = min(n, n_start + column_chunk)
                        packed_columns = np.empty(
                            (n_end - n_start, block_count, packed_block_bytes), dtype=np.uint8
                        )
                        for group_index in range(group_count):
                            source_codes = np.asarray(code_values[group_index, :, n_start:n_end]).T
                            if np.any(
                                (source_codes != int(TERNARY_NEGATIVE_ZERO_CODE))
                                & (source_codes != int(TERNARY_NEGATIVE_CODE))
                                & (source_codes != int(TERNARY_POSITIVE_ZERO_CODE))
                                & (source_codes != int(TERNARY_POSITIVE_CODE))
                            ):
                                raise ValueError(f"invalid exact ternary code: {source_name}")
                            negative_zero_count += int(
                                np.count_nonzero(source_codes == TERNARY_NEGATIVE_ZERO_CODE)
                            )
                            q4_codes = np.where(
                                source_codes == TERNARY_NEGATIVE_CODE,
                                ZERO_POINTS[4] - 1,
                                np.where(
                                    source_codes == TERNARY_POSITIVE_CODE,
                                    ZERO_POINTS[4] + 1,
                                    ZERO_POINTS[4],
                                ),
                            ).astype(np.uint8)
                            for subblock in range(blocks_per_source_group):
                                block_index = group_index * blocks_per_source_group + subblock
                                block_start = subblock * config.block_size
                                source_block = source_codes[
                                    :, block_start : block_start + config.block_size
                                ]
                                q4_block = q4_codes[
                                    :, block_start : block_start + config.block_size
                                ]
                                scales = np.asarray(
                                    scale_values[group_index, 0, n_start:n_end], dtype=np.float32
                                )
                                reconstructed = (q4_block.astype(np.int8) - ZERO_POINTS[4]).astype(
                                    np.float32
                                )
                                reconstructed *= scales[:, None]
                                source_values = source_block.astype(np.float32)
                                source_values = np.where(
                                    source_block == TERNARY_NEGATIVE_ZERO_CODE,
                                    np.float32(-0.0),
                                    source_values,
                                )
                                source_values *= scales[:, None]
                                if not np.array_equal(reconstructed, source_values):
                                    raise ValueError(
                                        f"q4 reconstruction differs from exact-int8 source: {source_name}"
                                    )
                                packed_columns[:, block_index, :] = pack_codes(q4_block, 4)
                        payload = memoryview(packed_columns).cast("B")
                        writer.write(payload)
                        packed_digest.update(payload)
                    code_offset, code_length = writer.finish_tensor(code_offset)
                    if negative_zero_count != int(report.get("negative_zero_count", -1)):
                        raise ValueError(f"exact ternary negative-zero count changed: {source_name}")

                    scales_offset = writer.start_tensor()
                    scales_digest = hashlib.sha256()
                    for n_start in range(0, n, column_chunk):
                        n_end = min(n, n_start + column_chunk)
                        source_scales = np.asarray(
                            scale_values[:, 0, n_start:n_end], dtype=np.float32
                        ).T
                        if not np.isfinite(source_scales).all() or np.any(source_scales < 0):
                            raise ValueError(f"invalid exact ternary scales: {source_name}")
                        scales_block = np.repeat(source_scales, blocks_per_source_group, axis=1)
                        payload = memoryview(np.ascontiguousarray(scales_block)).cast("B")
                        writer.write(payload)
                        scales_digest.update(payload)
                    scales_offset, scales_length = writer.finish_tensor(scales_offset)

                    zero_offset = writer.start_tensor()
                    zero_digest = hashlib.sha256()
                    for n_start in range(0, n, column_chunk):
                        n_end = min(n, n_start + column_chunk)
                        zero_block = np.full(
                            (n_end - n_start, block_count), ZERO_POINTS[4], dtype=np.uint8
                        )
                        payload = memoryview(pack_zero_points(zero_block, 4)).cast("B")
                        writer.write(payload)
                        zero_digest.update(payload)
                    zero_offset, zero_length = writer.finish_tensor(zero_offset)
                finally:
                    for mapped in (code_values, scale_values):
                        mapped_file = getattr(mapped, "_mmap", None)
                        if mapped_file is not None and not mapped_file.closed:
                            mapped_file.close()

                output_initializers.extend(
                    [
                        make_external_tensor(
                            packed_name,
                            TensorProto.UINT8,
                            (n, block_count, config.block_size // 2),
                            temp_location,
                            code_offset,
                            code_length,
                        ),
                        make_external_tensor(
                            scales_name,
                            TensorProto.FLOAT,
                            (n * block_count,),
                            temp_location,
                            scales_offset,
                            scales_length,
                        ),
                        make_external_tensor(
                            zero_points_name,
                            TensorProto.UINT8,
                            (n * ((block_count + 1) // 2),),
                            temp_location,
                            zero_offset,
                            zero_length,
                        ),
                    ]
                )
                signed_zero_differences += negative_zero_count
                packed_code_bytes += code_length
                scale_bytes += scales_length
                zero_point_bytes += zero_length
                tensor_reports.append(
                    {
                        "source_initializer": source_name,
                        "source_shape": [k, n],
                        "source_sha256": report["source_sha256"],
                        "source_code_initializer": report["code_initializer"],
                        "source_scale_initializer": report["scale_initializer"],
                        "source_negative_zero_count": negative_zero_count,
                        "signed_zero_bit_differences": negative_zero_count,
                        "packed_initializer": packed_name,
                        "packed_code_bytes": code_length,
                        "packed_code_sha256": packed_digest.hexdigest(),
                        "scales_initializer": scales_name,
                        "scale_bytes": scales_length,
                        "scales_sha256": scales_digest.hexdigest(),
                        "zero_points_initializer": zero_points_name,
                        "zero_point_bytes": zero_length,
                        "zero_points_sha256": zero_digest.hexdigest(),
                        "exact_numeric_reconstruction": True,
                    }
                )

            for tensor in model.graph.initializer:
                if tensor.name not in kept_source_names:
                    continue
                copied = copy.deepcopy(tensor)
                if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
                    tensor_slice = resolve_external_data_range(
                        source_path.parent,
                        tensor,
                        logical_bytes=logical_tensor_bytes(tensor),
                        require_exact_length=True,
                    )
                    offset, length = writer.copy_range(tensor_slice)
                    _set_external_reference(copied, temp_location, offset, length)
                output_initializers.append(copied)

        del rewritten.graph.initializer[:]
        rewritten.graph.initializer.extend(output_initializers)
        onnx.save_model(rewritten, str(temporary_graph), save_as_external_data=False)
        onnx.checker.check_model(str(temporary_graph))
        import onnxruntime as ort

        ort.InferenceSession(str(temporary_graph), providers=["CPUExecutionProvider"])

        for tensor in rewritten.graph.initializer:
            if tensor.data_location != TensorProto.EXTERNAL and not tensor.external_data:
                continue
            external = {entry.key: entry.value for entry in tensor.external_data}
            if external.get("location") == temp_location:
                _set_external_reference(
                    tensor,
                    data_path.name,
                    int(external.get("offset", "0")),
                    int(external.get("length", str(logical_tensor_bytes(tensor)))),
                )
        onnx.save_model(rewritten, str(temporary_graph), save_as_external_data=False)
        os.replace(temporary_data, data_path)
        published_data = True
        os.replace(temporary_graph, output_path)
    except BaseException:
        temporary_graph.unlink(missing_ok=True)
        temporary_data.unlink(missing_ok=True)
        if published_data and not output_path.exists():
            data_path.unlink(missing_ok=True)
        raise

    transformed_node_names = [
        node_name for node_name in reports_by_node if node_name in selected_nodes
    ]
    return {
        "method": "ternary-exact-int8-to-nbits4",
        "bits": 4,
        "block_size": config.block_size,
        "source_group_size": SOURCE_GROUP_SIZE,
        "accuracy_level": config.accuracy_level,
        "source_graph_sha256": source_graph_sha256,
        "source_external_data_sha256": source_external_hashes,
        "graph_sha256": _sha256_file(output_path),
        "external_data_file": data_path.name,
        "external_data_sha256": {data_path.name: _sha256_file(data_path)},
        "transformed_matmul_count": len(transformed_node_names),
        "transformed_initializer_count": len(tensor_reports),
        "transformed_nodes": transformed_node_names,
        "excluded_nodes": sorted(excluded_node_names),
        "excluded_initializers": sorted(excluded_initializer_names),
        "exact_numeric_reconstruction": True,
        "signed_zero_bit_differences": signed_zero_differences,
        "packed_code_bytes": packed_code_bytes,
        "scale_bytes": scale_bytes,
        "zero_point_bytes": zero_point_bytes,
        "transformed_tensors": tensor_reports,
    }
