"""Lossless INT8 storage for exact group-128 ternary MatMul weights."""

from __future__ import annotations

import copy
import hashlib
import os
import uuid
from collections import defaultdict
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
    _close_memmap,
    _graph_names,
    _is_exact_ternary_weight,
    _load_float32_weight,
    _unique_graph_name,
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


def compress_ternary_matmuls_streaming(
    source_model: str | Path,
    output_model: str | Path,
    output_data: str | Path,
    *,
    inventory: dict[str, Any],
    exclude_nodes: set[str] | None = None,
    config: TernaryExactStorageConfig | None = None,
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
        model = onnx.load(str(source_path), load_external_data=False)
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
        source_external_data_sha256 = _external_source_hashes(source_path, model)

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
        for initializer_name, rows in rows_by_weight.items():
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

            for initializer_name, rows in targets_by_weight.items():
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
        onnx.save_model(model, str(temporary_graph), save_as_external_data=False)
        onnx.checker.check_model(str(temporary_graph))
        import onnxruntime as ort

        ort.InferenceSession(str(temporary_graph), providers=["CPUExecutionProvider"])

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

        os.replace(temporary_data, data_path)
        published_data = True
        os.replace(temporary_graph, output_path)
        return report
    except BaseException:
        temporary_graph.unlink(missing_ok=True)
        temporary_data.unlink(missing_ok=True)
        if published_data and not output_path.exists():
            data_path.unlink(missing_ok=True)
        raise
