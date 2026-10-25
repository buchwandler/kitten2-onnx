"""Reference codec for compact group-local ternary weight storage."""

from __future__ import annotations

import copy
import hashlib
import json
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
    map_external_float32_tensor,
    resolve_external_data_range,
    resolve_external_tensor,
)
from kitten2_onnx.quantize_int8_streaming import ExternalDataWriter

TERNARY_TRITPACK_FORMAT_VERSION = "kitten2-ternary-tritpack-v1"
TRITS_PER_BYTE = 5
PACKED_BYTES_PER_GROUP = 26
MAX_PACKED_CODE = 242


@dataclass(frozen=True)
class TernaryTritpackConfig:
    group_size: int = 128
    trits_per_byte: int = TRITS_PER_BYTE
    packed_bytes_per_group: int = PACKED_BYTES_PER_GROUP
    scale_dtype: str = "float32"
    format_version: str = TERNARY_TRITPACK_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.group_size != 128:
            raise ValueError(f"group_size must be 128, found {self.group_size}")
        if self.trits_per_byte != TRITS_PER_BYTE:
            raise ValueError(f"trits_per_byte must be {TRITS_PER_BYTE}, found {self.trits_per_byte}")
        if self.packed_bytes_per_group != PACKED_BYTES_PER_GROUP:
            raise ValueError(
                "packed_bytes_per_group must be "
                f"{PACKED_BYTES_PER_GROUP}, found {self.packed_bytes_per_group}"
            )
        if self.scale_dtype != "float32":
            raise ValueError(f"scale_dtype must be 'float32', found {self.scale_dtype!r}")
        if self.format_version != TERNARY_TRITPACK_FORMAT_VERSION:
            raise ValueError(
                f"format_version must be {TERNARY_TRITPACK_FORMAT_VERSION!r}, "
                f"found {self.format_version!r}"
            )


def build_decode_lut() -> np.ndarray:
    """Return the 243 valid byte codes decoded as five signed ternary digits."""
    values = np.arange(MAX_PACKED_CODE + 1, dtype=np.int64)[:, None]
    powers = 3 ** np.arange(TRITS_PER_BYTE, dtype=np.int64)
    return ((values // powers) % 3 - 1).astype(np.int8)


def validate_ternary_group(
    group: np.ndarray,
    *,
    scales: np.ndarray | None = None,
    config: TernaryTritpackConfig | None = None,
) -> np.ndarray:
    """Validate a float32 ``[128, N]`` ternary group and return its scales."""
    config = config or TernaryTritpackConfig()
    values = np.asarray(group)
    if values.dtype != np.float32:
        raise TypeError(f"group must have dtype float32, found {values.dtype}")
    if values.ndim != 2 or values.shape[0] != config.group_size or values.shape[1] <= 0:
        raise ValueError(f"ternary group must have shape [128, N], found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("ternary group values must be finite")

    expected_scales = np.abs(values).max(axis=0).astype(np.float32, copy=False)
    if scales is None:
        validated_scales = expected_scales
    else:
        validated_scales = np.asarray(scales)
        if validated_scales.dtype != np.float32:
            raise TypeError(f"scales must have dtype float32, found {validated_scales.dtype}")
        if validated_scales.shape != (values.shape[1],):
            raise ValueError(
                f"scales must have shape ({values.shape[1]},), found {validated_scales.shape}"
            )
        if not np.isfinite(validated_scales).all() or np.any(validated_scales < 0):
            raise ValueError("scales must be finite and nonnegative")
        if not np.array_equal(validated_scales, expected_scales):
            raise ValueError("scales must equal the maximum absolute value in each column")

    valid_values = (
        (values == 0)
        | (values == validated_scales[None, :])
        | (values == -validated_scales[None, :])
    )
    if not valid_values.all():
        raise ValueError("group values must be exactly {-scale, 0, +scale} in each column")
    return validated_scales


def pack_group(
    group: np.ndarray,
    *,
    config: TernaryTritpackConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack one ``[128, N]`` ternary group to ``[N, 26]`` uint8 codes."""
    config = config or TernaryTritpackConfig()
    values = np.asarray(group)
    scales = validate_ternary_group(values, config=config)

    # Trit digits are -1 -> 0, 0 -> 1, +1 -> 2. Padding uses zero-valued
    # trits, so its two digits are deterministic and decode harmlessly.
    digits = np.ones((values.shape[1], config.group_size + 2), dtype=np.uint8)
    digits[:, : config.group_size] = np.where(values.T < 0, 0, np.where(values.T > 0, 2, 1))
    groups_of_five = digits.reshape(values.shape[1], config.packed_bytes_per_group, TRITS_PER_BYTE)
    powers = np.asarray([1, 3, 9, 27, 81], dtype=np.uint16)
    packed = (groups_of_five.astype(np.uint16) * powers).sum(axis=2).astype(np.uint8)
    return packed, scales


def unpack_group(
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    config: TernaryTritpackConfig | None = None,
) -> np.ndarray:
    """Decode ``[N, 26]`` uint8 codes and scales to a float32 ``[128, N]`` group."""
    config = config or TernaryTritpackConfig()
    codes = np.asarray(packed)
    scale_values = np.asarray(scales)
    if codes.dtype != np.uint8:
        raise TypeError(f"packed codes must have dtype uint8, found {codes.dtype}")
    if codes.ndim != 2 or codes.shape[0] <= 0 or codes.shape[1] != config.packed_bytes_per_group:
        raise ValueError(
            f"packed codes must have shape [N, {config.packed_bytes_per_group}], found {codes.shape}"
        )
    if np.any(codes > MAX_PACKED_CODE):
        raise ValueError(f"packed codes must be in the range 0..{MAX_PACKED_CODE}")
    if scale_values.dtype != np.float32:
        raise TypeError(f"scales must have dtype float32, found {scale_values.dtype}")
    if scale_values.shape != (codes.shape[0],):
        raise ValueError(f"scales must have shape ({codes.shape[0]},), found {scale_values.shape}")
    if not np.isfinite(scale_values).all() or np.any(scale_values < 0):
        raise ValueError("scales must be finite and nonnegative")

    digits = build_decode_lut()[codes].reshape(codes.shape[0], config.group_size + 2)
    ternary = digits[:, : config.group_size].T.astype(np.float32)
    return ternary * scale_values[None, :]


def pack_weight_reference(
    weight: np.ndarray,
    *,
    config: TernaryTritpackConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference-pack a float32 ``[K, N]`` weight into group-local codes/scales."""
    config = config or TernaryTritpackConfig()
    values = np.asarray(weight)
    if values.dtype != np.float32:
        raise TypeError(f"weight must have dtype float32, found {values.dtype}")
    if values.ndim != 2 or any(dimension <= 0 for dimension in values.shape):
        raise ValueError(f"weight must have two positive dimensions [K, N], found {values.shape}")
    if values.shape[0] % config.group_size:
        raise ValueError(f"weight K dimension must be divisible by {config.group_size}")

    group_count = values.shape[0] // config.group_size
    column_count = values.shape[1]
    packed = np.empty(
        (group_count, column_count, config.packed_bytes_per_group), dtype=np.uint8
    )
    scales = np.empty((group_count, 1, column_count), dtype=np.float32)
    for group_index in range(group_count):
        start = group_index * config.group_size
        end = start + config.group_size
        packed[group_index], scales[group_index, 0] = pack_group(
            values[start:end], config=config
        )
    return packed, scales


def unpack_weight_reference(
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    config: TernaryTritpackConfig | None = None,
) -> np.ndarray:
    """Reference-unpack codes/scales to their original two-dimensional weight shape."""
    config = config or TernaryTritpackConfig()
    codes = np.asarray(packed)
    scale_values = np.asarray(scales)
    if codes.dtype != np.uint8:
        raise TypeError(f"packed codes must have dtype uint8, found {codes.dtype}")
    if codes.ndim != 3 or any(dimension <= 0 for dimension in codes.shape):
        raise ValueError(f"packed weight must have shape [groups, N, 26], found {codes.shape}")
    group_count, column_count, packed_bytes = codes.shape
    if packed_bytes != config.packed_bytes_per_group:
        raise ValueError(
            f"packed weight trailing dimension must be {config.packed_bytes_per_group}, "
            f"found {packed_bytes}"
        )
    if scale_values.dtype != np.float32:
        raise TypeError(f"scales must have dtype float32, found {scale_values.dtype}")
    if scale_values.shape != (group_count, 1, column_count):
        raise ValueError(
            "scales must have shape "
            f"({group_count}, 1, {column_count}), found {scale_values.shape}"
        )

    weight = np.empty((group_count * config.group_size, column_count), dtype=np.float32)
    for group_index in range(group_count):
        start = group_index * config.group_size
        end = start + config.group_size
        weight[start:end] = unpack_group(
            codes[group_index], scale_values[group_index, 0], config=config
        )
    return weight



def _graph_names(graph: onnx.GraphProto) -> set[str]:
    names = {value.name for value in (*graph.input, *graph.output, *graph.value_info)}
    names.update(tensor.name for tensor in graph.initializer)
    for node in graph.node:
        if node.name:
            names.add(node.name)
        names.update(name for name in (*node.input, *node.output) if name)
    return names


def _unique_graph_name(base: str, used_names: set[str]) -> str:
    name = base
    suffix = 1
    while name in used_names:
        name = f"{base}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name


def rewrite_matmul_to_tritpack_storage(
    model: onnx.ModelProto,
    *,
    node_name: str,
    config: TernaryTritpackConfig | None = None,
) -> onnx.ModelProto:
    """Rewrite one inline float32 MatMul weight using lookup-based tritpack decoding."""
    config = config or TernaryTritpackConfig()
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
        raise ValueError("external-data weights require a streaming tritpack transform")
    if tensor.data_type != TensorProto.FLOAT:
        raise TypeError("target MatMul weight initializer must be float32")

    weight = np.asarray(numpy_helper.to_array(tensor))
    codes, scales = pack_weight_reference(weight, config=config)
    group_count, column_count, _ = codes.shape
    used_names = _graph_names(rewritten.graph)

    codes_name = _unique_graph_name(f"{weight_name}_tritpack_codes", used_names)
    scales_name = _unique_graph_name(f"{weight_name}_tritpack_scales", used_names)
    lut_name = _unique_graph_name("ternary_tritpack_decode_lut", used_names)
    decoded_shape_name = _unique_graph_name(f"{weight_name}_tritpack_decoded_shape", used_names)
    starts_name = _unique_graph_name(f"{weight_name}_tritpack_slice_starts", used_names)
    ends_name = _unique_graph_name(f"{weight_name}_tritpack_slice_ends", used_names)
    axes_name = _unique_graph_name(f"{weight_name}_tritpack_slice_axes", used_names)
    weight_shape_name = _unique_graph_name(f"{weight_name}_tritpack_weight_shape", used_names)

    code_indices_name = _unique_graph_name(f"{weight_name}_tritpack_code_indices", used_names)
    decoded_name = _unique_graph_name(f"{weight_name}_tritpack_decoded", used_names)
    flattened_name = _unique_graph_name(f"{weight_name}_tritpack_flattened", used_names)
    logical_trits_name = _unique_graph_name(f"{weight_name}_tritpack_logical_trits", used_names)
    ordered_trits_name = _unique_graph_name(f"{weight_name}_tritpack_ordered_trits", used_names)
    float_trits_name = _unique_graph_name(f"{weight_name}_tritpack_float_trits", used_names)
    grouped_weight_name = _unique_graph_name(f"{weight_name}_tritpack_grouped_weight", used_names)
    reconstructed_name = _unique_graph_name(f"{weight_name}_tritpack_weight", used_names)

    rewritten.graph.initializer.extend(
        [
            numpy_helper.from_array(codes, name=codes_name),
            numpy_helper.from_array(scales, name=scales_name),
            numpy_helper.from_array(build_decode_lut(), name=lut_name),
            numpy_helper.from_array(
                np.asarray([group_count, column_count, config.group_size + 2], dtype=np.int64),
                name=decoded_shape_name,
            ),
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), name=starts_name),
            numpy_helper.from_array(
                np.asarray([config.group_size], dtype=np.int64), name=ends_name
            ),
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), name=axes_name),
            numpy_helper.from_array(np.asarray(weight.shape, dtype=np.int64), name=weight_shape_name),
        ]
    )

    additions = [
        helper.make_node(
            "Cast",
            [codes_name],
            [code_indices_name],
            name=_unique_graph_name(f"{node_name}_tritpack_indices_cast", used_names),
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Gather",
            [lut_name, code_indices_name],
            [decoded_name],
            name=_unique_graph_name(f"{node_name}_tritpack_gather", used_names),
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            [decoded_name, decoded_shape_name],
            [flattened_name],
            name=_unique_graph_name(f"{node_name}_tritpack_flatten", used_names),
        ),
        helper.make_node(
            "Slice",
            [flattened_name, starts_name, ends_name, axes_name],
            [logical_trits_name],
            name=_unique_graph_name(f"{node_name}_tritpack_slice_padding", used_names),
        ),
        helper.make_node(
            "Transpose",
            [logical_trits_name],
            [ordered_trits_name],
            name=_unique_graph_name(f"{node_name}_tritpack_transpose", used_names),
            perm=[0, 2, 1],
        ),
        helper.make_node(
            "Cast",
            [ordered_trits_name],
            [float_trits_name],
            name=_unique_graph_name(f"{node_name}_tritpack_float_cast", used_names),
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Mul",
            [float_trits_name, scales_name],
            [grouped_weight_name],
            name=_unique_graph_name(f"{node_name}_tritpack_scale", used_names),
        ),
        helper.make_node(
            "Reshape",
            [grouped_weight_name, weight_shape_name],
            [reconstructed_name],
            name=_unique_graph_name(f"{node_name}_tritpack_restore_shape", used_names),
        ),
    ]

    node_index = next(index for index, candidate in enumerate(rewritten.graph.node) if candidate is node)
    node.input[1] = reconstructed_name
    for offset, addition in enumerate(additions):
        rewritten.graph.node.insert(node_index + offset, addition)

    consumers = [candidate for candidate in rewritten.graph.node if weight_name in candidate.input]
    graph_values = {value.name for value in (*rewritten.graph.input, *rewritten.graph.output)}
    if not consumers and weight_name not in graph_values:
        rewritten.graph.initializer.remove(tensor)
    return rewritten



def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_data_hashes(model_path: Path, model: onnx.ModelProto) -> dict[str, str]:
    locations: dict[str, Path] = {}
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL and not tensor.external_data:
            continue
        tensor_slice = resolve_external_data_range(
            model_path.parent,
            tensor,
            logical_bytes=logical_tensor_bytes(tensor),
        )
        locations[tensor_slice.location] = tensor_slice.path
    return {location: _sha256_file(path) for location, path in sorted(locations.items())}


def _hash_external_range(tensor_slice) -> str:
    digest = hashlib.sha256()
    with tensor_slice.path.open("rb") as stream:
        stream.seek(tensor_slice.offset)
        remaining = tensor_slice.length
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"external tensor ended early: {tensor_slice.path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _set_external_reference(
    tensor: onnx.TensorProto, location: str, offset: int, length: int
) -> None:
    entries = {
        entry.key: entry.value
        for entry in tensor.external_data
        if entry.key not in {"location", "offset", "length", "basepath"}
    }
    entries.update({"location": location, "offset": str(offset), "length": str(length)})
    del tensor.external_data[:]
    for key, value in entries.items():
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    tensor.data_location = TensorProto.EXTERNAL


def _load_float32_matrix(
    model_path: Path, tensor: onnx.TensorProto
) -> tuple[np.ndarray, bool]:
    if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) != 2:
        raise ValueError(f"MatMul weight must be a rank-2 float32 initializer: {tensor.name}")
    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        tensor_slice = resolve_external_tensor(model_path, tensor)
        return map_external_float32_tensor(tensor_slice, tuple(tensor.dims)), True
    return np.asarray(numpy_helper.to_array(tensor)), False


def _close_matrix(weight: np.ndarray, is_memmap: bool) -> None:
    if is_memmap:
        weight._mmap.close()


def _sha256_matrix(weight: np.ndarray) -> str:
    digest = hashlib.sha256()
    columns = weight.shape[1]
    rows_per_chunk = max(1, (8 * 1024 * 1024) // (columns * np.dtype(np.float32).itemsize))
    for row_start in range(0, weight.shape[0], rows_per_chunk):
        block = np.ascontiguousarray(weight[row_start : row_start + rows_per_chunk])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _report_stage(
    callback: Callable[[str, dict[str, Any]], None] | None,
    stage: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback(stage, details)


def _verify_tritpack_candidate(
    source_path: Path,
    candidate_path: Path,
    initializer_reports: list[dict[str, Any]],
    *,
    column_chunk: int,
) -> None:
    candidate = onnx.load(str(candidate_path), load_external_data=False)
    candidate_initializers = {tensor.name: tensor for tensor in candidate.graph.initializer}
    source = onnx.load(str(source_path), load_external_data=False)
    source_initializers = {tensor.name: tensor for tensor in source.graph.initializer}

    for report in initializer_reports:
        source_tensor = source_initializers[report["source_initializer"]]
        codes_tensor = candidate_initializers[report["code_initializer"]]
        scales_tensor = candidate_initializers[report["scale_initializer"]]
        k, n = report["source_shape"]
        group_count = k // 128
        expected_code_shape = (group_count, n, PACKED_BYTES_PER_GROUP)
        expected_scale_shape = (group_count, 1, n)
        if (
            codes_tensor.data_type != TensorProto.UINT8
            or tuple(codes_tensor.dims) != expected_code_shape
            or codes_tensor.data_location != TensorProto.EXTERNAL
        ):
            raise ValueError(f"tritpack code initializer has invalid storage: {codes_tensor.name}")
        if (
            scales_tensor.data_type != TensorProto.FLOAT
            or tuple(scales_tensor.dims) != expected_scale_shape
            or scales_tensor.data_location != TensorProto.EXTERNAL
        ):
            raise ValueError(f"tritpack scale initializer has invalid storage: {scales_tensor.name}")

        code_slice = resolve_external_data_range(
            candidate_path.parent,
            codes_tensor,
            logical_bytes=logical_tensor_bytes(codes_tensor),
            require_exact_length=True,
        )
        scale_slice = resolve_external_data_range(
            candidate_path.parent,
            scales_tensor,
            logical_bytes=logical_tensor_bytes(scales_tensor),
            require_exact_length=True,
        )
        if _hash_external_range(code_slice) != report["code_sha256"]:
            raise ValueError(f"tritpack code hash mismatch: {codes_tensor.name}")
        if _hash_external_range(scale_slice) != report["scale_sha256"]:
            raise ValueError(f"tritpack scale hash mismatch: {scales_tensor.name}")

        codes = np.memmap(
            code_slice.path,
            mode="r",
            dtype=np.uint8,
            offset=code_slice.offset,
            shape=expected_code_shape,
            order="C",
        )
        scales = np.memmap(
            scale_slice.path,
            mode="r",
            dtype="<f4",
            offset=scale_slice.offset,
            shape=expected_scale_shape,
            order="C",
        )
        source_weight, source_is_memmap = _load_float32_matrix(source_path, source_tensor)
        try:
            for group_index in range(group_count):
                row_start = group_index * 128
                row_end = row_start + 128
                for column_start in range(0, n, column_chunk):
                    column_end = min(n, column_start + column_chunk)
                    code_block = np.asarray(codes[group_index, column_start:column_end])
                    scale_block = np.asarray(scales[group_index, 0, column_start:column_end])
                    if np.any(code_block > MAX_PACKED_CODE):
                        raise ValueError(f"invalid tritpack code byte in {codes_tensor.name}")
                    if not np.isfinite(scale_block).all() or np.any(scale_block < 0):
                        raise ValueError(f"invalid tritpack scale in {scales_tensor.name}")
                    expected = np.asarray(source_weight[row_start:row_end, column_start:column_end])
                    if not np.array_equal(scale_block, np.abs(expected).max(axis=0)):
                        raise ValueError(f"tritpack scales do not match source weight: {source_tensor.name}")
                    actual = unpack_group(code_block, scale_block)
                    if not np.array_equal(actual, expected):
                        raise ValueError(f"tritpack reconstruction mismatch: {source_tensor.name}")
                    nonzero = expected != 0
                    if not np.array_equal(actual[nonzero].view(np.uint32), expected[nonzero].view(np.uint32)):
                        raise ValueError(f"tritpack nonzero reconstruction is not bit-identical: {source_tensor.name}")
        finally:
            _close_matrix(source_weight, source_is_memmap)
            codes._mmap.close()
            scales._mmap.close()




def build_ternary_tritpack_inventory(
    source_model: str | Path,
    *,
    expected_eligible_matmul_count: int = 196,
    column_chunk: int = 64,
    config: TernaryTritpackConfig | None = None,
) -> dict[str, Any]:
    """Inventory exact ternary MatMuls directly from one source ONNX graph."""
    config = config or TernaryTritpackConfig()
    if expected_eligible_matmul_count < 0:
        raise ValueError("expected eligible MatMul count must be nonnegative")
    if column_chunk <= 0:
        raise ValueError("column_chunk must be positive")

    source_path = Path(source_model).resolve()
    model = onnx.load(str(source_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    inspected: dict[str, tuple[bool, list[int] | None, str | None]] = {}
    rows: list[dict[str, Any]] = []
    eligible_count = 0

    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "MatMul":
            continue
        node_name = node.name or f"MatMul#{node_index}"
        initializer_name = node.input[1] if len(node.input) == 2 else None
        tensor = initializers.get(initializer_name or "")
        eligible = False
        shape: list[int] | None = None
        source_sha256: str | None = None
        if tensor is not None and tensor.data_type == TensorProto.FLOAT and len(tensor.dims) == 2:
            if initializer_name not in inspected:
                tensor_shape = [int(value) for value in tensor.dims]
                is_eligible = (
                    tensor_shape[0] > 0
                    and tensor_shape[1] > 0
                    and tensor_shape[0] % config.group_size == 0
                )
                weight, is_memmap = _load_float32_matrix(source_path, tensor)
                try:
                    source_hash = _sha256_matrix(weight)
                    if is_eligible:
                        for row_start in range(0, tensor_shape[0], config.group_size):
                            row_end = row_start + config.group_size
                            for column_start in range(0, tensor_shape[1], column_chunk):
                                column_end = min(tensor_shape[1], column_start + column_chunk)
                                try:
                                    pack_group(
                                        weight[row_start:row_end, column_start:column_end],
                                        config=config,
                                    )
                                except ValueError:
                                    is_eligible = False
                                    break
                            if not is_eligible:
                                break
                finally:
                    _close_matrix(weight, is_memmap)
                inspected[initializer_name] = (is_eligible, tensor_shape, source_hash)
            eligible, shape, source_sha256 = inspected[initializer_name]
        if eligible:
            eligible_count += 1
        rows.append(
            {
                "node": node_name,
                "initializer": initializer_name,
                "shape": shape,
                "source_sha256": source_sha256,
                "eligible_exact_ternary": eligible,
            }
        )

    if eligible_count != expected_eligible_matmul_count:
        raise ValueError(
            f"Expected {expected_eligible_matmul_count} exact ternary MatMuls, found {eligible_count}"
        )

    source_graph_sha256 = _sha256_file(source_path)
    source_external_data_sha256 = _external_data_hashes(source_path, model)
    inventory_identity = {
        "format": "kitten2-onnx-ternary-tritpack-source-inventory-v1",
        "source_graph_sha256": source_graph_sha256,
        "source_external_data_sha256": source_external_data_sha256,
        "nodes": rows,
    }
    inventory_json = json.dumps(
        inventory_identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **inventory_identity,
        "source_graph": str(source_path),
        "matmul_count": len(rows),
        "eligible_exact_ternary_matmul_count": eligible_count,
        "eligible_exact_ternary_initializer_count": len(
            {row["initializer"] for row in rows if row["eligible_exact_ternary"]}
        ),
        "inventory_sha256": hashlib.sha256(inventory_json).hexdigest(),
    }


def compress_ternary_matmuls_to_tritpack_streaming(
    source_model: str | Path,
    output_model: str | Path,
    output_data: str | Path,
    *,
    inventory: dict[str, Any],
    column_chunk: int = 64,
    exclude_nodes: set[str] | None = None,
    config: TernaryTritpackConfig | None = None,
    stage_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Stream-rewrite inventoried ternary MatMuls to group-local tritpack storage."""
    config = config or TernaryTritpackConfig()
    if column_chunk <= 0:
        raise ValueError("column_chunk must be positive")

    source_path = Path(source_model).resolve()
    output_path = Path(output_model).resolve()
    data_path = Path(output_data).resolve()
    if source_path in {output_path, data_path} or output_path == data_path:
        raise ValueError("source model and candidate outputs must be distinct")
    if output_path.parent != data_path.parent:
        raise ValueError("candidate graph and external data must share a directory")
    if output_path.exists() or data_path.exists():
        raise FileExistsError("candidate output path already exists")

    eligible_count = int(inventory.get("eligible_exact_ternary_matmul_count", -1))
    rows_by_node: dict[str, dict[str, Any]] = {}
    for row in inventory.get("nodes", []):
        if not row.get("eligible_exact_ternary"):
            continue
        node_name = row.get("node")
        if not node_name or node_name in rows_by_node:
            raise ValueError(f"inventory has a missing or duplicate MatMul node: {node_name!r}")
        rows_by_node[node_name] = row
    if not rows_by_node or len(rows_by_node) != eligible_count:
        raise ValueError("inventory eligible MatMul count does not match its node rows")
    excluded = set(exclude_nodes or ())
    unknown_exclusions = excluded - rows_by_node.keys()
    if unknown_exclusions:
        raise ValueError(f"unknown eligible MatMul exclusions: {sorted(unknown_exclusions)}")
    selected_rows = {name: row for name, row in rows_by_node.items() if name not in excluded}
    if not selected_rows:
        raise ValueError("no exact group-128 ternary MatMuls selected")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_name(f".{data_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_graph = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    published_data = False
    published_graph = False
    try:
        _report_stage(stage_callback, "load_graph")
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
        source_external_data_sha256 = _external_data_hashes(source_path, model)
        inventory_external_hashes = inventory.get("source_external_data_sha256")
        if inventory_external_hashes is not None and inventory_external_hashes != source_external_data_sha256:
            raise ValueError("inventory source external data hashes do not match the source model")

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
            shape = [int(value) for value in tensor.dims]
            if tensor.data_type != TensorProto.FLOAT or len(shape) != 2 or shape[0] % 128:
                raise ValueError(f"MatMul weight has unsupported tritpack shape: {initializer_name}")
            if row.get("shape") != shape:
                raise ValueError(f"inventory shape does not match initializer {initializer_name!r}")
            targets_by_weight[initializer_name].append((node_name, node_index, node))

        consumers: dict[str, set[int]] = defaultdict(set)
        for index, node in enumerate(model.graph.node):
            for name in node.input:
                if name:
                    consumers[name].add(index)
        graph_values = {
            value.name for value in (*model.graph.input, *model.graph.output)
        }
        removed_weights = set(targets_by_weight)
        for initializer_name, rows in targets_by_weight.items():
            target_indices = {index for _, index, _ in rows}
            if initializer_name in graph_values or consumers[initializer_name] != target_indices:
                raise ValueError(
                    f"tritpack target weight has non-MatMul consumers: {initializer_name}"
                )

        used_names = _graph_names(model.graph)
        lut_name = _unique_graph_name("ternary_tritpack_decode_lut", used_names)
        starts_name = _unique_graph_name("ternary_tritpack_slice_starts", used_names)
        ends_name = _unique_graph_name("ternary_tritpack_slice_ends", used_names)
        axes_name = _unique_graph_name("ternary_tritpack_slice_axes", used_names)
        extra_initializers = [
            numpy_helper.from_array(build_decode_lut(), name=lut_name),
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), name=starts_name),
            numpy_helper.from_array(
                np.asarray([config.group_size], dtype=np.int64), name=ends_name
            ),
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), name=axes_name),
        ]
        names_by_weight: dict[str, dict[str, str]] = {}
        for initializer_name in targets_by_weight:
            names_by_weight[initializer_name] = {
                role: _unique_graph_name(f"{initializer_name}_tritpack_{role}", used_names)
                for role in (
                    "codes",
                    "scales",
                    "decoded_shape",
                    "weight_shape",
                    "code_indices",
                    "decoded",
                    "flattened",
                    "logical_trits",
                    "ordered_trits",
                    "float_trits",
                    "grouped_weight",
                    "weight",
                )
            }

        external_location = temporary_data.name
        new_nodes_by_index: dict[int, list[onnx.NodeProto]] = defaultdict(list)
        initializer_reports: list[dict[str, Any]] = []
        transformed_nodes: list[str] = []
        source_fp32_bytes = 0
        total_code_bytes = 0
        total_scale_bytes = 0

        with ExternalDataWriter(temporary_data) as writer:
            _report_stage(stage_callback, "copy_unmodified_initializers")
            new_initializers = list(extra_initializers)
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

            for weight_index, (initializer_name, rows) in enumerate(
                targets_by_weight.items(), start=1
            ):
                _report_stage(
                    stage_callback,
                    f"encode_weight_{weight_index:03d}_of_{len(targets_by_weight):03d}",
                    initializer=initializer_name,
                    nodes=[name for name, _, _ in rows],
                )
                tensor = initializers[initializer_name]
                k, n = (int(value) for value in tensor.dims)
                group_count = k // config.group_size
                names = names_by_weight[initializer_name]
                weight, is_memmap = _load_float32_matrix(source_path, tensor)
                try:
                    source_sha256 = _sha256_matrix(weight)
                    expected_hashes = {
                        row.get("source_sha256")
                        for row in selected_rows.values()
                        if row.get("initializer") == initializer_name
                    }
                    expected_hashes.discard(None)
                    if expected_hashes and expected_hashes != {source_sha256}:
                        raise ValueError(
                            f"inventory source hash does not match initializer {initializer_name!r}"
                        )

                    code_offset = writer.start_tensor()
                    code_digest = hashlib.sha256()
                    for group_index in range(group_count):
                        row_start = group_index * config.group_size
                        row_end = row_start + config.group_size
                        for column_start in range(0, n, column_chunk):
                            column_end = min(n, column_start + column_chunk)
                            group = np.asarray(weight[row_start:row_end, column_start:column_end])
                            code_block, _ = pack_group(group, config=config)
                            code_block = np.ascontiguousarray(code_block)
                            payload = memoryview(code_block).cast("B")
                            writer.write(payload)
                            code_digest.update(payload)
                    code_offset, code_bytes = writer.finish_tensor(code_offset)
                    expected_code_bytes = group_count * n * config.packed_bytes_per_group
                    if code_bytes != expected_code_bytes:
                        raise ValueError(f"tritpack code byte count mismatch: {initializer_name}")

                    scale_offset = writer.start_tensor()
                    scale_digest = hashlib.sha256()
                    for group_index in range(group_count):
                        row_start = group_index * config.group_size
                        row_end = row_start + config.group_size
                        for column_start in range(0, n, column_chunk):
                            column_end = min(n, column_start + column_chunk)
                            group = np.asarray(weight[row_start:row_end, column_start:column_end])
                            scale_block = np.ascontiguousarray(
                                validate_ternary_group(group, config=config)
                            )
                            payload = memoryview(scale_block).cast("B")
                            writer.write(payload)
                            scale_digest.update(payload)
                    scale_offset, scale_bytes = writer.finish_tensor(scale_offset)
                    expected_scale_bytes = group_count * n * np.dtype(np.float32).itemsize
                    if scale_bytes != expected_scale_bytes:
                        raise ValueError(f"tritpack scale byte count mismatch: {initializer_name}")
                finally:
                    _close_matrix(weight, is_memmap)

                codes_tensor = make_external_tensor(
                    names["codes"],
                    TensorProto.UINT8,
                    (group_count, n, config.packed_bytes_per_group),
                    external_location,
                    code_offset,
                    code_bytes,
                )
                scales_tensor = make_external_tensor(
                    names["scales"],
                    TensorProto.FLOAT,
                    (group_count, 1, n),
                    external_location,
                    scale_offset,
                    scale_bytes,
                )
                new_initializers.extend(
                    [
                        codes_tensor,
                        scales_tensor,
                        numpy_helper.from_array(
                            np.asarray([group_count, n, config.group_size + 2], dtype=np.int64),
                            name=names["decoded_shape"],
                        ),
                        numpy_helper.from_array(
                            np.asarray([k, n], dtype=np.int64), name=names["weight_shape"]
                        ),
                    ]
                )

                node_name, first_index, _ = min(rows, key=lambda row: row[1])
                nodes = [
                    helper.make_node(
                        "Cast",
                        [names["codes"]],
                        [names["code_indices"]],
                        name=_unique_graph_name(f"{node_name}_tritpack_indices_cast", used_names),
                        to=TensorProto.INT64,
                    ),
                    helper.make_node(
                        "Gather",
                        [lut_name, names["code_indices"]],
                        [names["decoded"]],
                        name=_unique_graph_name(f"{node_name}_tritpack_gather", used_names),
                        axis=0,
                    ),
                    helper.make_node(
                        "Reshape",
                        [names["decoded"], names["decoded_shape"]],
                        [names["flattened"]],
                        name=_unique_graph_name(f"{node_name}_tritpack_flatten", used_names),
                    ),
                    helper.make_node(
                        "Slice",
                        [names["flattened"], starts_name, ends_name, axes_name],
                        [names["logical_trits"]],
                        name=_unique_graph_name(f"{node_name}_tritpack_slice_padding", used_names),
                    ),
                    helper.make_node(
                        "Transpose",
                        [names["logical_trits"]],
                        [names["ordered_trits"]],
                        name=_unique_graph_name(f"{node_name}_tritpack_transpose", used_names),
                        perm=[0, 2, 1],
                    ),
                    helper.make_node(
                        "Cast",
                        [names["ordered_trits"]],
                        [names["float_trits"]],
                        name=_unique_graph_name(f"{node_name}_tritpack_float_cast", used_names),
                        to=TensorProto.FLOAT,
                    ),
                    helper.make_node(
                        "Mul",
                        [names["float_trits"], names["scales"]],
                        [names["grouped_weight"]],
                        name=_unique_graph_name(f"{node_name}_tritpack_scale", used_names),
                    ),
                    helper.make_node(
                        "Reshape",
                        [names["grouped_weight"], names["weight_shape"]],
                        [names["weight"]],
                        name=_unique_graph_name(f"{node_name}_tritpack_restore_shape", used_names),
                    ),
                ]
                new_nodes_by_index[first_index].extend(nodes)
                for transformed_node, _, matmul in rows:
                    matmul.input[1] = names["weight"]
                    transformed_nodes.append(transformed_node)

                initializer_reports.append(
                    {
                        "source_initializer": initializer_name,
                        "weight_value": names["weight"],
                        "source_shape": [k, n],
                        "source_sha256": source_sha256,
                        "code_initializer": names["codes"],
                        "code_bytes": code_bytes,
                        "code_sha256": code_digest.hexdigest(),
                        "scale_initializer": names["scales"],
                        "scale_bytes": scale_bytes,
                        "scale_sha256": scale_digest.hexdigest(),
                        "transformed_nodes": [name for name, _, _ in rows],
                    }
                )
                total_code_bytes += code_bytes
                total_scale_bytes += scale_bytes
                source_fp32_bytes += k * n * np.dtype(np.float32).itemsize

        model.graph.ClearField("initializer")
        model.graph.initializer.extend(new_initializers)
        rewritten_nodes: list[onnx.NodeProto] = []
        for index, node in enumerate(model.graph.node):
            rewritten_nodes.extend(new_nodes_by_index.get(index, ()))
            rewritten_nodes.append(node)
        model.graph.ClearField("node")
        model.graph.node.extend(rewritten_nodes)

        _report_stage(stage_callback, "write_candidate_graph")
        onnx.save_model(model, str(temporary_graph), save_as_external_data=False)
        _report_stage(stage_callback, "onnx_checker")
        onnx.checker.check_model(str(temporary_graph))
        _report_stage(stage_callback, "verify_tritpack_storage")
        _verify_tritpack_candidate(
            source_path,
            temporary_graph,
            initializer_reports,
            column_chunk=column_chunk,
        )

        temporary_location = temporary_data.relative_to(output_path.parent).as_posix()
        final_location = data_path.relative_to(output_path.parent).as_posix()
        for tensor in model.graph.initializer:
            if tensor.data_location != TensorProto.EXTERNAL and not tensor.external_data:
                continue
            external = {entry.key: entry.value for entry in tensor.external_data}
            if external.get("location") == temporary_location:
                _set_external_reference(
                    tensor,
                    final_location,
                    int(external.get("offset", "0")),
                    int(external.get("length", str(logical_tensor_bytes(tensor)))),
                )
        onnx.save_model(model, str(temporary_graph), save_as_external_data=False)

        report = {
            "format": "kitten2-onnx-ternary-tritpack-storage-v1",
            "method": "ternary-tritpack-exact",
            "format_version": config.format_version,
            "source_graph_sha256": source_graph_sha256,
            "source_external_data_sha256": source_external_data_sha256,
            "graph_sha256": _sha256_file(temporary_graph),
            "external_data_sha256": _sha256_file(temporary_data),
            "transformed_matmul_count": len(transformed_nodes),
            "transformed_initializer_count": len(initializer_reports),
            "transformed_nodes": transformed_nodes,
            "group_size": config.group_size,
            "trits_per_byte": config.trits_per_byte,
            "packed_bytes_per_group": config.packed_bytes_per_group,
            "code_bytes": total_code_bytes,
            "packed_code_bytes": total_code_bytes,
            "scale_bytes": total_scale_bytes,
            "logical_fp32_operand_bytes": source_fp32_bytes,
            "compression_ratio": (
                (total_code_bytes + total_scale_bytes) / source_fp32_bytes
                if source_fp32_bytes
                else None
            ),
            "numeric_reconstruction_exact": True,
            "signed_zero_preserved": False,
            "storage_reconstruction_verified": True,
            "inventory_sha256": inventory.get("inventory_sha256"),
            "excluded_nodes": sorted(excluded),
            "excluded_initializers": sorted(
                {rows_by_node[name]["initializer"] for name in excluded}
            ),
            "initializer_reports": initializer_reports,
        }
        os.replace(temporary_data, data_path)
        published_data = True
        os.replace(temporary_graph, output_path)
        published_graph = True
        onnx.checker.check_model(str(output_path))
        return report
    except BaseException:
        temporary_graph.unlink(missing_ok=True)
        temporary_data.unlink(missing_ok=True)
        if published_graph:
            output_path.unlink(missing_ok=True)
        if published_data:
            data_path.unlink(missing_ok=True)
        raise
