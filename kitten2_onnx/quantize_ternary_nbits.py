"""Encode exact group-128 ternary MatMul weights for ONNX Runtime MatMulNBits."""

from __future__ import annotations

import copy
import hashlib
import os
import uuid
from collections import defaultdict
from collections.abc import Sequence
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

SOURCE_GROUP_SIZE = 128
BLOCK_SIZE = SOURCE_GROUP_SIZE  # Backwards-compatible name for the exact source group.
SUPPORTED_BLOCK_SIZES = (16, 32, 64, 128)


@dataclass(frozen=True)
class TernaryNBitsConfig:
    bits: int = 4
    block_size: int = 128
    source_group_size: int = SOURCE_GROUP_SIZE
    accuracy_level: int = 1

    def __post_init__(self) -> None:
        if self.bits not in (2, 4):
            raise ValueError(f"bits must be 2 or 4, found {self.bits}")
        if self.source_group_size != SOURCE_GROUP_SIZE:
            raise ValueError(
                f"source_group_size must remain {SOURCE_GROUP_SIZE}, found {self.source_group_size}"
            )
        if self.block_size not in SUPPORTED_BLOCK_SIZES:
            raise ValueError(
                f"block_size must be one of {SUPPORTED_BLOCK_SIZES}, found {self.block_size}"
            )
        if self.source_group_size % self.block_size:
            raise ValueError(
                f"source_group_size {self.source_group_size} must be divisible by block_size {self.block_size}"
            )
        if self.accuracy_level not in (0, 1):
            raise ValueError(f"accuracy_level must be 0 or 1, found {self.accuracy_level}")


ZERO_POINTS = {2: 1, 4: 8}


def codes_per_byte(bits: int) -> int:
    if bits not in ZERO_POINTS:
        raise ValueError(f"bits must be 2 or 4, found {bits}")
    return 8 // bits


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack low-bit-first unsigned codes into bytes."""
    lanes = codes_per_byte(bits)
    codes = np.asarray(codes)
    if codes.dtype.kind not in "iu":
        raise TypeError(f"codes must have an integer dtype, found {codes.dtype}")
    if codes.ndim == 0:
        raise ValueError("codes must have at least one dimension")
    if np.any(codes < 0) or np.any(codes >= (1 << bits)):
        raise ValueError(f"codes must fit in {bits} bits")
    padding = (-codes.shape[-1]) % lanes
    if padding:
        codes = np.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, padding)])
    packed = np.zeros(codes.shape[:-1] + (codes.shape[-1] // lanes,), dtype=np.uint8)
    for lane in range(lanes):
        packed |= (codes[..., lane::lanes].astype(np.uint8) << (lane * bits))
    return packed


def unpack_codes(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Unpack low-bit-first unsigned codes from bytes."""
    lanes = codes_per_byte(bits)
    packed = np.asarray(packed)
    if packed.dtype != np.uint8:
        raise TypeError(f"packed codes must have dtype uint8, found {packed.dtype}")
    if count < 0 or packed.ndim == 0 or packed.shape[-1] != (count + lanes - 1) // lanes:
        raise ValueError(f"packed code shape is incompatible with count {count}: {packed.shape}")
    codes = np.empty(packed.shape[:-1] + (count,), dtype=np.uint8)
    mask = (1 << bits) - 1
    for lane in range(lanes):
        indices = np.arange(lane, count, lanes)
        codes[..., indices] = (packed[..., : len(indices)] >> (lane * bits)) & mask
    return codes


def pack_zero_points(zero_points: np.ndarray, bits: int) -> np.ndarray:
    """Pack per-block zero points using the same bit geometry as codes."""
    return pack_codes(zero_points, bits)


@dataclass(frozen=True)
class PackedTernaryNBits:
    """Packed MatMulNBits operands for a source weight shaped ``[K, N]``."""

    packed: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray
    shape: tuple[int, int]
    block_size: int = SOURCE_GROUP_SIZE
    bits: int = 4


# Preserve the previous public type name.
PackedTernaryNBits4 = PackedTernaryNBits


def pack_ternary_nbits(
    weight: np.ndarray,
    config: TernaryNBitsConfig | None = None,
) -> PackedTernaryNBits:
    """Pack exact source-group ternary weights into the configured NBits blocks."""
    config = config or TernaryNBitsConfig()
    weight = np.asarray(weight)
    if weight.dtype != np.float32:
        raise TypeError(f"weight must have dtype float32, found {weight.dtype}")
    if weight.ndim != 2 or any(dimension <= 0 for dimension in weight.shape):
        raise ValueError(f"weight must have two positive dimensions [K, N], found {weight.shape}")
    k, n = weight.shape
    if k % config.source_group_size:
        raise ValueError(
            f"weight K dimension must be divisible by source_group_size {config.source_group_size}, found {k}"
        )
    if not np.isfinite(weight).all():
        raise ValueError("weight values must be finite")

    zero_point = ZERO_POINTS[config.bits]
    source_groups = weight.reshape(k // config.source_group_size, config.source_group_size, n)
    source_magnitudes = np.abs(source_groups)
    source_scales = source_magnitudes.max(axis=1)
    if np.any((source_magnitudes != 0) & (source_magnitudes != source_scales[:, None, :])):
        raise ValueError(
            f"weight is not exactly ternary within every group of {config.source_group_size}"
        )

    block_count = k // config.block_size
    blocks = weight.reshape(block_count, config.block_size, n).transpose(2, 0, 1)
    codes = np.where(blocks < 0, zero_point - 1, np.where(blocks > 0, zero_point + 1, zero_point))
    packed = pack_codes(codes.astype(np.uint8, copy=False), config.bits)

    scales = np.repeat(
        source_scales.T,
        config.source_group_size // config.block_size,
        axis=1,
    )
    zero_point_blocks = np.full((n, block_count), zero_point, dtype=np.uint8)
    packed_zero_points = pack_zero_points(zero_point_blocks, config.bits)
    return PackedTernaryNBits(
        packed=np.ascontiguousarray(packed),
        scales=np.ascontiguousarray(scales, dtype=np.float32),
        zero_points=np.ascontiguousarray(packed_zero_points),
        shape=(k, n),
        block_size=config.block_size,
        bits=config.bits,
    )


def unpack_ternary_nbits(packed: PackedTernaryNBits) -> np.ndarray:
    """Reconstruct the FP32 source matrix from its exact-scale NBits encoding."""
    k, n = packed.shape
    block_size = packed.block_size
    config = TernaryNBitsConfig(bits=packed.bits, block_size=block_size)
    if k <= 0 or n <= 0 or k % config.source_group_size:
        raise ValueError(f"invalid source shape {packed.shape}")
    block_count = k // block_size
    lanes = codes_per_byte(config.bits)
    expected_codes_shape = (n, block_count, (block_size + lanes - 1) // lanes)
    expected_scales_shape = (n, block_count)
    expected_zero_points_shape = (n, (block_count + lanes - 1) // lanes)
    if packed.packed.dtype != np.uint8 or packed.packed.shape != expected_codes_shape:
        raise ValueError(f"packed codes must be uint8 with shape {expected_codes_shape}")
    if packed.scales.dtype != np.float32 or packed.scales.shape != expected_scales_shape:
        raise ValueError(f"scales must be float32 with shape {expected_scales_shape}")
    if packed.zero_points.dtype != np.uint8 or packed.zero_points.shape != expected_zero_points_shape:
        raise ValueError(f"zero points must be uint8 with shape {expected_zero_points_shape}")

    zero_point = ZERO_POINTS[config.bits]
    codes = unpack_codes(packed.packed, config.bits, block_size)
    if np.any((codes != zero_point - 1) & (codes != zero_point) & (codes != zero_point + 1)):
        raise ValueError("packed codes contain values outside the exact ternary encoding")

    expected_zero_points = pack_zero_points(
        np.full((n, block_count), zero_point, dtype=np.uint8), config.bits
    )
    if not np.array_equal(packed.zero_points, expected_zero_points):
        raise ValueError("zero points do not match the ternary encoding")

    values = (codes.astype(np.int8) - zero_point).astype(np.float32)
    values *= packed.scales[..., None]
    return values.transpose(1, 2, 0).reshape(k, n)


def pack_ternary_nbits4(
    weight: np.ndarray,
    config: TernaryNBitsConfig | None = None,
) -> PackedTernaryNBits:
    config = config or TernaryNBitsConfig()
    if config.bits != 4:
        raise ValueError("pack_ternary_nbits4 requires bits=4")
    return pack_ternary_nbits(weight, config)


def unpack_ternary_nbits4(packed: PackedTernaryNBits) -> np.ndarray:
    if packed.bits != 4:
        raise ValueError("unpack_ternary_nbits4 requires bits=4")
    return unpack_ternary_nbits(packed)


def rewrite_matmul_to_nbits(
    model: onnx.ModelProto,
    *,
    config: TernaryNBitsConfig | None = None,
    node_name: str | None = None,
) -> onnx.ModelProto:
    """Return a copy of a tiny model with one constant FP32 MatMul rewritten to NBits."""
    config = config or TernaryNBitsConfig()
    rewritten = copy.deepcopy(model)
    matmuls = [node for node in rewritten.graph.node if node.op_type == "MatMul"]
    if node_name is not None:
        matmuls = [node for node in matmuls if node.name == node_name]
    if len(matmuls) != 1:
        target = f" named {node_name!r}" if node_name is not None else ""
        raise ValueError(f"expected exactly one target MatMul{target}, found {len(matmuls)}")

    node = matmuls[0]
    if len(node.input) != 2:
        raise ValueError("target MatMul must have exactly two inputs")
    initializers = {tensor.name: tensor for tensor in rewritten.graph.initializer}
    weight_name = node.input[1]
    weight_tensor = initializers.get(weight_name)
    if weight_tensor is None:
        raise ValueError("target MatMul weight must be a graph initializer")
    if weight_tensor.data_location == TensorProto.EXTERNAL:
        raise ValueError("external-data weights are not supported by the tiny graph rewrite")
    if weight_tensor.data_type != TensorProto.FLOAT:
        raise TypeError("target MatMul weight initializer must be float32")

    packed = pack_ternary_nbits(numpy_helper.to_array(weight_tensor), config)
    used_names = {
        name
        for graph_node in rewritten.graph.node
        for name in (*graph_node.input, *graph_node.output, graph_node.name)
        if name
    }
    used_names.update(tensor.name for tensor in rewritten.graph.initializer)
    used_names.update(value.name for value in (*rewritten.graph.input, *rewritten.graph.output))

    def unique_name(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    suffix = f"nbits{config.bits}"
    packed_name = unique_name(f"{weight_name}_{suffix}")
    scales_name = unique_name(f"{weight_name}_{suffix}_scales")
    zero_points_name = unique_name(f"{weight_name}_{suffix}_zero_points")
    packed_tensor = numpy_helper.from_array(packed.packed, name=packed_name)
    scales_tensor = numpy_helper.from_array(packed.scales.reshape(-1), name=scales_name)
    zero_points_tensor = numpy_helper.from_array(packed.zero_points.reshape(-1), name=zero_points_name)
    rewritten.graph.initializer.extend([packed_tensor, scales_tensor, zero_points_tensor])

    consumers = [
        graph_node
        for graph_node in rewritten.graph.node
        if graph_node is not node and weight_name in graph_node.input
    ]
    if not consumers:
        rewritten.graph.initializer.remove(weight_tensor)

    k, n = packed.shape
    replacement = helper.make_node(
        "MatMulNBits",
        [node.input[0], packed_name, scales_name, zero_points_name],
        list(node.output),
        name=f"{node.name}_{suffix}" if node.name else f"MatMulNBits{config.bits}",
        domain="com.microsoft",
        K=k,
        N=n,
        bits=config.bits,
        block_size=config.block_size,
        accuracy_level=config.accuracy_level,
    )
    node_index = next(index for index, candidate in enumerate(rewritten.graph.node) if candidate is node)
    rewritten.graph.node.remove(node)
    rewritten.graph.node.insert(node_index, replacement)

    if not any(opset.domain == "com.microsoft" for opset in rewritten.opset_import):
        rewritten.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return rewritten


def rewrite_matmul_to_nbits4(
    model: onnx.ModelProto,
    *,
    config: TernaryNBitsConfig | None = None,
    node_name: str | None = None,
) -> onnx.ModelProto:
    config = config or TernaryNBitsConfig()
    if config.bits != 4:
        raise ValueError("rewrite_matmul_to_nbits4 requires bits=4")
    return rewrite_matmul_to_nbits(model, config=config, node_name=node_name)


def rewrite_matmul_to_nbits2(
    model: onnx.ModelProto,
    *,
    config: TernaryNBitsConfig | None = None,
    node_name: str | None = None,
) -> onnx.ModelProto:
    config = config or TernaryNBitsConfig(bits=2)
    if config.bits != 2:
        raise ValueError("rewrite_matmul_to_nbits2 requires bits=2")
    return rewrite_matmul_to_nbits(model, config=config, node_name=node_name)


@dataclass(frozen=True)
class _TernaryWeight:
    name: str
    k: int
    n: int
    source_sha256: str


def _load_float32_weight(model_path: Path, tensor: TensorProto) -> tuple[np.ndarray, bool]:
    if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) != 2:
        raise ValueError(f"MatMul weight must be a rank-2 float32 initializer: {tensor.name}")
    shape = tuple(tensor.dims)
    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        tensor_slice = resolve_external_tensor(model_path, tensor)
        return map_external_float32_tensor(tensor_slice, shape), True
    return np.asarray(numpy_helper.to_array(tensor)), False


def _close_memmap(weight: np.ndarray, is_memmap: bool) -> None:
    if is_memmap:
        weight._mmap.close()


def _array_sha256(weight: np.ndarray) -> str:
    digest = hashlib.sha256()
    rows = max(1, (8 * 1024 * 1024) // (weight.shape[1] * np.dtype(np.float32).itemsize))
    for start in range(0, weight.shape[0], rows):
        block = np.ascontiguousarray(weight[start : start + rows])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _is_exact_ternary_weight(weight: np.ndarray) -> tuple[bool, str]:
    k, n = weight.shape
    if k % BLOCK_SIZE:
        return False, "K dimension is not divisible by 128"
    digest = hashlib.sha256()
    column_chunk = min(n, 8192)
    for k_start in range(0, k, BLOCK_SIZE):
        row_block = weight[k_start : k_start + BLOCK_SIZE]
        if n <= 8192:
            digest.update(memoryview(row_block).cast("B"))
        for n_start in range(0, n, column_chunk):
            group = np.asarray(row_block[:, n_start : n_start + column_chunk])
            if not np.isfinite(group).all():
                return False, "weight contains non-finite values"
            magnitude = np.abs(group)
            scales = magnitude.max(axis=0)
            if np.any((magnitude != 0) & (magnitude != scales[None, :])):
                return False, "weight is not exact group-128 ternary"
    return True, digest.hexdigest() if n <= 8192 else _array_sha256(weight)


def _graph_names(graph: onnx.GraphProto) -> set[str]:
    names = {value.name for value in (*graph.input, *graph.output, *graph.value_info)}
    names.update(tensor.name for tensor in graph.initializer)
    for node in graph.node:
        names.update(name for name in (*node.input, *node.output, node.name) if name)
    return names


def _unique_graph_name(base: str, used_names: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _set_external_reference(tensor: TensorProto, location: str, offset: int, length: int) -> None:
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


def _copy_initializer(
    model_path: Path,
    tensor: TensorProto,
    writer: ExternalDataWriter,
    external_location: str,
) -> TensorProto:
    result = copy.deepcopy(tensor)
    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        tensor_slice = resolve_external_data_range(
            model_path.parent,
            tensor,
            logical_bytes=logical_tensor_bytes(tensor),
        )
        offset, length = writer.copy_range(tensor_slice)
        _set_external_reference(result, external_location, offset, length)
    return result


def _encode_weight_to_external(
    weight: np.ndarray,
    *,
    weight_name: str,
    names: tuple[str, str, str],
    writer: ExternalDataWriter,
    external_location: str,
    column_chunk: int,
    source_sha256: str,
    config: TernaryNBitsConfig,
) -> tuple[list[TensorProto], dict[str, Any]]:
    k, n = weight.shape
    block_count = k // config.block_size
    blocks_per_source_group = config.source_group_size // config.block_size
    packed_block_bytes = config.block_size * config.bits // 8
    zero_point = ZERO_POINTS[config.bits]
    packed_name, scales_name, zero_points_name = names
    code_digest = hashlib.sha256()
    scale_digest = hashlib.sha256()
    zero_digest = hashlib.sha256()
    scales = np.empty((n, block_count), dtype=np.float32)
    signed_zero_differences = 0

    code_offset = writer.start_tensor()
    for n_start in range(0, n, column_chunk):
        n_end = min(n, n_start + column_chunk)
        packed_columns = np.empty(
            (n_end - n_start, block_count, packed_block_bytes), dtype=np.uint8
        )
        for source_group_index, k_start in enumerate(range(0, k, config.source_group_size)):
            source_group = np.asarray(
                weight[k_start : k_start + config.source_group_size, n_start:n_end]
            ).T
            magnitude = np.abs(source_group)
            group_scales = magnitude.max(axis=1)
            if not np.isfinite(source_group).all() or np.any(
                (magnitude != 0) & (magnitude != group_scales[:, None])
            ):
                raise ValueError(f"Source weight changed or is not ternary: {weight_name}")
            for subblock in range(blocks_per_source_group):
                block_index = source_group_index * blocks_per_source_group + subblock
                block_start = subblock * config.block_size
                source_block = source_group[:, block_start : block_start + config.block_size]
                codes = np.where(
                    source_block < 0,
                    zero_point - 1,
                    np.where(source_block > 0, zero_point + 1, zero_point),
                ).astype(np.uint8, copy=False)
                packed_columns[:, block_index, :] = pack_codes(codes, config.bits)
                scales[n_start:n_end, block_index] = group_scales
                reconstructed = (codes.astype(np.int8) - zero_point).astype(np.float32)
                reconstructed *= group_scales[:, None]
                if not np.array_equal(reconstructed, source_block):
                    raise ValueError(
                        f"NBits{config.bits} weight reconstruction differs from source: {weight_name}"
                    )
                signed_zero_differences += int(
                    np.count_nonzero(
                        (source_block == 0)
                        & (reconstructed == 0)
                        & (np.signbit(source_block) != np.signbit(reconstructed))
                    )
                )
        code_bytes = memoryview(packed_columns).cast("B")
        writer.write(code_bytes)
        code_digest.update(code_bytes)
    code_offset, code_length = writer.finish_tensor(code_offset)

    scale_offset = writer.start_tensor()
    scale_bytes = memoryview(scales).cast("B")
    writer.write(scale_bytes)
    scale_digest.update(scale_bytes)
    scale_offset, scale_length = writer.finish_tensor(scale_offset)

    zero_point_blocks = np.full((n, block_count), zero_point, dtype=np.uint8)
    zero_points = pack_zero_points(zero_point_blocks, config.bits)
    zero_offset = writer.start_tensor()
    zero_bytes = memoryview(zero_points).cast("B")
    writer.write(zero_bytes)
    zero_digest.update(zero_bytes)
    zero_offset, zero_length = writer.finish_tensor(zero_offset)

    tensors = [
        make_external_tensor(
            packed_name,
            TensorProto.UINT8,
            (n, block_count, packed_block_bytes),
            external_location,
            code_offset,
            code_length,
        ),
        make_external_tensor(
            scales_name,
            TensorProto.FLOAT,
            (n * block_count,),
            external_location,
            scale_offset,
            scale_length,
        ),
        make_external_tensor(
            zero_points_name,
            TensorProto.UINT8,
            (
                n
                * (
                    (block_count + codes_per_byte(config.bits) - 1)
                    // codes_per_byte(config.bits)
                ),
            ),
            external_location,
            zero_offset,
            zero_length,
        ),
    ]
    report = {
        "source_initializer": weight_name,
        "source_shape": [k, n],
        "source_sha256": source_sha256,
        "bits": config.bits,
        "block_size": config.block_size,
        "source_group_size": config.source_group_size,
        "accuracy_level": config.accuracy_level,
        "packed_initializer": packed_name,
        "packed_code_bytes": code_length,
        "packed_code_sha256": code_digest.hexdigest(),
        "scales_initializer": scales_name,
        "scale_bytes": scale_length,
        "scales_sha256": scale_digest.hexdigest(),
        "zero_points_initializer": zero_points_name,
        "zero_point_bytes": zero_length,
        "zero_points_sha256": zero_digest.hexdigest(),
        "exact_reconstruction": True,
        "signed_zero_bit_differences": signed_zero_differences,
    }
    return tensors, report


def quantize_ternary_matmuls_streaming(
    input_model: str | Path,
    output_model: str | Path,
    *,
    output_external_data: str,
    config: TernaryNBitsConfig | None = None,
    expected_source_ternary_matmul_count: int | None = None,
    exclude_nodes: Sequence[str] = (),
    exclude_initializers: Sequence[str] = (),
    column_chunk: int = 64,
) -> dict[str, Any]:
    """Rewrite exact ternary constant-B MatMuls without materializing external weights."""
    config = config or TernaryNBitsConfig()
    source_path = Path(input_model).resolve()
    output_path = Path(output_model).resolve()
    if source_path == output_path:
        raise ValueError("Input and output ONNX model paths must differ")
    if output_path.exists():
        raise FileExistsError(output_path)
    if column_chunk <= 0:
        raise ValueError("column_chunk must be positive")
    location_probe = make_external_tensor(
        "location_probe", TensorProto.UINT8, (1,), output_external_data, 0, 1
    )
    external_location = next(
        entry.value for entry in location_probe.external_data if entry.key == "location"
    )
    external_path = (output_path.parent / external_location).resolve()
    try:
        external_path.relative_to(output_path.parent.resolve())
    except ValueError as error:
        raise ValueError("Output external-data path escapes the output model directory") from error
    if external_path.exists():
        raise FileExistsError(external_path)

    model = onnx.load(str(source_path), load_external_data=False)
    initializer_by_name = {tensor.name: tensor for tensor in model.graph.initializer}
    weight_cache: dict[str, _TernaryWeight | None] = {}
    targets_by_weight: dict[str, list[tuple[int, onnx.NodeProto]]] = defaultdict(list)
    skipped_matmuls: list[dict[str, str]] = []
    non_ternary_initializer_count = 0
    matmul_count = 0
    source_eligible_matmul_count = 0
    excluded_matmuls: list[dict[str, str]] = []
    exclude_node_names = set(exclude_nodes)
    exclude_initializer_names = set(exclude_initializers)
    matched_exclude_nodes: set[str] = set()
    matched_exclude_initializers: set[str] = set()

    for index, node in enumerate(model.graph.node):
        if node.op_type != "MatMul":
            continue
        matmul_count += 1
        node_label = node.name or f"MatMul#{index}"
        if node_label in exclude_node_names:
            matched_exclude_nodes.add(node_label)
        if node.name and node.name in exclude_node_names:
            matched_exclude_nodes.add(node.name)
        if len(node.input) != 2 or not node.input[1]:
            skipped_matmuls.append({"node": node_label, "reason": "MatMul B input is absent or unsupported"})
            continue
        weight_name = node.input[1]
        if weight_name in exclude_initializer_names:
            matched_exclude_initializers.add(weight_name)
        if weight_name not in initializer_by_name:
            skipped_matmuls.append({"node": node_label, "reason": "MatMul B is not a direct initializer"})
            continue
        if weight_name not in weight_cache:
            tensor = initializer_by_name[weight_name]
            if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) != 2:
                weight_cache[weight_name] = None
                skipped_matmuls.append({"node": node_label, "reason": "MatMul B is not a rank-2 float32 tensor"})
                continue
            k, n = tensor.dims
            if k <= 0 or n <= 0 or k % BLOCK_SIZE:
                weight_cache[weight_name] = None
                skipped_matmuls.append({"node": node_label, "reason": "MatMul B has unsupported group-128 shape"})
                continue
            weight, is_memmap = _load_float32_weight(source_path, tensor)
            try:
                is_ternary, source_sha256 = _is_exact_ternary_weight(weight)
            finally:
                _close_memmap(weight, is_memmap)
            if not is_ternary:
                non_ternary_initializer_count += 1
                weight_cache[weight_name] = None
                skipped_matmuls.append({"node": node_label, "reason": "MatMul B is not exact group-128 ternary"})
                continue
            weight_cache[weight_name] = _TernaryWeight(weight_name, k, n, source_sha256)
        if weight_cache[weight_name] is None:
            if not any(row["node"] == node_label for row in skipped_matmuls):
                skipped_matmuls.append({"node": node_label, "reason": "shared MatMul B is not a ternary target"})
            continue
        source_eligible_matmul_count += 1
        reasons = []
        if node_label in exclude_node_names or node.name in exclude_node_names:
            reasons.append("node exclusion")
        if weight_name in exclude_initializer_names:
            reasons.append("initializer exclusion")
        if reasons:
            reason = "explicitly excluded by " + " and ".join(reasons)
            weight_info = weight_cache[weight_name]
            excluded_matmuls.append(
                {
                    "node": node_label,
                    "initializer": weight_name,
                    "reason": reason,
                    "source_shape": [weight_info.k, weight_info.n],
                    "source_sha256": weight_info.source_sha256,
                }
            )
            skipped_matmuls.append({"node": node_label, "reason": reason})
            continue
        targets_by_weight[weight_name].append((index, node))
    unknown_nodes = exclude_node_names - matched_exclude_nodes
    unknown_initializers = exclude_initializer_names - matched_exclude_initializers
    if unknown_nodes:
        raise ValueError(f"Unknown NBits MatMul node exclusions: {sorted(unknown_nodes)}")
    if unknown_initializers:
        raise ValueError(f"Unknown NBits MatMul initializer exclusions: {sorted(unknown_initializers)}")

    transformed_matmul_count = sum(len(nodes) for nodes in targets_by_weight.values())
    if expected_source_ternary_matmul_count is not None and (
        source_eligible_matmul_count != expected_source_ternary_matmul_count
    ):
        raise ValueError(
            "Expected "
            f"{expected_source_ternary_matmul_count} source-eligible ternary MatMuls, "
            f"found {source_eligible_matmul_count}"
        )
    if not source_eligible_matmul_count:
        raise ValueError("No exact group-128 ternary constant-B MatMuls were found")

    fallback_initializers: dict[str, dict[str, Any]] = {}
    for exclusion in excluded_matmuls:
        name = exclusion["initializer"]
        if name in fallback_initializers:
            continue
        k, n = exclusion["source_shape"]
        block_count = k // config.block_size
        nbits_bytes_if_quantized = (
            n * block_count * (config.block_size * config.bits // 8)
            + n * block_count * np.dtype(np.float32).itemsize
            + n
            * (
                (block_count + codes_per_byte(config.bits) - 1)
                // codes_per_byte(config.bits)
            )
        )
        nbits_bytes_removed = 0 if name in targets_by_weight else nbits_bytes_if_quantized
        fallback_fp32_bytes = k * n * np.dtype(np.float32).itemsize
        fallback_initializers[name] = {
            "initializer": name,
            "source_shape": [k, n],
            "source_sha256": exclusion["source_sha256"],
            "fallback_fp32_bytes": fallback_fp32_bytes,
            "nbits_bytes_if_quantized": nbits_bytes_if_quantized,
            "nbits_bytes_removed": nbits_bytes_removed,
            "net_bundle_byte_cost": fallback_fp32_bytes - nbits_bytes_removed,
        }
    fallback_storage = {
        "fallback_initializer_count": len(fallback_initializers),
        "fallback_fp32_bytes": sum(
            item["fallback_fp32_bytes"] for item in fallback_initializers.values()
        ),
        "nbits_bytes_removed": sum(
            item["nbits_bytes_removed"] for item in fallback_initializers.values()
        ),
        "net_bundle_byte_cost": sum(
            item["net_bundle_byte_cost"] for item in fallback_initializers.values()
        ),
        "initializers": list(fallback_initializers.values()),
    }

    used_names = _graph_names(model.graph)
    quantized_names = {
        weight_name: (
            _unique_graph_name(f"{weight_name}_nbits{config.bits}", used_names),
            _unique_graph_name(f"{weight_name}_nbits{config.bits}_scales", used_names),
            _unique_graph_name(f"{weight_name}_nbits{config.bits}_zero_points", used_names),
        )
        for weight_name in targets_by_weight
    }
    target_indices = {index for rows in targets_by_weight.values() for index, _ in rows}
    consumers: dict[str, set[int]] = defaultdict(set)
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                consumers[name].add(index)
    graph_values = {value.name for value in (*model.graph.input, *model.graph.output)}
    retained_target_weights = {
        name
        for name in targets_by_weight
        if name in graph_values
        or any(
            index not in target_indices
            or model.graph.node[index].input[1] != name
            or name in model.graph.node[index].input[:1]
            for index in consumers[name]
        )
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.tmp.onnx")
    tensor_reports: dict[str, dict[str, Any]] = {}
    new_initializers: list[TensorProto] = []
    try:
        with ExternalDataWriter(external_path) as writer:
            for tensor in model.graph.initializer:
                weight_info = weight_cache.get(tensor.name)
                if tensor.name not in targets_by_weight:
                    new_initializers.append(
                        _copy_initializer(source_path, tensor, writer, external_location)
                    )
                    continue

                if tensor.name in retained_target_weights:
                    new_initializers.append(
                        _copy_initializer(source_path, tensor, writer, external_location)
                    )
                weight, is_memmap = _load_float32_weight(source_path, tensor)
                try:
                    quantized_tensors, report = _encode_weight_to_external(
                        weight,
                        weight_name=tensor.name,
                        names=quantized_names[tensor.name],
                        writer=writer,
                        external_location=external_location,
                        column_chunk=column_chunk,
                        source_sha256=weight_info.source_sha256,
                        config=config,
                    )
                finally:
                    _close_memmap(weight, is_memmap)
                new_initializers.extend(quantized_tensors)
                tensor_reports[tensor.name] = report

        rewritten = copy.deepcopy(model)
        del rewritten.graph.initializer[:]
        rewritten.graph.initializer.extend(new_initializers)
        new_nodes: list[onnx.NodeProto] = []
        node_reports = []
        for index, node in enumerate(model.graph.node):
            if index not in target_indices:
                new_nodes.append(copy.deepcopy(node))
                continue
            weight_name = node.input[1]
            weight_info = weight_cache[weight_name]
            packed_name, scales_name, zero_points_name = quantized_names[weight_name]
            replacement = helper.make_node(
                "MatMulNBits",
                [node.input[0], packed_name, scales_name, zero_points_name],
                list(node.output),
                name=(
                    f"{node.name}_nbits{config.bits}"
                    if node.name
                    else f"MatMul_{index}_nbits{config.bits}"
                ),
                domain="com.microsoft",
                K=weight_info.k,
                N=weight_info.n,
                bits=config.bits,
                block_size=config.block_size,
                accuracy_level=config.accuracy_level,
            )
            new_nodes.append(replacement)
            node_reports.append(
                {
                    "node": node.name or f"MatMul#{index}",
                    "source_initializer": weight_name,
                    "source_shape": [weight_info.k, weight_info.n],
                    "activation": node.input[0],
                    "output": node.output[0],
                    "source_sha256": weight_info.source_sha256,
                }
            )
        del rewritten.graph.node[:]
        rewritten.graph.node.extend(new_nodes)
        if not any(opset.domain == "com.microsoft" for opset in rewritten.opset_import):
            rewritten.opset_import.append(helper.make_opsetid("com.microsoft", 1))

        onnx.save_model(rewritten, str(temporary_path))
        onnx.checker.check_model(str(temporary_path))
        import onnxruntime as ort

        ort.InferenceSession(str(temporary_path), providers=["CPUExecutionProvider"])
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        external_path.unlink(missing_ok=True)
        raise

    return {
        "method": "ternary-nbits",
        "bits": config.bits,
        "block_size": config.block_size,
        "source_group_size": config.source_group_size,
        "accuracy_level": config.accuracy_level,
        "source_graph_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "output_graph_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "external_data_file": external_location,
        "transformed_matmul_count": transformed_matmul_count,
        "transformed_initializer_count": len(targets_by_weight),
        "non_ternary_initializer_count": non_ternary_initializer_count,
        "accidentally_transformed_non_ternary_count": 0,
        "matmul_count": matmul_count,
        "source_eligible_matmul_count": source_eligible_matmul_count,
        "excluded_matmul_count": len(excluded_matmuls),
        "exclude_nodes": sorted(exclude_node_names),
        "exclude_initializers": sorted(exclude_initializer_names),
        "excluded_matmuls": excluded_matmuls,
        "fallback_storage": fallback_storage,
        "transformed_nodes": node_reports,
        "transformed_tensors": [tensor_reports[name] for name in targets_by_weight],
        "skipped_matmuls": skipped_matmuls,
    }
