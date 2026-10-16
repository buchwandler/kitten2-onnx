"""Encode exact group-128 ternary MatMul weights for ONNX Runtime MatMulNBits."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BLOCK_SIZE = 128
BITS = 4
ZERO_POINT = 8


@dataclass(frozen=True)
class PackedTernaryNBits4:
    """Packed MatMulNBits operands for a source weight shaped ``[K, N]``."""

    packed: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray
    shape: tuple[int, int]


def pack_ternary_nbits4(weight: np.ndarray) -> PackedTernaryNBits4:
    """Pack an exact group-128 ternary FP32 MatMul weight without requantizing scales."""
    weight = np.asarray(weight)
    if weight.dtype != np.float32:
        raise TypeError(f"weight must have dtype float32, found {weight.dtype}")
    if weight.ndim != 2 or any(dimension <= 0 for dimension in weight.shape):
        raise ValueError(f"weight must have two positive dimensions [K, N], found {weight.shape}")
    k, n = weight.shape
    if k % BLOCK_SIZE:
        raise ValueError(f"weight K dimension must be divisible by {BLOCK_SIZE}, found {k}")
    if not np.isfinite(weight).all():
        raise ValueError("weight values must be finite")

    block_count = k // BLOCK_SIZE
    blocks = weight.reshape(block_count, BLOCK_SIZE, n).transpose(2, 0, 1)
    magnitudes = np.abs(blocks)
    scales = magnitudes.max(axis=2)
    if np.any((magnitudes != 0) & (magnitudes != scales[..., None])):
        raise ValueError("weight is not exactly ternary within every group of 128")

    codes = np.where(blocks < 0, ZERO_POINT - 1, np.where(blocks > 0, ZERO_POINT + 1, ZERO_POINT))
    codes = codes.astype(np.uint8, copy=False)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)

    packed_zero_points = np.zeros((n, (block_count + 1) // 2), dtype=np.uint8)
    for block in range(block_count):
        packed_zero_points[:, block // 2] |= np.uint8(ZERO_POINT << (4 * (block % 2)))

    return PackedTernaryNBits4(
        packed=np.ascontiguousarray(packed),
        scales=np.ascontiguousarray(scales, dtype=np.float32),
        zero_points=packed_zero_points,
        shape=(k, n),
    )


def unpack_ternary_nbits4(packed: PackedTernaryNBits4) -> np.ndarray:
    """Reconstruct the FP32 source matrix from its exact-scale NBits4 encoding."""
    k, n = packed.shape
    if k <= 0 or n <= 0 or k % BLOCK_SIZE:
        raise ValueError(f"invalid source shape {packed.shape}")
    block_count = k // BLOCK_SIZE
    expected_codes_shape = (n, block_count, BLOCK_SIZE // 2)
    expected_scales_shape = (n, block_count)
    expected_zero_points_shape = (n, (block_count + 1) // 2)
    if packed.packed.dtype != np.uint8 or packed.packed.shape != expected_codes_shape:
        raise ValueError(f"packed codes must be uint8 with shape {expected_codes_shape}")
    if packed.scales.dtype != np.float32 or packed.scales.shape != expected_scales_shape:
        raise ValueError(f"scales must be float32 with shape {expected_scales_shape}")
    if packed.zero_points.dtype != np.uint8 or packed.zero_points.shape != expected_zero_points_shape:
        raise ValueError(f"zero points must be uint8 with shape {expected_zero_points_shape}")

    low = packed.packed & 0x0F
    high = packed.packed >> 4
    codes = np.empty((n, block_count, BLOCK_SIZE), dtype=np.uint8)
    codes[..., 0::2] = low
    codes[..., 1::2] = high
    if np.any((codes != ZERO_POINT - 1) & (codes != ZERO_POINT) & (codes != ZERO_POINT + 1)):
        raise ValueError("packed codes contain values outside the exact ternary encoding")

    expected_zero_points = np.zeros(expected_zero_points_shape, dtype=np.uint8)
    for block in range(block_count):
        expected_zero_points[:, block // 2] |= np.uint8(ZERO_POINT << (4 * (block % 2)))
    if not np.array_equal(packed.zero_points, expected_zero_points):
        raise ValueError("zero points do not match the ternary encoding")

    values = (codes.astype(np.int8) - ZERO_POINT).astype(np.float32)
    values *= packed.scales[..., None]
    return values.transpose(1, 2, 0).reshape(k, n)


def rewrite_matmul_to_nbits4(
    model: onnx.ModelProto,
    *,
    node_name: str | None = None,
) -> onnx.ModelProto:
    """Return a copy of a tiny model with one constant FP32 MatMul rewritten to NBits4."""
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

    packed = pack_ternary_nbits4(numpy_helper.to_array(weight_tensor))
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

    packed_name = unique_name(f"{weight_name}_nbits4")
    scales_name = unique_name(f"{weight_name}_nbits4_scales")
    zero_points_name = unique_name(f"{weight_name}_nbits4_zero_points")
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
        name=f"{node.name}_nbits4" if node.name else "MatMulNBits4",
        domain="com.microsoft",
        K=k,
        N=n,
        bits=BITS,
        block_size=BLOCK_SIZE,
        accuracy_level=1,
    )
    node_index = next(index for index, candidate in enumerate(rewritten.graph.node) if candidate is node)
    rewritten.graph.node.remove(node)
    rewritten.graph.node.insert(node_index, replacement)

    if not any(opset.domain == "com.microsoft" for opset in rewritten.opset_import):
        rewritten.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return rewritten
