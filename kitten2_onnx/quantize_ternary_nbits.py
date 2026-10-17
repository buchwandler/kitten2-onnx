"""Encode exact group-128 ternary MatMul weights for ONNX Runtime MatMulNBits."""

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
    map_external_float32_tensor,
    resolve_external_data_range,
    resolve_external_tensor,
)
from kitten2_onnx.quantize_int8_streaming import ExternalDataWriter

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
) -> tuple[list[TensorProto], dict[str, Any]]:
    k, n = weight.shape
    block_count = k // BLOCK_SIZE
    packed_name, scales_name, zero_points_name = names
    code_digest = hashlib.sha256()
    scale_digest = hashlib.sha256()
    zero_digest = hashlib.sha256()
    scales = np.empty((n, block_count), dtype=np.float32)
    signed_zero_differences = 0

    code_offset = writer.start_tensor()
    for n_start in range(0, n, column_chunk):
        n_end = min(n, n_start + column_chunk)
        packed_columns = np.empty((n_end - n_start, block_count, BLOCK_SIZE // 2), dtype=np.uint8)
        for block_index, k_start in enumerate(range(0, k, BLOCK_SIZE)):
            source_group = np.asarray(weight[k_start : k_start + BLOCK_SIZE, n_start:n_end]).T
            magnitude = np.abs(source_group)
            group_scales = magnitude.max(axis=1)
            if not np.isfinite(source_group).all() or np.any(
                (magnitude != 0) & (magnitude != group_scales[:, None])
            ):
                raise ValueError(f"Source weight changed or is not ternary: {weight_name}")
            codes = np.where(
                source_group < 0,
                ZERO_POINT - 1,
                np.where(source_group > 0, ZERO_POINT + 1, ZERO_POINT),
            ).astype(np.uint8, copy=False)
            packed_columns[:, block_index, :] = codes[:, 0::2] | (codes[:, 1::2] << 4)
            scales[n_start:n_end, block_index] = group_scales
            reconstructed = (codes.astype(np.int8) - ZERO_POINT).astype(np.float32)
            reconstructed *= group_scales[:, None]
            if not np.array_equal(reconstructed, source_group):
                raise ValueError(f"NBits4 weight reconstruction differs from source: {weight_name}")
            signed_zero_differences += int(
                np.count_nonzero(
                    (source_group == 0)
                    & (reconstructed == 0)
                    & (np.signbit(source_group) != np.signbit(reconstructed))
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

    zero_points = np.zeros((n, (block_count + 1) // 2), dtype=np.uint8)
    for block_index in range(block_count):
        zero_points[:, block_index // 2] |= np.uint8(ZERO_POINT << (4 * (block_index % 2)))
    zero_offset = writer.start_tensor()
    zero_bytes = memoryview(zero_points).cast("B")
    writer.write(zero_bytes)
    zero_digest.update(zero_bytes)
    zero_offset, zero_length = writer.finish_tensor(zero_offset)

    tensors = [
        make_external_tensor(
            packed_name,
            TensorProto.UINT8,
            (n, block_count, BLOCK_SIZE // 2),
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
            (n * ((block_count + 1) // 2),),
            external_location,
            zero_offset,
            zero_length,
        ),
    ]
    report = {
        "source_initializer": weight_name,
        "source_shape": [k, n],
        "source_sha256": source_sha256,
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
    expected_matmul_count: int | None = None,
    column_chunk: int = 64,
) -> dict[str, Any]:
    """Rewrite exact ternary constant-B MatMuls without materializing external weights."""
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

    for index, node in enumerate(model.graph.node):
        if node.op_type != "MatMul":
            continue
        matmul_count += 1
        node_label = node.name or f"MatMul#{index}"
        if len(node.input) != 2 or not node.input[1]:
            skipped_matmuls.append({"node": node_label, "reason": "MatMul B input is absent or unsupported"})
            continue
        weight_name = node.input[1]
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
        targets_by_weight[weight_name].append((index, node))

    transformed_matmul_count = sum(len(nodes) for nodes in targets_by_weight.values())
    if not transformed_matmul_count:
        raise ValueError("No exact group-128 ternary constant-B MatMuls were found")
    if expected_matmul_count is not None and transformed_matmul_count != expected_matmul_count:
        raise ValueError(
            f"Expected {expected_matmul_count} ternary MatMuls, found {transformed_matmul_count}"
        )

    used_names = _graph_names(model.graph)
    quantized_names = {
        weight_name: (
            _unique_graph_name(f"{weight_name}_nbits4", used_names),
            _unique_graph_name(f"{weight_name}_nbits4_scales", used_names),
            _unique_graph_name(f"{weight_name}_nbits4_zero_points", used_names),
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
                name=f"{node.name}_nbits4" if node.name else f"MatMul_{index}_nbits4",
                domain="com.microsoft",
                K=weight_info.k,
                N=weight_info.n,
                bits=BITS,
                block_size=BLOCK_SIZE,
                accuracy_level=1,
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
        "method": "ternary-nbits4",
        "source_graph_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "output_graph_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "external_data_file": external_location,
        "transformed_matmul_count": transformed_matmul_count,
        "transformed_initializer_count": len(targets_by_weight),
        "non_ternary_initializer_count": non_ternary_initializer_count,
        "accidentally_transformed_non_ternary_count": 0,
        "matmul_count": matmul_count,
        "transformed_nodes": node_reports,
        "transformed_tensors": [tensor_reports[name] for name in targets_by_weight],
        "skipped_matmuls": skipped_matmuls,
    }
