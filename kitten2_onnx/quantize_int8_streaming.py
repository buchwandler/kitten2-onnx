"""Bounded-memory per-output-channel QInt8 quantization for ONNX weights."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.external_data import (
    ExternalTensorSlice,
    copy_external_data,
    logical_tensor_bytes,
    make_external_tensor,
    map_external_float32_tensor,
    resolve_external_data_range,
    resolve_external_tensor,
)


@dataclass(frozen=True)
class QuantizedExternalWeight:
    scales: np.ndarray
    zero_points: np.ndarray
    offset: int
    length: int


class ExternalDataWriter:
    """Write aligned tensor payloads to one external-data file."""

    def __init__(self, path: str | Path, *, alignment: int = 64) -> None:
        if alignment <= 0:
            raise ValueError("External-data alignment must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.alignment = alignment
        self._stream: BinaryIO = self.path.open("wb")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._stream.close()

    def align(self) -> None:
        padding = (-self._stream.tell()) % self.alignment
        if padding:
            self._stream.write(b"\0" * padding)

    def start_tensor(self) -> int:
        self.align()
        return self._stream.tell()

    def write(self, data: bytes | memoryview) -> None:
        self._stream.write(data)

    def finish_tensor(self, offset: int) -> tuple[int, int]:
        return offset, self._stream.tell() - offset

    def append_bytes(self, data: bytes | memoryview) -> tuple[int, int]:
        offset = self.start_tensor()
        self.write(data)
        return self.finish_tensor(offset)

    def copy_range(
        self,
        source: ExternalTensorSlice,
        *,
        buffer_size: int = 8 * 1024 * 1024,
    ) -> tuple[int, int]:
        offset = self.start_tensor()
        copied = copy_external_data(source, self._stream, buffer_size=buffer_size)
        return offset, copied


def _symmetric_qint8_parameters(
    channel_min: np.ndarray,
    channel_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rmin = np.minimum(channel_min, np.float32(0))
    rmax = np.maximum(channel_max, np.float32(0))
    absmax = np.maximum(np.abs(rmin), np.abs(rmax))
    symmetric_min = -absmax
    symmetric_max = absmax
    real_range = symmetric_max.astype(np.float64) - symmetric_min.astype(np.float64)
    scale64 = real_range / np.float64(254)
    scale = scale64.astype(np.float32)
    degenerate = scale64 < np.finfo(np.float32).tiny
    scale[degenerate] = np.float32(1)
    return scale, np.zeros(scale.shape, dtype=np.int8)


def quantize_external_tensor_to_int8(
    model_path: str | Path,
    tensor: TensorProto,
    writer: ExternalDataWriter,
    *,
    chunk_rows: int = 64,
) -> QuantizedExternalWeight:
    """Quantize one external float32 rank-2 initializer in bounded row chunks."""
    if chunk_rows <= 0:
        raise ValueError("Quantization chunk_rows must be positive")
    tensor_slice = resolve_external_tensor(model_path, tensor)
    weight = map_external_float32_tensor(tensor_slice, tuple(tensor.dims))
    rows, channels = weight.shape
    channel_min = np.full(channels, np.inf, dtype=np.float32)
    channel_max = np.full(channels, -np.inf, dtype=np.float32)

    for row_start in range(0, rows, chunk_rows):
        row_end = min(rows, row_start + chunk_rows)
        block = weight[row_start:row_end, :]
        np.minimum(channel_min, np.min(block, axis=0), out=channel_min)
        np.maximum(channel_max, np.max(block, axis=0), out=channel_max)

    scales, zero_points = _symmetric_qint8_parameters(channel_min, channel_max)
    output_offset = writer.start_tensor()
    work = np.empty((min(chunk_rows, rows), channels), dtype=np.float32)
    quantized = np.empty(work.shape, dtype=np.int8)
    for row_start in range(0, rows, chunk_rows):
        row_end = min(rows, row_start + chunk_rows)
        row_count = row_end - row_start
        block = weight[row_start:row_end, :]
        block_work = work[:row_count]
        block_quantized = quantized[:row_count]
        np.divide(block, scales, out=block_work)
        np.rint(block_work, out=block_work)
        np.clip(block_work, np.iinfo(np.int8).min, np.iinfo(np.int8).max, out=block_work)
        block_quantized[:] = block_work
        writer.write(memoryview(block_quantized))

    output_offset, output_length = writer.finish_tensor(output_offset)
    expected_length = math.prod(tensor.dims)
    if output_length != expected_length:
        raise ValueError(
            f"Quantized external weight has {output_length} bytes, expected {expected_length}"
        )
    return QuantizedExternalWeight(scales, zero_points, output_offset, output_length)


@dataclass(frozen=True)
class MatMulTarget:
    index: int
    name: str
    activation_name: str
    weight_name: str
    output_name: str


def _unique_name(base: str, used_names: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _all_graph_names(graph: onnx.GraphProto) -> set[str]:
    names = {value.name for value in (*graph.input, *graph.output, *graph.value_info)}
    names.update(tensor.name for tensor in graph.initializer)
    for node in graph.node:
        names.update(name for name in (*node.input, *node.output, node.name) if name)
    return names


def _value_element_types(graph: onnx.GraphProto) -> dict[str, int]:
    types = {tensor.name: tensor.data_type for tensor in graph.initializer}
    for value in (*graph.input, *graph.value_info, *graph.output):
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    return types


def _is_constant_value(
    name: str,
    producers: dict[str, onnx.NodeProto],
    initializer_names: set[str],
    graph_input_names: set[str],
    memo: dict[str, bool],
    active: set[str],
) -> bool:
    if name in initializer_names:
        return True
    if name in graph_input_names:
        return False
    if name in memo:
        return memo[name]
    if name in active:
        return False
    node = producers.get(name)
    if node is None:
        return False
    if node.op_type == "Constant":
        memo[name] = True
        return True
    inputs = [value for value in node.input if value]
    if not inputs:
        memo[name] = False
        return False
    active.add(name)
    is_constant = all(
        _is_constant_value(value, producers, initializer_names, graph_input_names, memo, active)
        for value in inputs
    )
    active.remove(name)
    memo[name] = is_constant
    return is_constant


def _set_external_reference(
    tensor: TensorProto,
    location: str,
    offset: int,
    length: int,
) -> None:
    entries = {
        entry.key: entry.value for entry in tensor.external_data if entry.key != "basepath"
    }
    entries.update({"location": location, "offset": str(offset), "length": str(length)})
    del tensor.external_data[:]
    for key, value in entries.items():
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    tensor.data_location = TensorProto.EXTERNAL


def _matmul_targets(
    graph: onnx.GraphProto,
    model_path: Path,
    *,
    exclude_nodes: Collection[str],
    exclude_initializers: Collection[str],
) -> tuple[list[MatMulTarget], list[str], int]:
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output if output}
    graph_input_names = {value.name for value in graph.input}
    types = _value_element_types(graph)
    targets: list[MatMulTarget] = []
    excluded: list[str] = []
    total = 0
    constant_cache: dict[str, bool] = {}

    for index, node in enumerate(graph.node):
        if node.op_type != "MatMul":
            continue
        total += 1
        label = node.name or f"MatMul#{index}"
        if (
            len(node.input) != 2
            or len(node.output) != 1
            or not node.input[0]
            or not node.input[1]
            or not node.output[0]
        ):
            raise ValueError(f"Streaming INT8 requires a two-input, single-output MatMul: node={label}")
        activation_name, weight_name = node.input[:2]
        if label in exclude_nodes or node.name in exclude_nodes or weight_name in exclude_initializers:
            excluded.append(label)
            continue
        if weight_name not in initializers:
            if _is_constant_value(
                weight_name,
                producers,
                set(initializers),
                graph_input_names,
                constant_cache,
                set(),
            ):
                raise ValueError(
                    "Streaming INT8 does not yet support MatMul B through a constant-producing "
                    f"subgraph: node={label}, input={weight_name}"
                )
            if weight_name not in graph_input_names and weight_name not in producers:
                raise ValueError(f"MatMul B has no graph producer or initializer: node={label}, input={weight_name}")
            excluded.append(label)
            continue

        weight = initializers[weight_name]
        if weight.data_type != TensorProto.FLOAT:
            raise ValueError(f"Streaming MatMul weights must be float32: node={label}, tensor={weight_name}")
        if len(weight.dims) != 2:
            raise ValueError(f"Streaming MatMul weights must have rank 2: node={label}, tensor={weight_name}")
        resolve_external_tensor(model_path, weight)
        if types.get(activation_name) != TensorProto.FLOAT:
            raise ValueError(
                f"Streaming MatMul activation must be float32: node={label}, input={activation_name}"
            )
        output_type = types.get(node.output[0])
        if output_type is not None and output_type != TensorProto.FLOAT:
            raise ValueError(f"Streaming MatMul output must be float32: node={label}, output={node.output[0]}")
        targets.append(MatMulTarget(index, label, activation_name, weight_name, node.output[0]))

    if not targets:
        raise ValueError("Streaming INT8 found no supported constant-B MatMul nodes")
    return targets, excluded, total


def _standard_opset_version(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    raise ValueError("ONNX model has no standard-domain opset import")


def quantize_dynamic_matmul_streaming(
    input_model: str | Path,
    output_model: str | Path,
    *,
    output_external_data: str,
    chunk_rows: int = 64,
    exclude_nodes: Collection[str] = (),
    exclude_initializers: Collection[str] = (),
) -> dict[str, object]:
    """Stream direct external MatMul weights and rewrite them to dynamic INT8 ops."""
    if chunk_rows <= 0:
        raise ValueError("Quantization chunk_rows must be positive")
    source_path = Path(input_model).resolve()
    output_path = Path(output_model).resolve()
    if source_path == output_path:
        raise ValueError("Input and output ONNX graph paths must differ")
    if output_path.exists():
        raise FileExistsError(output_path)

    location_probe = make_external_tensor(
        "location_probe", TensorProto.INT8, (1,), output_external_data, 0, 1
    )
    external_location = next(
        entry.value for entry in location_probe.external_data if entry.key == "location"
    )
    output_data_path = (output_path.parent / external_location).resolve()
    try:
        output_data_path.relative_to(output_path.parent.resolve())
    except ValueError as error:
        raise ValueError("Output external-data path escapes the output graph directory") from error
    if output_data_path in (source_path, output_path) or output_data_path.exists():
        raise FileExistsError(output_data_path)

    model = onnx.load(str(source_path), load_external_data=False)
    if _standard_opset_version(model) < 11:
        raise ValueError("Streaming dynamic MatMul quantization requires ONNX opset 11 or newer")
    model = onnx.shape_inference.infer_shapes(model)
    targets, excluded, matmul_count = _matmul_targets(
        model.graph,
        source_path,
        exclude_nodes=exclude_nodes,
        exclude_initializers=exclude_initializers,
    )

    targets_by_index = {target.index: target for target in targets}
    target_indices_by_weight: dict[str, set[int]] = defaultdict(set)
    target_names_by_weight: dict[str, tuple[str, str, str]] = {}
    target_names_by_activation: dict[str, tuple[str, str, str]] = {}
    used_names = _all_graph_names(model.graph)
    for target in targets:
        target_indices_by_weight[target.weight_name].add(target.index)
        if target.weight_name not in target_names_by_weight:
            target_names_by_weight[target.weight_name] = (
                _unique_name(f"{target.weight_name}_quantized", used_names),
                _unique_name(f"{target.weight_name}_scale", used_names),
                _unique_name(f"{target.weight_name}_zero_point", used_names),
            )
        if target.activation_name not in target_names_by_activation:
            target_names_by_activation[target.activation_name] = (
                _unique_name(f"{target.activation_name}_quantized", used_names),
                _unique_name(f"{target.activation_name}_scale", used_names),
                _unique_name(f"{target.activation_name}_zero_point", used_names),
            )

    consumers: dict[str, set[int]] = defaultdict(set)
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                consumers[name].add(index)
    graph_value_names = {value.name for value in (*model.graph.input, *model.graph.output)}
    removed_weights = {
        weight_name
        for weight_name, indices in target_indices_by_weight.items()
        if weight_name not in graph_value_names
        and consumers[weight_name]
        and consumers[weight_name] <= indices
    }

    input_external_paths: set[Path] = set()
    for tensor in model.graph.initializer:
        if tensor.external_data or tensor.data_location == TensorProto.EXTERNAL:
            external_slice = resolve_external_data_range(
                source_path.parent,
                tensor,
                logical_bytes=logical_tensor_bytes(tensor),
            )
            input_external_paths.add(external_slice.path)
    if output_data_path in input_external_paths:
        raise ValueError("Output external-data path must not overwrite an input tensor file")

    new_initializers: list[TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ExternalDataWriter(output_data_path) as writer:
        for tensor in model.graph.initializer:
            if tensor.name in removed_weights:
                continue
            copied = copy.deepcopy(tensor)
            if tensor.external_data or tensor.data_location == TensorProto.EXTERNAL:
                external_slice = resolve_external_data_range(
                    source_path.parent,
                    tensor,
                    logical_bytes=logical_tensor_bytes(tensor),
                )
                offset, length = writer.copy_range(external_slice)
                _set_external_reference(copied, external_location, offset, length)
            new_initializers.append(copied)

        quantized_weight_results: dict[str, QuantizedExternalWeight] = {}
        initializer_by_name = {tensor.name: tensor for tensor in model.graph.initializer}
        for target in targets:
            if target.weight_name in quantized_weight_results:
                continue
            result = quantize_external_tensor_to_int8(
                source_path,
                initializer_by_name[target.weight_name],
                writer,
                chunk_rows=chunk_rows,
            )
            quantized_weight_results[target.weight_name] = result
            quantized_name, scale_name, zero_point_name = target_names_by_weight[target.weight_name]
            weight = initializer_by_name[target.weight_name]
            new_initializers.append(
                make_external_tensor(
                    quantized_name,
                    TensorProto.INT8,
                    tuple(weight.dims),
                    external_location,
                    result.offset,
                    result.length,
                )
            )
            new_initializers.append(numpy_helper.from_array(result.scales, name=scale_name))
            new_initializers.append(
                numpy_helper.from_array(result.zero_points, name=zero_point_name)
            )

        activation_nodes: dict[str, tuple[str, str, str]] = {}
        for index, original_node in enumerate(model.graph.node):
            target = targets_by_index.get(index)
            if target is None:
                new_nodes.append(copy.deepcopy(original_node))
                continue

            if target.activation_name not in activation_nodes:
                activation_names = target_names_by_activation[target.activation_name]
                activation_nodes[target.activation_name] = activation_names
                new_nodes.append(
                    helper.make_node(
                        "DynamicQuantizeLinear",
                        [target.activation_name],
                        list(activation_names),
                        name=_unique_name(f"{target.name}_DynamicQuantizeLinear", used_names),
                    )
                )
            activation_quantized, activation_scale, activation_zero_point = activation_nodes[
                target.activation_name
            ]
            weight_quantized, weight_scale, weight_zero_point = target_names_by_weight[
                target.weight_name
            ]
            integer_output = _unique_name(f"{target.name}_output_quantized", used_names)
            cast_output = _unique_name(f"{target.name}_output_float", used_names)
            scale_output = _unique_name(f"{target.name}_scales_mul", used_names)
            new_nodes.extend(
                [
                    helper.make_node(
                        "Mul",
                        [activation_scale, weight_scale],
                        [scale_output],
                        name=_unique_name(f"{target.name}_ScaleMul", used_names),
                    ),
                    helper.make_node(
                        "MatMulInteger",
                        [
                            activation_quantized,
                            weight_quantized,
                            activation_zero_point,
                            weight_zero_point,
                        ],
                        [integer_output],
                        name=_unique_name(f"{target.name}_MatMulInteger", used_names),
                    ),
                    helper.make_node(
                        "Cast",
                        [integer_output],
                        [cast_output],
                        name=_unique_name(f"{target.name}_Cast", used_names),
                        to=TensorProto.FLOAT,
                    ),
                    helper.make_node(
                        "Mul",
                        [cast_output, scale_output],
                        [target.output_name],
                        name=_unique_name(f"{target.name}_OutputMul", used_names),
                    ),
                ]
            )

    rewritten = copy.deepcopy(model)
    del rewritten.graph.node[:]
    rewritten.graph.node.extend(new_nodes)
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(new_initializers)
    onnx.save_model(rewritten, str(output_path))
    onnx.checker.check_model(str(output_path))
    import onnxruntime as ort

    ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    activation_counts: dict[str, int] = defaultdict(int)
    for target in targets:
        activation_counts[target.activation_name] += 1
    return {
        "matmul_count": matmul_count,
        "supported_constant_b_count": len(targets),
        "transformed_matmul_count": len(targets),
        "unique_weight_count": len(target_names_by_weight),
        "unique_quantized_weight_count": len(quantized_weight_results),
        "shared_activation_count": sum(count > 1 for count in activation_counts.values()),
        "transformed_nodes": [
            {
                "node": target.name,
                "activation": target.activation_name,
                "weight": target.weight_name,
                "output": target.output_name,
            }
            for target in targets
        ],
        "excluded_nodes": excluded,
        "excluded_matmul_count": len(excluded),
        "unsupported_nodes": [],
        "external_data_file": external_location,
    }
