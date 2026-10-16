"""Safe helpers for reading and writing ONNX external tensor data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

import numpy as np
import onnx
from onnx import TensorProto


@dataclass(frozen=True)
class ExternalTensorSlice:
    path: Path
    location: str
    offset: int
    length: int


_PACKED_DTYPE_BITS = {
    "INT4": 4,
    "UINT4": 4,
    "FLOAT4E2M1": 4,
    "FLOAT8E4M3FN": 8,
    "FLOAT8E4M3FNUZ": 8,
    "FLOAT8E5M2": 8,
    "FLOAT8E5M2FNUZ": 8,
    "FLOAT8E8M0": 8,
}


def logical_tensor_bytes(tensor: TensorProto) -> int:
    count = math.prod(tensor.dims) if tensor.dims else 1
    dtype_name = TensorProto.DataType.Name(tensor.data_type)
    if dtype_name in _PACKED_DTYPE_BITS:
        return (count * _PACKED_DTYPE_BITS[dtype_name] + 7) // 8
    if tensor.data_type == TensorProto.STRING:
        return sum(len(value) for value in tensor.string_data)
    dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
    return count * np.dtype(dtype).itemsize


def _relative_path(location: str) -> Path:
    relative = Path(location)
    windows_path = PureWindowsPath(location)
    if (
        not location
        or relative.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in relative.parts
    ):
        raise ValueError(f"ONNX external-data location must be relative and contained: {location!r}")
    return relative


def _relative_location(model_dir: Path, location: str) -> tuple[Path, str]:
    relative = _relative_path(location)
    root = model_dir.resolve()
    path = (root / relative).resolve()
    try:
        normalized = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"ONNX external-data location escapes the model directory: {location!r}") from error
    return path, normalized.as_posix()

def resolve_external_data_range(
    model_dir: str | Path,
    tensor: TensorProto,
    *,
    logical_bytes: int,
    require_exact_length: bool = False,
) -> ExternalTensorSlice:
    """Resolve and bounds-check the external byte range for a tensor."""
    if tensor.data_location != TensorProto.EXTERNAL:
        raise ValueError(f"Tensor {tensor.name!r} is not stored as ONNX external data")
    external = {entry.key: entry.value for entry in tensor.external_data}
    if "location" not in external:
        raise ValueError(f"Tensor {tensor.name!r} has no external-data location")

    location = external["location"]
    path, normalized_location = _relative_location(Path(model_dir), location)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX external-data file does not exist: {path}")

    try:
        offset = int(external.get("offset", "0"))
        length = int(external.get("length", str(logical_bytes)))
    except ValueError as error:
        raise ValueError(f"Invalid ONNX external-data range for tensor {tensor.name!r}") from error
    if offset < 0 or length < 0:
        raise ValueError(f"Negative ONNX external-data range for tensor {tensor.name!r}")
    if require_exact_length and length != logical_bytes:
        raise ValueError(
            f"External-data length for tensor {tensor.name!r} is {length}, expected {logical_bytes}"
        )
    if offset + length > path.stat().st_size:
        raise ValueError(f"ONNX external-data range is outside its file for tensor {tensor.name!r}")
    return ExternalTensorSlice(path, normalized_location, offset, length)


def resolve_external_tensor(model_path: str | Path, tensor: TensorProto) -> ExternalTensorSlice:
    """Resolve a direct external float32 rank-2 initializer for streaming v1."""
    if tensor.data_type != TensorProto.FLOAT:
        raise ValueError(f"Streaming MatMul weights must be float32: tensor={tensor.name!r}")
    if len(tensor.dims) != 2:
        raise ValueError(f"Streaming MatMul weights must have rank 2: tensor={tensor.name!r}")
    if tensor.HasField("raw_data") or any(
        getattr(tensor, field)
        for field in ("float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data")
    ):
        raise ValueError(f"Streaming external tensor has inline data fields: tensor={tensor.name!r}")

    logical_bytes = math.prod(tensor.dims) * np.dtype(np.float32).itemsize
    return resolve_external_data_range(
        Path(model_path).parent,
        tensor,
        logical_bytes=logical_bytes,
        require_exact_length=True,
    )


def map_external_float32_tensor(
    tensor_slice: ExternalTensorSlice,
    shape: tuple[int, int],
) -> np.memmap:
    expected_bytes = math.prod(shape) * np.dtype(np.float32).itemsize
    if len(shape) != 2 or tensor_slice.length != expected_bytes:
        raise ValueError("External float32 MatMul weight shape does not match its data length")
    return np.memmap(
        tensor_slice.path,
        mode="r",
        dtype="<f4",
        offset=tensor_slice.offset,
        shape=shape,
        order="C",
    )


def copy_external_data(
    tensor_slice: ExternalTensorSlice,
    destination: BinaryIO,
    *,
    buffer_size: int = 8 * 1024 * 1024,
) -> int:
    """Copy exactly one validated external-data range into a binary stream."""
    if buffer_size <= 0:
        raise ValueError("External-data copy buffer size must be positive")
    remaining = tensor_slice.length
    copied = 0
    with tensor_slice.path.open("rb") as source:
        source.seek(tensor_slice.offset)
        while remaining:
            block = source.read(min(buffer_size, remaining))
            if not block:
                raise ValueError(f"External-data file ended early: {tensor_slice.path}")
            destination.write(block)
            copied += len(block)
            remaining -= len(block)
    return copied


def make_external_tensor(
    name: str,
    data_type: int,
    shape: tuple[int, ...],
    location: str,
    offset: int,
    length: int,
) -> TensorProto:
    """Build an ONNX tensor reference without materializing its payload."""
    normalized_location = _relative_path(location).as_posix()
    if offset < 0 or length < 0:
        raise ValueError("ONNX external-data offset and length must be nonnegative")
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(data_type))
    expected_bytes = math.prod(shape) * dtype.itemsize
    if length != expected_bytes:
        raise ValueError(f"External tensor length is {length}, expected {expected_bytes}")

    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = data_type
    tensor.dims.extend(shape)
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", normalized_location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    return tensor
