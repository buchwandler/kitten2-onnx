from __future__ import annotations

import io

import numpy as np
import pytest
from onnx import TensorProto

from kitten2_onnx.external_data import (
    copy_external_data,
    make_external_tensor,
    map_external_float32_tensor,
    resolve_external_data_range,
    resolve_external_tensor,
)


def _external_tensor(
    location: str,
    *,
    offset: int = 0,
    length: int = 8,
    dtype: int = TensorProto.FLOAT,
    shape: tuple[int, ...] = (1, 2),
) -> TensorProto:
    tensor = TensorProto()
    tensor.name = "weight"
    tensor.data_type = dtype
    tensor.dims.extend(shape)
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    return tensor


def test_external_float32_tensor_uses_declared_offset_and_memmap(tmp_path):
    values = np.asarray([[1.25, -2.5], [3.0, 4.5]], dtype=np.float32)
    prefix = b"prefix-bytes"
    suffix = b"suffix-bytes"
    external_path = tmp_path / "nested" / "weights.bin"
    external_path.parent.mkdir()
    external_path.write_bytes(prefix + values.tobytes() + suffix)
    tensor = make_external_tensor(
        "weight",
        TensorProto.FLOAT,
        values.shape,
        "nested/weights.bin",
        len(prefix),
        values.nbytes,
    )

    tensor_slice = resolve_external_tensor(tmp_path / "lm.onnx", tensor)
    mapped = map_external_float32_tensor(tensor_slice, tuple(values.shape))

    assert tensor_slice.location == "nested/weights.bin"
    assert tensor_slice.offset == len(prefix)
    assert tensor_slice.length == values.nbytes
    assert not mapped.flags.writeable
    np.testing.assert_array_equal(mapped, values)


def test_external_data_copy_preserves_only_the_validated_range(tmp_path):
    prefix = b"prefix"
    payload = bytes(range(32))
    suffix = b"suffix"
    path = tmp_path / "weights.bin"
    path.write_bytes(prefix + payload + suffix)
    tensor = _external_tensor("weights.bin", offset=len(prefix), length=len(payload))
    tensor_slice = resolve_external_data_range(
        tmp_path, tensor, logical_bytes=len(payload), require_exact_length=True
    )
    output = io.BytesIO()

    copied = copy_external_data(tensor_slice, output, buffer_size=7)

    assert copied == len(payload)
    assert output.getvalue() == payload


def test_external_tensor_builder_creates_external_reference_without_payload():
    tensor = make_external_tensor(
        "quantized",
        TensorProto.INT8,
        (2, 3),
        "lm.int8.onnx.data",
        64,
        6,
    )

    assert tensor.data_type == TensorProto.INT8
    assert list(tensor.dims) == [2, 3]
    assert tensor.data_location == TensorProto.EXTERNAL
    assert not tensor.HasField("raw_data")
    assert {item.key: item.value for item in tensor.external_data} == {
        "location": "lm.int8.onnx.data",
        "offset": "64",
        "length": "6",
    }


def test_external_range_rejects_past_end_and_mismatched_logical_size(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"01234567")
    tensor = _external_tensor("weights.bin", offset=4, length=8)

    with pytest.raises(ValueError, match="outside its file"):
        resolve_external_data_range(tmp_path, tensor, logical_bytes=8)

    tensor = _external_tensor("weights.bin", length=4)
    with pytest.raises(ValueError, match="expected 8"):
        resolve_external_data_range(
            tmp_path, tensor, logical_bytes=8, require_exact_length=True
        )


def test_external_range_rejects_absolute_and_escaping_locations(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"12345678")
    absolute = _external_tensor(str(tmp_path / "weights.bin"))
    traversal = _external_tensor("../weights.bin")

    with pytest.raises(ValueError, match="must be relative"):
        resolve_external_data_range(tmp_path, absolute, logical_bytes=8)
    with pytest.raises(ValueError, match="relative and contained"):
        resolve_external_data_range(tmp_path, traversal, logical_bytes=8)


def test_streaming_float32_resolver_rejects_unsupported_tensor_encodings(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"\0" * 32)
    wrong_dtype = _external_tensor("weights.bin", dtype=TensorProto.INT8, shape=(2, 2), length=4)
    wrong_rank = _external_tensor("weights.bin", shape=(1, 1, 2, 2), length=16)
    inline_fields = _external_tensor("weights.bin")
    inline_fields.float_data.append(1.0)

    with pytest.raises(ValueError, match="must be float32"):
        resolve_external_tensor(tmp_path / "lm.onnx", wrong_dtype)
    with pytest.raises(ValueError, match="rank 2"):
        resolve_external_tensor(tmp_path / "lm.onnx", wrong_rank)
    with pytest.raises(ValueError, match="inline data fields"):
        resolve_external_tensor(tmp_path / "lm.onnx", inline_fields)


def test_external_tensor_builder_rejects_invalid_location_and_length():
    with pytest.raises(ValueError, match="must be relative"):
        make_external_tensor("weight", TensorProto.INT8, (2,), "/outside.bin", 0, 2)
    with pytest.raises(ValueError, match="expected 2"):
        make_external_tensor("weight", TensorProto.INT8, (2,), "weights.bin", 0, 3)
