from __future__ import annotations

from dataclasses import replace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from kitten2_onnx.quantize_ternary_nbits import (
    BLOCK_SIZE,
    TernaryNBitsConfig,
    pack_ternary_nbits4,
    quantize_ternary_matmuls_streaming,
    rewrite_matmul_to_nbits4,
    unpack_ternary_nbits4,
)


def _weight(k: int, n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(731 + k)
    block_count = k // BLOCK_SIZE
    signs = rng.choice(np.asarray([-1, 0, 1], dtype=np.float32), size=(block_count, BLOCK_SIZE, n))
    scales = rng.choice(np.asarray([0.125, 0.5, 1.75], dtype=np.float32), size=(block_count, n))
    if block_count > 1:
        signs[0, :, 0] = 0
        scales[0, 0] = 0
    return (signs * scales[:, None, :]).reshape(k, n)



def _nonternary_weight() -> np.ndarray:
    weight = np.zeros((128, 3), dtype=np.float32)
    weight[0, 0] = 0.5
    weight[1, 0] = 0.625
    return weight

def _matmul_model(weight: np.ndarray) -> onnx.ModelProto:
    k, n = weight.shape
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="tiny_matmul")],
        "tiny-ternary-matmul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, n])],
        [numpy_helper.from_array(weight, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model


def test_pack_unpack_preserves_exact_group_scales_and_values():
    weight = _weight(384)

    packed = pack_ternary_nbits4(weight)
    reconstructed = unpack_ternary_nbits4(packed)

    np.testing.assert_array_equal(reconstructed.view(np.uint32), weight.view(np.uint32))
    np.testing.assert_array_equal(packed.scales[0, 0], 0.0)
    assert packed.packed.shape == (3, 3, 64)
    assert packed.zero_points.shape == (3, 2)
    np.testing.assert_array_equal(packed.zero_points[:, 0], 0x88)
    np.testing.assert_array_equal(packed.zero_points[:, 1], 0x08)



def test_pack_accepts_signed_zero_values():
    weight = np.zeros((128, 1), dtype=np.float32)
    weight[0, 0] = np.float32(-0.0)

    reconstructed = unpack_ternary_nbits4(pack_ternary_nbits4(weight))

    assert weight[0, 0] == reconstructed[0, 0] == 0
    assert np.signbit(weight[0, 0])
    assert not np.signbit(reconstructed[0, 0])


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_pack_unpack_smaller_blocks_preserve_source_group_scales(block_size: int):
    weight = _weight(256)
    config = TernaryNBitsConfig(block_size=block_size)

    packed = pack_ternary_nbits4(weight, config)
    reconstructed = unpack_ternary_nbits4(packed)

    np.testing.assert_array_equal(reconstructed.view(np.uint32), weight.view(np.uint32))
    source_scales = np.abs(weight.reshape(2, BLOCK_SIZE, 3)).max(axis=1).T
    expected_scales = np.repeat(source_scales, BLOCK_SIZE // block_size, axis=1)
    np.testing.assert_array_equal(packed.scales, expected_scales)
    assert packed.packed.shape == (3, 256 // block_size, block_size // 2)
    assert packed.block_size == block_size


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bits": 2}, "only 4-bit"),
        ({"block_size": 8}, "block_size must be one of"),
        ({"source_group_size": 64}, "source_group_size must remain 128"),
        ({"accuracy_level": 2}, "accuracy_level must be 0 or 1"),
    ],
)
def test_nbits_config_rejects_unsupported_settings(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        TernaryNBitsConfig(**kwargs)

def test_pack_rejects_unsupported_weights():
    with pytest.raises(ValueError, match="divisible by source_group_size 128"):
        pack_ternary_nbits4(np.zeros((127, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="exactly ternary"):
        pack_ternary_nbits4(_nonternary_weight())
    with pytest.raises(ValueError, match="finite"):
        pack_ternary_nbits4(np.full((128, 2), np.nan, dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        pack_ternary_nbits4(np.zeros((128, 2), dtype=np.float16))


def test_unpack_rejects_codes_outside_the_ternary_encoding():
    packed = pack_ternary_nbits4(_weight(128))
    invalid_codes = packed.packed.copy()
    invalid_codes[0, 0, 0] = 0

    with pytest.raises(ValueError, match="outside the exact ternary encoding"):
        unpack_ternary_nbits4(replace(packed, packed=invalid_codes))

@pytest.mark.parametrize(
    ("k", "block_size", "accuracy_level"),
    [
        (128, block_size, accuracy_level)
        for block_size in (16, 32, 64, 128)
        for accuracy_level in (0, 1)
    ]
    + [(384, 128, 1)],
)
def test_rewritten_matmul_executes_on_ort_cpu_and_matches_fp32(
    k: int, block_size: int, accuracy_level: int
):
    rng = np.random.default_rng(2026 + k)
    weight = _weight(k)
    activation = rng.normal(size=(3, k)).astype(np.float32)
    source = _matmul_model(weight)

    rewritten = rewrite_matmul_to_nbits4(
        source,
        config=TernaryNBitsConfig(block_size=block_size, accuracy_level=accuracy_level),
    )

    onnx.checker.check_model(rewritten)
    assert [node.op_type for node in source.graph.node] == ["MatMul"]
    assert [node.op_type for node in rewritten.graph.node] == ["MatMulNBits"]
    node = rewritten.graph.node[0]
    assert node.domain == "com.microsoft"
    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    assert attributes == {
        "K": k,
        "N": weight.shape[1],
        "bits": 4,
        "block_size": block_size,
        "accuracy_level": accuracy_level,
    }

    session = ort.InferenceSession(
        rewritten.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    assert session.get_providers()[0] == "CPUExecutionProvider"
    actual = session.run(None, {"A": activation})[0]
    expected = activation @ weight
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_rewrite_rejects_non_ternary_initializer():
    model = _matmul_model(_nonternary_weight())

    with pytest.raises(ValueError, match="exactly ternary"):
        rewrite_matmul_to_nbits4(model)



def _external_initializer(name: str, weight: np.ndarray, location: str, offset: int) -> TensorProto:
    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = TensorProto.FLOAT
    tensor.dims.extend(weight.shape)

    tensor.raw_data = weight.tobytes()
    tensor.data_location = TensorProto.EXTERNAL
    external_data_helper.set_external_data(
        tensor,
        location=location,
        offset=offset,
        length=weight.nbytes,
    )

    tensor.ClearField("raw_data")
    return tensor


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_streaming_rewrite_preserves_non_targets_and_executes_on_cpu(tmp_path, block_size: int):
    k = 256
    ternary = _weight(k, 3)
    dense = np.zeros((k, 2), dtype=np.float32)
    dense[0, 0] = 0.5
    dense[1, 0] = 0.625
    prefix = b"external-prefix"
    middle = b"external-middle"
    suffix = b"external-suffix"
    ternary_offset = len(prefix)
    dense_offset = ternary_offset + ternary.nbytes + len(middle)
    external_source = tmp_path / "source"
    external_source.mkdir()
    external_path = external_source / "weights.bin"
    external_path.write_bytes(
        prefix + ternary.tobytes() + middle + dense.tobytes() + suffix
    )

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["A", "W_ternary"], ["Y_ternary"], name="ternary_matmul"),
            helper.make_node("MatMul", ["A", "W_dense"], ["Y_dense"], name="dense_matmul"),
            helper.make_node("MatMul", ["A", "W_dynamic"], ["Y_dynamic"], name="dynamic_matmul"),
        ],
        "streaming-ternary-test",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, k]),
            helper.make_tensor_value_info("W_dynamic", TensorProto.FLOAT, [k, 1]),
        ],
        [
            helper.make_tensor_value_info("Y_ternary", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("Y_dense", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("Y_dynamic", TensorProto.FLOAT, [2, 1]),
        ],
        [
            _external_initializer("W_ternary", ternary, "weights.bin", ternary_offset),
            _external_initializer("W_dense", dense, "weights.bin", dense_offset),
        ],
    )
    source_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    source_model.ir_version = min(source_model.ir_version, 10)
    source_path = external_source / "lm.onnx"
    onnx.save_model(source_model, str(source_path))
    original_graph_bytes = source_path.read_bytes()
    original_data_bytes = external_path.read_bytes()

    mismatch_output = tmp_path / "mismatch" / "lm.onnx"
    with pytest.raises(ValueError, match="Expected 2 source-eligible ternary MatMuls, found 1"):
        quantize_ternary_matmuls_streaming(
            source_path,
            mismatch_output,
            output_external_data="lm.nbits4.onnx.data",
            expected_source_ternary_matmul_count=2,
        )
    assert not mismatch_output.exists()

    output_dir = tmp_path / "output"
    output_path = output_dir / "lm.onnx"
    report = quantize_ternary_matmuls_streaming(
        source_path,
        output_path,
        output_external_data="lm.nbits4.onnx.data",
        config=TernaryNBitsConfig(block_size=block_size),
        expected_source_ternary_matmul_count=1,
        column_chunk=2,
    )

    assert source_path.read_bytes() == original_graph_bytes
    assert external_path.read_bytes() == original_data_bytes
    assert report["matmul_count"] == 3
    assert report["transformed_matmul_count"] == 1
    assert report["transformed_initializer_count"] == 1
    assert report["non_ternary_initializer_count"] == 1
    assert report["accidentally_transformed_non_ternary_count"] == 0
    tensor_report = report["transformed_tensors"][0]
    assert tensor_report["source_initializer"] == "W_ternary"
    assert tensor_report["source_shape"] == [k, 3]
    assert tensor_report["exact_reconstruction"]
    assert tensor_report["packed_code_bytes"] == 3 * (k // block_size) * (block_size // 2)
    assert tensor_report["scale_bytes"] == 3 * (k // block_size) * 4
    assert tensor_report["zero_point_bytes"] == 3 * ((k // block_size + 1) // 2)
    assert tensor_report["block_size"] == block_size

    rewritten = onnx.load(str(output_path), load_external_data=False)
    assert [node.op_type for node in rewritten.graph.node] == ["MatMulNBits", "MatMul", "MatMul"]
    locations = {
        entry.value
        for tensor in rewritten.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    assert locations == {"lm.nbits4.onnx.data"}
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(72)
    activation = rng.normal(size=(2, k)).astype(np.float32)
    dynamic_weight = rng.normal(size=(k, 1)).astype(np.float32)
    actual = session.run(None, {"A": activation, "W_dynamic": dynamic_weight})
    np.testing.assert_allclose(actual[0], activation @ ternary, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual[1], activation @ dense)
    np.testing.assert_allclose(actual[2], activation @ dynamic_weight, rtol=1e-6, atol=1e-6)
