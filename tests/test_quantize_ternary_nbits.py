from __future__ import annotations

from dataclasses import replace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.quantize_ternary_nbits import (
    BLOCK_SIZE,
    pack_ternary_nbits4,
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


def test_pack_rejects_unsupported_weights():
    with pytest.raises(ValueError, match="divisible by 128"):
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


@pytest.mark.parametrize("k", [128, 384])
def test_rewritten_matmul_executes_on_ort_cpu_and_matches_fp32(k: int):
    rng = np.random.default_rng(2026 + k)
    weight = _weight(k)
    activation = rng.normal(size=(3, k)).astype(np.float32)
    source = _matmul_model(weight)

    rewritten = rewrite_matmul_to_nbits4(source)

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
        "block_size": 128,
        "accuracy_level": 1,
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
