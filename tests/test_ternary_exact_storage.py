from __future__ import annotations

import hashlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from kitten2_onnx.quantize_ternary_nbits import _is_exact_ternary_weight
from kitten2_onnx.ternary_exact_storage import (
    TernaryExactStorageConfig,
    _encode_weight,
    compress_ternary_matmuls_streaming,
    rewrite_matmul_to_exact_ternary_storage,
)


def _weight(k: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(731 + k + n)
    groups = k // 128
    signs = rng.choice(np.asarray([-1, 0, 1], dtype=np.float32), size=(groups, 128, n))
    scales = rng.choice(np.asarray([0.125, 0.5, 1.75], dtype=np.float32), size=(groups, n))
    signs[:, 0, :] = 1
    signs[:, 1, :] = 0
    signs[:, 2, :] = -1
    return (signs * scales[:, None, :]).reshape(k, n)


def _matmul_model(weight: np.ndarray, *, extra_consumer: bool = False) -> onnx.ModelProto:
    k, n = weight.shape
    nodes = [helper.make_node("MatMul", ["A", "W"], ["Y"], name="tiny_matmul")]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, n])]
    if extra_consumer:
        nodes.append(helper.make_node("Identity", ["W"], ["W_copy"], name="weight_consumer"))
        outputs.append(helper.make_tensor_value_info("W_copy", TensorProto.FLOAT, [k, n]))
    graph = helper.make_graph(
        nodes,
        "tiny-exact-ternary-matmul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, k])],
        outputs,
        [numpy_helper.from_array(weight, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model


def _external_model(path, weight: np.ndarray) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    external_path = path.parent / "source.data"
    prefix = b"source-prefix"
    suffix = b"source-suffix"
    payload = weight.tobytes(order="C")
    external_path.write_bytes(prefix + payload + suffix)
    tensor = TensorProto()
    tensor.name = "W"
    tensor.data_type = TensorProto.FLOAT
    tensor.dims.extend(weight.shape)
    tensor.raw_data = payload
    tensor.data_location = TensorProto.EXTERNAL
    external_data_helper.set_external_data(
        tensor,
        location=external_path.name,
        offset=len(prefix),
        length=len(payload),
    )
    tensor.ClearField("raw_data")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="tiny_matmul")],
        "tiny-external-exact-ternary-matmul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, weight.shape[0]])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, weight.shape[1]])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, str(path))
    return external_path.read_bytes()


@pytest.mark.parametrize("shape", [(128, 16), (256, 16), (128, 32)])
def test_encode_exact_ternary_weight_is_bit_identical(shape: tuple[int, int]):
    weight = _weight(*shape)
    codes, scales = _encode_weight(weight, TernaryExactStorageConfig())
    reconstructed = codes.astype(np.float32) * scales

    assert codes.shape == (shape[0] // 128, 128, shape[1])
    assert scales.shape == (shape[0] // 128, 1, shape[1])
    assert set(np.unique(codes)) <= {-1, 0, 1}
    np.testing.assert_array_equal(
        reconstructed.reshape(shape).view(np.uint32), weight.view(np.uint32)
    )
    for group_codes in codes:
        assert {-1, 0, 1} <= set(np.unique(group_codes))


def test_config_rejects_non_group_128_and_other_dtypes():
    with pytest.raises(ValueError, match="group_size must be 128"):
        TernaryExactStorageConfig(group_size=64)
    with pytest.raises(ValueError, match="code_dtype must be 'int8'"):
        TernaryExactStorageConfig(code_dtype="uint8")
    with pytest.raises(ValueError, match="scale_dtype must be 'float32'"):
        TernaryExactStorageConfig(scale_dtype="float16")


def test_encode_rejects_nonternary_and_negative_zero():
    nonternary = np.zeros((128, 2), dtype=np.float32)
    nonternary[:2, 0] = [0.5, 0.625]
    with pytest.raises(ValueError, match="not exact group-128 ternary"):
        _encode_weight(nonternary, TernaryExactStorageConfig())

    negative_zero = _weight(128, 2)
    negative_zero[3, 0] = np.float32(-0.0)
    with pytest.raises(ValueError, match="negative zero"):
        _encode_weight(negative_zero, TernaryExactStorageConfig())


def test_rewrite_matmul_uses_primitive_ops_and_matches_baseline():
    weight = _weight(256, 16)
    activation = np.random.default_rng(2026).normal(size=(2, 256)).astype(np.float32)
    source = _matmul_model(weight)
    source_bytes = source.SerializeToString()

    rewritten = rewrite_matmul_to_exact_ternary_storage(source, node_name="tiny_matmul")

    onnx.checker.check_model(rewritten)
    assert [node.op_type for node in source.graph.node] == ["MatMul"]
    assert source.SerializeToString() == source_bytes
    assert [node.op_type for node in rewritten.graph.node] == [
        "Cast",
        "Mul",
        "Reshape",
        "MatMul",
    ]
    assert "W" not in {tensor.name for tensor in rewritten.graph.initializer}
    session = ort.InferenceSession(
        rewritten.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    assert session.get_providers()[0] == "CPUExecutionProvider"
    actual = session.run(None, {"A": activation})[0]
    expected = activation @ weight
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_rewrite_retains_initializer_with_other_consumer():
    weight = _weight(128, 3)
    rewritten = rewrite_matmul_to_exact_ternary_storage(
        _matmul_model(weight, extra_consumer=True), node_name="tiny_matmul"
    )
    assert "W" in {tensor.name for tensor in rewritten.graph.initializer}
    onnx.checker.check_model(rewritten)


def test_streaming_external_candidate_is_atomic_and_preserves_source(tmp_path):
    weight = _weight(256, 16)
    source_path = tmp_path / "source" / "lm.onnx"
    source_data_before = _external_model(source_path, weight)
    source_graph_before = source_path.read_bytes()
    source_model = onnx.load(str(source_path), load_external_data=False)
    source_weight = source_model.graph.initializer[0]
    is_ternary, source_sha256 = _is_exact_ternary_weight(weight)
    assert is_ternary
    inventory = {
        "inventory_sha256": hashlib.sha256(b"inventory").hexdigest(),
        "eligible_exact_ternary_matmul_count": 1,
        "nodes": [
            {
                "node": "tiny_matmul",
                "eligible_exact_ternary": True,
                "initializer": source_weight.name,
                "shape": list(weight.shape),
                "source_sha256": source_sha256,
            }
        ],
    }
    output_path = tmp_path / "candidate" / "lm.onnx"
    output_data = tmp_path / "candidate" / "lm.onnx.data"

    report = compress_ternary_matmuls_streaming(
        source_path,
        output_path,
        output_data,
        inventory=inventory,
    )

    assert report["transformed_matmul_count"] == 1
    assert report["transformed_nodes"] == ["tiny_matmul"]
    assert report["exact_reconstruction"] is True
    assert report["source_graph_sha256"] == hashlib.sha256(source_graph_before).hexdigest()
    assert report["inventory_sha256"] == inventory["inventory_sha256"]
    assert report["code_bytes"] == weight.size
    assert report["scale_bytes"] == weight.shape[0] // 128 * weight.shape[1] * 4
    assert source_path.read_bytes() == source_graph_before
    assert (source_path.parent / "source.data").read_bytes() == source_data_before

    candidate = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(str(output_path))
    assert "W" not in {tensor.name for tensor in candidate.graph.initializer}
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    activation = np.random.default_rng(2027).normal(size=(2, 256)).astype(np.float32)
    actual = session.run(None, {"A": activation})[0]
    expected = activation @ weight
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_streaming_failure_cleans_candidate_and_preserves_source(tmp_path):
    weight = _weight(128, 4)
    weight[4, 0] = np.float32(-0.0)
    source_path = tmp_path / "source" / "lm.onnx"
    source_data_before = _external_model(source_path, weight)
    source_graph_before = source_path.read_bytes()
    is_ternary, source_sha256 = _is_exact_ternary_weight(weight)
    assert is_ternary
    inventory = {
        "eligible_exact_ternary_matmul_count": 1,
        "nodes": [
            {
                "node": "tiny_matmul",
                "eligible_exact_ternary": True,
                "initializer": "W",
                "shape": list(weight.shape),
                "source_sha256": source_sha256,
            }
        ],
    }
    output_path = tmp_path / "candidate" / "lm.onnx"
    output_data = tmp_path / "candidate" / "lm.onnx.data"

    with pytest.raises(ValueError, match="negative zero"):
        compress_ternary_matmuls_streaming(
            source_path,
            output_path,
            output_data,
            inventory=inventory,
        )

    assert not output_path.exists()
    assert not output_data.exists()
    assert not list(output_path.parent.glob(".*.tmp"))
    assert source_path.read_bytes() == source_graph_before
    assert (source_path.parent / "source.data").read_bytes() == source_data_before



def test_session_creation_failure_cleans_candidate_outputs(tmp_path, monkeypatch):
    weight = _weight(128, 4)
    source_path = tmp_path / "source" / "lm.onnx"
    _external_model(source_path, weight)
    source_graph_before = source_path.read_bytes()
    source_model = onnx.load(str(source_path), load_external_data=False)
    _, source_sha256 = _is_exact_ternary_weight(weight)
    inventory = {
        "eligible_exact_ternary_matmul_count": 1,
        "nodes": [
            {
                "node": "tiny_matmul",
                "eligible_exact_ternary": True,
                "initializer": source_model.graph.initializer[0].name,
                "shape": list(weight.shape),
                "source_sha256": source_sha256,
            }
        ],
    }
    output_path = tmp_path / "candidate" / "lm.onnx"
    output_data = tmp_path / "candidate" / "lm.onnx.data"

    def fail_session(*args, **kwargs):
        raise RuntimeError("session creation failed")

    monkeypatch.setattr(ort, "InferenceSession", fail_session)
    with pytest.raises(RuntimeError, match="session creation failed"):
        compress_ternary_matmuls_streaming(
            source_path,
            output_path,
            output_data,
            inventory=inventory,
        )

    assert not output_path.exists()
    assert not output_data.exists()
    assert not list(output_path.parent.glob(".*.tmp"))
    assert source_path.read_bytes() == source_graph_before
