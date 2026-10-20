from __future__ import annotations

import hashlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.quantize_ternary_nbits import TernaryNBitsConfig, _is_exact_ternary_weight
from kitten2_onnx.ternary_exact_storage import (
    compress_ternary_matmuls_streaming,
    transcode_exact_ternary_matmuls_to_nbits4_streaming,
)


def _source_weight() -> np.ndarray:
    rng = np.random.default_rng(2048)
    signs = rng.choice(np.asarray([-1, 0, 1], dtype=np.float32), size=(2, 128, 4))
    signs[:, 0, :] = 1
    signs[:, 1, :] = -1
    scales = np.asarray([[0.25, 0.5, 1.0, 2.0], [1.5, 2.0, 2.5, 3.0]], dtype=np.float32)
    weight = (signs * scales[:, None, :]).reshape(256, 4)
    weight[2, 0] = np.float32(-0.0)
    return weight


def _make_exact_int8_model(tmp_path):
    weight = _source_weight()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "lm.onnx"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="tiny_matmul")],
        "exact-int8-to-nbits4",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 256])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(weight, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, str(source_path))
    source_graph_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    eligible, source_sha256 = _is_exact_ternary_weight(weight)
    assert eligible
    inventory = {
        "inventory_sha256": hashlib.sha256(b"tiny-inventory").hexdigest(),
        "source_graph_sha256": source_graph_sha256,
        "eligible_exact_ternary_matmul_count": 1,
        "nodes": [
            {
                "node": "tiny_matmul",
                "initializer": "W",
                "eligible_exact_ternary": True,
                "shape": list(weight.shape),
                "source_sha256": source_sha256,
            }
        ],
    }
    exact_dir = tmp_path / "exact-int8"
    exact_path = exact_dir / "lm.onnx"
    exact_data = exact_dir / "lm.ternary-exact-int8.onnx.data"
    exact_report = compress_ternary_matmuls_streaming(
        source_path,
        exact_path,
        exact_data,
        inventory=inventory,
    )
    exact_metadata = {
        **exact_report,
        "method": "ternary-exact-int8",
        "graph_sha256": hashlib.sha256(exact_path.read_bytes()).hexdigest(),
        "external_data_sha256": {
            exact_data.name: hashlib.sha256(exact_data.read_bytes()).hexdigest()
        },
    }
    return weight, source_path, exact_path, exact_data, exact_metadata


def test_streaming_exact_int8_to_nbits4_preserves_values_and_source(tmp_path):
    weight, source_path, exact_path, _exact_data, exact_metadata = _make_exact_int8_model(tmp_path)
    source_files = {
        path.relative_to(exact_path.parent).as_posix(): path.read_bytes()
        for path in exact_path.parent.iterdir()
        if path.is_file()
    }
    output_dir = tmp_path / "q4"
    output_path = output_dir / "lm.onnx"
    output_data = output_dir / "lm.nbits4.onnx.data"

    report = transcode_exact_ternary_matmuls_to_nbits4_streaming(
        exact_path,
        output_path,
        output_data,
        exact_storage_metadata=exact_metadata,
        config=TernaryNBitsConfig(bits=4, block_size=32),
        column_chunk=2,
    )

    assert report["method"] == "ternary-exact-int8-to-nbits4"
    assert report["transformed_matmul_count"] == 1
    assert report["transformed_initializer_count"] == 1
    assert report["signed_zero_bit_differences"] == 1
    assert report["exact_numeric_reconstruction"] is True
    assert report["source_graph_sha256"] == exact_metadata["graph_sha256"]
    assert (
        report["external_data_sha256"][output_data.name]
        == hashlib.sha256(output_data.read_bytes()).hexdigest()
    )
    assert {
        path.relative_to(exact_path.parent).as_posix(): path.read_bytes()
        for path in exact_path.parent.iterdir()
        if path.is_file()
    } == source_files
    assert source_path.is_file()

    quantized = onnx.load(str(output_path), load_external_data=True)
    assert [node.op_type for node in quantized.graph.node] == ["MatMulNBits"]
    tensors = {tensor.name: numpy_helper.to_array(tensor) for tensor in quantized.graph.initializer}
    tensor_report = report["transformed_tensors"][0]
    block_count = 8
    assert tensors[tensor_report["packed_initializer"]].shape == (4, block_count, 16)
    expected_scales = np.repeat(
        np.asarray([[0.25, 1.5], [0.5, 2.0], [1.0, 2.5], [2.0, 3.0]], dtype=np.float32),
        4,
        axis=1,
    )
    np.testing.assert_array_equal(
        tensors[tensor_report["scales_initializer"]].reshape(4, block_count), expected_scales
    )
    np.testing.assert_array_equal(
        tensors[tensor_report["zero_points_initializer"]].reshape(4, 4),
        np.full((4, 4), 0x88, dtype=np.uint8),
    )
    assert tensor_report["source_code_initializer"] not in tensors
    assert tensor_report["source_scale_initializer"] not in tensors

    activation = np.random.default_rng(73).normal(size=(2, 256)).astype(np.float32)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"A": activation})[0]
    np.testing.assert_allclose(actual, activation @ weight, rtol=1e-5, atol=1e-5)


def test_streaming_exact_int8_to_nbits4_failure_cleans_candidate_files(tmp_path, monkeypatch):
    _, _, exact_path, _, exact_metadata = _make_exact_int8_model(tmp_path)
    output_dir = tmp_path / "failed-q4"
    output_path = output_dir / "lm.onnx"
    output_data = output_dir / "lm.nbits4.onnx.data"
    exact_graph_before = exact_path.read_bytes()

    def fail_session(*args, **kwargs):
        raise RuntimeError("injected ORT failure")

    monkeypatch.setattr(ort, "InferenceSession", fail_session)
    with pytest.raises(RuntimeError, match="injected ORT failure"):
        transcode_exact_ternary_matmuls_to_nbits4_streaming(
            exact_path,
            output_path,
            output_data,
            exact_storage_metadata=exact_metadata,
        )

    assert exact_path.read_bytes() == exact_graph_before
    assert not output_path.exists()
    assert not output_data.exists()
    assert not list(output_dir.glob(".*.tmp"))


def test_streaming_exact_int8_to_nbits4_rejects_stale_source_hash(tmp_path):
    _, _, exact_path, _, exact_metadata = _make_exact_int8_model(tmp_path)
    output_dir = tmp_path / "stale-q4"
    output_path = output_dir / "lm.onnx"
    output_data = output_dir / "lm.nbits4.onnx.data"
    stale_metadata = {**exact_metadata, "graph_sha256": "0" * 64}

    with pytest.raises(ValueError, match="graph hash does not match metadata"):
        transcode_exact_ternary_matmuls_to_nbits4_streaming(
            exact_path,
            output_path,
            output_data,
            exact_storage_metadata=stale_metadata,
        )

    assert not output_path.exists()
    assert not output_data.exists()
