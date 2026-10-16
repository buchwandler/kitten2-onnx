from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.quant_utils import compute_data_quant_params, quantize_nparray

from kitten2_onnx.external_data import make_external_tensor
from kitten2_onnx.quantize_int8_streaming import (
    ExternalDataWriter,
    quantize_dynamic_matmul_streaming,
    quantize_external_tensor_to_int8,
)


def _weight_fixtures() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(731)
    return [
        ("shape_2x2", rng.normal(size=(2, 2)).astype(np.float32)),
        ("shape_3x5", rng.normal(size=(3, 5)).astype(np.float32)),
        ("shape_5x3", rng.normal(size=(5, 3)).astype(np.float32)),
        ("shape_1x7", rng.normal(size=(1, 7)).astype(np.float32)),
        ("shape_7x1", rng.normal(size=(7, 1)).astype(np.float32)),
        ("all_zero", np.zeros((3, 2), dtype=np.float32)),
        (
            "zero_and_constant_channels",
            np.asarray([[0, 2, -3, 1], [0, 2, -3, -1], [0, 2, -3, 0]], dtype=np.float32),
        ),
        (
            "rounding_boundaries",
            np.asarray(
                [
                    [0.5, -0.5, 1.5, -1.5, 127.0, -127.0, 1.0, -1.0],
                    [127.0, -127.0, 127.0, -127.0, 127.5, -127.5, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        ),
        (
            "outlier_and_subnormal_channels",
            np.asarray(
                [[0.25, 1e-40, -1e-40], [1000.0, -2e-40, 0.0], [1.0, 3e-40, 1e-40]],
                dtype=np.float32,
            ),
        ),
        ("seeded_random_7x9", rng.normal(size=(7, 9)).astype(np.float32)),
    ]


def _write_matmul_model(
    model_path: Path,
    weight: np.ndarray,
    *,
    external_offset: bool = False,
) -> None:
    if external_offset:
        prefix = b"unrelated-prefix"
        payload = weight.tobytes(order="C")
        suffix = b"unrelated-suffix"
        (model_path.parent / "weights.bin").write_bytes(prefix + payload + suffix)
        initializer = TensorProto()
        initializer.name = "W"
        initializer.data_type = TensorProto.FLOAT
        initializer.dims.extend(weight.shape)
        initializer.data_location = TensorProto.EXTERNAL
        initializer.raw_data = payload
        external_data_helper.set_external_data(
            initializer,
            location="weights.bin",
            offset=len(prefix),
            length=len(payload),
        )
        initializer.ClearField("raw_data")
    else:
        initializer = numpy_helper.from_array(weight, name="W")

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"])],
        "tiny-matmul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, weight.shape[0]])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, weight.shape[1]])],
        [initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, str(model_path))


def _ort_quantized_tensors(
    tmp_path: Path,
    weight: np.ndarray,
    *,
    external_offset: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, onnx.ModelProto]:
    source = tmp_path / "source.onnx"
    optimized = tmp_path / "quantized.onnx"
    _write_matmul_model(source, weight, external_offset=external_offset)
    quantize_dynamic(
        str(source),
        str(optimized),
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )
    model = onnx.load(str(optimized), load_external_data=True)
    integer_matmuls = [node for node in model.graph.node if node.op_type == "MatMulInteger"]
    assert len(integer_matmuls) == 1
    quantized_weight_name = integer_matmuls[0].input[1]
    zero_point_name = integer_matmuls[0].input[3]
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    scale_name = next(
        name for name in initializers if name.endswith("_scale") and name != "A_scale"
    )
    return (
        initializers[quantized_weight_name],
        initializers[scale_name],
        initializers[zero_point_name],
        model,
    )


def _expected_ort_weight(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantized = np.empty(weight.shape, dtype=np.int8)
    scales = np.empty(weight.shape[1], dtype=np.float32)
    zero_points = np.empty(weight.shape[1], dtype=np.int8)
    for channel in range(weight.shape[1]):
        zero_point, scale = compute_data_quant_params(
            weight[:, channel], TensorProto.INT8, symmetric=True, reduce_range=False
        )
        scales[channel] = scale
        zero_points[channel] = zero_point
        quantized[:, channel] = quantize_nparray(
            TensorProto.INT8, weight[:, channel], scale, zero_point
        )
    return quantized, scales, zero_points


@pytest.mark.parametrize(
    ("case_name", "weight"),
    _weight_fixtures(),
    ids=[case[0] for case in _weight_fixtures()],
)
def test_ort_qint8_matmul_reference_semantics(tmp_path, case_name, weight):
    quantized, scales, zero_points, _ = _ort_quantized_tensors(tmp_path, weight)
    expected_quantized, expected_scales, expected_zero_points = _expected_ort_weight(weight)

    assert quantized.dtype == np.int8
    assert quantized.shape == weight.shape
    assert scales.dtype == np.float32
    assert scales.shape == (weight.shape[1],)
    assert zero_points.dtype == np.int8
    assert zero_points.shape == (weight.shape[1],)
    np.testing.assert_array_equal(quantized, expected_quantized, err_msg=case_name)
    np.testing.assert_array_equal(scales, expected_scales, err_msg=case_name)
    np.testing.assert_array_equal(zero_points, expected_zero_points, err_msg=case_name)


def test_ort_qint8_matmul_reference_reads_external_weight_slice(tmp_path):
    weight = np.asarray(
        [[0.5, -2.0, 3.0, 0.0, 1.0], [127.0, -1.0, -3.0, 0.0, -1.0], [2.0, 0.25, 1.0, 0.0, 0.5]],
        dtype=np.float32,
    )
    quantized, scales, zero_points, _ = _ort_quantized_tensors(
        tmp_path, weight, external_offset=True
    )
    expected_quantized, expected_scales, expected_zero_points = _expected_ort_weight(weight)

    np.testing.assert_array_equal(quantized, expected_quantized)
    np.testing.assert_array_equal(scales, expected_scales)
    np.testing.assert_array_equal(zero_points, expected_zero_points)
    raw_data = (tmp_path / "weights.bin").read_bytes()
    assert raw_data.startswith(b"unrelated-prefix")
    assert raw_data.endswith(b"unrelated-suffix")


@pytest.mark.parametrize(
    ("case_name", "weight"),
    _weight_fixtures(),
    ids=[case[0] for case in _weight_fixtures()],
)
def test_streaming_qint8_weight_matches_ort_across_chunk_sizes(tmp_path, case_name, weight):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "lm.onnx"
    _write_matmul_model(source_path, weight, external_offset=True)
    source_model = onnx.load(str(source_path), load_external_data=False)
    source_weight = source_model.graph.initializer[0]

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    expected_q, expected_scales, expected_zero_points, _ = _ort_quantized_tensors(
        reference_dir, weight, external_offset=True
    )

    for chunk_rows in (1, 2, 3, 7, 64):
        output_path = tmp_path / f"quantized-{chunk_rows}.bin"
        with ExternalDataWriter(output_path) as writer:
            header_offset, header_length = writer.append_bytes(b"header")
            result = quantize_external_tensor_to_int8(
                source_path, source_weight, writer, chunk_rows=chunk_rows
            )

        output = output_path.read_bytes()
        actual_q = np.frombuffer(
            output[result.offset : result.offset + result.length], dtype=np.int8
        ).reshape(weight.shape)
        assert header_offset == 0
        assert header_length == len(b"header")
        assert result.offset == 64
        assert result.length == weight.size
        np.testing.assert_array_equal(actual_q, expected_q, err_msg=case_name)
        np.testing.assert_array_equal(result.scales, expected_scales, err_msg=case_name)
        np.testing.assert_array_equal(result.zero_points, expected_zero_points, err_msg=case_name)



def test_streaming_quantized_weight_matches_ort_runtime_output(tmp_path):
    weight = np.asarray(
        [[0.125, -0.5, 1.0], [2.0, 0.25, -3.0], [-1.5, 0.75, 0.5]], dtype=np.float32
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "lm.onnx"
    _write_matmul_model(source_path, weight, external_offset=True)
    source_model = onnx.load(str(source_path), load_external_data=False)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    expected_q, expected_scales, expected_zero_points, reference_model = _ort_quantized_tensors(
        reference_dir, weight, external_offset=True
    )

    output_path = tmp_path / "streamed.data"
    with ExternalDataWriter(output_path) as writer:
        streamed = quantize_external_tensor_to_int8(
            source_path, source_model.graph.initializer[0], writer, chunk_rows=2
        )
    actual_q = np.frombuffer(
        output_path.read_bytes()[streamed.offset : streamed.offset + streamed.length],
        dtype=np.int8,
    ).reshape(weight.shape)
    np.testing.assert_array_equal(actual_q, expected_q)
    np.testing.assert_array_equal(streamed.scales, expected_scales)
    np.testing.assert_array_equal(streamed.zero_points, expected_zero_points)

    matmul_integer = next(node for node in reference_model.graph.node if node.op_type == "MatMulInteger")
    weight_name = matmul_integer.input[1]
    zero_point_name = matmul_integer.input[3]
    scale_name = next(
        tensor.name
        for tensor in reference_model.graph.initializer
        if tensor.name.endswith("_scale") and tensor.name != "A_scale"
    )
    replacements = {
        weight_name: actual_q,
        scale_name: streamed.scales,
        zero_point_name: streamed.zero_points,
    }
    for initializer in reference_model.graph.initializer:
        if initializer.name in replacements:
            initializer.CopyFrom(
                numpy_helper.from_array(replacements[initializer.name], name=initializer.name)
            )

    streamed_model_path = tmp_path / "streamed.onnx"
    onnx.save_model(reference_model, str(streamed_model_path))
    reference_session = ort.InferenceSession(
        str(reference_dir / "quantized.onnx"), providers=["CPUExecutionProvider"]
    )
    streamed_session = ort.InferenceSession(
        str(streamed_model_path), providers=["CPUExecutionProvider"]
    )
    activation = np.asarray([[0.25, -2.0, 1.5]], dtype=np.float32)
    expected_output = reference_session.run(None, {"A": activation})
    actual_output = streamed_session.run(None, {"A": activation})
    np.testing.assert_array_equal(actual_output[0], expected_output[0])



def _write_graph_rewrite_bundle(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    root.mkdir(parents=True)
    weights = {
        "W_shared": np.asarray([[0.125, -0.5, 1.0], [2.0, 0.25, -3.0]], dtype=np.float32),
        "W_second": np.asarray([[1.5, -2.0, 0.75], [-0.5, 3.0, 2.25]], dtype=np.float32),
        "W_untouched": np.asarray([[0.25, 0.5, 0.75], [1.0, 1.25, 1.5]], dtype=np.float32),
    }
    payload = bytearray(b"external-prefix")
    initializers = []
    for name, weight in weights.items():
        offset = len(payload)
        raw = weight.tobytes(order="C")
        payload.extend(raw)
        initializers.append(
            make_external_tensor(name, TensorProto.FLOAT, weight.shape, "weights.bin", offset, len(raw))
        )
        payload.extend(b"gap")
    (root / "weights.bin").write_bytes(payload + b"external-suffix")

    nodes = [
        helper.make_node("MatMul", ["A", "W_shared"], ["Y_shared_1"], name="mm_shared_1"),
        helper.make_node("MatMul", ["A", "W_second"], ["Y_second"], name="mm_second"),
        helper.make_node("MatMul", ["A2", "W_shared"], ["Y_shared_2"], name="mm_shared_2"),
        helper.make_node("MatMul", ["A3", "B_dynamic"], ["Y_dynamic"], name="mm_dynamic"),
        helper.make_node("Identity", ["W_shared"], ["shared_copy"], name="keep_shared"),
        helper.make_node("Identity", ["W_untouched"], ["untouched_copy"], name="keep_untouched"),
    ]
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in (
            ("A", [1, 2]),
            ("A2", [1, 2]),
            ("A3", [1, 2]),
            ("B_dynamic", [2, 3]),
        )
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in (
            ("Y_shared_1", [1, 3]),
            ("Y_second", [1, 3]),
            ("Y_shared_2", [1, 3]),
            ("Y_dynamic", [1, 3]),
            ("shared_copy", [2, 3]),
            ("untouched_copy", [2, 3]),
        )
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "streaming-graph", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = min(model.ir_version, 10)
    model_path = root / "lm.onnx"
    onnx.save_model(model, str(model_path))
    return model_path, weights


def _external_tensor_bytes(model_path: Path, tensor) -> bytes:
    external = {entry.key: entry.value for entry in tensor.external_data}
    location = model_path.parent / external["location"]
    offset = int(external.get("offset", "0"))
    length = int(external["length"])
    return location.read_bytes()[offset : offset + length]


def test_streaming_graph_rewrite_handles_shared_and_mixed_consumers(tmp_path):
    source_dir = tmp_path / "source"
    source_path, weights = _write_graph_rewrite_bundle(source_dir)
    source_graph_bytes = source_path.read_bytes()
    source_data_bytes = (source_dir / "weights.bin").read_bytes()
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference_path = reference_dir / "ort.onnx"
    quantize_dynamic(
        str(source_path),
        str(reference_path),
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )

    output_path = tmp_path / "optimized" / "lm.onnx"
    report = quantize_dynamic_matmul_streaming(
        source_path,
        output_path,
        output_external_data="lm.int8.onnx.data",
        chunk_rows=1,
    )

    graph = onnx.load(str(output_path), load_external_data=False)
    initializer_names = {tensor.name for tensor in graph.graph.initializer}
    assert report["matmul_count"] == 4
    assert report["transformed_matmul_count"] == 3
    assert report["unique_quantized_weight_count"] == 2
    assert report["shared_activation_count"] == 1
    assert report["excluded_nodes"] == ["mm_dynamic"]
    assert sum(node.op_type == "DynamicQuantizeLinear" for node in graph.graph.node) == 2
    assert sum(node.op_type == "MatMulInteger" for node in graph.graph.node) == 3
    assert "W_shared" in initializer_names
    assert "W_second" not in initializer_names
    assert "W_untouched" in initializer_names
    assert "W_shared_quantized" in initializer_names
    assert "W_second_quantized" in initializer_names
    assert [value.name for value in graph.graph.output] == [
        "Y_shared_1",
        "Y_second",
        "Y_shared_2",
        "Y_dynamic",
        "shared_copy",
        "untouched_copy",
    ]

    source_initializers = {
        tensor.name: tensor
        for tensor in onnx.load(str(source_path), load_external_data=False).graph.initializer
    }
    output_initializers = {tensor.name: tensor for tensor in graph.graph.initializer}
    for name in ("W_shared", "W_untouched"):
        assert _external_tensor_bytes(output_path, output_initializers[name]) == weights[name].tobytes()
        assert _external_tensor_bytes(source_path, source_initializers[name]) == weights[name].tobytes()

    inputs = {
        "A": np.asarray([[0.25, -2.0]], dtype=np.float32),
        "A2": np.asarray([[1.0, 0.5]], dtype=np.float32),
        "A3": np.asarray([[-0.5, 2.0]], dtype=np.float32),
        "B_dynamic": np.asarray([[1.0, 0.25, -0.5], [0.5, -1.0, 2.0]], dtype=np.float32),
    }
    expected = ort.InferenceSession(str(reference_path), providers=["CPUExecutionProvider"]).run(
        None, inputs
    )
    actual = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"]).run(
        None, inputs
    )
    for actual_value, expected_value in zip(actual[:4], expected[:4], strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)
    np.testing.assert_array_equal(actual[4], weights["W_shared"])
    np.testing.assert_array_equal(actual[5], weights["W_untouched"])
    assert source_path.read_bytes() == source_graph_bytes
    assert (source_dir / "weights.bin").read_bytes() == source_data_bytes


def test_streaming_graph_rewrite_rejects_constant_weight_subgraphs_before_output(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weight = np.arange(6, dtype=np.float32).reshape(3, 2)
    raw = weight.tobytes()
    (source_dir / "weights.bin").write_bytes(raw)
    initializer = make_external_tensor("W", TensorProto.FLOAT, weight.shape, "weights.bin", 0, len(raw))
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["W"], ["W_transposed"], perm=[1, 0]),
            helper.make_node("MatMul", ["A", "W_transposed"], ["Y"], name="constant_b_matmul"),
        ],
        "constant-b-subgraph",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
        [initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    source_path = source_dir / "lm.onnx"
    onnx.save_model(model, str(source_path))
    output_path = tmp_path / "optimized" / "lm.onnx"

    with pytest.raises(ValueError, match="constant-producing subgraph: node=constant_b_matmul"):
        quantize_dynamic_matmul_streaming(
            source_path,
            output_path,
            output_external_data="lm.int8.onnx.data",
        )

    assert not output_path.exists()
    assert not output_path.parent.exists()



def test_streaming_graph_rewrite_keeps_initializer_that_is_graph_output(tmp_path):
    source_path, weights = _write_graph_rewrite_bundle(tmp_path / "source")
    model = onnx.load(str(source_path), load_external_data=False)
    model.graph.output.add().CopyFrom(
        helper.make_tensor_value_info("W_second", TensorProto.FLOAT, [2, 3])
    )
    onnx.save_model(model, str(source_path))
    output_path = tmp_path / "optimized" / "lm.onnx"

    quantize_dynamic_matmul_streaming(
        source_path,
        output_path,
        output_external_data="lm.int8.onnx.data",
    )

    graph = onnx.load(str(output_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in graph.graph.initializer}
    assert "W_second" in initializers
    assert "W_second" in {value.name for value in graph.graph.output}
    assert _external_tensor_bytes(output_path, initializers["W_second"]) == weights[
        "W_second"
    ].tobytes()



def test_streaming_graph_rewrite_generates_stable_unique_names(tmp_path):
    source_path, _ = _write_graph_rewrite_bundle(tmp_path / "source")
    model = onnx.load(str(source_path), load_external_data=False)
    external_path = source_path.parent / "weights.bin"
    payload = bytearray(external_path.read_bytes())
    for name in (
        "A_quantized",
        "A_scale",
        "A_zero_point",
        "W_shared_quantized",
        "W_shared_scale",
        "W_shared_zero_point",
    ):
        value = np.asarray([1.0], dtype=np.float32)
        offset = len(payload)
        raw = value.tobytes()
        payload.extend(raw)
        model.graph.initializer.add().CopyFrom(
            make_external_tensor(name, TensorProto.FLOAT, (1,), "weights.bin", offset, len(raw))
        )
    external_path.write_bytes(payload)
    onnx.save_model(model, str(source_path))

    graph_signatures = []
    for output_dir in (tmp_path / "first", tmp_path / "second"):
        output_path = output_dir / "lm.onnx"
        quantize_dynamic_matmul_streaming(
            source_path,
            output_path,
            output_external_data="lm.int8.onnx.data",
        )
        graph = onnx.load(str(output_path), load_external_data=False).graph
        dql = next(node for node in graph.node if node.op_type == "DynamicQuantizeLinear")
        first_integer_matmul = next(node for node in graph.node if node.op_type == "MatMulInteger")
        assert list(dql.output) == ["A_quantized_1", "A_scale_1", "A_zero_point_1"]
        assert first_integer_matmul.input[1] == "W_shared_quantized_1"
        assert len([name for node in graph.node for name in node.output]) == len(
            {name for node in graph.node for name in node.output}
        )
        graph_signatures.append(
            (
                [(node.op_type, node.name, tuple(node.input), tuple(node.output)) for node in graph.node],
                [tensor.name for tensor in graph.initializer],
            )
        )

    assert graph_signatures[0] == graph_signatures[1]
