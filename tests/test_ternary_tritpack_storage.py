from __future__ import annotations

import hashlib
import multiprocessing
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from kitten2_onnx.external_data import (
    logical_tensor_bytes,
    map_external_float32_tensor,
    resolve_external_data_range,
)
from kitten2_onnx.materialized_weights import (
    _cache_identity,
    materialize_tritpack_model,
)
from kitten2_onnx.ternary_tritpack_storage import (
    MAX_PACKED_CODE,
    PACKED_BYTES_PER_GROUP,
    TERNARY_TRITPACK_FORMAT_VERSION,
    TRITS_PER_BYTE,
    TernaryTritpackConfig,
    build_decode_lut,
    build_ternary_tritpack_inventory,
    compress_ternary_matmuls_to_tritpack_streaming,
    pack_group,
    pack_weight_reference,
    rewrite_matmul_to_tritpack_storage,
    unpack_group,
    unpack_weight_reference,
    validate_ternary_group,
)


def _group(rows: int = 128, columns: int = 7, seed: int = 731) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1, 0, 1], dtype=np.float32), size=(rows, columns))
    scales = rng.choice(np.asarray([0.125, 0.5, 1.75], dtype=np.float32), size=columns)
    return signs * scales[None, :]


def test_config_has_fixed_group_local_format():
    config = TernaryTritpackConfig()

    assert config.group_size == 128
    assert config.trits_per_byte == TRITS_PER_BYTE == 5
    assert config.packed_bytes_per_group == PACKED_BYTES_PER_GROUP == 26
    assert config.scale_dtype == "float32"
    assert config.format_version == TERNARY_TRITPACK_FORMAT_VERSION


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_size": 64}, "group_size must be 128"),
        ({"trits_per_byte": 4}, "trits_per_byte must be 5"),
        ({"packed_bytes_per_group": 25}, "packed_bytes_per_group must be 26"),
        ({"scale_dtype": "float16"}, "scale_dtype must be 'float32'"),
        ({"format_version": "future"}, "format_version must be"),
    ],
)
def test_config_rejects_unsupported_format_settings(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        TernaryTritpackConfig(**kwargs)


def test_decode_lut_exhaustively_covers_all_valid_byte_codes():
    lut = build_decode_lut()

    assert lut.shape == (MAX_PACKED_CODE + 1, TRITS_PER_BYTE)
    assert lut.dtype == np.int8
    assert set(np.unique(lut)) == {-1, 0, 1}
    for code in range(MAX_PACKED_CODE + 1):
        expected = [((code // (3**digit)) % 3) - 1 for digit in range(TRITS_PER_BYTE)]
        np.testing.assert_array_equal(lut[code], expected)


def test_all_243_five_trit_patterns_pack_and_unpack():
    patterns = build_decode_lut()
    group = np.zeros((128, MAX_PACKED_CODE + 1), dtype=np.float32)
    group[:TRITS_PER_BYTE] = patterns.T

    packed, scales = pack_group(group)
    reconstructed = unpack_group(packed, scales)

    np.testing.assert_array_equal(packed[:, 0], np.arange(MAX_PACKED_CODE + 1, dtype=np.uint8))
    np.testing.assert_array_equal(reconstructed, group)


def test_pack_group_round_trips_negative_zero_positive_and_mixed_columns():
    group = np.empty((128, 4), dtype=np.float32)
    group[:, 0] = -0.125
    group[:, 1] = 0.0
    group[:, 2] = 1.75
    group[:, 3] = _group(columns=1, seed=15)[:, 0]

    packed, scales = pack_group(group)
    reconstructed = unpack_group(packed, scales)

    assert packed.shape == (4, PACKED_BYTES_PER_GROUP)
    assert packed.dtype == np.uint8
    assert scales.shape == (4,)
    assert scales.dtype == np.float32
    assert np.all(packed <= MAX_PACKED_CODE)
    np.testing.assert_array_equal(reconstructed, group)
    nonzero = group != 0
    np.testing.assert_array_equal(reconstructed[nonzero].view(np.uint32), group[nonzero].view(np.uint32))


def test_padding_is_zero_trits_and_deterministic():
    group = np.empty((128, 3), dtype=np.float32)
    group[:, 0] = -1.0
    group[:, 1] = 0.0
    group[:, 2] = 1.0

    packed, scales = pack_group(group)
    packed_again, _ = pack_group(group)

    np.testing.assert_array_equal(packed, packed_again)
    np.testing.assert_array_equal(packed[:, -1], np.asarray([108, 121, 134], dtype=np.uint8))
    np.testing.assert_array_equal(unpack_group(packed, scales), group)


def test_validate_group_returns_finite_nonnegative_column_scales():
    group = _group(columns=5)

    scales = validate_ternary_group(group)

    assert scales.dtype == np.float32
    assert np.isfinite(scales).all()
    assert np.all(scales >= 0)
    np.testing.assert_array_equal(scales, np.abs(group).max(axis=0))


def test_validate_group_rejects_nonternary_values():
    group = np.zeros((128, 2), dtype=np.float32)
    group[0, 0] = 0.5
    group[1, 0] = 0.25

    with pytest.raises(ValueError, match="exactly \\{-scale, 0, \\+scale\\}"):
        validate_ternary_group(group)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_validate_group_rejects_nonfinite_values(bad_value: float):
    group = np.zeros((128, 1), dtype=np.float32)
    group[0, 0] = bad_value

    with pytest.raises(ValueError, match="values must be finite"):
        validate_ternary_group(group)


@pytest.mark.parametrize("bad_scale", [-1.0, np.inf, np.nan])
def test_unpack_group_rejects_invalid_scales(bad_scale: float):
    packed = np.zeros((1, PACKED_BYTES_PER_GROUP), dtype=np.uint8)
    scales = np.asarray([bad_scale], dtype=np.float32)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        unpack_group(packed, scales)


def test_validate_group_rejects_invalid_explicit_scales():
    group = _group(columns=2)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        validate_ternary_group(group, scales=np.asarray([-1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="maximum absolute value"):
        validate_ternary_group(group, scales=np.asarray([0.0, 0.0], dtype=np.float32))


def test_unpack_group_rejects_malformed_code_bytes():
    packed = np.zeros((1, PACKED_BYTES_PER_GROUP), dtype=np.uint8)
    packed[0, 3] = 243

    with pytest.raises(ValueError, match="range 0..242"):
        unpack_group(packed, np.asarray([1.0], dtype=np.float32))


def test_pack_and_unpack_reject_invalid_group_shapes_and_dtypes():
    with pytest.raises(ValueError, match="shape \\[128, N\\]"):
        pack_group(np.zeros((127, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="dtype float32"):
        pack_group(np.zeros((128, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="packed codes must have shape"):
        unpack_group(np.zeros((2, 25), dtype=np.uint8), np.ones(2, dtype=np.float32))
    with pytest.raises(TypeError, match="dtype uint8"):
        unpack_group(np.zeros((2, PACKED_BYTES_PER_GROUP), dtype=np.int8), np.ones(2, dtype=np.float32))


def test_unpack_group_rejects_scale_shape_and_dtype_mismatches():
    packed = np.zeros((2, PACKED_BYTES_PER_GROUP), dtype=np.uint8)

    with pytest.raises(ValueError, match="scales must have shape"):
        unpack_group(packed, np.ones(1, dtype=np.float32))
    with pytest.raises(TypeError, match="scales must have dtype float32"):
        unpack_group(packed, np.ones(2, dtype=np.float64))


def test_reference_weight_round_trip_restores_shape_and_multiple_groups():
    signs = np.random.default_rng(2026).choice(
        np.asarray([-1, 0, 1], dtype=np.float32), size=(4, 128, 19)
    )
    scales_per_group = np.random.default_rng(2027).choice(
        np.asarray([0.125, 0.5, 1.75], dtype=np.float32), size=(4, 19)
    )
    weight = (signs * scales_per_group[:, None, :]).reshape(512, 19)

    packed, scales = pack_weight_reference(weight)
    reconstructed = unpack_weight_reference(packed, scales)

    assert packed.shape == (4, 19, PACKED_BYTES_PER_GROUP)
    assert packed.dtype == np.uint8
    assert scales.shape == (4, 1, 19)
    assert scales.dtype == np.float32
    assert reconstructed.shape == weight.shape
    np.testing.assert_array_equal(reconstructed, weight)
    nonzero = weight != 0
    np.testing.assert_array_equal(reconstructed[nonzero].view(np.uint32), weight[nonzero].view(np.uint32))


def test_reference_weight_rejects_invalid_dtype_shape_and_group_dimension():
    with pytest.raises(TypeError, match="weight must have dtype float32"):
        pack_weight_reference(np.zeros((128, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="two positive dimensions"):
        pack_weight_reference(np.zeros((128,), dtype=np.float32))
    with pytest.raises(ValueError, match="divisible by 128"):
        pack_weight_reference(np.zeros((129, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="exactly \\{-scale, 0, \\+scale\\}"):
        invalid = np.zeros((128, 1), dtype=np.float32)
        invalid[0, 0] = 1.0
        invalid[1, 0] = 0.5
        pack_weight_reference(invalid)


def test_reference_unpack_rejects_bad_shapes_and_codes():
    packed = np.zeros((2, 3, PACKED_BYTES_PER_GROUP), dtype=np.uint8)
    scales = np.ones((2, 1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="scales must have shape"):
        unpack_weight_reference(packed, np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="range 0..242"):
        packed[1, 2, 25] = 255
        unpack_weight_reference(packed, scales)
    with pytest.raises(TypeError, match="dtype uint8"):
        unpack_weight_reference(packed.astype(np.int8), scales)



def _matmul_model(weight: np.ndarray, *, node_name: str = "tiny_matmul") -> onnx.ModelProto:
    k, n = weight.shape
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name=node_name)],
        "tiny-tritpack-matmul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, n])],
        [numpy_helper.from_array(weight, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model


def test_rewrite_matmul_to_tritpack_storage_matches_standard_matmul():
    weight = _group(rows=256, columns=13, seed=2026)
    activation = np.random.default_rng(2027).normal(size=(2, 256)).astype(np.float32)
    source = _matmul_model(weight)
    source_bytes = source.SerializeToString()

    rewritten = rewrite_matmul_to_tritpack_storage(source, node_name="tiny_matmul")
    reconstructed_name = next(
        node.input[1] for node in rewritten.graph.node if node.op_type == "MatMul"
    )
    rewritten.graph.output.append(
        helper.make_tensor_value_info(reconstructed_name, TensorProto.FLOAT, list(weight.shape))
    )

    onnx.checker.check_model(rewritten)
    assert source.SerializeToString() == source_bytes
    assert "W" not in {tensor.name for tensor in rewritten.graph.initializer}
    assert "MatMulNBits" not in {node.op_type for node in rewritten.graph.node}
    assert [node.op_type for node in rewritten.graph.node] == [
        "Cast",
        "Gather",
        "Reshape",
        "Slice",
        "Transpose",
        "Cast",
        "Mul",
        "Reshape",
        "MatMul",
    ]

    baseline = ort.InferenceSession(source.SerializeToString(), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    expected = baseline.run(None, {"A": activation})[0]
    actual, reconstructed = candidate.run(None, {"A": activation})

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)
    np.testing.assert_array_equal(reconstructed, weight)
    nonzero = weight != 0
    np.testing.assert_array_equal(reconstructed[nonzero].view(np.uint32), weight[nonzero].view(np.uint32))


def _load_smallest_real_packed_weight(source_path: Path) -> tuple[str, np.ndarray]:
    source = onnx.load(str(source_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in source.graph.initializer}
    candidates = [
        (node.name, initializers[node.input[1]])
        for node in source.graph.node
        if node.op_type == "MatMul"
        and len(node.input) == 2
        and node.name
        and node.input[1] in initializers
        and initializers[node.input[1]].data_type == TensorProto.FLOAT
        and len(initializers[node.input[1]].dims) == 2
        and initializers[node.input[1]].dims[0] % 128 == 0
    ]
    candidates.sort(key=lambda row: int(np.prod(row[1].dims)))

    for node_name, tensor in candidates:
        if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
            tensor_slice = resolve_external_data_range(
                source_path.parent,
                tensor,
                logical_bytes=logical_tensor_bytes(tensor),
                require_exact_length=True,
            )
            mapped = map_external_float32_tensor(
                tensor_slice, tuple(int(dimension) for dimension in tensor.dims)
            )
            try:
                weight = np.array(mapped, copy=True)
            finally:
                mapped._mmap.close()
        else:
            weight = np.asarray(numpy_helper.to_array(tensor)).copy()
        try:
            pack_weight_reference(weight)
        except (TypeError, ValueError):
            continue
        return node_name, weight
    raise AssertionError(f"no exact ternary packed MatMul found in {source_path}")


def test_real_packed_transformer_matmul_matches_standard_matmul():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "build/variant-bundles/packed-source/lm/lm.onnx"
    )
    if not source_path.is_file():
        pytest.skip("local packed source ONNX model is unavailable")

    source_node_name, weight = _load_smallest_real_packed_weight(source_path)
    node_name = f"real_{source_node_name}"
    source = _matmul_model(weight, node_name=node_name)
    rewritten = rewrite_matmul_to_tritpack_storage(source, node_name=node_name)
    onnx.checker.check_model(rewritten)

    activation = np.random.default_rng(2028).normal(size=(2, weight.shape[0])).astype(np.float32)
    baseline = ort.InferenceSession(source.SerializeToString(), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"])
    expected = baseline.run(None, {"A": activation})[0]
    actual = candidate.run(None, {"A": activation})[0]

    assert "MatMulNBits" not in {node.op_type for node in rewritten.graph.node}
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)



def _external_matmul_model(path: Path, weight: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_path = path.parent / "source.data"
    prefix = b"prefix"
    suffix = b"suffix"
    payload = weight.tobytes(order="C")
    data_path.write_bytes(prefix + payload + suffix)

    tensor = TensorProto()
    tensor.name = "W"
    tensor.data_type = TensorProto.FLOAT
    tensor.dims.extend(weight.shape)
    tensor.raw_data = payload
    tensor.data_location = TensorProto.EXTERNAL
    external_data_helper.set_external_data(
        tensor,
        location=data_path.name,
        offset=len(prefix),
        length=len(payload),
    )
    tensor.ClearField("raw_data")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="tiny_matmul")],
        "external-tritpack-matmul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, weight.shape[0]])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, weight.shape[1]])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, str(path))
    return data_path


def _one_weight_inventory(source_path: Path, weight: np.ndarray) -> dict:
    return {
        "source_graph_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "eligible_exact_ternary_matmul_count": 1,
        "inventory_sha256": hashlib.sha256(b"one-weight-inventory").hexdigest(),
        "nodes": [
            {
                "node": "tiny_matmul",
                "initializer": "W",
                "shape": list(weight.shape),
                "source_sha256": hashlib.sha256(weight.tobytes(order="C")).hexdigest(),
                "eligible_exact_ternary": True,
            }
        ],
    }


def test_streaming_tritpack_conversion_verifies_storage_and_standard_matmul(tmp_path):
    weight = _group(rows=256, columns=7, seed=2030)
    source_path = tmp_path / "source" / "lm.onnx"
    source_data = _external_matmul_model(source_path, weight)
    source_graph_before = source_path.read_bytes()
    source_data_before = source_data.read_bytes()
    output_path = tmp_path / "candidate" / "lm.onnx"
    output_data = tmp_path / "candidate" / "lm.tritpack.data"
    inventory = build_ternary_tritpack_inventory(
        source_path,
        expected_eligible_matmul_count=1,
        column_chunk=3,
    )

    report = compress_ternary_matmuls_to_tritpack_streaming(
        source_path,
        output_path,
        output_data,
        inventory=inventory,
        column_chunk=3,
    )

    assert report["transformed_matmul_count"] == 1
    assert report["transformed_initializer_count"] == 1
    assert report["storage_reconstruction_verified"] is True
    assert report["numeric_reconstruction_exact"] is True
    assert report["code_bytes"] == 2 * 7 * PACKED_BYTES_PER_GROUP
    assert report["packed_code_bytes"] == report["code_bytes"]
    assert report["scale_bytes"] == 2 * 7 * np.dtype(np.float32).itemsize
    assert report["logical_fp32_operand_bytes"] == weight.nbytes
    assert report["inventory_sha256"] == inventory["inventory_sha256"]
    assert report["graph_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert report["external_data_sha256"] == hashlib.sha256(output_data.read_bytes()).hexdigest()
    assert output_data.stat().st_size < weight.nbytes
    assert source_path.read_bytes() == source_graph_before
    assert source_data.read_bytes() == source_data_before

    candidate_model = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(str(output_path))
    initializers = {tensor.name: tensor for tensor in candidate_model.graph.initializer}
    assert "W" not in initializers
    assert initializers[report["initializer_reports"][0]["code_initializer"]].data_type == TensorProto.UINT8
    assert "MatMulNBits" not in {node.op_type for node in candidate_model.graph.node}

    activation = np.random.default_rng(2031).normal(size=(2, 256)).astype(np.float32)
    baseline = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    expected = baseline.run(None, {"A": activation})[0]
    actual = candidate.run(None, {"A": activation})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)


def test_streaming_tritpack_failure_removes_partial_candidate(tmp_path):
    weight = np.zeros((128, 2), dtype=np.float32)
    weight[0, 0] = 1.0
    weight[1, 0] = 0.5
    source_path = tmp_path / "source" / "lm.onnx"
    source_data = _external_matmul_model(source_path, weight)
    source_graph_before = source_path.read_bytes()
    source_data_before = source_data.read_bytes()
    output_path = tmp_path / "candidate" / "lm.onnx"
    output_data = tmp_path / "candidate" / "lm.tritpack.data"

    with pytest.raises(ValueError, match="exactly \\{-scale, 0, \\+scale\\}"):
        compress_ternary_matmuls_to_tritpack_streaming(
            source_path,
            output_path,
            output_data,
            inventory=_one_weight_inventory(source_path, weight),
            column_chunk=1,
        )

    assert not output_path.exists()
    assert not output_data.exists()
    assert source_path.read_bytes() == source_graph_before
    assert source_data.read_bytes() == source_data_before



def _materialization_fixture(directory: Path, weight: np.ndarray):
    source_path = directory / "source" / "lm.onnx"
    _external_matmul_model(source_path, weight)
    packed_path = directory / "packed" / "lm.onnx"
    packed_data = directory / "packed" / "lm.onnx.data"
    inventory = build_ternary_tritpack_inventory(
        source_path,
        expected_eligible_matmul_count=1,
        column_chunk=3,
    )
    report = compress_ternary_matmuls_to_tritpack_streaming(
        source_path,
        packed_path,
        packed_data,
        inventory=inventory,
        column_chunk=3,
    )
    entry = report["initializer_reports"][0]
    storage_metadata = {
        "method": report["method"],
        "format_version": report["format_version"],
        "source_group_size": report["group_size"],
        "trits_per_byte": report["trits_per_byte"],
        "packed_bytes_per_group": report["packed_bytes_per_group"],
        "graph_sha256": report["graph_sha256"],
        "payload_file": packed_data.name,
        "payload_sha256": report["external_data_sha256"],
        "runtime_materialization_required": True,
        "transformed_matmul_count": report["transformed_matmul_count"],
        "transformed_initializer_count": report["transformed_initializer_count"],
        "logical_fp32_operand_bytes": report["logical_fp32_operand_bytes"],
        "materialized_external_bytes": report["logical_fp32_operand_bytes"],
        "weights": [
            {
                key: entry[key]
                for key in (
                    "source_initializer",
                    "weight_value",
                    "source_shape",
                    "source_sha256",
                    "code_initializer",
                    "scale_initializer",
                    "transformed_nodes",
                )
            }
        ],
    }
    return source_path, packed_path, packed_data, storage_metadata


def _materialization_worker(model_path, cache_root, storage_metadata, barrier, results):
    barrier.wait()
    result = materialize_tritpack_model(
        model_path,
        cache_root,
        storage_metadata=storage_metadata,
    )
    results.put((str(result.cache_dir), result.cache_hit))


def test_materialization_reconstructs_standard_matmul_and_reuses_cache(tmp_path):
    weight = _group(rows=128, columns=3, seed=2040)
    source_path, packed_path, packed_data, storage = _materialization_fixture(tmp_path, weight)
    cache_root = tmp_path / "local-cache"
    source_before = source_path.read_bytes()
    source_data_before = (source_path.parent / "source.data").read_bytes()
    packed_graph_before = packed_path.read_bytes()
    packed_data_before = packed_data.read_bytes()

    materialized = materialize_tritpack_model(
        packed_path,
        cache_root,
        storage_metadata=storage,
    )

    assert materialized.cache_hit is False
    assert materialized.cache_dir.is_relative_to(cache_root)
    assert not materialized.cache_dir.is_relative_to(packed_path.parent)
    assert source_path.read_bytes() == source_before
    assert (source_path.parent / "source.data").read_bytes() == source_data_before
    assert packed_path.read_bytes() == packed_graph_before
    assert packed_data.read_bytes() == packed_data_before

    model = onnx.load(str(materialized.model_path), load_external_data=False)
    onnx.checker.check_model(str(materialized.model_path))
    nodes = {node.name: node for node in model.graph.node}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    assert nodes["tiny_matmul"].input[1] == "W"
    assert initializers["W"].data_location == TensorProto.EXTERNAL
    assert all("tritpack" not in node.name for node in model.graph.node)
    assert "MatMulNBits" not in {node.op_type for node in model.graph.node}

    activation = np.random.default_rng(2041).normal(size=(2, 128)).astype(np.float32)
    baseline = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(materialized.model_path), providers=["CPUExecutionProvider"])
    expected = baseline.run(None, {"A": activation})[0]
    actual = candidate.run(None, {"A": activation})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)

    reused = materialize_tritpack_model(
        packed_path,
        cache_root,
        storage_metadata=storage,
    )
    assert reused.cache_hit is True
    assert reused.cache_dir == materialized.cache_dir


def test_materialization_rejects_wrong_payload_hash_and_corrupt_cache(tmp_path):
    weight = _group(rows=128, columns=2, seed=2042)
    _, packed_path, _, storage = _materialization_fixture(tmp_path, weight)
    cache_root = tmp_path / "local-cache"
    wrong_hash_storage = dict(storage, payload_sha256="0" * 64)
    with pytest.raises(ValueError, match="payload hash differs"):
        materialize_tritpack_model(
            packed_path,
            cache_root,
            storage_metadata=wrong_hash_storage,
        )

    materialized = materialize_tritpack_model(
        packed_path,
        cache_root,
        storage_metadata=storage,
    )
    materialized.data_path.write_bytes(materialized.data_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="cache files are corrupt"):
        materialize_tritpack_model(
            packed_path,
            cache_root,
            storage_metadata=storage,
        )


def test_materialization_does_not_reuse_interrupted_temporary_directory(tmp_path):
    weight = _group(rows=128, columns=2, seed=2043)
    _, packed_path, _, storage = _materialization_fixture(tmp_path, weight)
    cache_root = tmp_path / "local-cache"
    cache_key, _, _, _ = _cache_identity(packed_path, storage)
    interrupted = cache_root / f".{cache_key}.tmp-interrupted"
    interrupted.mkdir(parents=True)
    (interrupted / "partial").write_text("incomplete", encoding="utf-8")

    materialized = materialize_tritpack_model(
        packed_path,
        cache_root,
        storage_metadata=storage,
    )

    assert materialized.cache_dir != interrupted
    assert (interrupted / "partial").read_text(encoding="utf-8") == "incomplete"
    assert materialized.model_path.is_file()


def test_materialization_rejects_cache_inside_model_bundle(tmp_path):
    weight = _group(rows=128, columns=2, seed=2045)
    _, packed_path, _, storage = _materialization_fixture(tmp_path, weight)

    with pytest.raises(ValueError, match="cache must remain outside"):
        materialize_tritpack_model(
            packed_path,
            packed_path.parent / "cache",
            storage_metadata=storage,
        )


def test_materialization_is_safe_across_processes(tmp_path):
    weight = _group(rows=128, columns=2, seed=2044)
    _, packed_path, _, storage = _materialization_fixture(tmp_path, weight)
    cache_root = tmp_path / "local-cache"
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_materialization_worker,
            args=(packed_path, cache_root, storage, barrier, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    assert all(process.exitcode == 0 for process in processes)
    results_list = [results.get(timeout=10) for _ in processes]
    assert len({result[0] for result in results_list}) == 1
    assert sorted(result[1] for result in results_list) == [False, True]
