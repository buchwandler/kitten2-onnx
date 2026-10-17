from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.external_data import make_external_tensor
from scripts.optimize_bundle import main, optimize_bundle


def _write_tiny_bundle(root: Path) -> None:
    lm_dir = root / "lm"
    lm_dir.mkdir(parents=True)
    graph_path = lm_dir / "lm.onnx"
    inputs = [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, 2])]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])]
    weight = numpy_helper.from_array(
        np.asarray([[0.125, -0.25], [0.5, 0.75]], dtype=np.float32),
        name="weight",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["inputs_embeds", "weight"], ["logits"])],
        "tiny-lm",
        inputs,
        outputs,
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(
        model,
        str(graph_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="lm.onnx.data",
        size_threshold=0,
    )
    metadata = {
        "graph": "lm.onnx",
        "input_mode": "inputs_embeds",
        "input_names": ["inputs_embeds"],
        "output_names": ["logits"],
        "runtime_profile": "dev-fp32",
        "runtime_provider": "CPUExecutionProvider",
        "io_dtype": "float32",
    }
    metadata_path = lm_dir / "lm.meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    bundle = {
        "format": "kitten2-onnx-bundle-v1",
        "lm": {"model": "lm/lm.onnx", "metadata": "lm/lm.meta.json"},
        "export": {"runtime_profile": "dev-fp32"},
        "files": files,
    }
    (root / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")




def _write_ternary_tiny_bundle(root: Path, matmul_count: int = 196) -> np.ndarray:
    _write_tiny_bundle(root)
    lm_dir = root / "lm"
    weight = np.zeros((128, 2), dtype=np.float32)
    weight[0] = [0.5, -0.25]
    weight[1] = [-0.5, 0.25]
    output_names = [f"logits_{index}" for index in range(matmul_count)]
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul",
                ["inputs_embeds", "weight"],
                [output_name],
                name=f"ternary_matmul_{index}",
            )
            for index, output_name in enumerate(output_names)
        ],
        "tiny-ternary-lm",
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, 128])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2]) for name in output_names],
        [numpy_helper.from_array(weight, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(
        model,
        str(lm_dir / "lm.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="lm.onnx.data",
        size_threshold=0,
    )
    metadata_path = lm_dir / "lm.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_names"] = output_names
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    bundle_path = root / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["files"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != bundle_path
    }
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    return weight

def test_optimize_bundle_relocates_quantized_external_data(tmp_path, monkeypatch):
    from onnxruntime import quantization

    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_bundle(source)

    def fake_quantize_dynamic(input_path, output_path, **kwargs):
        model = onnx.load(str(input_path))
        onnx.save_model(
            model,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(output_path).name}.data",
            size_threshold=0,
        )

    monkeypatch.setattr(quantization, "quantize_dynamic", fake_quantize_dynamic)
    output = tmp_path / "optimized"

    optimize_bundle(source, output, lm_method="int8-dynamic")

    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "lm/lm.meta.json").read_text(encoding="utf-8"))
    locations = metadata["external_data_files"]

    assert locations == ["lm.quantized.onnx.data"]
    assert (output / "lm" / locations[0]).is_file()
    assert not (output / "lm/lm.onnx.data").exists()
    assert "lm/lm.quantized.onnx.data" in bundle["files"]
    assert all(
        hashlib.sha256((output / relative_path).read_bytes()).hexdigest() == expected_hash
        for relative_path, expected_hash in bundle["files"].items()
    )


def test_optimize_bundle_quantizes_lm_and_rewrites_manifest_and_external_paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_bundle(source)
    original_graph_hash = hashlib.sha256((source / "lm/lm.onnx").read_bytes()).hexdigest()
    output = tmp_path / "optimized"
    reports = tmp_path / "reports"

    report = optimize_bundle(
        source,
        output,
        lm_method="int8-dynamic",
        report_dir=reports,
    )

    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "lm/lm.meta.json").read_text(encoding="utf-8"))
    graph = onnx.load(str(output / "lm/lm.onnx"), load_external_data=False)
    manifest = bundle["files"]
    locations = {
        entry.value
        for tensor in graph.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }

    assert bundle["export"]["lm_optimization"] == "int8-dynamic"
    assert metadata["quantization"]["method"] == "int8-dynamic"
    assert metadata["quantization"]["source_graph_sha256"] == original_graph_hash
    assert metadata["external_data_files"] == sorted(locations)
    assert all(not Path(location).is_absolute() for location in locations)
    assert all((output / "lm" / location).is_file() for location in locations)
    assert all(
        hashlib.sha256((output / relative_path).read_bytes()).hexdigest() == expected_hash
        for relative_path, expected_hash in manifest.items()
    )
    assert hashlib.sha256((source / "lm/lm.onnx").read_bytes()).hexdigest() == original_graph_hash
    assert report["parity"]["status"] == "not_run"
    assert report["validation"]["status"] == "not_run"
    assert (reports / "optimization.json").is_file()
    assert (reports / "size-report.json").is_file()



def test_optimize_bundle_chunked_updates_metadata_hashes_and_size_reports(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_bundle(source)
    source_files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "optimized"
    reports = tmp_path / "reports"

    report = optimize_bundle(
        source,
        output,
        lm_method="int8-dynamic-chunked",
        chunk_rows=1,
        report_dir=reports,
    )

    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "lm/lm.meta.json").read_text(encoding="utf-8"))
    graph_path = output / "lm/lm.onnx"
    graph = onnx.load(str(graph_path), load_external_data=False)
    quantization = metadata["quantization"]
    manifest = bundle["files"]
    assert bundle["export"]["lm_optimization"] == "int8-dynamic-chunked"
    assert quantization["method"] == "int8-dynamic-chunked"
    assert quantization["backend"] == "streaming"
    assert quantization["chunk_rows"] == 1
    assert quantization["transformed_matmul_count"] == 1
    assert report["quantization"] == quantization
    assert quantization["source_graph_sha256"] == hashlib.sha256(
        source_files["lm/lm.onnx"]
    ).hexdigest()
    assert quantization["graph_sha256"] == hashlib.sha256(graph_path.read_bytes()).hexdigest()
    assert quantization["external_data_files"] == ["lm.int8.onnx.data"]
    assert quantization["external_data_sha256"]["lm.int8.onnx.data"] == hashlib.sha256(
        (output / "lm/lm.int8.onnx.data").read_bytes()
    ).hexdigest()
    assert quantization["source_external_data_sha256"]["lm.onnx.data"] == hashlib.sha256(
        source_files["lm/lm.onnx.data"]
    ).hexdigest()
    assert not (output / "lm/lm.onnx.data").exists()
    assert (output / "lm/lm.int8.onnx.data").is_file()
    assert "lm/lm.int8.onnx.data" in manifest
    assert "lm/lm.onnx.data" not in manifest
    assert all(
        hashlib.sha256((output / relative_path).read_bytes()).hexdigest() == expected_hash
        for relative_path, expected_hash in manifest.items()
    )
    assert report["output_bundle_bytes"] == json.loads(
        (reports / "size-report.json").read_text(encoding="utf-8")
    )["bundle_bytes"]
    assert (reports / "optimization.json").is_file()
    assert [node.op_type for node in graph.graph.node] == [
        "DynamicQuantizeLinear",
        "Mul",
        "MatMulInteger",
        "Cast",
        "Mul",
    ]
    assert all(
        path.read_bytes() == source_files[path.relative_to(source).as_posix()]
        for path in source.rglob("*")
        if path.is_file()
    )


def test_optimize_bundle_chunked_removes_only_proven_stale_external_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_bundle(source)
    lm_path = source / "lm/lm.onnx"
    source_graph = onnx.load(str(lm_path), load_external_data=False)
    source_weight = source_graph.graph.initializer[0]
    external = {entry.key: entry.value for entry in source_weight.external_data}
    aux_weight = make_external_tensor(
        "aux_weight",
        source_weight.data_type,
        source_weight.dims,
        external["location"],
        int(external["offset"]),
        int(external["length"]),
    )
    aux_graph = helper.make_graph(
        [helper.make_node("Identity", ["aux_weight"], ["aux_output"])],
        "auxiliary-model",
        [],
        [helper.make_tensor_value_info("aux_output", TensorProto.FLOAT, [2, 2])],
        [aux_weight],
    )
    aux_model = helper.make_model(aux_graph, opset_imports=[helper.make_opsetid("", 18)])
    aux_model.ir_version = min(aux_model.ir_version, 10)
    onnx.save_model(aux_model, str(source / "lm/aux.onnx"))
    stale_path = source / "lm/lm.int8.onnx.data"
    stale_path.write_bytes(b"stale unreferenced optimizer output")
    source_files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    output = tmp_path / "optimized"
    optimize_bundle(source, output, lm_method="int8-dynamic-chunked")

    metadata = json.loads((output / "lm/lm.meta.json").read_text(encoding="utf-8"))
    assert metadata["external_data_files"] == ["lm.int8-1.onnx.data"]
    assert (output / "lm/lm.onnx.data").is_file()
    assert not (output / "lm/lm.int8.onnx.data").exists()
    assert (output / "lm/lm.int8-1.onnx.data").is_file()
    manifest = json.loads((output / "bundle.json").read_text(encoding="utf-8"))["files"]
    assert "lm/lm.onnx.data" in manifest
    assert "lm/lm.int8-1.onnx.data" in manifest
    assert "lm/lm.int8.onnx.data" not in manifest
    assert all(
        path.read_bytes() == source_files[path.relative_to(source).as_posix()]
        for path in source.rglob("*")
        if path.is_file()
    )


def test_optimize_bundle_chunked_keeps_final_graph_on_validation_failure(tmp_path, monkeypatch):
    import onnxruntime as ort

    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_bundle(source)
    original_graph = (source / "lm/lm.onnx").read_bytes()
    output = tmp_path / "optimized"

    def fail_session(*args, **kwargs):
        raise RuntimeError("injected ORT validation failure")

    monkeypatch.setattr(ort, "InferenceSession", fail_session)
    with pytest.raises(RuntimeError, match="injected ORT validation failure"):
        optimize_bundle(source, output, lm_method="int8-dynamic-chunked")

    assert (output / "lm/lm.onnx").read_bytes() == original_graph
    assert (output / "lm/lm.onnx.data").is_file()
    assert not (output / "lm/lm.int8.onnx.data").exists()
    assert not list((output / "lm").glob(".lm.chunked-*.tmp.onnx"))
    assert (source / "lm/lm.onnx").read_bytes() == original_graph


def test_optimize_cli_exposes_chunked_mode_without_removing_legacy_mode(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["optimize_bundle", "--help"])
    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 0
    help_output = capsys.readouterr().out
    assert "int8-dynamic" in help_output
    assert "int8-dynamic-chunked" in help_output

    assert "ternary-nbits4" in help_output
    assert "--chunk-rows" in help_output
    assert "--exclude-matmul-initializer" in help_output



def test_optimize_bundle_chunked_accepts_initializer_exclusions(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_bundle(source)
    source_graph = (source / "lm/lm.onnx").read_bytes()
    output = tmp_path / "optimized"

    with pytest.raises(ValueError, match="no supported constant-B MatMul nodes"):
        optimize_bundle(
            source,
            output,
            lm_method="int8-dynamic-chunked",
            exclude_matmul_initializers=("weight",),
        )

    assert (output / "lm/lm.onnx").read_bytes() == source_graph
    assert (output / "lm/lm.onnx.data").is_file()
    assert not (output / "lm/lm.int8.onnx.data").exists()



def test_optimize_bundle_ternary_nbits4_reports_size_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    weight = _write_ternary_tiny_bundle(source)
    source_files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "optimized"
    reports = tmp_path / "reports"

    report = optimize_bundle(
        source,
        output,
        lm_method="ternary-nbits4",
        report_dir=reports,
    )

    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "lm/lm.meta.json").read_text(encoding="utf-8"))
    graph_path = output / "lm/lm.onnx"
    graph = onnx.load(str(graph_path), load_external_data=False)
    quantization = metadata["quantization"]
    assert bundle["export"]["lm_optimization"] == "ternary-nbits4"
    assert quantization["method"] == "ternary-nbits4"
    assert quantization["bits"] == 4
    assert quantization["block_size"] == 128
    assert quantization["transformed_matmul_count"] == 196
    assert quantization["transformed_initializer_count"] == 1
    assert quantization["accidentally_transformed_non_ternary_count"] == 0
    assert len(quantization["transformed_nodes"]) == 196
    assert quantization["external_data_files"] == ["lm.nbits4.onnx.data"]
    assert quantization["external_data_sha256"]["lm.nbits4.onnx.data"] == hashlib.sha256(
        (output / "lm/lm.nbits4.onnx.data").read_bytes()
    ).hexdigest()
    assert [node.op_type for node in graph.graph.node] == ["MatMulNBits"] * 196
    assert not (output / "lm/lm.onnx.data").exists()
    assert "lm/lm.nbits4.onnx.data" in bundle["files"]
    assert "lm/lm.onnx.data" not in bundle["files"]
    assert all(
        hashlib.sha256((output / relative_path).read_bytes()).hexdigest() == expected_hash
        for relative_path, expected_hash in bundle["files"].items()
    )

    size_report = json.loads((reports / "size-report.json").read_text(encoding="utf-8"))
    nbits4 = size_report["onnx"]["lm"]["matmul_nbits4"]
    assert nbits4["node_count"] == 196
    assert nbits4["storage_bytes_by_role"] == {
        "packed_codes": 128,
        "scales": 8,
        "zero_points": 2,
    }
    assert nbits4["total_storage_bytes"] == 138
    assert report["output_bundle_bytes"] == size_report["bundle_bytes"]

    import onnxruntime as ort

    onnx.checker.check_model(str(graph_path))
    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    activation = np.arange(128, dtype=np.float32).reshape(1, 128) / 128
    outputs = session.run(None, {"inputs_embeds": activation})
    assert len(outputs) == 196
    for actual in outputs:
        np.testing.assert_allclose(actual, activation @ weight, rtol=1e-5, atol=1e-5)
    assert {
        relative_path: path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
        for relative_path in [path.relative_to(source).as_posix()]
    } == source_files


def test_optimize_bundle_ternary_nbits4_cleans_failed_external_output(tmp_path, monkeypatch):
    import onnxruntime as ort

    source = tmp_path / "source"
    source.mkdir()
    _write_ternary_tiny_bundle(source)
    source_graph = (source / "lm/lm.onnx").read_bytes()
    source_data = (source / "lm/lm.onnx.data").read_bytes()
    output = tmp_path / "optimized"

    def fail_session(*args, **kwargs):
        raise RuntimeError("injected NBits4 CPU session failure")

    monkeypatch.setattr(ort, "InferenceSession", fail_session)
    with pytest.raises(RuntimeError, match="injected NBits4 CPU session failure"):
        optimize_bundle(source, output, lm_method="ternary-nbits4")

    assert (source / "lm/lm.onnx").read_bytes() == source_graph
    assert (source / "lm/lm.onnx.data").read_bytes() == source_data
    assert (output / "lm/lm.onnx").read_bytes() == source_graph
    assert (output / "lm/lm.onnx.data").is_file()
    assert not (output / "lm/lm.nbits4.onnx.data").exists()
    assert not list((output / "lm").glob(".lm.nbits4-*.tmp.onnx"))
