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




def _write_ternary_tiny_bundle(
    root: Path, matmul_count: int = 196, *, signed_zero: bool = False
) -> np.ndarray:
    _write_tiny_bundle(root)
    lm_dir = root / "lm"
    weight = np.zeros((128, 2), dtype=np.float32)
    weight[0] = [0.5, -0.25]
    weight[1] = [-0.5, 0.25]
    if signed_zero:
        weight[2, 0] = np.float32(-0.0)
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



def _write_exact_ternary_inventory(
    path: Path,
    bundle: Path,
    weight: np.ndarray,
    matmul_count: int,
) -> Path:
    from kitten2_onnx.quantize_ternary_nbits import _is_exact_ternary_weight

    _, source_hash = _is_exact_ternary_weight(weight)
    rows = [
        {
            "node": f"ternary_matmul_{index}",
            "initializer": "weight",
            "eligible_exact_ternary": True,
            "shape": list(weight.shape),
            "source_sha256": source_hash,
        }
        for index in range(matmul_count)
    ]
    inventory = {
        "format": "kitten2-onnx-exact-ternary-inventory-v1",
        "inventory_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "source_graph_sha256": hashlib.sha256((bundle / "lm/lm.onnx").read_bytes()).hexdigest(),
        "eligible_exact_ternary_matmul_count": matmul_count,
        "nodes": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory), encoding="utf-8")
    return path


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


def test_optimize_bundle_chunked_does_not_publish_on_validation_failure(tmp_path, monkeypatch):
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

    assert not output.exists()
    assert not list(tmp_path.glob(".optimized.*.partial"))
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
    assert "ternary-tritpack-exact" in help_output
    assert "ternary-exact-int8-to-nbits4" in help_output
    assert "--chunk-rows" in help_output
    assert "--exclude-matmul-initializer" in help_output
    assert "--lm-method" in help_output
    assert "--nbits-block-size" in help_output
    assert "--nbits-accuracy-level" in help_output



def test_optimize_cli_passes_nbits_configuration_and_exclusions(monkeypatch, capsys):
    import scripts.optimize_bundle as optimize_module

    calls = {}

    def fake_optimize_bundle(input_bundle, output_bundle, **kwargs):
        calls.update(input_bundle=input_bundle, output_bundle=output_bundle, **kwargs)
        return {"ok": True}

    monkeypatch.setattr(optimize_module, "optimize_bundle", fake_optimize_bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_bundle",
            "source",
            "output",
            "--lm-method",
            "ternary-nbits",
            "--nbits",
            "4",
            "--nbits-block-size",
            "32",
            "--nbits-accuracy-level",
            "0",
            "--exclude-matmul-node",
            "node-1",
            "--exclude-matmul-initializer",
            "weight-1",
        ],
    )

    main()
    capsys.readouterr()

    assert calls["lm_method"] == "ternary-nbits"
    assert calls["nbits"] == 4
    assert calls["nbits_block_size"] == 32
    assert calls["nbits_accuracy_level"] == 0
    assert calls["exclude_matmul_nodes"] == ["node-1"]
    assert calls["exclude_matmul_initializers"] == ["weight-1"]



def test_optimize_bundle_chunked_accepts_initializer_exclusions(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_bundle(source)
    output = tmp_path / "optimized"

    with pytest.raises(ValueError, match="no supported constant-B MatMul nodes"):
        optimize_bundle(
            source,
            output,
            lm_method="int8-dynamic-chunked",
            exclude_matmul_initializers=("weight",),
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".optimized.*.partial"))


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
    assert quantization["method"] == "ternary-nbits"
    assert quantization["bits"] == 4
    assert quantization["block_size"] == 128
    assert quantization["source_group_size"] == 128
    assert quantization["accuracy_level"] == 1
    assert quantization["fallback_storage"]["fallback_initializer_count"] == 0
    assert quantization["fallback_storage"]["net_bundle_byte_cost"] == 0
    assert quantization["provenance"]["onnx_version"]
    assert quantization["provenance"]["onnxruntime_version"]
    assert quantization["provenance"]["repository_commit"]
    assert quantization["transformed_matmul_count"] == 196
    assert quantization["source_eligible_matmul_count"] == 196
    assert quantization["excluded_matmul_count"] == 0
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
    nbits4 = size_report["onnx"]["lm"]["matmul_nbits"]
    assert nbits4["node_count"] == 196
    assert size_report["storage_breakdown"]["fallback_storage"] == quantization[
        "fallback_storage"
    ]
    assert nbits4["configurations"][0]["accuracy_level"] == 1
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


def test_optimize_ternary_nbits_excluded_node_preserves_shared_initializer(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_ternary_tiny_bundle(source)
    source_graph = (source / "lm/lm.onnx").read_bytes()
    source_data = (source / "lm/lm.onnx.data").read_bytes()
    output = tmp_path / "optimized"

    report = optimize_bundle(
        source,
        output,
        lm_method="ternary-nbits",
        nbits_block_size=32,
        exclude_matmul_nodes=("ternary_matmul_0",),
    )

    quantization = report["quantization"]
    graph = onnx.load(str(output / "lm/lm.onnx"), load_external_data=False)
    assert quantization["source_eligible_matmul_count"] == 196
    assert quantization["transformed_matmul_count"] == 195
    assert quantization["excluded_matmul_count"] == 1
    assert quantization["block_size"] == 32
    excluded = quantization["excluded_matmuls"]
    assert len(excluded) == 1
    assert {
        key: excluded[0][key] for key in ("node", "initializer", "reason")
    } == {
        "node": "ternary_matmul_0",
        "initializer": "weight",
        "reason": "explicitly excluded by node exclusion",
    }
    assert excluded[0]["source_shape"] == [128, 2]
    assert len(excluded[0]["source_sha256"]) == 64
    assert quantization["fallback_storage"]["fallback_fp32_bytes"] == 1024
    assert quantization["fallback_storage"]["nbits_bytes_removed"] == 0
    assert quantization["fallback_storage"]["net_bundle_byte_cost"] == 1024
    provenance = quantization["provenance"]
    assert provenance["onnx_version"]
    assert provenance["onnxruntime_version"]
    assert provenance["python_version"]
    assert provenance["cpu_architecture"]
    assert len(provenance["repository_commit"]) == 40
    assert graph.graph.node[0].op_type == "MatMul"
    assert all(node.op_type == "MatMulNBits" for node in graph.graph.node[1:])
    assert any(tensor.name == "weight" for tensor in graph.graph.initializer)
    excluded_weight = next(tensor for tensor in graph.graph.initializer if tensor.name == "weight")
    assert excluded_weight.data_location == TensorProto.EXTERNAL
    external_location = next(
        entry.value for entry in excluded_weight.external_data if entry.key == "location"
    )
    assert (output / "lm" / external_location).is_file()
    assert (source / "lm/lm.onnx").read_bytes() == source_graph
    assert (source / "lm/lm.onnx.data").read_bytes() == source_data


def test_optimize_ternary_nbits_excluded_initializer_keeps_all_matmuls(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_ternary_tiny_bundle(source)
    output = tmp_path / "optimized"

    report = optimize_bundle(
        source,
        output,
        lm_method="ternary-nbits",
        exclude_matmul_initializers=("weight",),
    )

    quantization = report["quantization"]
    graph = onnx.load(str(output / "lm/lm.onnx"), load_external_data=False)
    assert quantization["source_eligible_matmul_count"] == 196
    assert quantization["transformed_matmul_count"] == 0
    assert quantization["excluded_matmul_count"] == 196
    assert quantization["fallback_storage"]["fallback_fp32_bytes"] == 1024
    assert quantization["fallback_storage"]["nbits_bytes_removed"] == 138
    assert quantization["fallback_storage"]["net_bundle_byte_cost"] == 886
    assert all(node.op_type == "MatMul" for node in graph.graph.node)

    assert any(tensor.name == "weight" for tensor in graph.graph.initializer)

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
    assert not output.exists()
    assert not list(tmp_path.glob(".optimized.*.partial"))


def test_optimize_bundle_ternary_exact_int8_routes_accounts_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    weight = _write_ternary_tiny_bundle(source, signed_zero=True)
    source_files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    inventory = _write_exact_ternary_inventory(
        tmp_path / "inventory.json", source, weight, matmul_count=196
    )
    output = tmp_path / "optimized"
    reports = tmp_path / "reports"

    report = optimize_bundle(
        source,
        output,
        lm_method="ternary-exact-int8",
        ternary_inventory_path=inventory,
        report_dir=reports,
    )

    quantization = report["quantization"]
    metadata = json.loads((output / "lm/lm.meta.json").read_text(encoding="utf-8"))
    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    graph = onnx.load(str(output / "lm/lm.onnx"), load_external_data=False)
    assert bundle["export"]["lm_optimization"] == "ternary-exact-int8"
    assert quantization["method"] == "ternary-exact-int8"
    assert quantization["group_size"] == 128
    assert quantization["code_dtype"] == "int8"
    assert quantization["scale_dtype"] == "float32"
    assert metadata["runtime_memory_profile"] == "low-memory-exact-ternary"
    assert quantization["runtime_memory_profile"] == metadata["runtime_memory_profile"]
    assert metadata["runtime_options"] == {
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "execution_mode": "sequential",
        "enable_mem_pattern": False,
        "enable_cpu_mem_arena": False,
        "graph_optimization_level": "ORT_DISABLE_ALL",
    }
    assert quantization["encoding_version"] == 2
    assert quantization["code_values"] == [-2, -1, 0, 1]
    assert quantization["negative_zero_code"] == -2
    assert quantization["negative_zero_count"] == 1
    assert quantization["negative_zero_preserved"] is True
    assert quantization["exact_reconstruction"] is True
    assert quantization["transformed_matmul_count"] == 196
    assert quantization["transformed_initializer_count"] == 1
    assert quantization["code_bytes"] == 256
    assert quantization["scale_bytes"] == 8
    assert quantization["source_fp32_bytes_replaced"] == 1024
    assert quantization["operand_storage_reduction_bytes"] == 760
    assert quantization["exact_storage_ratio"] == pytest.approx(264 / 1024)
    assert metadata["quantization"] == quantization
    assert quantization["external_data_files"] == ["lm.ternary-exact-int8.onnx.data"]
    assert not (output / "lm/lm.onnx.data").exists()
    assert all(
        node.op_type in {"Equal", "Cast", "Mul", "Add", "Sub", "Reshape", "MatMul"}
        for node in graph.graph.node
    )
    assert sum(node.op_type == "MatMul" for node in graph.graph.node) == 196

    import onnxruntime as ort

    session = ort.InferenceSession(str(output / "lm/lm.onnx"), providers=["CPUExecutionProvider"])
    activation = np.arange(128, dtype=np.float32).reshape(1, 128) / 128
    actual = session.run(None, {"inputs_embeds": activation})
    assert len(actual) == 196
    for result in actual:
        np.testing.assert_array_equal(result, activation @ weight)

    size_report = json.loads((reports / "size-report.json").read_text(encoding="utf-8"))
    exact_storage = size_report["onnx"]["lm"]["ternary_exact_int8"]
    assert exact_storage["node_count"] == 196
    assert exact_storage["encoding_versions"] == [2]
    assert exact_storage["negative_zero_code"] == -2
    assert exact_storage["negative_zero_preserved"] is True
    assert exact_storage["negative_zero_reconstruction"] == "arithmetic"
    assert exact_storage["storage_bytes_by_role"] == {
        "int8_codes": 256,
        "float32_scales": 8,
    }
    assert exact_storage["source_fp32_bytes_replaced"] == 1024
    assert exact_storage["storage_reduction_bytes"] == 760
    assert size_report["storage_breakdown"]["ternary_exact_int8"] == exact_storage
    assert report["output_bundle_bytes"] == size_report["bundle_bytes"]
    assert {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == source_files

def test_optimize_bundle_exact_int8_to_nbits4_streams_and_preserves_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    weight = _write_ternary_tiny_bundle(source, signed_zero=True)
    inventory = _write_exact_ternary_inventory(
        tmp_path / "inventory.json", source, weight, matmul_count=196
    )
    source_files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    exact = tmp_path / "exact-int8"
    optimize_bundle(
        source,
        exact,
        lm_method="ternary-exact-int8",
        ternary_inventory_path=inventory,
    )
    exact_files = {
        path.relative_to(exact).as_posix(): path.read_bytes()
        for path in exact.rglob("*")
        if path.is_file()
    }

    output = tmp_path / "q4"
    report = optimize_bundle(
        exact,
        output,
        lm_method="ternary-exact-int8-to-nbits4",
        nbits_block_size=32,
    )

    quantization = report["quantization"]
    metadata = json.loads((output / "lm/lm.meta.json").read_text(encoding="utf-8"))
    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    graph = onnx.load(str(output / "lm/lm.onnx"), load_external_data=False)
    assert bundle["export"]["lm_optimization"] == "ternary-exact-int8-to-nbits4"
    assert quantization["method"] == "ternary-exact-int8-to-nbits4"
    assert quantization["bits"] == 4
    assert quantization["block_size"] == 32
    assert quantization["transformed_matmul_count"] == 196
    assert quantization["signed_zero_bit_differences"] == 1
    provenance = quantization["provenance"]
    assert provenance["onnx_version"]
    assert provenance["onnxruntime_version"]
    assert provenance["python_version"]
    assert provenance["repository_commit"]
    assert quantization["source_external_data_sha256"] == {
        "lm.ternary-exact-int8.onnx.data": hashlib.sha256(
            (exact / "lm/lm.ternary-exact-int8.onnx.data").read_bytes()
        ).hexdigest()
    }
    assert metadata["quantization"] == quantization
    assert metadata["external_data_files"] == ["lm.nbits4.onnx.data"]
    assert not (output / "lm/lm.ternary-exact-int8.onnx.data").exists()
    assert [node.op_type for node in graph.graph.node] == ["MatMulNBits"] * 196
    assert "lm/lm.nbits4.onnx.data" in bundle["files"]
    assert "lm/lm.ternary-exact-int8.onnx.data" not in bundle["files"]

    import onnxruntime as ort

    session = ort.InferenceSession(str(output / "lm/lm.onnx"), providers=["CPUExecutionProvider"])
    activation = np.arange(128, dtype=np.float32).reshape(1, 128) / 128
    actual = session.run(None, {"inputs_embeds": activation})
    assert len(actual) == 196
    for result in actual:
        np.testing.assert_allclose(result, activation @ weight, rtol=1e-5, atol=1e-5)

    assert {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == source_files
    assert {
        path.relative_to(exact).as_posix(): path.read_bytes()
        for path in exact.rglob("*")
        if path.is_file()
    } == exact_files


def test_optimize_bundle_exact_int8_to_nbits4_failure_keeps_source_and_removes_stage(
    tmp_path, monkeypatch
):
    import onnxruntime as ort

    source = tmp_path / "source"
    source.mkdir()
    weight = _write_ternary_tiny_bundle(source, signed_zero=True)
    inventory = _write_exact_ternary_inventory(
        tmp_path / "inventory.json", source, weight, matmul_count=196
    )
    exact = tmp_path / "exact-int8"
    optimize_bundle(
        source,
        exact,
        lm_method="ternary-exact-int8",
        ternary_inventory_path=inventory,
    )
    exact_files = {
        path.relative_to(exact).as_posix(): path.read_bytes()
        for path in exact.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "q4-failed"

    def fail_session(*args, **kwargs):
        raise RuntimeError("injected q4 ORT failure")

    monkeypatch.setattr(ort, "InferenceSession", fail_session)
    with pytest.raises(RuntimeError, match="injected q4 ORT failure"):
        optimize_bundle(
            exact,
            output,
            lm_method="ternary-exact-int8-to-nbits4",
            nbits_block_size=32,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".q4-failed.*.partial"))
    assert {
        path.relative_to(exact).as_posix(): path.read_bytes()
        for path in exact.rglob("*")
        if path.is_file()
    } == exact_files



def test_optimize_bundle_preserves_existing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_bundle(source)
    output = tmp_path / "optimized"
    output.mkdir()
    sentinel = output / "previous-success"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        optimize_bundle(source, output, lm_method="int8-dynamic")

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".optimized.*.partial"))


def test_optimize_bundle_ternary_exact_int8_does_not_create_runtime_session(tmp_path, monkeypatch):
    import onnxruntime as ort

    source = tmp_path / "source"
    source.mkdir()
    weight = _write_ternary_tiny_bundle(source)
    inventory = _write_exact_ternary_inventory(
        tmp_path / "inventory.json", source, weight, matmul_count=196
    )
    source_graph = (source / "lm/lm.onnx").read_bytes()
    source_data = (source / "lm/lm.onnx.data").read_bytes()
    output = tmp_path / "optimized"

    def fail_session(*args, **kwargs):
        pytest.fail("exact ternary optimization must not create an ONNX Runtime session")

    monkeypatch.setattr(ort, "InferenceSession", fail_session)
    report = optimize_bundle(
        source,
        output,
        lm_method="ternary-exact-int8",
        ternary_inventory_path=inventory,
    )

    assert report["output_bundle"] == str(output)
    assert output.is_dir()
    progress = Path(report["build_progress"])
    stages = [json.loads(line)["stage"] for line in progress.read_text().splitlines()]
    assert "scan_weight_001_of_001" in stages
    assert "encode_weight_001_of_001" in stages
    assert "onnx_checker" in stages
    assert stages[-1] == "complete"
    assert (output / "lm/lm.onnx").is_file()
    assert (source / "lm/lm.onnx").read_bytes() == source_graph
    assert (source / "lm/lm.onnx.data").read_bytes() == source_data
    assert not list(tmp_path.glob(".optimized.*.partial"))

@pytest.mark.parametrize(
    "failure_point", ["scan", "encode", "graph_save", "checker", "size_audit", "after_size_audit"]
)
def test_optimize_bundle_failure_does_not_publish_or_mutate_source(
    tmp_path, monkeypatch, failure_point
):
    import scripts.optimize_bundle as optimize_module
    from kitten2_onnx import ternary_exact_storage
    from scripts import audit_bundle_size

    source = tmp_path / "source"
    source.mkdir()
    weight = _write_ternary_tiny_bundle(source)
    inventory = _write_exact_ternary_inventory(
        tmp_path / "inventory.json", source, weight, matmul_count=196
    )
    source_files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "optimized"

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "scan":
        monkeypatch.setattr(ternary_exact_storage, "_weight_info", fail)
    elif failure_point == "encode":
        monkeypatch.setattr(ternary_exact_storage, "_write_exact_weight", fail)
    elif failure_point == "graph_save":
        monkeypatch.setattr(onnx, "save_model", fail)
    elif failure_point == "checker":
        monkeypatch.setattr(onnx.checker, "check_model", fail)
    elif failure_point == "size_audit":
        monkeypatch.setattr(audit_bundle_size, "audit_bundle", fail)
    else:
        original_write_json = optimize_module._write_json

        def fail_after_size_report(path, value):
            original_write_json(path, value)
            if Path(path).name == "size-report.json":
                raise RuntimeError("injected after_size_audit failure")

        monkeypatch.setattr(optimize_module, "_write_json", fail_after_size_report)

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        optimize_bundle(
            source,
            output,
            lm_method="ternary-exact-int8",
            ternary_inventory_path=inventory,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".optimized.*.partial"))
    assert {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == source_files


def test_optimize_bundle_ternary_exact_int8_include_selects_only_requested_nodes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    weight = _write_ternary_tiny_bundle(source, matmul_count=3)
    inventory = _write_exact_ternary_inventory(
        tmp_path / "inventory.json", source, weight, matmul_count=3
    )
    source_graph = (source / "lm/lm.onnx").read_bytes()
    output = tmp_path / "one-node"

    report = optimize_bundle(
        source,
        output,
        lm_method="ternary-exact-int8",
        ternary_inventory_path=inventory,
        include_matmul_nodes=("ternary_matmul_1",),
    )

    quantization = report["quantization"]
    graph = onnx.load(str(output / "lm/lm.onnx"), load_external_data=False)
    assert quantization["transformed_nodes"] == ["ternary_matmul_1"]
    assert quantization["transformed_matmul_count"] == 1
    assert quantization["excluded_nodes"] == ["ternary_matmul_0", "ternary_matmul_2"]
    assert sum(node.op_type == "MatMul" for node in graph.graph.node) == 3
    assert sum(node.op_type == "Cast" for node in graph.graph.node) == 2
    assert any(tensor.name == "weight" for tensor in graph.graph.initializer)
    assert (source / "lm/lm.onnx").read_bytes() == source_graph
    size_report = json.loads(Path(report["size_report"]).read_text(encoding="utf-8"))
    exact_storage = size_report["onnx"]["lm"]["ternary_exact_int8"]
    assert exact_storage["node_count"] == 1
    assert exact_storage["source_fp32_bytes_replaced"] == 0
    assert exact_storage["storage_reduction_bytes"] == -264



def test_optimize_bundle_v2_builds_q4_and_preserves_variant_provenance(tmp_path):
    import onnxruntime as ort

    source = tmp_path / "source-v2"
    source.mkdir()
    _write_ternary_tiny_bundle(source)
    lm_dir = source / "lm"
    vocabulary_path = lm_dir / "vocabulary.bf16.npy"
    np.save(vocabulary_path, np.zeros((3, 8), dtype=np.uint16), allow_pickle=False)
    head_graph = helper.make_graph(
        [],
        "shared-head",
        [
            helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, 8]),
            helper.make_tensor_value_info("vocabulary_weight", TensorProto.FLOAT, [3, 8]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
    )
    head_model = helper.make_model(head_graph, opset_imports=[helper.make_opsetid("", 18)])
    head_model.ir_version = min(head_model.ir_version, 10)
    onnx.save_model(head_model, str(lm_dir / "lm_output.onnx"))

    source_hash = "a" * 64
    minimum_runtime = ort.__version__
    metadata_path = lm_dir / "lm.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "input_mode": "inputs_embeds",
            "vocabulary_mode": "shared",
            "hidden_size": 8,
            "vocab_size": 3,
            "embedding_file": vocabulary_path.name,
            "embedding_storage_dtype": "bfloat16-u16",
            "embedding_runtime_dtype": "float32",
            "source_model_revision": "model-v2",
            "source_weight_variant": "packed",
            "source_weight_sha256": source_hash,
            "onnxruntime_min_tested_version": minimum_runtime,
            "output_head_file": "lm_output.onnx",
            "output_head_input_names": ["hidden_states", "vocabulary_weight"],
            "output_head_output_names": ["logits"],
            "output_head_input_shapes": {
                "hidden_states": [None, 8],
                "vocabulary_weight": [3, 8],
            },
            "output_head_output_shape": [None, 3],
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    bundle_path = source / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["format"] = "kitten2-onnx-bundle-v2"
    bundle["revisions"] = {"source_model_revision": "model-v2"}
    bundle["lm"].update(
        {
            "core_model": "lm/lm.onnx",
            "head_model": "lm/lm_output.onnx",
            "embeddings": "lm/vocabulary.bf16.npy",
            "weight_variant": "packed",
            "source_weight_sha256": source_hash,
            "embedding_reconstruction_version": None,
            "onnxruntime_min_tested_version": minimum_runtime,
            "storage": {
                "transformer": {"method": "float", "dtype": "float32"},
                "vocabulary": {
                    "method": "bfloat16-u16",
                    "file": "lm/vocabulary.bf16.npy",
                    "lossy": False,
                },
            },
        }
    )
    bundle["files"] = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*"))
        if path.is_file() and path != bundle_path
    }
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    output = tmp_path / "optimized-v2"
    report = optimize_bundle(source, output, lm_method="ternary-nbits4")
    output_bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    output_metadata = json.loads((output / "lm/lm.meta.json").read_text(encoding="utf-8"))

    assert output_bundle["format"] == "kitten2-onnx-bundle-v2"
    assert output_bundle["lm"]["storage"]["transformer"]["bits"] == 4
    assert output_bundle["lm"]["storage"]["transformer"]["zero_point"] == 8
    assert output_bundle["lm"]["storage"]["transformer"]["source_weight_sha256"] == source_hash
    assert output_metadata["quantization"]["source_weight_sha256"] == source_hash
    assert report["quantization"]["method"] == "ternary-nbits"
    assert report["size_report"]


    source_graph_path = source / "lm/lm.onnx"
    source_model = onnx.load(str(source_graph_path))
    non_ternary_weight = np.linspace(-0.4, 0.6, 256, dtype=np.float32).reshape(128, 2)
    source_model.graph.initializer.append(
        numpy_helper.from_array(non_ternary_weight, name="non_ternary_weight")
    )
    source_model.graph.node.append(
        helper.make_node(
            "MatMul",
            ["inputs_embeds", "non_ternary_weight"],
            ["diagnostic_logits"],
            name="non_ternary_matmul",
        )
    )
    source_model.graph.output.append(
        helper.make_tensor_value_info("diagnostic_logits", TensorProto.FLOAT, [1, 2])
    )
    onnx.save_model(
        source_model,
        str(source_graph_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="lm.onnx.data",
        size_threshold=0,
    )
    source_metadata_path = source / "lm/lm.meta.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_metadata["output_names"].append("diagnostic_logits")
    source_metadata_path.write_text(json.dumps(source_metadata), encoding="utf-8")
    source_bundle_path = source / "bundle.json"
    source_bundle = json.loads(source_bundle_path.read_text(encoding="utf-8"))
    source_bundle["files"] = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*"))
        if path.is_file() and path != source_bundle_path
    }
    source_bundle_path.write_text(json.dumps(source_bundle), encoding="utf-8")


    tritpack_output = tmp_path / "optimized-tritpack-v2"
    tritpack_reports = tmp_path / "tritpack-reports"
    tritpack_result = optimize_bundle(
        source,
        tritpack_output,
        lm_method="ternary-tritpack-exact",
        report_dir=tritpack_reports,
        chunk_rows=1,
    )
    tritpack_bundle = json.loads((tritpack_output / "bundle.json").read_text(encoding="utf-8"))
    tritpack_metadata = json.loads(
        (tritpack_output / "lm/lm.meta.json").read_text(encoding="utf-8")
    )
    tritpack_storage = tritpack_bundle["lm"]["storage"]["transformer"]
    tritpack_graph = onnx.load(
        str(tritpack_output / "lm/lm.onnx"), load_external_data=False
    )

    assert tritpack_result["quantization"]["method"] == "ternary-tritpack-exact"
    assert tritpack_storage["method"] == "ternary-tritpack-exact"
    assert tritpack_storage["transformed_matmul_count"] == 196
    assert tritpack_storage["transformed_initializer_count"] == 1
    assert tritpack_storage["runtime_materialization_required"] is True
    assert tritpack_storage["source_weight_sha256"] == source_hash
    assert len(tritpack_storage["weights"]) == 1
    assert tritpack_metadata["quantization"]["source_weight_sha256"] == source_hash
    assert sum(node.op_type == "MatMul" for node in tritpack_graph.graph.node) == 197
    assert not any(node.op_type == "MatMulNBits" for node in tritpack_graph.graph.node)
    assert "weight" not in {item.name for item in tritpack_graph.graph.initializer}
    assert (tritpack_output / "lm" / tritpack_storage["payload_file"]).is_file()
    from kitten2_onnx.materialized_weights import materialize_tritpack_model

    materialized = materialize_tritpack_model(
        tritpack_output / "lm/lm.onnx",
        tmp_path / "materialized-cache",
        storage_metadata=tritpack_storage,
    )
    sample = np.random.default_rng(8).normal(size=(1, 128)).astype(np.float32)
    source_session = ort.InferenceSession(
        str(source / "lm/lm.onnx"), providers=["CPUExecutionProvider"]
    )
    materialized_session = ort.InferenceSession(
        str(materialized.model_path), providers=["CPUExecutionProvider"]
    )
    expected_outputs = source_session.run(None, {"inputs_embeds": sample})
    actual_outputs = materialized_session.run(None, {"inputs_embeds": sample})
    assert len(actual_outputs) == 197
    for expected, actual in zip(expected_outputs, actual_outputs, strict=True):
        np.testing.assert_array_equal(actual, expected)

    size_report = json.loads(Path(tritpack_result["size_report"]).read_text(encoding="utf-8"))
    assert size_report["storage_breakdown"]["ternary_tritpack"][
        "compact_transformer_storage_bytes"
    ] == tritpack_storage["packed_code_bytes"] + tritpack_storage["scale_bytes"]


    from kitten2_onnx.validate_bundle import validate_bundle

    validation = validate_bundle(
        tritpack_output,
        materialization_cache_dir=tmp_path / "validator-cache",
    )
    assert "lm" in validation["graphs"]
    assert validation["vocabulary_storage"]["physical_copies"] == 1
    assert validation["graphs"]["lm"]["materialization"]["cache_key"]
    assert not any("tritpack" in error.lower() for error in validation["errors"]), validation[
        "errors"
    ]
