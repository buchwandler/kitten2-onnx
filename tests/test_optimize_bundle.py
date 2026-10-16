from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.optimize_bundle import optimize_bundle


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
