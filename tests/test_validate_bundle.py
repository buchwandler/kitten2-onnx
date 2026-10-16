import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from kitten2_onnx.validate_bundle import validate_bundle


def test_validate_bundle_reports_corrupt_hash_and_unsafe_path(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    payload = bundle_dir / "payload.bin"
    payload.write_bytes(b"payload")
    token_map = {
        "audio_id_base": 10,
        "num_audio_tokens": 5,
        "speech_start_id": 1,
        "speech_end_id": 2,
        "text_start_id": 3,
        "text_end_id": 4,
        "start_id": 5,
        "stop_id": 6,
        "final_seg_id": 15,
    }
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
            {
                "format": "kitten2-onnx-bundle-v1",
                "files": {
                    "payload.bin": "0" * 64,
                    "../outside.bin": hashlib.sha256(b"outside").hexdigest(),
                },
                "token_map": token_map,
                "revisions": {
                    "source_model_revision": "model-rev",
                    "upstream_code_revision": "upstream-rev",
                    "s3gen_revision": "s3gen-rev",
                },
                "versions": {"python": "3.13", "packages": {"onnx": "1.0"}},
                "voices": {},
            }
        ),
        encoding="utf-8",
    )

    report = validate_bundle(bundle_dir)

    assert not report["valid"]
    assert any("SHA256 mismatch" in error for error in report["errors"])
    assert any("must be relative and contained" in error for error in report["errors"])


def test_validate_relocated_exported_bundle_when_configured():
    bundle_dir = os.environ.get("KITTEN2_VALIDATION_BUNDLE")
    if not bundle_dir:
        pytest.skip("set KITTEN2_VALIDATION_BUNDLE to run the full ONNX bundle validator")

    report = validate_bundle(bundle_dir)

    assert report["valid"], report["errors"]
    assert report["files"]["listed"] == report["files"]["verified"]
    assert len(report["graphs"]) == 4
    assert "Bruno" in report["voices"]


def test_check_graph_rejects_io_that_differs_from_metadata(tmp_path, monkeypatch):
    from kitten2_onnx.validate_bundle import _check_graph

    graph_path = tmp_path / "graph.onnx"
    graph_path.write_bytes(b"graph")
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="actual_input", shape=[1, 3])],
        get_outputs=lambda: [SimpleNamespace(name="actual_output", shape=[1, 2])],
        get_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(checker=SimpleNamespace(check_model=lambda path: None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=lambda path, providers: session),
    )
    errors = []

    _check_graph(
        root=tmp_path,
        label="test",
        graph_path=graph_path,
        metadata={},
        providers=["CPUExecutionProvider"],
        expected_inputs=["expected_input"],
        expected_outputs=["expected_output"],
        input_shapes={"actual_input": [1, 4]},
        output_shape=[1, 3],
        errors=errors,
    )

    assert any("graph inputs differ from metadata" in error for error in errors)
    assert any("graph outputs differ from metadata" in error for error in errors)
    assert any("dimension 1 differs from metadata" in error for error in errors)


@pytest.mark.parametrize("failure", ["checker", "runtime"])
def test_check_graph_reports_checker_and_session_failures(tmp_path, monkeypatch, failure):
    from kitten2_onnx.validate_bundle import _check_graph

    graph_path = tmp_path / "graph.onnx"
    graph_path.write_bytes(b"graph")

    def check_model(path):
        if failure == "checker":
            raise ValueError("invalid graph")

    checker = SimpleNamespace(ValidationError=ValueError, check_model=check_model)
    if failure == "runtime":
        runtime = SimpleNamespace(
            InferenceSession=lambda path, providers: (_ for _ in ()).throw(RuntimeError("invalid session"))
        )
    else:
        runtime = SimpleNamespace(InferenceSession=lambda path, providers: None)
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(checker=checker))
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    errors = []

    result = _check_graph(
        root=tmp_path,
        label="test",
        graph_path=graph_path,
        metadata={},
        providers=["CPUExecutionProvider"],
        expected_inputs=[],
        expected_outputs=[],
        input_shapes=None,
        output_shape=None,
        errors=errors,
    )

    assert result is None
    assert len(errors) == 1
    assert ("ONNX checker failed" if failure == "checker" else "ONNX Runtime session failed") in errors[0]
