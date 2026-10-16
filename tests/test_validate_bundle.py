import hashlib
import importlib
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


def test_validate_bundle_accepts_input_ids_without_external_embeddings(tmp_path, monkeypatch):
    validator_module = importlib.import_module("kitten2_onnx.validate_bundle")
    monkeypatch.setattr(
        validator_module,
        "_check_graph",
        lambda **kwargs: {"label": kwargs["label"], "valid": True},
    )
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    files = {}

    def add_file(relative_path, content):
        path = bundle_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[relative_path] = hashlib.sha256(content).hexdigest()

    lm_metadata = {
        "graph": "lm.onnx",
        "input_mode": "input_ids",
        "input_names": ["input_ids", "speaker_hidden"],
        "output_names": ["logits"],
        "source_model_revision": "model-rev",
        "hidden_size": 4,
    }
    speaker_metadata = {
        "graph": "speaker.onnx",
        "input_names": ["speaker_embedding"],
        "output_names": ["speaker_hidden"],
        "hidden_size": 4,
        "speaker_dim": 2,
    }
    flow_metadata = {"graph": "flow.onnx", "source_model_revision": "model-rev"}
    mel_metadata = {
        "graph": "mel2wav.onnx",
        "source_model_revision": "model-rev",
        "sample_rate": 24000,
    }
    artifacts = {
        "lm/lm.onnx": b"lm",
        "lm/lm.meta.json": json.dumps(lm_metadata).encode(),
        "speaker/speaker.onnx": b"speaker",
        "speaker/speaker.meta.json": json.dumps(speaker_metadata).encode(),
        "decoder/flow/flow.onnx": b"flow",
        "decoder/flow/flow.meta.json": json.dumps(flow_metadata).encode(),
        "decoder/mel2wav/mel2wav.onnx": b"mel2wav",
        "decoder/mel2wav/mel2wav.meta.json": json.dumps(mel_metadata).encode(),
        "tokenizer/tokenizer.json": b"{}",
    }
    for relative_path, content in artifacts.items():
        add_file(relative_path, content)
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
    bundle = {
        "format": "kitten2-onnx-bundle-v1",
        "sample_rate": 24000,
        "files": files,
        "revisions": {
            "source_model_revision": "model-rev",
            "upstream_code_revision": "upstream-rev",
            "s3gen_revision": "s3gen-rev",
        },
        "versions": {
            "python": "3.13",
            "platform": "linux",
            "packages": {
                name: "test"
                for name in (
                    "kitten-text-processing",
                    "onnx",
                    "onnxruntime",
                    "torch",
                    "transformers",
                )
            },
        },
        "lm": {"model": "lm/lm.onnx", "metadata": "lm/lm.meta.json"},
        "speaker": {
            "model": "speaker/speaker.onnx",
            "metadata": "speaker/speaker.meta.json",
        },
        "decoder": {
            "flow": "decoder/flow/flow.onnx",
            "flow_metadata": "decoder/flow/flow.meta.json",
            "mel2wav": "decoder/mel2wav/mel2wav.onnx",
            "mel2wav_metadata": "decoder/mel2wav/mel2wav.meta.json",
        },
        "tokenizer": "tokenizer",
        "tokenizer_assets": ["tokenizer.json"],
        "token_map": token_map,
        "voices": {},
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")

    report = validate_bundle(bundle_dir)

    assert "LM input_ids and speaker injection contract" in report["checks"]
    assert not any(
        "input_ids LM must not use a separate embedding table" in error
        for error in report["errors"]
    )


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


def test_check_graph_validates_input_ids_types(tmp_path, monkeypatch):
    from kitten2_onnx.validate_bundle import _check_graph

    graph_path = tmp_path / "lm.onnx"
    graph_path.write_bytes(b"graph")
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", type="tensor(float)", shape=[1, 3]),
            SimpleNamespace(name="speaker_hidden", type="tensor(float16)", shape=[1, 4]),
        ],
        get_outputs=lambda: [SimpleNamespace(name="logits", type="tensor(float)", shape=[1, 8])],
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
        label="lm",
        graph_path=graph_path,
        metadata={"input_mode": "input_ids", "io_dtype": "float32"},
        providers=["CPUExecutionProvider"],
        expected_inputs=["input_ids", "speaker_hidden"],
        expected_outputs=["logits"],
        input_shapes=None,
        output_shape=None,
        errors=errors,
    )

    assert "LM input_ids graph input must be int64" in errors
    assert "LM speaker_hidden type differs from io_dtype metadata" in errors


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
