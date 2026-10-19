import hashlib
import importlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from kitten2_onnx.decoder_reference import DECODER_REFERENCE_FORMAT, write_decoder_reference
from kitten2_onnx.validate_bundle import validate_bundle


def _write_voice_bundle(bundle_dir, voice_specs, *, declare_dynamic_axes=True):
    bundle_dir.mkdir(parents=True)
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
    silence_count = 3
    token_mel_ratio = 2
    samples_per_frame = 480
    first = voice_specs[0]
    first_prompt_length = first.get("prompt_length", 3)
    first_speech_count = first.get("speech_count", 3)
    first_prompt_frames = first.get("prompt_frames", 5)
    first_noise_frames = token_mel_ratio * (first_prompt_length + first_speech_count + silence_count)
    first_mel_frames = token_mel_ratio * (first_speech_count + silence_count) - 1
    flow_axes = {
        "speech_tokens": {1: "speech_tokens"},
        "prompt_token": {1: "prompt_tokens"},
        "prompt_feat": {1: "prompt_mel_frames"},
        "flow_noise": {2: "initial_noise_frames"},
        "mel": {2: "generated_mel_frames"},
    }
    mel_axes = {"mel": {2: "mel_frames"}, "sine_noise": {2: "vocoder_samples"}}
    if not declare_dynamic_axes:
        flow_axes = {}
        mel_axes = {}
    flow_metadata = {
        "graph": "flow.onnx",
        "source_model_revision": "model-rev",
        "upstream_code_revision": "upstream-rev",
        "s3gen_revision": "s3gen-rev",
        "input_names": [
            "speech_tokens",
            "speech_token_lengths",
            "prompt_token",
            "prompt_token_len",
            "prompt_feat",
            "embedding",
            "flow_noise",
        ],
        "output_names": ["mel"],
        "output_shape": [1, 80, first_mel_frames],
        "input_shapes": {
            "speech_tokens": [1, first_speech_count + silence_count],
            "speech_token_lengths": [1],
            "prompt_token": [1, first_prompt_length],
            "prompt_token_len": [1],
            "prompt_feat": [1, first_prompt_frames, 80],
            "embedding": [1, 192],
            "flow_noise": [1, 80, first_noise_frames],
        },
        "dynamic_axes": flow_axes,
        "silence_tail": {"count": silence_count},
        "token_mel_ratio": token_mel_ratio,
        "speech_vocab_size": token_map["num_audio_tokens"],
    }
    mel_metadata = {
        "graph": "mel2wav.onnx",
        "source_model_revision": "model-rev",
        "upstream_code_revision": "upstream-rev",
        "s3gen_revision": "s3gen-rev",
        "input_names": ["mel", "phase_noise", "sine_noise"],
        "output_names": ["waveform"],
        "input_shapes": {
            "mel": [1, 80, first_mel_frames],
            "phase_noise": [1, 9, 1],
            "sine_noise": [1, 9, first_mel_frames * samples_per_frame],
        },
        "dynamic_axes": mel_axes,
        "samples_per_mel_frame": samples_per_frame,
        "sample_rate": 24000,
    }
    graph_metadata = {
        "lm/lm.onnx": b"lm",
        "lm/lm.meta.json": json.dumps(
            {
                "graph": "lm.onnx",
                "input_mode": "input_ids",
                "input_names": ["input_ids", "speaker_hidden"],
                "output_names": ["logits"],
                "source_model_revision": "model-rev",
                "hidden_size": 4,
            }
        ).encode(),
        "speaker/speaker.onnx": b"speaker",
        "speaker/speaker.meta.json": json.dumps(
            {
                "graph": "speaker.onnx",
                "input_names": ["speaker_embedding"],
                "output_names": ["speaker_hidden"],
                "speaker_dim": 4,
                "hidden_size": 4,
            }
        ).encode(),
        "decoder/flow/flow.onnx": b"flow",
        "decoder/flow/flow.meta.json": json.dumps(flow_metadata).encode(),
        "decoder/mel2wav/mel2wav.onnx": b"mel2wav",
        "decoder/mel2wav/mel2wav.meta.json": json.dumps(mel_metadata).encode(),
        "tokenizer/tokenizer.json": b"{}",
    }
    for relative_path, content in graph_metadata.items():
        path = bundle_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    for spec in voice_specs:
        voice = spec["name"]
        prompt_length = spec.get("prompt_length", 3)
        prompt_frames = spec.get("prompt_frames", 5)
        speech_count = spec.get("speech_count", 3)
        noise_frames = token_mel_ratio * (prompt_length + speech_count + silence_count)
        mel_frames = token_mel_ratio * (speech_count + silence_count) - 1
        samples = mel_frames * samples_per_frame
        flow_channels = spec.get("flow_channels", 80)
        arrays = {
            "conditioning_prompt_token": (np.arange(prompt_length, dtype=np.int64) % 5)[None, :],
            "conditioning_prompt_token_len": np.asarray([prompt_length], dtype=np.int64),
            "conditioning_prompt_feat": np.zeros((1, prompt_frames, 80), dtype=np.float32),
            "conditioning_embedding": np.zeros((1, spec.get("embedding_width", 192)), dtype=np.float32),
            "speech_tokens": (np.arange(speech_count, dtype=np.int64) % 5)[None, :],
            "speech_token_lengths": np.asarray([speech_count], dtype=np.int64),
            "flow_noise": np.zeros((1, flow_channels, noise_frames), dtype=np.float32),
            "flow_mel": np.zeros((1, flow_channels, mel_frames), dtype=np.float32),
            "mel2wav_input": np.zeros((1, flow_channels, mel_frames), dtype=np.float32),
            "vocoder_phase_noise": np.zeros((1, 9, 1), dtype=np.float32),
            "vocoder_sine_noise": np.zeros((1, 9, samples), dtype=np.float32),
            "mel2wav_waveform": np.zeros((1, samples), dtype=np.float32),
            "waveform": np.zeros((1, samples), dtype=np.float32),
            "speaker_embedding": np.zeros((4,), dtype=np.float32),
            "reference_tokens": np.asarray(spec.get("reference_tokens", [1, 2]), dtype=np.int64),
        }
        if spec.get("nonfinite"):
            arrays["waveform"][0, 0] = np.nan
        if spec.get("speech_length_delta"):
            arrays["speech_token_lengths"][0] += spec["speech_length_delta"]
        if spec.get("mel2wav_mismatch"):
            arrays["mel2wav_input"][0, 0, 0] = 1.0
        reference_path = bundle_dir / "voices" / f"{voice}.npz"
        write_decoder_reference(
            reference_path,
            arrays,
            {
                "format": DECODER_REFERENCE_FORMAT,
                "voice": spec.get("metadata_voice", voice),
                "source_model_revision": "model-rev",
                "upstream_code_revision": "upstream-rev",
                "s3gen_revision": "s3gen-rev",
            },
        )

    files = {
        path.relative_to(bundle_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file()
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
        "speaker": {"model": "speaker/speaker.onnx", "metadata": "speaker/speaker.meta.json"},
        "decoder": {
            "flow": "decoder/flow/flow.onnx",
            "flow_metadata": "decoder/flow/flow.meta.json",
            "mel2wav": "decoder/mel2wav/mel2wav.onnx",
            "mel2wav_metadata": "decoder/mel2wav/mel2wav.meta.json",
        },
        "tokenizer": "tokenizer",
        "tokenizer_assets": ["tokenizer.json"],
        "token_map": token_map,
        "voices": {spec["name"]: f"voices/{spec['name']}.npz" for spec in voice_specs},
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")


def _validate_voice_bundle(bundle_dir, monkeypatch):
    validator_module = importlib.import_module("kitten2_onnx.validate_bundle")
    monkeypatch.setattr(
        validator_module,
        "_check_graph",
        lambda **kwargs: {"graph": kwargs["graph_path"].name, "provider": "CPUExecutionProvider"},
    )
    return validator_module.validate_bundle(bundle_dir)


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

def test_check_graph_uses_low_memory_profile_for_exact_ternary_lm(tmp_path, monkeypatch):
    import onnxruntime as ort

    from kitten2_onnx.validate_bundle import _check_graph

    graph_path = tmp_path / "lm.onnx"
    graph_path.write_bytes(b"graph")
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", type="tensor(int64)", shape=[1, 3]),
            SimpleNamespace(name="speaker_hidden", type="tensor(float)", shape=[1, 4]),
        ],
        get_outputs=lambda: [SimpleNamespace(name="logits", type="tensor(float)", shape=[1, 8])],
        get_providers=lambda: ["CPUExecutionProvider"],
    )
    captured = {}

    def create_session(path, *, sess_options, providers):
        captured.update(path=path, sess_options=sess_options, providers=providers)
        return session

    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(checker=SimpleNamespace(ValidationError=ValueError, check_model=lambda path: None)),
    )
    monkeypatch.setattr(ort, "InferenceSession", create_session)
    errors = []

    result = _check_graph(
        root=tmp_path,
        label="lm",
        graph_path=graph_path,
        metadata={
            "input_mode": "input_ids",
            "io_dtype": "float32",
            "runtime_memory_profile": "low-memory-exact-ternary",
        },
        providers=["CPUExecutionProvider"],
        expected_inputs=["input_ids", "speaker_hidden"],
        expected_outputs=["logits"],
        input_shapes=None,
        output_shape=None,
        errors=errors,
    )

    options = captured["sess_options"]
    assert result is not None
    assert not errors
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert not options.enable_mem_pattern
    assert not options.enable_cpu_mem_arena
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL



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



def test_validate_bundle_accepts_different_metadata_declared_time_lengths(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    _write_voice_bundle(
        bundle_dir,
        [
            {"name": "Bruno", "prompt_length": 3, "prompt_frames": 5, "speech_count": 3},
            {"name": "Victoria", "prompt_length": 6, "prompt_frames": 9, "speech_count": 7},
        ],
    )

    report = _validate_voice_bundle(bundle_dir, monkeypatch)

    assert report["valid"], report["errors"]
    assert report["voices"]["Bruno"]["prompt_tokens"] == 3
    assert report["voices"]["Victoria"]["prompt_tokens"] == 6
    assert "voice conditioning shapes" in report["checks"]


def test_validate_bundle_requires_declared_dynamic_time_axes(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    _write_voice_bundle(
        bundle_dir,
        [{"name": "Bruno", "prompt_length": 3}, {"name": "Victoria", "prompt_length": 6}],
        declare_dynamic_axes=False,
    )

    report = _validate_voice_bundle(bundle_dir, monkeypatch)

    assert not report["valid"]
    assert any("conditioning shape mismatch for prompt_token" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("voice_spec", "expected_error"),
    [
        ({"name": "Victoria", "embedding_width": 191}, "conditioning shape mismatch for embedding"),
        ({"name": "Victoria", "metadata_voice": "Bruno"}, "metadata identity differs from the manifest"),
        ({"name": "Victoria", "reference_tokens": [1, 5]}, "outside the audio vocabulary"),
        ({"name": "Victoria", "nonfinite": True}, "contains non-finite values"),
        ({"name": "Victoria", "speech_length_delta": 1}, "speech token length does not match its tokens"),
        ({"name": "Victoria", "mel2wav_mismatch": True}, "mel2wav input differs from the flow output mel"),
        ({"name": "Victoria", "flow_channels": 79}, "flow noise shape differs from static flow dimensions"),
    ],
)
def test_validate_bundle_rejects_malformed_voice_tensors(
    tmp_path, monkeypatch, voice_spec, expected_error
):
    bundle_dir = tmp_path / "bundle"
    _write_voice_bundle(bundle_dir, [{"name": "Bruno"}, voice_spec])

    report = _validate_voice_bundle(bundle_dir, monkeypatch)

    assert not report["valid"]
    assert any(expected_error in error for error in report["errors"])
