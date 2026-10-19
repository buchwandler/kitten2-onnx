from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kitten2_onnx.decoder_reference import (
    DECODER_REFERENCE_FORMAT,
    REQUIRED_ARRAYS,
    load_decoder_reference,
)
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from kitten2_onnx.s3gen_reference import S3GEN_REPOSITORY, S3GEN_REVISION
from scripts import capture_all_voices as capture_all_module
from scripts import capture_decoder_reference as decoder_capture


def _arrays() -> dict[str, np.ndarray]:
    arrays = {name: np.zeros((1,), dtype=np.float32) for name in REQUIRED_ARRAYS}
    arrays["speaker_embedding"] = np.zeros((4,), dtype=np.float32)
    arrays["reference_tokens"] = np.asarray([1, 2], dtype=np.int64)
    arrays["conditioning_prompt_token"] = np.asarray([[1, 2]], dtype=np.int64)
    arrays["flow_mel"] = np.zeros((1, 3, 4), dtype=np.float32)
    arrays["mel2wav_input"] = arrays["flow_mel"].copy()
    return arrays


def _metadata(voice: str, *, token_count: int = 2) -> dict[str, object]:
    return {
        "format": DECODER_REFERENCE_FORMAT,
        "voice": voice,
        "source_model_revision": "pinned-commit",
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "s3gen_repository": S3GEN_REPOSITORY,
        "s3gen_revision": S3GEN_REVISION,
        "token_count": token_count,
        "flow_seed": 314159,
        "vocoder_seed": 271828,
    }


def _model_dir(root: Path, voices: list[str]) -> Path:
    model_dir = root / "model"
    voice_dir = model_dir / "voices"
    voice_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"voices": "voices/voices.json"}), encoding="utf-8"
    )
    (voice_dir / "voices.json").write_text(
        json.dumps({voice: {} for voice in voices}), encoding="utf-8"
    )
    (model_dir / "model_revision.json").write_text(
        json.dumps(
            {
                "source_model_revision": "pinned-commit",
                "upstream_code_revision": UPSTREAM_CODE_REVISION,
            }
        ),
        encoding="utf-8",
    )
    return model_dir


def test_capture_all_voices_reuses_context_in_sorted_sequence(tmp_path, monkeypatch, capsys):
    model_dir = _model_dir(tmp_path, ["Victoria", "Bruno", "Alfred"])
    output_dir = tmp_path / "captures"
    context = (object(), Path("weights.safetensors"))
    hashed_paths = []

    def hash_weights(path):
        hashed_paths.append(path)
        return "weights-sha256"

    monkeypatch.setattr(decoder_capture, "_sha256", hash_weights)
    loaded_contexts: list[tuple[object, Path]] = []
    captured_voices: list[str] = []

    def load_context():
        loaded_contexts.append(context)
        return context

    def capture_reference(model_dir, voice, **kwargs):
        captured_voices.append(voice)
        assert kwargs["s3gen_context"] is context
        assert kwargs["s3gen_weights_sha256"] == "weights-sha256"
        return _arrays(), _metadata(voice, token_count=kwargs["token_count"])

    monkeypatch.setattr(decoder_capture, "load_s3gen_model", load_context)
    monkeypatch.setattr(decoder_capture, "capture_reference", capture_reference)

    report = capture_all_module.capture_all_voices(model_dir, output_dir, token_count=2)

    assert captured_voices == ["Alfred", "Bruno", "Victoria"]
    assert len(loaded_contexts) == 1
    assert report["complete"] is True
    assert hashed_paths == [context[1]]
    assert report["peak_rss_bytes"] > 0
    assert [entry["status"] for entry in report["voices"]] == ["captured"] * 3
    assert all(Path(entry["npz"]).is_file() for entry in report["voices"])
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [entry["voice"] for entry in progress] == captured_voices


def test_capture_all_voices_skips_valid_and_recaptures_stale_metadata(tmp_path, monkeypatch):
    model_dir = _model_dir(tmp_path, ["Alfred", "Bruno"])
    output_dir = tmp_path / "captures"
    context = (object(), Path("weights.safetensors"))
    monkeypatch.setattr(decoder_capture, "_sha256", lambda _path: "weights-sha256")
    captures: list[str] = []

    def load_context():
        return context

    def capture_reference(model_dir, voice, **kwargs):
        captures.append(voice)
        return _arrays(), _metadata(voice, token_count=kwargs["token_count"])

    monkeypatch.setattr(decoder_capture, "load_s3gen_model", load_context)
    monkeypatch.setattr(decoder_capture, "capture_reference", capture_reference)
    initial = capture_all_module.capture_all_voices(model_dir, output_dir, token_count=2)
    assert [entry["status"] for entry in initial["voices"]] == ["captured", "captured"]

    def unexpected_capture(*args, **kwargs):
        raise AssertionError("valid references should be resumable without recapture")

    monkeypatch.setattr(decoder_capture, "load_s3gen_model", unexpected_capture)
    monkeypatch.setattr(decoder_capture, "capture_reference", unexpected_capture)
    resumed = capture_all_module.capture_all_voices(model_dir, output_dir, token_count=2)
    assert [entry["status"] for entry in resumed["voices"]] == ["skipped", "skipped"]

    bruno_metadata_path = output_dir / "Bruno.meta.json"
    bruno_metadata = json.loads(bruno_metadata_path.read_text(encoding="utf-8"))
    bruno_metadata["source_model_revision"] = "stale-commit"
    bruno_metadata_path.write_text(json.dumps(bruno_metadata), encoding="utf-8")
    monkeypatch.setattr(decoder_capture, "load_s3gen_model", load_context)
    monkeypatch.setattr(decoder_capture, "capture_reference", capture_reference)
    captures.clear()

    repaired = capture_all_module.capture_all_voices(model_dir, output_dir, token_count=2)

    assert captures == ["Bruno"]
    assert [entry["status"] for entry in repaired["voices"]] == ["skipped", "captured"]
    assert load_decoder_reference(output_dir / "Bruno.npz").metadata["source_model_revision"] == "pinned-commit"


def test_capture_all_voices_rejects_bad_metadata_before_writing(tmp_path, monkeypatch):
    model_dir = _model_dir(tmp_path, ["Bruno"])
    output_dir = tmp_path / "captures"
    monkeypatch.setattr(decoder_capture, "_sha256", lambda _path: "weights-sha256")
    monkeypatch.setattr(decoder_capture, "load_s3gen_model", lambda: (object(), Path("weights")))
    monkeypatch.setattr(
        decoder_capture,
        "capture_reference",
        lambda *args, **kwargs: (_arrays(), _metadata("WrongVoice", token_count=2)),
    )

    with pytest.raises(ValueError, match="metadata mismatch for 'voice'"):
        capture_all_module.capture_all_voices(model_dir, output_dir, token_count=2)

    assert not (output_dir / "Bruno.npz").exists()
    report = json.loads((output_dir / "capture_report.json").read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert report["voices"] == []


def test_write_capture_keeps_existing_pair_when_staging_fails(tmp_path, monkeypatch):
    output_path = tmp_path / "Bruno.npz"
    arrays = _arrays()
    metadata = _metadata("Bruno")
    decoder_capture.write_capture(output_path, arrays, metadata)
    original_npz = output_path.read_bytes()
    metadata_path = output_path.with_suffix(".meta.json")
    original_metadata = metadata_path.read_bytes()

    def fail_after_staging(path, arrays, metadata):
        np.savez_compressed(path, **arrays)
        path.with_suffix(".meta.json").write_text("staged", encoding="utf-8")
        raise RuntimeError("staging failure")

    monkeypatch.setattr(decoder_capture, "write_decoder_reference", fail_after_staging)
    with pytest.raises(RuntimeError, match="staging failure"):
        decoder_capture.write_capture(output_path, arrays, metadata)

    assert output_path.read_bytes() == original_npz
    assert metadata_path.read_bytes() == original_metadata
    assert not list(tmp_path.glob(".Bruno.*.tmp*"))
