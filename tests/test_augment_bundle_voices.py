from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kitten2_onnx.decoder_reference import (
    DECODER_REFERENCE_FORMAT,
    REQUIRED_ARRAYS,
    load_decoder_reference,
    write_decoder_reference,
)
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from scripts import augment_bundle_voices as augmenter

SOURCE_REVISION = "model-rev"
S3GEN_REVISION = "s3gen-rev"


def _write_reference(path: Path, voice: str) -> None:
    arrays = {name: np.zeros((1,), dtype=np.float32) for name in REQUIRED_ARRAYS}
    arrays["speaker_embedding"] = np.zeros((4,), dtype=np.float32)
    arrays["reference_tokens"] = np.asarray([1, 2], dtype=np.int64)
    arrays["flow_mel"] = np.zeros((1, 80, 4), dtype=np.float32)
    arrays["mel2wav_input"] = arrays["flow_mel"].copy()
    write_decoder_reference(
        path,
        arrays,
        {
            "format": DECODER_REFERENCE_FORMAT,
            "voice": voice,
            "source_model_revision": SOURCE_REVISION,
            "upstream_code_revision": UPSTREAM_CODE_REVISION,
            "s3gen_revision": S3GEN_REVISION,
        },
    )


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _inputs(root: Path, voices: list[str]) -> tuple[Path, Path, Path]:
    source = root / "source"
    (source / "decoder").mkdir(parents=True)
    (source / "voices").mkdir()
    (source / "decoder" / "flow.onnx").write_bytes(b"preserve this graph")
    _write_reference(source / "voices" / "Bruno.npz", "Bruno")

    source_files = _hashes(source)
    bundle = {
        "format": "kitten2-onnx-bundle-v1",
        "default_voice": "Bruno",
        "revisions": {
            "source_model_revision": SOURCE_REVISION,
            "upstream_code_revision": UPSTREAM_CODE_REVISION,
            "s3gen_revision": S3GEN_REVISION,
        },
        "voices": {"Bruno": "voices/Bruno.npz"},
        "files": {path: digest for path, digest in source_files.items() if path != "bundle.json"},
    }
    (source / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    (source / "unlisted.bin").write_bytes(b"preserve unlisted files too")

    model_dir = root / "model"
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"voices": "voices/voices.json"}), encoding="utf-8"
    )
    (model_dir / "voices" / "voices.json").write_text(
        json.dumps({voice: {} for voice in voices}), encoding="utf-8"
    )
    (model_dir / "model_revision.json").write_text(
        json.dumps(
            {
                "source_model_revision": SOURCE_REVISION,
                "upstream_code_revision": UPSTREAM_CODE_REVISION,
            }
        ),
        encoding="utf-8",
    )

    capture_dir = root / "captures"
    capture_dir.mkdir()
    for voice in voices:
        _write_reference(capture_dir / f"{voice}.npz", voice)
    (capture_dir / "capture_report.json").write_text(
        json.dumps(
            {
                "format": augmenter.CAPTURE_REPORT_FORMAT,
                "complete": True,
                "source_model_revision": SOURCE_REVISION,
                "upstream_code_revision": UPSTREAM_CODE_REVISION,
                "voices": [{"voice": voice, "status": "captured"} for voice in voices],
            }
        ),
        encoding="utf-8",
    )
    return source, capture_dir, model_dir


def _valid_report(path: Path) -> dict[str, object]:
    bundle = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    return {
        "valid": True,
        "errors": [],
        "checks": ["test validation"],
        "files": {"listed": len(bundle["files"]), "verified": len(bundle["files"])},
    }


def test_augmentation_preserves_source_files_and_adds_only_missing_voices(
    tmp_path, monkeypatch
):
    voices = ["Bruno", "Victoria", "Walter"]
    source, capture_dir, model_dir = _inputs(tmp_path, voices)
    original_hashes = _hashes(source)
    validation_paths: list[Path] = []

    def validate(path):
        validation_paths.append(Path(path))
        return _valid_report(Path(path))

    monkeypatch.setattr(augmenter, "validate_bundle", validate)
    output = tmp_path / "augmented"

    report = augmenter.augment_bundle_voices(source, capture_dir, model_dir, output)

    assert report["added_voices"] == ["Victoria", "Walter"]
    assert report["existing_voices_preserved"] == ["Bruno"]
    assert report["voice_count"] == 3
    assert report["source_hashes_preserved"] is True
    assert validation_paths[0] == source.resolve()
    assert not validation_paths[1].exists()
    assert _hashes(source) == original_hashes
    for relative_path, digest in original_hashes.items():
        if relative_path != "bundle.json":
            assert _hashes(output)[relative_path] == digest

    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["voices"] == {
        "Bruno": "voices/Bruno.npz",
        "Victoria": "voices/Victoria.npz",
        "Walter": "voices/Walter.npz",
    }
    assert load_decoder_reference(output / "voices/Victoria.npz").metadata["voice"] == "Victoria"
    assert load_decoder_reference(output / "voices/Walter.npz").metadata["voice"] == "Walter"
    for relative_path, expected_hash in bundle["files"].items():
        actual_hash = hashlib.sha256((output / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash
    assert not list(tmp_path.glob(".augmented.*.staging"))


def test_augmentation_does_not_promote_a_bundle_that_fails_validation(tmp_path, monkeypatch):
    voices = ["Bruno", "Victoria"]
    source, capture_dir, model_dir = _inputs(tmp_path, voices)
    original_hashes = _hashes(source)
    validation_calls = 0

    def validate(path):
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return _valid_report(Path(path))
        return {"valid": False, "errors": ["invalid staged output"], "checks": [], "files": {}}

    monkeypatch.setattr(augmenter, "validate_bundle", validate)
    output = tmp_path / "augmented"

    with pytest.raises(ValueError, match="Augmented bundle validation failed"):
        augmenter.augment_bundle_voices(source, capture_dir, model_dir, output)

    assert not output.exists()
    assert _hashes(source) == original_hashes
    assert not list(tmp_path.glob(".augmented.*.staging"))


def test_augmentation_refuses_to_overwrite_an_existing_destination(tmp_path, monkeypatch):
    source, capture_dir, model_dir = _inputs(tmp_path, ["Bruno", "Victoria"])
    output = tmp_path / "augmented"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        augmenter.augment_bundle_voices(source, capture_dir, model_dir, output)
