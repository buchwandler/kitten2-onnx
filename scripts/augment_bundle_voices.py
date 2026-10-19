from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from kitten2_onnx.decoder_reference import load_decoder_reference
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from kitten2_onnx.validate_bundle import validate_bundle

AUGMENTATION_FORMAT = "kitten2-onnx-voice-augmentation-v1"
CAPTURE_REPORT_FORMAT = "kitten2-onnx-all-voice-capture-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _contained_by(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _validate_voice_capture(
    capture_dir: Path,
    voice: str,
    *,
    source_model_revision: str,
    s3gen_revision: str,
) -> Path:
    reference_path = capture_dir / f"{voice}.npz"
    reference = load_decoder_reference(reference_path)
    metadata = reference.metadata
    expected = {
        "voice": voice,
        "source_model_revision": source_model_revision,
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "s3gen_revision": s3gen_revision,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Captured voice {voice!r} metadata mismatch for {key!r}: "
                f"expected {value!r}, found {metadata.get(key)!r}"
            )
    for name, array in reference.arrays.items():
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"Captured voice {voice!r} array {name!r} contains non-finite values")
    return reference_path


def augment_bundle_voices(
    source_bundle: str | Path,
    capture_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_dir = Path(source_bundle).resolve()
    capture_dir = Path(capture_dir).resolve()
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    protected_dirs = (source_dir, capture_dir, model_dir)
    if any(
        _contained_by(output_dir, protected) or _contained_by(protected, output_dir)
        for protected in protected_dirs
    ):
        raise ValueError("output_dir must not overlap the source bundle, captures, or model directory")
    if output_dir.exists():
        raise FileExistsError(f"Output bundle already exists: {output_dir}")

    source_validation = validate_bundle(source_dir)
    if not source_validation["valid"]:
        details = "\n".join(f"- {error}" for error in source_validation["errors"])
        raise ValueError(f"Source bundle validation failed:\n{details}")
    source_hashes = _directory_hashes(source_dir)
    bundle_path = source_dir / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((model_dir / "model_revision.json").read_text(encoding="utf-8"))
    source_model_revision = model_manifest.get("source_model_revision")
    if not source_model_revision:
        raise ValueError("Pinned model metadata has no source_model_revision")
    if model_manifest.get("upstream_code_revision") != UPSTREAM_CODE_REVISION:
        raise ValueError(
            f"Model upstream revision differs from pinned code {UPSTREAM_CODE_REVISION!r}"
        )
    revisions = bundle.get("revisions")
    if not isinstance(revisions, dict):
        raise TypeError("Source bundle revisions must be an object")
    if revisions.get("source_model_revision") != source_model_revision:
        raise ValueError("Model and source bundle revisions do not match")
    if revisions.get("upstream_code_revision") != UPSTREAM_CODE_REVISION:
        raise ValueError("Source bundle upstream code revision does not match the pinned revision")
    s3gen_revision = revisions.get("s3gen_revision")
    if not isinstance(s3gen_revision, str) or not s3gen_revision:
        raise ValueError("Source bundle has no S3Gen revision")

    voices_path = model_dir / model_config.get("voices", "voices/voices.json")
    voice_index = json.loads(voices_path.read_text(encoding="utf-8"))
    if (
        not isinstance(voice_index, dict)
        or not voice_index
        or any(not isinstance(name, str) or Path(name).name != name for name in voice_index)
    ):
        raise ValueError(f"Pinned voice index is empty or malformed: {voices_path}")
    voice_names = sorted(voice_index)

    capture_report_path = capture_dir / "capture_report.json"
    capture_report = json.loads(capture_report_path.read_text(encoding="utf-8"))
    if capture_report.get("format") != CAPTURE_REPORT_FORMAT or capture_report.get("complete") is not True:
        raise ValueError(f"All-voice capture report is incomplete or unsupported: {capture_report_path}")
    if capture_report.get("source_model_revision") != source_model_revision:
        raise ValueError("Captured voices use a different source model revision")
    if capture_report.get("upstream_code_revision") != UPSTREAM_CODE_REVISION:
        raise ValueError("Captured voices use a different upstream code revision")
    captured_voices = capture_report.get("voices")
    if not isinstance(captured_voices, list) or len(captured_voices) != len(voice_names):
        raise ValueError("All-voice capture report has an incomplete voice list")
    if any(not isinstance(entry, dict) for entry in captured_voices):
        raise ValueError("All-voice capture report contains malformed voice entries")
    captured_names = [entry.get("voice") for entry in captured_voices]
    if captured_names != voice_names:
        raise ValueError("Captured voice names do not exactly match the pinned upstream index")
    if any(entry.get("status") not in {"captured", "skipped"} for entry in captured_voices):
        raise ValueError("All-voice capture report has an invalid capture status")

    voices = bundle.get("voices")
    if not isinstance(voices, dict) or not voices:
        raise ValueError("Source bundle voices must be a non-empty object")
    unexpected_voices = set(voices) - set(voice_names)
    if unexpected_voices:
        raise ValueError(f"Source bundle contains voices absent from the pinned index: {sorted(unexpected_voices)}")
    existing_voices = sorted(voices)
    added_voices = [name for name in voice_names if name not in voices]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.staging")
    try:
        shutil.copytree(source_dir, staging_dir, copy_function=shutil.copy2)
        staged_voices = bundle["voices"]
        for voice in added_voices:
            capture_path = _validate_voice_capture(
                capture_dir,
                voice,
                source_model_revision=source_model_revision,
                s3gen_revision=s3gen_revision,
            )
            metadata_path = capture_path.with_suffix(".meta.json")
            staged_path = staging_dir / "voices" / f"{voice}.npz"
            staged_metadata_path = staged_path.with_suffix(".meta.json")
            if staged_path.exists() or staged_metadata_path.exists():
                raise FileExistsError(f"Unmanifested voice files already exist for {voice!r}")
            shutil.copy2(capture_path, staged_path)
            shutil.copy2(metadata_path, staged_metadata_path)
            staged_voices[voice] = f"voices/{voice}.npz"

        if bundle.get("default_voice") not in staged_voices:
            bundle["default_voice"] = voice_names[0]
        staged_hashes = _directory_hashes(staging_dir)
        bundle["files"] = {
            relative_path: digest
            for relative_path, digest in staged_hashes.items()
            if relative_path != "bundle.json"
        }
        (staging_dir / "bundle.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        for relative_path, expected_hash in source_hashes.items():
            if relative_path == "bundle.json":
                continue
            if staged_hashes.get(relative_path) != expected_hash:
                raise ValueError(f"Source file changed during augmentation: {relative_path}")
        new_paths = set(staged_hashes) - set(source_hashes)
        if any(not path.startswith("voices/") for path in new_paths):
            raise ValueError(f"Augmentation added files outside voices/: {sorted(new_paths)}")

        validation = validate_bundle(staging_dir)
        if not validation["valid"]:
            details = "\n".join(f"- {error}" for error in validation["errors"])
            raise ValueError(f"Augmented bundle validation failed:\n{details}")

        os.replace(staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    return {
        "format": AUGMENTATION_FORMAT,
        "source_bundle": str(source_dir),
        "output_bundle": str(output_dir),
        "source_model_revision": source_model_revision,
        "voice_count": len(staged_voices),
        "existing_voices_preserved": existing_voices,
        "added_voices": added_voices,
        "source_file_hashes": source_hashes,
        "source_hashes_preserved": True,
        "validation": {
            "valid": validation["valid"],
            "checks": validation["checks"],
            "files": validation["files"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment a local ONNX bundle with captured built-in voice packages."
    )
    parser.add_argument("source_bundle", type=Path, help="Validated local source bundle")
    parser.add_argument("capture_dir", type=Path, help="Output directory from capture_all_voices")
    parser.add_argument("--model-dir", type=Path, default=Path(".models/kitten-tts-2"))
    parser.add_argument("--output", type=Path, required=True, help="New local augmented bundle directory")
    args = parser.parse_args()

    report = augment_bundle_voices(
        args.source_bundle,
        args.capture_dir,
        args.model_dir,
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
