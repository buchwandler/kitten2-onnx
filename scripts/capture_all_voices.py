from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from kitten2_onnx.decoder_reference import (
    DECODER_REFERENCE_FORMAT,
    REQUIRED_ARRAYS,
    load_decoder_reference,
)
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from kitten2_onnx.s3gen_reference import S3GEN_REPOSITORY, S3GEN_REVISION
from scripts import capture_decoder_reference as decoder_capture

CAPTURE_REPORT_FORMAT = "kitten2-onnx-all-voice-capture-v1"


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system() == "Darwin" else peak * 1024


def _expected_metadata(
    voice: str,
    source_model_revision: str,
    *,
    token_count: int,
    flow_seed: int,
    vocoder_seed: int,
) -> dict[str, Any]:
    return {
        "format": DECODER_REFERENCE_FORMAT,
        "voice": voice,
        "source_model_revision": source_model_revision,
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "s3gen_repository": S3GEN_REPOSITORY,
        "s3gen_revision": S3GEN_REVISION,
        "token_count": token_count,
        "flow_seed": flow_seed,
        "vocoder_seed": vocoder_seed,
    }


def _validate_capture(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> None:
    if missing := REQUIRED_ARRAYS - arrays.keys():
        raise ValueError(f"Decoder reference is missing arrays: {sorted(missing)}")
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Decoder reference metadata mismatch for {key!r}: "
                f"expected {value!r}, found {metadata.get(key)!r}"
            )
    for name, array in arrays.items():
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"Decoder reference array {name!r} contains non-finite values")
    if not np.array_equal(arrays["flow_mel"], arrays["mel2wav_input"]):
        raise ValueError("The captured mel2wav input differs from the flow output mel")


def _capture_is_complete(path: Path, expected_metadata: dict[str, Any]) -> bool:
    metadata_path = path.with_suffix(".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        reference = load_decoder_reference(path)
        _validate_capture(reference.arrays, reference.metadata, expected_metadata)
    except (EOFError, KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def capture_all_voices(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    token_count: int = 32,
    flow_seed: int = 314159,
    vocoder_seed: int = 271828,
) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    voices_path = model_dir / config.get("voices", "voices/voices.json")
    voice_index = json.loads(voices_path.read_text(encoding="utf-8"))
    model_manifest_path = model_dir / "model_revision.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    source_model_revision = model_manifest.get("source_model_revision")
    if not source_model_revision:
        raise ValueError(f"No pinned source model revision is recorded in {model_manifest_path}")
    if model_manifest.get("upstream_code_revision") != UPSTREAM_CODE_REVISION:
        raise ValueError(
            f"Model assets use upstream revision {model_manifest.get('upstream_code_revision')}, "
            f"expected {UPSTREAM_CODE_REVISION}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "capture_report.json"
    report: dict[str, Any] = {
        "format": CAPTURE_REPORT_FORMAT,
        "source_model_revision": source_model_revision,
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "capture_options": {
            "token_count": token_count,
            "flow_seed": flow_seed,
            "vocoder_seed": vocoder_seed,
        },
        "complete": False,
        "voices": [],
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    _write_json_atomic(report_path, report)

    s3gen_context = None
    s3gen_weights_sha256 = None
    for index, voice in enumerate(sorted(voice_index), start=1):
        output_path = output_dir / f"{voice}.npz"
        expected_metadata = _expected_metadata(
            voice,
            source_model_revision,
            token_count=token_count,
            flow_seed=flow_seed,
            vocoder_seed=vocoder_seed,
        )
        if _capture_is_complete(output_path, expected_metadata):
            status = "skipped"
        else:
            if s3gen_context is None:
                s3gen_context = decoder_capture.load_s3gen_model()
                s3gen_weights_sha256 = decoder_capture._sha256(s3gen_context[1])
            arrays, metadata = decoder_capture.capture_reference(
                model_dir,
                voice,
                token_count=token_count,
                flow_seed=flow_seed,
                vocoder_seed=vocoder_seed,
                s3gen_context=s3gen_context,
                s3gen_weights_sha256=s3gen_weights_sha256,
            )
            _validate_capture(arrays, metadata, expected_metadata)
            decoder_capture.write_capture(output_path, arrays, metadata)
            if not _capture_is_complete(output_path, expected_metadata):
                raise ValueError(f"Captured decoder reference failed validation for {voice!r}")
            status = "captured"

        report["voices"].append(
            {
                "voice": voice,
                "status": status,
                "npz": str(output_path),
                "metadata": str(output_path.with_suffix(".meta.json")),
            }
        )
        report["peak_rss_bytes"] = _peak_rss_bytes()
        _write_json_atomic(report_path, report)
        print(
            json.dumps(
                {"completed": index, "total": len(voice_index), "voice": voice, "status": status}
            ),
            flush=True,
        )

    report["complete"] = True
    report["peak_rss_bytes"] = _peak_rss_bytes()
    _write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture built-in decoder references sequentially, reusing one S3Gen model."
    )
    parser.add_argument("--model-dir", type=Path, default=Path(".models/kitten-tts-2"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/voices"))
    parser.add_argument("--token-count", type=int, default=32)
    parser.add_argument("--flow-seed", type=int, default=314159)
    parser.add_argument("--vocoder-seed", type=int, default=271828)
    args = parser.parse_args()

    report = capture_all_voices(
        args.model_dir,
        args.output_dir,
        token_count=args.token_count,
        flow_seed=args.flow_seed,
        vocoder_seed=args.vocoder_seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
