"""Run a deterministic finite-audio smoke test for one or every bundle voice."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from kitten2_onnx.tts import Kitten2OnnxTTS


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _bundle_tree_sha256(bundle_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in bundle_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(bundle_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def smoke_bundle(
    bundle_dir: str | Path,
    output_path: str | Path,
    *,
    voice: str | None = None,
    max_new_tokens: int = 2,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    output_path = Path(output_path)
    bundle_manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    bundle_hash = _bundle_tree_sha256(bundle_dir)
    tts = Kitten2OnnxTTS(bundle_dir, providers=["CPUExecutionProvider"])
    voices = [voice] if voice is not None else sorted(tts.available_voices)
    missing = sorted(set(voices) - set(tts.available_voices))
    if missing:
        raise ValueError(f"Unknown smoke-test voices: {missing}")

    report: dict[str, Any] = {
        "format": "kitten2-onnx-voice-smoke-report-v1",
        "bundle": str(bundle_dir),
        "bundle_tree_sha256": bundle_hash,
        "weight_variant": bundle_manifest.get("lm", {}).get("weight_variant", "full"),
        "source_model_revision": bundle_manifest.get("revisions", {}).get("source_model_revision"),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "provider": "CPUExecutionProvider",
        "max_new_tokens": max_new_tokens,
        "expected_voices": voices,
        "cases": [],
        "status": "running",
    }
    existing: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            previous.get("format") == report["format"]
            and previous.get("bundle_tree_sha256") == bundle_hash
            and previous.get("max_new_tokens") == max_new_tokens
        ):
            existing = {
                case["voice"]: case
                for case in previous.get("cases", [])
                if case.get("status") == "passed"
            }
    report["cases"] = [existing[name] for name in voices if name in existing]
    _write_json(output_path, report)

    for voice_name in voices:
        if voice_name in existing:
            continue
        started = time.perf_counter()
        try:
            result = tts.generate_with_metadata(
                "Release voice smoke.",
                voice=voice_name,
                seed=1729,
                max_new_tokens=max_new_tokens,
                preset=bundle_manifest.get("default_preset", "stable"),
                normalize=False,
                use_emotion=False,
            )
            waveform = np.asarray(result.waveform, dtype=np.float32)
            finite = bool(np.all(np.isfinite(waveform)))
            passed = tts.sample_rate == 24000 and waveform.size > 0 and finite
            case = {
                "voice": voice_name,
                "status": "passed" if passed else "failed",
                "sample_rate": tts.sample_rate,
                "waveform_samples": int(waveform.size),
                "waveform_finite": finite,
                "token_count": len(result.s3_ids),
                "elapsed_seconds": time.perf_counter() - started,
            }
            if not passed:
                case["error"] = "Expected finite, non-empty 24 kHz audio"
        except Exception as error:  # noqa: BLE001
            case = {
                "voice": voice_name,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        report["cases"].append(case)
        report["cases"].sort(key=lambda item: voices.index(item["voice"]))
        _write_json(output_path, report)

    report["passed_voices"] = sum(case["status"] == "passed" for case in report["cases"])
    report["failed_voices"] = [
        case["voice"] for case in report["cases"] if case["status"] != "passed"
    ]
    report["peak_rss_bytes"] = _peak_rss_bytes()
    report["status"] = "passed" if not report["failed_voices"] and len(report["cases"]) == len(voices) else "failed"
    _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    report = smoke_bundle(
        args.bundle_dir,
        args.output,
        voice=args.voice,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
