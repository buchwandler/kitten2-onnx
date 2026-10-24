"""Validate a built-in-voice corpus against PyTorch and retain first-divergence logits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kitten2_onnx.reference import (
    UPSTREAM_CODE_REVISION,
    load_reference_artifacts,
    load_reference_model,
)
from kitten2_onnx.sampling import sampling_settings_from_config
from kitten2_onnx.tts import Kitten2OnnxTTS
from scripts.capture_reference import capture_reference


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")



def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_provenance(bundle_dir: Path) -> dict[str, Any]:
    if not bundle_dir.is_dir():
        return {
            "path": str(bundle_dir),
            "status": "unavailable",
            "bundle_json_sha256": None,
            "tree_sha256": None,
            "file_count": 0,
            "bytes": 0,
        }
    files = sorted(path for path in bundle_dir.rglob("*") if path.is_file())
    bundle_json = bundle_dir / "bundle.json"
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(bundle_dir).as_posix()
        file_hash = _sha256_file(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\\n")
        total_bytes += size
    return {
        "path": str(bundle_dir),
        "status": "available" if bundle_json.is_file() else "missing_bundle_json",
        "bundle_json_sha256": _sha256_file(bundle_json) if bundle_json.is_file() else None,
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": total_bytes,
    }


def _report_reference(report_path: str | Path | None) -> dict[str, Any] | None:
    if report_path is None:
        return None
    path = Path(report_path)
    return {
        "path": str(path),
        "status": "available" if path.is_file() else "missing",
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _actual_providers(tts: Kitten2OnnxTTS) -> list[str]:
    sessions = (
        getattr(getattr(tts, "lm", None), "session", None),
        getattr(getattr(tts, "lm", None), "output_session", None),
        getattr(getattr(tts, "lm", None), "speaker_session", None),
        getattr(getattr(tts, "decoder", None), "flow_session", None),
        getattr(getattr(tts, "decoder", None), "mel2wav_session", None),
    )
    providers: list[str] = []
    for session in sessions:
        if session is not None and hasattr(session, "get_providers"):
            for provider in session.get_providers():
                if provider not in providers:
                    providers.append(provider)
    return providers


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)

def _first_difference(expected: list[int], actual: list[int]) -> int | None:
    for index, (expected_id, actual_id) in enumerate(zip(expected, actual)):
        if expected_id != actual_id:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None



def _waveform_stats(waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    waveform = np.asarray(waveform, dtype=np.float32)
    return {
        "finite": bool(np.all(np.isfinite(waveform))),
        "samples": int(waveform.size),
        "duration_seconds": waveform.size / sample_rate,
        "peak": float(np.max(np.abs(waveform))) if waveform.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        if waveform.size
        else 0.0,
    }


def _waveform_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {"same_length": False, "rmse": None, "correlation": None}
    difference = left - right
    correlation = None
    if left.size and np.std(left) > 0 and np.std(right) > 0:
        correlation = float(np.corrcoef(left, right)[0, 1])
    return {
        "same_length": True,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))) if left.size else 0.0,
        "correlation": correlation,
    }


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(waveform, dtype=np.float32), sample_rate)

def validate_corpus(
    model_dir: str | Path,
    bundle_dir: str | Path,
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    providers: list[str] | None = None,
    reference_device: str = "cpu",
    reference_dtype: torch.dtype = torch.float32,
    quality_report: bool = False,
    reference_bundle_dir: str | Path | None = None,
    weight_model_dir: str | Path | None = None,
    weight_variant: str = "full",
    speaker_projection_model_dir: str | Path | None = None,
    lm_parity_report_path: str | Path | None = None,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    voice = corpus["voice"]
    max_new_tokens = int(corpus["max_new_tokens"])
    seed = int(corpus["seed"])

    artifacts = load_reference_artifacts(
        model_dir,
        voice,
        weight_variant=weight_variant,
        weight_model_dir=weight_model_dir,
        speaker_projection_model_dir=speaker_projection_model_dir,
    )
    reference_model = load_reference_model(
        artifacts,
        device=reference_device,
        dtype=reference_dtype,
    )
    if providers is None:
        providers = ["CPUExecutionProvider"]
    tts = Kitten2OnnxTTS(bundle_dir, providers=providers)
    reference_tts = None
    if quality_report:
        if reference_bundle_dir is None:
            raise ValueError("quality reports require the original --reference-bundle")
        reference_tts = Kitten2OnnxTTS(
            reference_bundle_dir,
            providers=["CPUExecutionProvider"],
        )
        if reference_tts.sample_rate != tts.sample_rate:
            raise ValueError("candidate and original bundles use different sample rates")
    preset = corpus["preset"]
    sampling_settings = sampling_settings_from_config(tts.bundle, preset)
    report: dict[str, Any] = {
        "format": "kitten2-onnx-e2e-parity-report-v2",
        "validation_mode": "quality_report" if quality_report else "exact_parity",
        "strict_parity_tolerances": {
            "logits_rtol": 1e-4,
            "logits_atol": 2e-4,
            "greedy_token_ids": "exact",
            "seeded_waveform_repeat_rmse_max": 1e-7,
            "seeded_waveform_repeat_correlation_min": 0.999999,
        },
        "quality_review": (
            {"status": "requires_manual_review", "thresholds": None}
            if quality_report
            else None
        ),
        "reference_bundle": str(reference_bundle_dir) if reference_bundle_dir else None,
        "voice": voice,
        "source_model_revision": artifacts.model_revision,
        "weight_variant": getattr(artifacts, "weight_variant", weight_variant),
        "weight_model_dir": str(getattr(artifacts, "repo_dir", model_dir)),
        "source_weight_sha256": getattr(artifacts, "source_weight_sha256", None),
        "source_weight_files": getattr(artifacts, "source_weight_files", []),
        "voice_artifacts_provenance": {
            "path": str(getattr(artifacts, "voice_artifacts_dir", model_dir)),
            "source_model_revision": getattr(
                artifacts, "voice_artifacts_source_model_revision", None
            ),
        },
        "speaker_projection_provenance": {
            "source": getattr(artifacts, "speaker_projection_source", None),
            "weight_variant": getattr(artifacts, "speaker_projection_weight_variant", None),
            "source_model_revision": getattr(
                artifacts, "speaker_projection_source_model_revision", None
            ),
            "sha256": getattr(artifacts, "speaker_projection_source_sha256", None),
        },
        "onnx_bundle_provenance": _bundle_provenance(Path(bundle_dir)),
        "onnx_bundle_revisions": tts.bundle.get("revisions", {}),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "actual_providers": _actual_providers(tts),
        "lm_parity_report": _report_reference(lm_parity_report_path),
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "normalizer_version": importlib.metadata.version("kitten-text-processing"),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "requested_providers": providers,
        "reference_device": str(reference_model.device),
        "reference_dtype": str(reference_model.model.dtype).removeprefix("torch."),
        "runtime_profile": tts.bundle.get("export", {}).get("runtime_profile"),
        "sampling": f"seeded preset={preset}",
        "sample_rate": tts.sample_rate,
        "cases": [],
        "first_divergence": None,
        "listening_smoke_test": {
            "status": "not_performed",
            "reason": "This automated environment has no audio playback interface.",
        },
    }
    _, _, conditioning = tts._voice(voice)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for case_index, case in enumerate(corpus["cases"]):
        normalize = case.get("normalize", True)
        use_emotion = case.get("use_emotion")
        reference_logits: list[np.ndarray] = []
        onnx_logits: list[np.ndarray] = []
        expected_arrays, expected_metadata = capture_reference(
            artifacts,
            text=case["text"],
            normalize=normalize,
            use_emotion=use_emotion,
            max_new_tokens=max_new_tokens,
            reference_model=reference_model,
            sampling_settings=sampling_settings,
            seed=seed,
            logits_callback=lambda index, values, sink=reference_logits: sink.append(values),
        )
        result = tts.generate_with_metadata(
            case["text"],
            voice=voice,
            seed=seed,
            max_new_tokens=max_new_tokens,
            preset=preset,
            normalize=normalize,
            use_emotion=use_emotion,
            logits_callback=lambda index, values, sink=onnx_logits: sink.append(values),
        )

        if result.expression_detected != expected_metadata["expression_detected"]:
            raise AssertionError(f"{case['name']}: expression detection differs from PyTorch reference")
        if result.emotion_control_used != expected_metadata["emotion_control_used"]:
            raise AssertionError(f"{case['name']}: emotion conditioning differs from PyTorch reference")
        reference_result = None
        if quality_report:
            if reference_tts is None:
                raise AssertionError("quality reference bundle was not initialized")
            reference_result = reference_tts.generate_with_metadata(
                case["text"],
                voice=voice,
                seed=seed,
                max_new_tokens=max_new_tokens,
                preset=preset,
                normalize=normalize,
                use_emotion=use_emotion,
            )
            if reference_result.normalized_text != expected_metadata["processed_text"]:
                raise AssertionError(f"{case['name']}: original bundle normalized text differs")
            if reference_result.expression_detected != expected_metadata["expression_detected"]:
                raise AssertionError(f"{case['name']}: original bundle expression detection differs")
            if reference_result.emotion_control_used != expected_metadata["emotion_control_used"]:
                raise AssertionError(f"{case['name']}: original bundle emotion conditioning differs")
            if reference_result.prompt_ids != expected_arrays["prompt_ids"].tolist():
                raise AssertionError(f"{case['name']}: original bundle prompt IDs differ")
        expected_ids = expected_arrays["generated_raw_ids"].tolist()
        divergence = _first_difference(expected_ids, result.raw_ids)
        if result.normalized_text != expected_metadata["processed_text"]:
            raise AssertionError(
                f"{case['name']}: normalized text differs: {result.normalized_text!r}"
            )
        if result.prompt_ids != expected_arrays["prompt_ids"].tolist():
            raise AssertionError(f"{case['name']}: prompt token IDs differ from PyTorch reference")
        if divergence is not None:
            diagnostic_name = (
                f"{output_path.stem}-case-{case_index:03d}-first-divergence.npz"
                if quality_report
                else "first-divergence.npz"
            )
            diagnostic_path = output_path.with_name(diagnostic_name)
            reference_step_logits = (
                reference_logits[divergence]
                if divergence < len(reference_logits)
                else np.empty(0, dtype=np.float32)
            )
            onnx_step_logits = (
                onnx_logits[divergence]
                if divergence < len(onnx_logits)
                else np.empty(0, dtype=np.float32)
            )
            np.savez_compressed(
                diagnostic_path,
                reference_logits=reference_step_logits,
                onnx_logits=onnx_step_logits,
                reference_raw_ids=np.asarray(expected_ids, dtype=np.int64),
                onnx_raw_ids=np.asarray(result.raw_ids, dtype=np.int64),
                prompt_ids=np.asarray(result.prompt_ids, dtype=np.int64),
            )
            divergence_report = {
                "case": case["name"],
                "step": divergence,
                "kind": "token",
                "reference_token": expected_ids[divergence] if divergence < len(expected_ids) else None,
                "onnx_token": result.raw_ids[divergence] if divergence < len(result.raw_ids) else None,
                "logits_file": diagnostic_path.name,
            }
            if report["first_divergence"] is None:
                report["first_divergence"] = divergence_report
            if not quality_report:
                _write_json(output_path, report)
                raise AssertionError(f"{case['name']}: first token divergence at step {divergence}")
        if len(reference_logits) != len(onnx_logits) and not quality_report:
            raise AssertionError(f"{case['name']}: PyTorch and ONNX logit trace lengths differ")
        logit_step_count = min(len(reference_logits), len(onnx_logits))
        if quality_report and divergence is not None:
            logit_step_count = min(logit_step_count, divergence + 1)
        squared_error = 0.0
        element_count = 0
        max_abs_error = 0.0
        for index in range(logit_step_count):
            torch_values = reference_logits[index]
            onnx_values = onnx_logits[index]
            difference = torch_values.astype(np.float64) - onnx_values.astype(np.float64)
            max_abs_error = max(max_abs_error, float(np.max(np.abs(difference))))
            squared_error += float(np.sum(np.square(difference)))
            element_count += difference.size
            if not quality_report and not np.allclose(onnx_values, torch_values, rtol=1e-4, atol=2e-4):
                diagnostic_path = output_path.with_name("first-divergence.npz")
                np.savez_compressed(
                    diagnostic_path,
                    reference_logits=torch_values,
                    onnx_logits=onnx_values,
                    reference_raw_ids=np.asarray(expected_ids, dtype=np.int64),
                    onnx_raw_ids=np.asarray(result.raw_ids, dtype=np.int64),
                    prompt_ids=np.asarray(result.prompt_ids, dtype=np.int64),
                )
                report["first_divergence"] = {
                    "case": case["name"],
                    "step": index,
                    "kind": "logits",
                    "logits_file": diagnostic_path.name,
                }
                _write_json(output_path, report)
                raise AssertionError(
                    f"{case['name']}: logits exceed parity tolerance at step {index}"
                )

        repeated = tts.decoder.decode(result.s3_ids, conditioning, seed=seed)
        repeated_audio = np.asarray(repeated, dtype=np.float32)[0]
        difference = result.waveform.astype(np.float64) - repeated_audio.astype(np.float64)
        repeat_rmse = float(np.sqrt(np.mean(np.square(difference))))
        repeat_correlation = float(np.corrcoef(result.waveform, repeated_audio)[0, 1])
        if not quality_report and (repeat_rmse > 1e-7 or repeat_correlation < 0.999999):
            raise AssertionError(f"{case['name']}: seeded waveform repeat parity failed")

        repeat_generation = None
        quality_audio = None
        if quality_report:
            repeat_generation = tts.generate_with_metadata(
                case["text"],
                voice=voice,
                seed=seed,
                max_new_tokens=max_new_tokens,
                preset=preset,
                normalize=normalize,
                use_emotion=use_emotion,
            )
            audio_dir = output_path.parent / f"{output_path.stem}-audio"
            candidate_audio_path = audio_dir / f"case-{case_index:03d}-candidate.wav"
            original_audio_path = audio_dir / f"case-{case_index:03d}-original.wav"
            _write_wav(candidate_audio_path, result.waveform, tts.sample_rate)
            if reference_result is None:
                raise AssertionError("quality reference audio was not generated")
            _write_wav(original_audio_path, reference_result.waveform, reference_tts.sample_rate)
            quality_audio = {
                "candidate_wav": str(candidate_audio_path),
                "original_wav": str(original_audio_path),
                "candidate": _waveform_stats(result.waveform, tts.sample_rate),
                "original": _waveform_stats(reference_result.waveform, reference_tts.sample_rate),
                "waveform_comparison": _waveform_comparison(
                    result.waveform,
                    reference_result.waveform,
                ),
                "repeat_generation_raw_ids_match": result.raw_ids == repeat_generation.raw_ids,
                "repeat_generation_waveform": _waveform_comparison(
                    result.waveform,
                    repeat_generation.waveform,
                ),
            }

        waveform_stats = _waveform_stats(result.waveform, tts.sample_rate)
        peak = waveform_stats["peak"]
        rms = waveform_stats["rms"]
        duration = waveform_stats["duration_seconds"]
        raw_ids = result.raw_ids
        s3_ids = result.s3_ids
        expected_s3_ids = [
            token - artifacts.token_map.audio_id_base
            for token in expected_ids
            if artifacts.token_map.audio_id_base <= token < artifacts.token_map.audio_id_end
        ]
        s3_ids_match_reference = s3_ids == expected_s3_ids
        corpus_raw_ids_match = "reference_raw_ids" not in case or raw_ids == case["reference_raw_ids"]
        corpus_s3_ids_match = "reference_s3_ids" not in case or s3_ids == case["reference_s3_ids"]
        if not quality_report:
            if len(result.s3_ids) != case.get("reference_token_count", len(result.s3_ids)):
                raise AssertionError(f"{case['name']}: S3 token count differs from the corpus")
            if result.waveform.size != case.get("reference_waveform_samples", result.waveform.size):
                raise AssertionError(f"{case['name']}: waveform sample count differs from the corpus")
            if not np.isclose(duration, case.get("reference_duration_seconds", duration), atol=1e-8):
                raise AssertionError(f"{case['name']}: waveform duration differs from the corpus")
            if not np.isclose(peak, case.get("reference_peak", peak), rtol=1e-5, atol=1e-6):
                raise AssertionError(f"{case['name']}: waveform peak differs from the corpus")
            if not np.isclose(rms, case.get("reference_rms", rms), rtol=1e-5, atol=1e-7):
                raise AssertionError(f"{case['name']}: waveform RMS differs from the corpus")
            if not s3_ids_match_reference:
                raise AssertionError(f"{case['name']}: ONNX S3 IDs differ from PyTorch raw IDs")
            if not corpus_raw_ids_match:
                raise AssertionError(f"{case['name']}: raw IDs differ from the checked-in corpus")
            if not corpus_s3_ids_match:
                raise AssertionError(f"{case['name']}: S3 IDs differ from the checked-in corpus")
        if result.prompt_length != int(expected_metadata["prompt_length"]):
            raise AssertionError(f"{case['name']}: prompt length differs from PyTorch reference")
        report["cases"].append(
            {
                "name": case["name"],
                "input_text": case["text"],
                "normalize": normalize,
                "use_emotion": use_emotion,
                "normalized_text": result.normalized_text,
                "expression_detected": expected_metadata["expression_detected"],
                "emotion_control_used": expected_metadata["emotion_control_used"],
                "prompt_length": result.prompt_length,
                "prompt_ids": result.prompt_ids,
                "raw_ids": raw_ids,
                "s3_ids": s3_ids,
                "token_count": len(s3_ids),
                "waveform_samples": int(result.waveform.size),
                "duration_seconds": duration,
                "peak": peak,
                "rms": rms,
                "logits_max_abs_error": max_abs_error,
                "logits_rmse": float(np.sqrt(squared_error / element_count)) if element_count else 0.0,
                "seeded_waveform_repeat_rmse": repeat_rmse,
                "seeded_waveform_repeat_correlation": repeat_correlation,
                "token_divergence_step": divergence,
                "token_ids_match_reference": divergence is None,
                "logit_trace_length_match": len(reference_logits) == len(onnx_logits),
                "logits_compared_steps": logit_step_count,
                "waveform_finite": waveform_stats["finite"],
                "quality_audio": quality_audio,
            }
        )
        _write_json(output_path, report)

    report["resource_usage"] = {
        "peak_rss_bytes": _peak_rss_bytes(),
        "measurement": "process maximum resident set size",
        "guard_status": "not_configured",
    }
    _write_json(output_path, report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("tests/fixtures/e2e_corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("build/e2e-parity/report.json"))
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--reference-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--quality-report", action="store_true")
    parser.add_argument("--reference-bundle", type=Path)
    parser.add_argument("--weights", choices=("full", "packed", "emb4"), default="full")
    parser.add_argument("--weight-model", type=Path)
    parser.add_argument("--speaker-projection-model", type=Path)
    parser.add_argument("--lm-parity-report", type=Path)
    args = parser.parse_args()
    report = validate_corpus(
        args.model_dir,
        args.bundle_dir,
        args.corpus,
        args.output,
        providers=args.providers,
        reference_device=args.reference_device,
        reference_dtype={"float32": torch.float32, "float16": torch.float16}[args.reference_dtype],
        quality_report=args.quality_report,
        reference_bundle_dir=args.reference_bundle,
        weight_variant=args.weights,
        weight_model_dir=args.weight_model,
        speaker_projection_model_dir=args.speaker_projection_model,
        lm_parity_report_path=args.lm_parity_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
