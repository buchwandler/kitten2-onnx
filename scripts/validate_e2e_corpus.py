"""Validate a built-in-voice corpus against PyTorch and retain first-divergence logits."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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


def _first_difference(expected: list[int], actual: list[int]) -> int | None:
    for index, (expected_id, actual_id) in enumerate(zip(expected, actual)):
        if expected_id != actual_id:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def validate_corpus(
    model_dir: str | Path,
    bundle_dir: str | Path,
    corpus_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    voice = corpus["voice"]
    max_new_tokens = int(corpus["max_new_tokens"])
    seed = int(corpus["seed"])

    artifacts = load_reference_artifacts(model_dir, voice)
    reference_model = load_reference_model(artifacts, device="cpu", dtype=torch.float32)
    tts = Kitten2OnnxTTS(bundle_dir, providers=["CPUExecutionProvider"])
    preset = corpus["preset"]
    sampling_settings = sampling_settings_from_config(tts.bundle, preset)
    report: dict[str, Any] = {
        "format": "kitten2-onnx-e2e-parity-report-v1",
        "voice": voice,
        "source_model_revision": artifacts.model_revision,
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "normalizer_version": importlib.metadata.version("kitten-text-processing"),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
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
    for case in corpus["cases"]:
        reference_logits: list[np.ndarray] = []
        onnx_logits: list[np.ndarray] = []
        expected_arrays, expected_metadata = capture_reference(
            artifacts,
            text=case["normalized_text"],
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
            logits_callback=lambda index, values, sink=onnx_logits: sink.append(values),
        )
        expected_ids = expected_arrays["generated_raw_ids"].tolist()
        divergence = _first_difference(expected_ids, result.raw_ids)
        if result.normalized_text != case["normalized_text"]:
            raise AssertionError(
                f"{case['name']}: normalized text differs: {result.normalized_text!r}"
            )
        if result.prompt_ids != expected_arrays["prompt_ids"].tolist():
            raise AssertionError(f"{case['name']}: prompt token IDs differ from PyTorch reference")
        if divergence is not None:
            diagnostic_path = output_path.with_name("first-divergence.npz")
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
            report["first_divergence"] = {
                "case": case["name"],
                "step": divergence,
                "kind": "token",
                "reference_token": (
                    expected_ids[divergence] if divergence < len(expected_ids) else None
                ),
                "onnx_token": result.raw_ids[divergence] if divergence < len(result.raw_ids) else None,
                "logits_file": diagnostic_path.name,
            }
            _write_json(output_path, report)
            raise AssertionError(f"{case['name']}: first token divergence at step {divergence}")
        if len(reference_logits) != len(onnx_logits):
            raise AssertionError(f"{case['name']}: PyTorch and ONNX logit trace lengths differ")
        squared_error = 0.0
        element_count = 0
        max_abs_error = 0.0
        for index, (torch_values, onnx_values) in enumerate(
            zip(reference_logits, onnx_logits, strict=True)
        ):
            difference = torch_values.astype(np.float64) - onnx_values.astype(np.float64)
            max_abs_error = max(max_abs_error, float(np.max(np.abs(difference))))
            squared_error += float(np.sum(np.square(difference)))
            element_count += difference.size
            if not np.allclose(onnx_values, torch_values, rtol=1e-4, atol=2e-4):
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
        if repeat_rmse > 1e-7 or repeat_correlation < 0.999999:
            raise AssertionError(f"{case['name']}: seeded waveform repeat parity failed")

        peak = float(np.max(np.abs(result.waveform)))
        rms = float(np.sqrt(np.mean(np.square(result.waveform, dtype=np.float64))))
        duration = result.waveform.size / tts.sample_rate
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
        raw_ids = result.raw_ids
        s3_ids = result.s3_ids
        expected_s3_ids = [
            token - artifacts.token_map.audio_id_base
            for token in expected_ids
            if artifacts.token_map.audio_id_base <= token < artifacts.token_map.audio_id_end
        ]
        if s3_ids != expected_s3_ids:
            raise AssertionError(f"{case['name']}: ONNX S3 IDs differ from PyTorch raw IDs")
        if "reference_raw_ids" in case and raw_ids != case["reference_raw_ids"]:
            raise AssertionError(f"{case['name']}: raw IDs differ from the checked-in corpus")
        if "reference_s3_ids" in case and s3_ids != case["reference_s3_ids"]:
            raise AssertionError(f"{case['name']}: S3 IDs differ from the checked-in corpus")
        if result.prompt_length != int(expected_metadata["prompt_length"]):
            raise AssertionError(f"{case['name']}: prompt length differs from PyTorch reference")
        if result.prompt_length != case.get("reference_prompt_length", result.prompt_length):
            raise AssertionError(f"{case['name']}: prompt length differs from the checked-in corpus")

        report["cases"].append(
            {
                "name": case["name"],
                "input_text": case["text"],
                "normalized_text": result.normalized_text,
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
                "logits_rmse": float(np.sqrt(squared_error / element_count)),
                "seeded_waveform_repeat_rmse": repeat_rmse,
                "seeded_waveform_repeat_correlation": repeat_correlation,
            }
        )
        _write_json(output_path, report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("tests/fixtures/e2e_corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("build/e2e-parity/report.json"))
    args = parser.parse_args()
    report = validate_corpus(args.model_dir, args.bundle_dir, args.corpus, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
