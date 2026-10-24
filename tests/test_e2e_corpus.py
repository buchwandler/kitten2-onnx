from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import kitten_text_processing
import pytest

from kitten2_onnx.text import has_expression_tag

CORPUS_PATH = Path(__file__).parent / "fixtures" / "e2e_corpus.json"


def test_bundle_provenance_hashes_the_bundle_and_its_files(tmp_path):
    import scripts.validate_e2e_corpus as validator

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    bundle_json = bundle_dir / "bundle.json"
    bundle_json.write_text('{"format": "test"}', encoding="utf-8")
    (bundle_dir / "lm.onnx").write_bytes(b"onnx-data")

    provenance = validator._bundle_provenance(bundle_dir)

    assert provenance["status"] == "available"
    assert provenance["file_count"] == 2
    assert provenance["bytes"] == bundle_json.stat().st_size + (bundle_dir / "lm.onnx").stat().st_size
    assert provenance["bundle_json_sha256"] == hashlib.sha256(bundle_json.read_bytes()).hexdigest()
    assert len(provenance["tree_sha256"]) == 64


def test_corpus_normalized_text_matches_pinned_normalizer():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for case in corpus["cases"]:
        actual = kitten_text_processing.normalize_text(case["text"]).strip()
        assert actual == case["normalized_text"], case["name"]


def test_real_e2e_corpus_has_exact_pytorch_onnx_token_parity(tmp_path):
    model_dir = os.environ.get("KITTEN2_MODEL_DIR")
    bundle_dir = os.environ.get("KITTEN2_TTS_BUNDLE")
    if not model_dir or not bundle_dir:
        pytest.skip("set KITTEN2_MODEL_DIR and KITTEN2_TTS_BUNDLE for the real e2e corpus")

    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validate_e2e_corpus",
            model_dir,
            bundle_dir,
            "--corpus",
            str(CORPUS_PATH),
            "--output",
            str(report_path),
        ],
        cwd=Path(__file__).parent.parent,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["first_divergence"] is None
    assert len(report["cases"]) == 5
    assert all(case["waveform_samples"] > 0 for case in report["cases"])



def test_validate_corpus_defaults_and_explicit_reference_settings(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    import scripts.validate_e2e_corpus as validator

    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {"voice": "voice", "max_new_tokens": 1, "seed": 1, "preset": "default", "cases": []}
        ),
        encoding="utf-8",
    )
    seen = {"reference_calls": []}

    class FakeTTS:
        def __init__(self, bundle_dir, *, providers):
            seen["providers"] = providers
            profile = "cuda-fp16" if "CUDAExecutionProvider" in providers else "dev-fp32"
            self.bundle = {"export": {"runtime_profile": profile}}
            self.sample_rate = 24000

        def _voice(self, name):
            return None, None, {}

    def fake_load_reference_model(artifacts, *, device, dtype):
        seen["reference_calls"].append((device, dtype))
        return SimpleNamespace(device=torch.device(device), model=SimpleNamespace(dtype=dtype))

    monkeypatch.setattr(
        validator,
        "load_reference_artifacts",
        lambda *args, **kwargs: SimpleNamespace(model_revision="revision"),
    )
    monkeypatch.setattr(validator, "load_reference_model", fake_load_reference_model)
    monkeypatch.setattr(validator, "Kitten2OnnxTTS", FakeTTS)
    monkeypatch.setattr(validator, "sampling_settings_from_config", lambda *args: object())

    default_report = validator.validate_corpus(
        "model", "bundle", corpus_path, tmp_path / "default-report.json"
    )
    assert seen["providers"] == ["CPUExecutionProvider"]
    assert seen["reference_calls"][-1] == ("cpu", torch.float32)
    assert default_report["reference_device"] == "cpu"
    assert default_report["reference_dtype"] == "float32"
    assert default_report["runtime_profile"] == "dev-fp32"
    assert default_report["weight_variant"] == "full"
    assert default_report["strict_parity_tolerances"]["logits_atol"] == 2e-4
    assert default_report["onnx_bundle_provenance"]["status"] == "unavailable"
    assert default_report["actual_providers"] == []
    assert default_report["resource_usage"]["peak_rss_bytes"] is not None

    fp16_report = validator.validate_corpus(
        "model",
        "bundle",
        corpus_path,
        tmp_path / "fp16-report.json",
        providers=["CUDAExecutionProvider"],
        reference_device="cuda",
        reference_dtype=torch.float16,
    )
    assert seen["reference_calls"][-1] == ("cuda", torch.float16)
    assert fp16_report["requested_providers"] == ["CUDAExecutionProvider"]
    assert fp16_report["reference_dtype"] == "float16"
    assert fp16_report["runtime_profile"] == "cuda-fp16"

def test_cli_preserves_repeated_provider_arguments_and_reference_settings(monkeypatch, capsys):
    import torch

    import scripts.validate_e2e_corpus as validator

    seen = {}

    def fake_validate(*args, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(validator, "validate_corpus", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_e2e_corpus",
            "model",
            "bundle",
            "--provider",
            "CUDAExecutionProvider",
            "--provider",
            "CPUExecutionProvider",
            "--reference-device",
            "cuda",
            "--reference-dtype",
            "float16",
            "--weights",
            "emb4",
            "--weight-model",
            "variant-model",
            "--speaker-projection-model",
            "full-model",
            "--lm-parity-report",
            "lm-parity.json",
        ],
    )

    validator.main()
    capsys.readouterr()

    assert seen["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert seen["reference_device"] == "cuda"
    assert seen["reference_dtype"] is torch.float16
    assert seen["weight_variant"] == "emb4"
    assert seen["weight_model_dir"] == Path("variant-model")
    assert seen["speaker_projection_model_dir"] == Path("full-model")
    assert seen["lm_parity_report_path"] == Path("lm-parity.json")



def test_quality_report_records_token_divergence_without_relaxing_strict_mode(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import numpy as np
    import torch

    import scripts.validate_e2e_corpus as validator

    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "voice": "voice",
                "max_new_tokens": 2,
                "seed": 1,
                "preset": "default",
                "cases": [
                    {
                        "name": "sample",
                        "text": "Hello",
                        "normalized_text": "Hello",
                        "normalize": False,
                        "use_emotion": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts = SimpleNamespace(
        model_revision="revision",
        token_map=SimpleNamespace(audio_id_base=100, audio_id_end=200),
    )

    preprocessing_calls = []
    class FakeTTS:
        def __init__(self, bundle_dir, *, providers):
            self.is_original = Path(bundle_dir).name == "original"
            self.bundle = {"export": {"runtime_profile": "dev-fp32"}}
            self.sample_rate = 24000
            self.decoder = SimpleNamespace(
                decode=lambda *args, **kwargs: np.asarray([[0.1, 0.2]], dtype=np.float32)
            )

        def _voice(self, name):
            return None, None, {}

        def generate_with_metadata(
            self,
            text,
            *,
            voice,
            seed,
            max_new_tokens,
            preset,
            normalize,
            use_emotion,
            logits_callback=None,
        ):
            preprocessing_calls.append(
                ("original" if self.is_original else "candidate", normalize, use_emotion)
            )
            raw_ids = [100, 101] if self.is_original else [100, 102]
            logits = (
                [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])]
                if self.is_original
                else [np.asarray([1.0, 0.0]), np.asarray([0.8, 0.2])]
            )
            if logits_callback is not None:
                for index, values in enumerate(logits):
                    logits_callback(index, values)
            waveform = np.asarray([0.1, 0.3] if self.is_original else [0.1, 0.2], dtype=np.float32)
            return SimpleNamespace(
                normalized_text="Hello",
                expression_detected=False,
                emotion_control_used=False,
                prompt_ids=[1, 2],
                raw_ids=raw_ids,
                s3_ids=[token - 100 for token in raw_ids],
                waveform=waveform,
                prompt_length=2,
            )

    def fake_capture_reference(*args, logits_callback, **kwargs):
        preprocessing_calls.append(
            ("reference", kwargs["normalize"], kwargs["use_emotion"])
        )
        for index, values in enumerate((np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))):
            logits_callback(index, values)
        return (
            {"generated_raw_ids": np.asarray([100, 101]), "prompt_ids": np.asarray([1, 2])},
            {"prompt_length": 2, "processed_text": "Hello", "expression_detected": False, "emotion_control_used": False},
        )

    monkeypatch.setattr(validator, "load_reference_artifacts", lambda *args, **kwargs: artifacts)
    monkeypatch.setattr(
        validator,
        "load_reference_model",
        lambda artifacts, *, device, dtype: SimpleNamespace(
            device=torch.device(device), model=SimpleNamespace(dtype=dtype)
        ),
    )
    monkeypatch.setattr(validator, "Kitten2OnnxTTS", FakeTTS)
    monkeypatch.setattr(validator, "capture_reference", fake_capture_reference)
    monkeypatch.setattr(validator, "sampling_settings_from_config", lambda *args: object())

    report = validator.validate_corpus(
        "model",
        "candidate",
        corpus_path,
        tmp_path / "quality.json",
        quality_report=True,
        reference_bundle_dir="original",
    )

    case_report = report["cases"][0]
    assert report["validation_mode"] == "quality_report"
    assert report["quality_review"]["status"] == "requires_manual_review"
    assert report["first_divergence"]["step"] == 1
    assert preprocessing_calls == [
        ("reference", False, False),
        ("candidate", False, False),
        ("original", False, False),
        ("candidate", False, False),
    ]
    assert case_report["token_ids_match_reference"] is False
    assert case_report["logits_compared_steps"] == 2
    assert case_report["waveform_finite"] is True
    assert Path(case_report["quality_audio"]["candidate_wav"]).is_file()
    assert Path(case_report["quality_audio"]["original_wav"]).is_file()

    with pytest.raises(AssertionError, match="first token divergence at step 1"):
        validator.validate_corpus(
            "model",
            "candidate",
            corpus_path,
            tmp_path / "strict.json",
        )



def test_expression_corpus_uses_recognized_controls():
    expression_path = Path(__file__).parent / "fixtures" / "expression_cases.json"
    expression_corpus = json.loads(expression_path.read_text(encoding="utf-8"))

    assert len(expression_corpus["cases"]) == 3
    assert all(has_expression_tag(case["text"]) for case in expression_corpus["cases"])



def test_multilingual_corpus_disables_normalization_for_each_language():
    multilingual_path = Path(__file__).parent / "fixtures" / "multilingual_cases.json"
    multilingual_corpus = json.loads(multilingual_path.read_text(encoding="utf-8"))

    assert {case["name"] for case in multilingual_corpus["cases"]} == {
        "german",
        "chinese",
        "arabic",
        "hindi",
    }
    assert all(case["normalize"] is False for case in multilingual_corpus["cases"])
