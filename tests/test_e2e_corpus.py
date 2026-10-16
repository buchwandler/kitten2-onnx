from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import kitten_text_processing
import pytest

CORPUS_PATH = Path(__file__).parent / "fixtures" / "e2e_corpus.json"


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
