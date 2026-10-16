from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from kitten2_onnx.decoder_reference import load_decoder_reference
from kitten2_onnx.export_mel2wav import _metrics


def test_captured_mel2wav_onnx_matches_pytorch_reference():
    reference_path = os.environ.get("KITTEN2_DECODER_REFERENCE")
    graph_dir = os.environ.get("KITTEN2_MEL2WAV_ONNX")
    if not reference_path or not graph_dir:
        pytest.skip("set KITTEN2_DECODER_REFERENCE and KITTEN2_MEL2WAV_ONNX")
    ort = pytest.importorskip("onnxruntime")

    reference = load_decoder_reference(Path(reference_path))
    arrays = reference.arrays
    graph_dir = Path(graph_dir)
    metadata = json.loads((graph_dir / "mel2wav.meta.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(
        str(graph_dir / metadata["graph"]), providers=["CPUExecutionProvider"]
    )
    waveform = session.run(
        ["waveform"],
        {
            "mel": arrays["mel2wav_input"],
            "phase_noise": arrays["vocoder_phase_noise"],
            "sine_noise": arrays["vocoder_sine_noise"],
        },
    )[0]

    expected = arrays["mel2wav_waveform"]
    assert waveform.shape == expected.shape
    assert np.isfinite(waveform).all()
    metrics = _metrics(expected, waveform)
    assert metrics["correlation"] > metadata["correlation_threshold"]
    assert metrics["relative_rmse"] < metadata["relative_rmse_threshold"]
    assert all(sample["abs_error"] < 1e-4 for sample in metrics["sample_metrics"])
