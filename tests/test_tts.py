from __future__ import annotations

import os

import numpy as np
import pytest

from kitten2_onnx.synthesize import main, validate_audio


def test_validate_audio_rejects_invalid_waveforms():
    with pytest.raises(ValueError, match="non-empty mono"):
        validate_audio(np.zeros((1, 0), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        validate_audio(np.asarray([np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="peak"):
        validate_audio(np.asarray([1.1], dtype=np.float32))


def test_synthesis_cli_generates_valid_builtin_voice_wav(tmp_path):
    bundle_value = os.environ.get("KITTEN2_TTS_BUNDLE")
    if not bundle_value:
        pytest.skip("set KITTEN2_TTS_BUNDLE to run built-in-voice ONNX synthesis")

    sf = pytest.importorskip("soundfile")
    output = tmp_path / "bruno.wav"
    main(
        [
            bundle_value,
            "Hello from Kitten.",
            "--voice",
            "Bruno",
            "--preset",
            "stable",
            "--seed",
            "7",
            "--max-new-tokens",
            "32",
            "--output",
            str(output),
        ]
    )

    audio, sample_rate = sf.read(output, dtype="float32")
    assert sample_rate == 24000
    assert audio.ndim == 1
    assert audio.size > 0
    assert np.isfinite(audio).all()
    assert 1e-5 <= np.max(np.abs(audio)) <= 1.0
    rms = np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
    assert 1e-6 <= rms <= 0.5
