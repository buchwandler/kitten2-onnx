from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from kitten2_onnx.decoder_reference import (
    DECODER_REFERENCE_FORMAT,
    load_decoder_reference,
    write_decoder_reference,
)


def _arrays() -> dict[str, np.ndarray]:
    flow_mel = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    return {
        "speaker_embedding": np.zeros((512,), dtype=np.float32),
        "reference_tokens": np.array([1, 2], dtype=np.int64),
        "conditioning_prompt_token": np.array([[1, 2]], dtype=np.int64),
        "conditioning_prompt_token_len": np.array([2], dtype=np.int64),
        "conditioning_prompt_feat": np.zeros((1, 4, 80), dtype=np.float32),
        "conditioning_embedding": np.zeros((1, 192), dtype=np.float32),
        "speech_tokens": np.array([[3, 4]], dtype=np.int64),
        "speech_token_lengths": np.array([2], dtype=np.int64),
        "flow_noise": np.zeros((1, 80, 4), dtype=np.float32),
        "flow_mel": flow_mel,
        "mel2wav_input": flow_mel.copy(),
        "vocoder_phase_noise": np.zeros((1, 9, 1), dtype=np.float32),
        "vocoder_sine_noise": np.zeros((1, 9, 1920), dtype=np.float32),
        "mel2wav_waveform": np.zeros((1, 1920), dtype=np.float32),
        "waveform": np.zeros((1, 1920), dtype=np.float32),
    }


def test_write_and_load_decoder_reference_records_array_contract(tmp_path):
    path = tmp_path / "Bruno.npz"
    arrays = _arrays()
    write_decoder_reference(path, arrays, {"format": DECODER_REFERENCE_FORMAT, "voice": "Bruno"})

    reference = load_decoder_reference(path)

    assert reference.metadata["voice"] == "Bruno"
    assert reference.metadata["arrays"]["conditioning_prompt_feat"] == {
        "shape": [1, 4, 80],
        "dtype": "float32",
    }
    for name, array in arrays.items():
        np.testing.assert_array_equal(reference.arrays[name], array)


def test_load_decoder_reference_rejects_shape_drift(tmp_path):
    path = tmp_path / "Bruno.npz"
    write_decoder_reference(path, _arrays(), {"format": DECODER_REFERENCE_FORMAT, "voice": "Bruno"})
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["arrays"]["flow_noise"]["shape"] = [1, 80, 5]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="shape mismatch"):
        load_decoder_reference(path)


def test_captured_bruno_decoder_reference_contract():
    value = os.environ.get("KITTEN2_DECODER_REFERENCE")
    if not value:
        pytest.skip("set KITTEN2_DECODER_REFERENCE to validate captured decoder tensors")
    reference = load_decoder_reference(Path(value))
    arrays = reference.arrays

    assert reference.metadata["voice"] == "Bruno"
    assert reference.metadata["reference_transcript"]
    assert arrays["speaker_embedding"].shape == (512,)
    assert arrays["reference_tokens"].ndim == 1
    assert reference.metadata["meanflow_steps"] == 2
    assert reference.metadata["s3_silence_tail"] == {
        "token_id": 4299,
        "count": 3,
        "mel_frames_per_token": 2,
    }
    assert arrays["speech_tokens"].shape[0] == 1
    assert arrays["speech_token_lengths"].tolist() == [arrays["speech_tokens"].shape[1]]
    assert arrays["flow_noise"].shape == (
        1,
        arrays["flow_mel"].shape[1],
        reference.metadata["s3_silence_tail"]["mel_frames_per_token"]
        * (
            arrays["conditioning_prompt_token_len"][0]
            + arrays["speech_token_lengths"][0]
            + reference.metadata["s3_silence_tail"]["count"]
        ),
    )
    assert arrays["mel2wav_input"].shape == arrays["flow_mel"].shape
    assert arrays["waveform"].shape[0] == 1
    assert arrays["mel2wav_waveform"].shape == arrays["waveform"].shape
    assert arrays["vocoder_phase_noise"].shape[0] == 1
    assert arrays["vocoder_phase_noise"].shape[2] == 1
    assert arrays["vocoder_sine_noise"].shape == (
        1,
        arrays["vocoder_phase_noise"].shape[1],
        arrays["mel2wav_waveform"].shape[1],
    )
    assert np.isfinite(arrays["mel2wav_waveform"]).all()
    assert np.isfinite(arrays["waveform"]).all()
    assert reference.metadata["sample_rate"] == 24000
