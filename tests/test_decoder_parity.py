from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kitten2_onnx.decoder_reference import load_decoder_reference
from kitten2_onnx.runtime import (
    S3GEN_SILENCE_TAIL_TOKENS,
    S3GEN_SILENCE_TOKEN,
    S3GEN_SPEECH_VOCAB_SIZE,
    Kitten2OnnxDecoder,
    append_s3_silence_tail,
    clean_s3_tokens,
)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([4, 8], [4, 8, S3GEN_SILENCE_TOKEN, S3GEN_SILENCE_TOKEN, S3GEN_SILENCE_TOKEN]),
        (
            np.asarray([[4, 8]], dtype=np.int64),
            np.asarray([[4, 8, S3GEN_SILENCE_TOKEN, S3GEN_SILENCE_TOKEN, S3GEN_SILENCE_TOKEN]]),
        ),
    ],
)
def test_append_s3_silence_tail_matches_upstream_contract(tokens, expected):
    original = np.asarray(tokens).copy()
    actual = append_s3_silence_tail(tokens)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape[-1] == len(tokens[-1] if np.asarray(tokens).ndim == 2 else tokens) + S3GEN_SILENCE_TAIL_TOKENS
    np.testing.assert_array_equal(tokens, original)


def test_append_s3_silence_tail_rejects_non_single_item_batches():
    with pytest.raises(ValueError, match="speech_tokens must have shape"):
        append_s3_silence_tail(np.zeros((2, 4), dtype=np.int64))


def test_clean_s3_tokens_drops_ids_outside_pinned_codec_vocabulary():
    assert S3GEN_SPEECH_VOCAB_SIZE == 6561
    actual = clean_s3_tokens([-1, 0, 6560, 6561, 8000])

    np.testing.assert_array_equal(actual, [0, 6560])


def test_decoder_returns_no_waveform_when_all_codec_ids_are_invalid():
    decoder = object.__new__(Kitten2OnnxDecoder)
    decoder.speech_vocab_size = S3GEN_SPEECH_VOCAB_SIZE

    assert decoder.decode([-1, 6561], {}) is None


def test_full_decoder_onnx_matches_fixed_pytorch_reference():
    reference_path = os.environ.get("KITTEN2_DECODER_REFERENCE")
    flow_dir = os.environ.get("KITTEN2_FLOW_ONNX")
    mel2wav_dir = os.environ.get("KITTEN2_MEL2WAV_ONNX")
    if not reference_path or not flow_dir or not mel2wav_dir:
        pytest.skip("set decoder reference, flow ONNX, and mel2wav ONNX environment variables")

    reference = load_decoder_reference(Path(reference_path))
    arrays = reference.arrays
    conditioning = {
        name: arrays[name]
        for name in (
            "conditioning_prompt_token",
            "conditioning_prompt_token_len",
            "conditioning_prompt_feat",
            "conditioning_embedding",
        )
    }
    decoder = Kitten2OnnxDecoder(flow_dir, mel2wav_dir, providers=["CPUExecutionProvider"])
    actual = decoder.decode(
        arrays["speech_tokens"],
        conditioning,
        flow_noise=arrays["flow_noise"],
        phase_noise=arrays["vocoder_phase_noise"],
        sine_noise=arrays["vocoder_sine_noise"],
    )
    expected = arrays["waveform"]

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    reference_rms = np.sqrt(np.mean(np.square(expected.astype(np.float64))))
    rmse = np.sqrt(np.mean(np.square(difference)))
    correlation = np.corrcoef(actual.reshape(-1), expected.reshape(-1))[0, 1]
    assert correlation > 0.999
    assert rmse / reference_rms < 1e-4
    sample_indices = (0, 1, 2, expected.shape[1] // 2, expected.shape[1] - 1)
    assert all(abs(difference[0, index]) < 1e-4 for index in sample_indices)
