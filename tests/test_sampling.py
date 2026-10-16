from __future__ import annotations

import numpy as np
import pytest

from kitten2_onnx.sampling import (
    SamplingSettings,
    is_eos_token,
    process_logits,
    sample,
    sampling_settings_from_config,
)
from kitten2_onnx.tokens import TokenMap


def _token_map(
    *, audio_id_base: int = 10, num_audio_tokens: int = 8, speech_end_id: int = 1, stop_id: int = 2
) -> TokenMap:
    return TokenMap(
        audio_id_base=audio_id_base,
        num_audio_tokens=num_audio_tokens,
        speech_start_id=3,
        speech_end_id=speech_end_id,
        text_start_id=4,
        text_end_id=5,
        start_id=6,
        stop_id=stop_id,
    )


def _settings(**overrides) -> SamplingSettings:
    values = {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "repetition_window": None,
        "token_run_penalty": 0.0,
        "token_run_grace": 3,
    }
    values.update(overrides)
    return SamplingSettings(**values)


def test_repetition_penalty_handles_positive_and_negative_logits():
    scores = np.asarray([2.0, -2.0, 0.5, -0.5, 3.0])
    processed = process_logits(
        scores, [0, 1, 2, 3], _token_map(speech_end_id=18, stop_id=19), _settings(repetition_penalty=2.0)
    )
    np.testing.assert_array_equal(processed, [1.0, -4.0, 0.25, -1.0, 3.0])


def test_repetition_penalty_exempts_silence_speech_end_and_stop():
    token_map = _token_map(audio_id_base=0, num_audio_tokens=5000, speech_end_id=5000, stop_id=5001)
    scores = np.ones(5002)
    history = [0, 4299, token_map.speech_end_id, token_map.stop_id]
    processed = process_logits(
        scores, history, token_map, _settings(repetition_penalty=2.0)
    )
    assert processed[0] == 0.5
    assert processed[4299] == 1.0
    assert processed[token_map.speech_end_id] == 1.0
    assert processed[token_map.stop_id] == 1.0


def test_finite_repetition_window_excludes_prompt_positions():
    scores = np.full(12, 2.0)
    history = [8, 9, 8, 9, 4, 5]
    processed = process_logits(
        scores,
        history,
        _token_map(),
        _settings(repetition_penalty=2.0, repetition_window=2),
        prompt_length=4,
    )
    assert processed[8] == 2.0
    assert processed[9] == 2.0
    assert processed[4] == 1.0
    assert processed[5] == 1.0


def test_token_run_penalty_has_grace_and_escalates():
    scores = np.zeros(20)
    token_map = _token_map()
    settings = _settings(token_run_penalty=1.3, token_run_grace=3)
    np.testing.assert_array_equal(process_logits(scores, [10, 10], token_map, settings), scores)
    assert process_logits(scores, [10, 10, 10], token_map, settings)[10] == pytest.approx(-1.3)
    assert process_logits(scores, [10, 10, 10, 10], token_map, settings)[10] == pytest.approx(-2.6)
    assert process_logits(scores, [3, 3, 3], token_map, settings)[3] == 0.0


def test_token_run_penalty_precedes_repetition_penalty():
    token_map = _token_map()
    settings = _settings(
        repetition_penalty=2.0,
        token_run_penalty=1.0,
        token_run_grace=2,
    )
    scores = np.zeros(20)
    scores[10] = 4.0
    processed = process_logits(scores, [10, 10, 10], token_map, settings)
    assert processed[10] == pytest.approx(1.0)


def test_temperature_scales_logits_before_filtering():
    processed = process_logits(
        np.asarray([4.0, 2.0, 0.0]), [], _token_map(), _settings(temperature=2.0)
    )
    np.testing.assert_array_equal(processed, [2.0, 1.0, 0.0])


def test_top_k_keeps_k_logits_and_ties():
    processed = process_logits(
        np.asarray([1.0, 4.0, 2.0, 4.0, -1.0]),
        [],
        _token_map(),
        _settings(top_k=2),
    )
    np.testing.assert_array_equal(processed, [-np.inf, 4.0, -np.inf, 4.0, -np.inf])


def test_top_p_keeps_smallest_high_probability_prefix():
    processed = process_logits(
        np.asarray([3.0, 2.0, 1.0]), [], _token_map(), _settings(top_p=0.8)
    )
    assert np.isfinite(processed[:2]).all()
    assert processed[2] == -np.inf


def test_min_p_filters_relative_to_the_highest_probability():
    processed = process_logits(
        np.asarray([3.0, 2.0, 1.0]), [], _token_map(), _settings(min_p=0.2)
    )
    assert np.isfinite(processed[:2]).all()
    assert processed[2] == -np.inf


def test_controlled_uniform_values_select_cdf_intervals():
    logits = np.zeros(3)
    token_map = _token_map()
    settings = _settings()
    assert sample(logits, [], token_map, settings, 0.1) == 0
    assert sample(logits, [], token_map, settings, 0.34) == 1
    assert sample(logits, [], token_map, settings, 0.99) == 2


def test_sampling_rejects_uniform_outside_unit_interval():
    with pytest.raises(ValueError, match=r"u must be in \[0, 1\)"):
        sample(np.zeros(2), [], _token_map(), _settings(), 1.0)


def test_eos_tokens_are_speech_end_and_stop_only():
    token_map = _token_map(speech_end_id=7, stop_id=8)
    assert is_eos_token(7, token_map)
    assert is_eos_token(8, token_map)
    assert not is_eos_token(10, token_map)


def test_settings_read_stable_and_expressive_presets():
    config = {
        "default_preset": "stable",
        "generation": {"repetition_window": 50},
        "decode_presets": {
            "stable": {"temperature": 0.8, "top_k": 50, "top_p": 0.8, "min_p": 0.0},
            "expressive": {"temperature": 0.9, "top_k": 50, "top_p": 0.9, "min_p": 0.0},
        },
    }
    stable = sampling_settings_from_config(config)
    expressive = sampling_settings_from_config(config, "expressive")
    assert stable == SamplingSettings()
    assert expressive.temperature == 0.9
    assert expressive.top_k == 50
    assert expressive.top_p == 0.9
    assert expressive.repetition_window == 50



def test_standard_warpers_match_pinned_transformers_order():
    torch = pytest.importorskip("torch")
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    logits = np.asarray([4.0, 2.5, 1.0, 0.25, -1.0], dtype=np.float32)
    scores = torch.from_numpy(logits.copy())[None, :]
    processors = (
        TemperatureLogitsWarper(0.7),
        TopKLogitsWarper(4),
        TopPLogitsWarper(0.8),
        MinPLogitsWarper(0.1),
    )
    for processor in processors:
        scores = processor(torch.tensor([[1, 2]]), scores)
    actual = process_logits(
        logits,
        [],
        _token_map(),
        _settings(temperature=0.7, top_k=4, top_p=0.8, min_p=0.1),
    )
    expected = scores.numpy()[0]
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_allclose(actual[np.isfinite(actual)], expected[np.isfinite(expected)], rtol=1e-6, atol=1e-6)
