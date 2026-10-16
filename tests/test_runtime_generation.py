from __future__ import annotations

import numpy as np
import pytest

from kitten2_onnx.runtime import GenerationResult, Kitten2OnnxLM
from kitten2_onnx.sampling import SamplingSettings
from kitten2_onnx.tokens import TokenMap


def _token_map() -> TokenMap:
    return TokenMap(
        audio_id_base=5,
        num_audio_tokens=3,
        speech_start_id=3,
        speech_end_id=9,
        text_start_id=4,
        text_end_id=5,
        start_id=6,
        stop_id=10,
        final_seg_id=8,
    )


def _settings(**overrides) -> SamplingSettings:
    values = {
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "repetition_window": None,
        "token_run_penalty": 0.0,
        "token_run_grace": 10,
    }
    values.update(overrides)
    return SamplingSettings(**values)


def _only_token(vocab_size: int, token_id: int) -> np.ndarray:
    logits = np.full(vocab_size, -np.inf)
    logits[token_id] = 0.0
    return logits[None, :]


class ScriptedLM(Kitten2OnnxLM):
    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def step(self, token_ids, *, past=None, speaker_embedding=None, position=None):
        self.calls.append(
            {
                "token_ids": token_ids.copy(),
                "past": past,
                "speaker_embedding": speaker_embedding,
                "position": position,
            }
        )
        step = len(self.calls) - 1
        return self.logits[step], {"step": step}


@pytest.mark.parametrize("eos_id", [9, 10])
def test_generate_returns_raw_ids_and_zero_based_audio_ids_and_stops_at_eos(eos_id):
    token_map = _token_map()
    speaker_embedding = np.arange(512, dtype=np.float32)
    lm = ScriptedLM([_only_token(12, 7), _only_token(12, eos_id)])

    result = lm.generate(
        [1, 2],
        token_map=token_map,
        max_new_tokens=5,
        speaker_embedding=speaker_embedding,
        settings=_settings(),
        seed=7,
    )

    assert result == GenerationResult(raw_ids=[7, eos_id], s3_ids=[2])
    assert len(lm.calls) == 2
    assert lm.calls[0]["token_ids"].tolist() == [[1, 2]]
    assert lm.calls[0]["speaker_embedding"] is speaker_embedding
    assert lm.calls[1]["token_ids"].tolist() == [[7]]
    assert lm.calls[1]["speaker_embedding"] is None


def test_generate_only_converts_ids_inside_the_audio_range():
    token_map = _token_map()
    lm = ScriptedLM([_only_token(12, token_map.final_seg_id), _only_token(12, 9)])
    result = lm.generate(
        [1], token_map=token_map, max_new_tokens=3, settings=_settings(), seed=0
    )
    assert result.raw_ids == [token_map.final_seg_id, token_map.speech_end_id]
    assert result.s3_ids == []


def test_seeded_sampling_repeats_and_changes_with_seed():
    token_map = _token_map()
    first_logits = np.full((1, 12), -np.inf)
    first_logits[0, 5:7] = 0.0
    end_logits = _only_token(12, token_map.stop_id)
    logits = [first_logits, end_logits]

    first = ScriptedLM(logits).generate(
        [1], token_map=token_map, max_new_tokens=2, settings=_settings(top_k=0), seed=0
    )
    repeated = ScriptedLM(logits).generate(
        [1], token_map=token_map, max_new_tokens=2, settings=_settings(top_k=0), seed=0
    )
    different = ScriptedLM(logits).generate(
        [1], token_map=token_map, max_new_tokens=2, settings=_settings(top_k=0), seed=2
    )

    assert first == repeated
    assert first.raw_ids[0] != different.raw_ids[0]
    assert first.s3_ids == [first.raw_ids[0] - token_map.audio_id_base]
    assert first.raw_ids[-1] == token_map.stop_id


def test_zero_token_request_does_not_run_the_lm():
    lm = ScriptedLM([_only_token(12, 7)])
    result = lm.generate([1], token_map=_token_map(), max_new_tokens=0)
    assert result == GenerationResult(raw_ids=[], s3_ids=[])
    assert lm.calls == []


def test_generate_reports_raw_logits_with_step_indices():
    logits = [_only_token(12, 7), _only_token(12, 10)]
    captured = []
    lm = ScriptedLM(logits)

    lm.generate(
        [1],
        token_map=_token_map(),
        max_new_tokens=2,
        settings=_settings(),
        logits_callback=lambda index, values: captured.append((index, values)),
    )

    assert [index for index, _ in captured] == [0, 1]
    np.testing.assert_array_equal(captured[0][1], logits[0][0])
    np.testing.assert_array_equal(captured[1][1], logits[1][0])
