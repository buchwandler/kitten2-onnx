"""KittenTTS 2 decode-time logit processing and controlled sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from kitten2_onnx.tokens import TokenMap

SILENCE_TOKEN = 4299


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.8
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    repetition_window: int | None = 50
    token_run_penalty: float = 1.3
    token_run_grace: int = 10

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between zero and one")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be between zero and one")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero")
        if self.repetition_window is not None and self.repetition_window < 0:
            raise ValueError("repetition_window must be non-negative")
        if self.token_run_penalty < 0:
            raise ValueError("token_run_penalty must be non-negative")
        if self.token_run_grace < 1:
            raise ValueError("token_run_grace must be positive")


def sampling_settings_from_config(
    config: Mapping[str, object], preset: str | None = None
) -> SamplingSettings:
    """Resolve the checkpoint's decode preset and generation repetition window."""
    preset_name = preset or config.get("default_preset")
    presets = config.get("decode_presets")
    if not isinstance(preset_name, str):
        raise TypeError("default_preset must be a string")
    if not isinstance(presets, Mapping):
        raise TypeError("model config decode_presets must be an object")
    values = presets.get(preset_name)
    if values is None:
        raise ValueError(f"unknown decode preset {preset_name!r}")
    if not isinstance(values, Mapping):
        raise TypeError(f"decode preset {preset_name!r} must be an object")
    generation = config.get("generation", {})
    if not isinstance(generation, Mapping):
        raise TypeError("model config generation settings must be an object")
    return SamplingSettings(
        temperature=float(values["temperature"]),
        top_k=int(values["top_k"]),
        top_p=float(values["top_p"]),
        min_p=float(values.get("min_p", 0.0)),
        repetition_window=int(generation.get("repetition_window", 50)),
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("logits must not contain NaN or positive infinity")
    maximum = np.max(scores)
    if not np.isfinite(maximum):
        raise ValueError("at least one logit must remain unfiltered")
    probabilities = np.exp(scores - maximum)
    probabilities /= probabilities.sum()
    return probabilities


def _apply_token_run_penalty(
    scores: np.ndarray,
    history: np.ndarray,
    token_map: TokenMap,
    settings: SamplingSettings,
) -> None:
    if not history.size or not settings.token_run_penalty:
        return
    token_id = int(history[-1])
    if not token_map.audio_id_base <= token_id < token_map.audio_id_end:
        return
    run_length = 1
    for previous_id in history[-2::-1]:
        if previous_id != token_id:
            break
        run_length += 1
    if run_length >= settings.token_run_grace:
        escalation = run_length - settings.token_run_grace + 1
        scores[token_id] -= escalation * settings.token_run_penalty


def _apply_repetition_penalty(
    scores: np.ndarray,
    history: np.ndarray,
    token_map: TokenMap,
    settings: SamplingSettings,
    prompt_length: int,
) -> None:
    if not history.size or settings.repetition_penalty == 1.0:
        return
    if settings.repetition_window:
        start = max(prompt_length, history.size - settings.repetition_window)
        history = history[start:]
    exempt = {
        token_map.speech_end_id,
        token_map.stop_id,
        token_map.audio_id_base + SILENCE_TOKEN,
    }
    for token_id in {int(value) for value in history}:
        if token_id in exempt:
            continue
        if token_id < 0 or token_id >= scores.size:
            raise ValueError(f"history token {token_id} is outside the logits vocabulary")
        score = scores[token_id]
        scores[token_id] = (
            score * settings.repetition_penalty
            if score < 0
            else score / settings.repetition_penalty
        )


def _apply_top_k(scores: np.ndarray, top_k: int) -> None:
    if not top_k or top_k >= scores.size:
        return
    threshold = np.partition(scores, -top_k)[-top_k]
    scores[scores < threshold] = -np.inf


def _apply_top_p(scores: np.ndarray, top_p: float) -> None:
    if top_p == 1.0:
        return
    sorted_indices = np.argsort(scores, kind="stable")
    sorted_scores = scores[sorted_indices]
    cumulative_probabilities = np.cumsum(_softmax(sorted_scores))
    remove = cumulative_probabilities <= 1.0 - top_p
    remove[-1] = False
    scores[sorted_indices[remove]] = -np.inf


def _apply_min_p(scores: np.ndarray, min_p: float) -> None:
    if min_p == 0.0:
        return
    probabilities = _softmax(scores)
    threshold = min_p * probabilities.max()
    scores[probabilities < threshold] = -np.inf


def process_logits(
    logits: np.ndarray,
    history: Sequence[int],
    token_map: TokenMap,
    settings: SamplingSettings,
    *,
    prompt_length: int = 0,
) -> np.ndarray:
    """Apply Kitten's ordered token-run, repetition, temperature, and truncation processors."""
    scores = np.asarray(logits, dtype=np.float64).copy()
    if scores.ndim != 1 or not scores.size:
        raise ValueError("logits must be a non-empty one-dimensional array")
    history_ids = np.asarray(history, dtype=np.int64).reshape(-1)
    if prompt_length < 0 or prompt_length > history_ids.size:
        raise ValueError("prompt_length must be within the history")

    _apply_token_run_penalty(scores, history_ids, token_map, settings)
    _apply_repetition_penalty(scores, history_ids, token_map, settings, prompt_length)
    if settings.temperature != 1.0:
        scores /= settings.temperature
    _apply_top_k(scores, settings.top_k)
    _apply_top_p(scores, settings.top_p)
    _apply_min_p(scores, settings.min_p)
    return scores


def sample(
    logits: np.ndarray,
    history: Sequence[int],
    token_map: TokenMap,
    settings: SamplingSettings,
    u: float,
    *,
    prompt_length: int = 0,
) -> int:
    """Sample one token using a controlled uniform variate in ``[0, 1)``."""
    if not 0.0 <= u < 1.0:
        raise ValueError("u must be in [0, 1)")
    probabilities = _softmax(
        process_logits(
            logits,
            history,
            token_map,
            settings,
            prompt_length=prompt_length,
        )
    )
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, u, side="right"))


def is_eos_token(token_id: int, token_map: TokenMap) -> bool:
    """Return whether a generated LM token terminates Kitten speech generation."""
    return int(token_id) in {token_map.speech_end_id, token_map.stop_id}
