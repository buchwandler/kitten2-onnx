"""Prompt assembly for reference-conditioned KittenTTS 2 generation."""

from __future__ import annotations

from collections.abc import Sequence

from kitten2_onnx.tokens import TokenMap


def build_reference_prefix(
    token_map: TokenMap,
    reference_text_ids: Sequence[int],
    reference_audio_tokens: Sequence[int],
) -> list[int]:
    """Build the in-context reference transcript and 0-based S3 token prefix."""
    audio_ids = [token_map.audio_id_base + token for token in reference_audio_tokens]
    if token_map.uses_reference_role_tokens:
        return [
            token_map.start_id,
            token_map.reference_text_start_id,
            *reference_text_ids,
            token_map.reference_text_end_id,
            token_map.reference_speech_start_id,
            *audio_ids,
            token_map.reference_speech_end_id,
        ]
    return [
        token_map.start_id,
        token_map.text_start_id,
        *reference_text_ids,
        token_map.speech_start_id,
        *audio_ids,
        token_map.speech_end_id,
    ]


def build_generation_prompt(
    token_map: TokenMap,
    target_text_ids: Sequence[int],
    reference_prefix: Sequence[int] | None = None,
    emotion_ids: Sequence[int] | None = None,
) -> list[int]:
    """Build the target segment, optionally preceded by a reference prefix."""
    head = list(reference_prefix) if reference_prefix else [token_map.start_id]
    final_marker = [token_map.final_seg_id] if token_map.final_seg_id else []
    return [
        *head,
        *(emotion_ids or []),
        token_map.text_start_id,
        *target_text_ids,
        *final_marker,
        token_map.speech_start_id,
    ]
