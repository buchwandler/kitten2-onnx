"""Token IDs used by the KittenTTS 2 speech language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenMap:
    """Resolved text, audio, and prompt marker IDs for one checkpoint."""

    audio_id_base: int
    num_audio_tokens: int
    speech_start_id: int
    speech_end_id: int
    text_start_id: int
    text_end_id: int
    start_id: int
    stop_id: int
    final_seg_id: int = 0
    reference_speech_start_id: int = 0
    reference_speech_end_id: int = 0
    reference_text_start_id: int = 0
    reference_text_end_id: int = 0

    @property
    def audio_id_end(self) -> int:
        """One past the final S3 audio token ID."""
        return self.audio_id_base + self.num_audio_tokens

    @property
    def uses_reference_role_tokens(self) -> bool:
        """Whether this checkpoint has dedicated reference-prefix markers."""
        return bool(self.reference_text_start_id)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TokenMap:
        """Build from the ``token_map`` block of a KittenTTS 2 config."""
        values = config["token_map"]
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown token_map keys: {sorted(unknown)}")
        return cls(**values)
