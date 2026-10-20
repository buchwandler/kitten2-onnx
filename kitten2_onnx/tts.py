"""High-level built-in-voice text-to-speech over ONNX Runtime graphs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kitten2_onnx.decoder_reference import load_decoder_reference
from kitten2_onnx.prompt import build_generation_prompt, build_reference_prefix
from kitten2_onnx.runtime import GenerationResult, Kitten2OnnxDecoder, Kitten2OnnxLM
from kitten2_onnx.sampling import SamplingSettings, sampling_settings_from_config
from kitten2_onnx.text import prepare_generation_text
from kitten2_onnx.tokens import TokenMap


@dataclass(frozen=True)
class SynthesisResult:
    normalized_text: str
    expression_detected: bool
    emotion_control_used: bool
    prompt_length: int
    prompt_ids: list[int]
    raw_ids: list[int]
    s3_ids: list[int]
    waveform: np.ndarray


class Kitten2OnnxTTS:
    """Synthesize built-in voices using ONNX Runtime for all neural stages."""

    def __init__(self, bundle_dir: str | Path, *, providers: list[str] | None = None):
        from transformers import AutoTokenizer

        self.bundle_dir = Path(bundle_dir)
        self.bundle: dict[str, Any] = json.loads(
            (self.bundle_dir / "bundle.json").read_text(encoding="utf-8")
        )
        if self.bundle.get("format") not in {"kitten2-onnx-bundle-v1", "kitten2-onnx-bundle-v2"}:
            raise ValueError("Unsupported Kitten2 ONNX bundle format")
        self.token_map = TokenMap.from_config({"token_map": self.bundle["token_map"]})
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.bundle_dir / "tokenizer"), local_files_only=True
        )

        lm_model = self.bundle_dir / self.bundle["lm"]["model"]
        speaker_model = self.bundle_dir / self.bundle["speaker"]["model"]
        flow_model = self.bundle_dir / self.bundle["decoder"]["flow"]
        mel2wav_model = self.bundle_dir / self.bundle["decoder"]["mel2wav"]
        self.lm = Kitten2OnnxLM(
            lm_model.parent, speaker_model=speaker_model, providers=providers
        )
        self.decoder = Kitten2OnnxDecoder(
            flow_model.parent, mel2wav_model.parent, providers=providers
        )
        self.sample_rate = self.decoder.sample_rate
        self._voices: dict[str, tuple[np.ndarray, list[int], dict[str, np.ndarray]]] = {}

    @property
    def available_voices(self) -> list[str]:
        return list(self.bundle["voices"])

    @property
    def available_presets(self) -> list[str]:
        return list(self.bundle.get("decode_presets", {}))

    def _voice(self, name: str) -> tuple[np.ndarray, list[int], dict[str, np.ndarray]]:
        if name not in self.bundle["voices"]:
            available = ", ".join(sorted(self.bundle["voices"]))
            raise ValueError(f"Unknown voice {name!r}; available voices: {available}")
        if name not in self._voices:
            path = self.bundle_dir / self.bundle["voices"][name]
            reference = load_decoder_reference(path)
            arrays = reference.arrays
            transcript = reference.metadata["reference_transcript"]
            reference_text_ids = self.tokenizer.encode(
                transcript.strip(), add_special_tokens=False
            )
            reference_prefix = build_reference_prefix(
                self.token_map,
                reference_text_ids,
                arrays["reference_tokens"].tolist(),
            )
            conditioning = {
                key: arrays[key]
                for key in (
                    "conditioning_prompt_token",
                    "conditioning_prompt_token_len",
                    "conditioning_prompt_feat",
                    "conditioning_embedding",
                )
            }
            self._voices[name] = (
                np.asarray(arrays["speaker_embedding"], dtype=np.float32),
                reference_prefix,
                conditioning,
            )
        return self._voices[name]

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        preset: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 512,
        normalize: bool = True,
        use_emotion: bool | None = None,
    ) -> np.ndarray:
        """Generate a mono waveform from text and a packaged built-in voice."""
        return self.generate_with_metadata(
            text,
            voice=voice,
            preset=preset,
            seed=seed,
            max_new_tokens=max_new_tokens,
            normalize=normalize,
            use_emotion=use_emotion,
        ).waveform

    def generate_with_metadata(
        self,
        text: str,
        *,
        voice: str | None = None,
        preset: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 512,
        sampling_settings: SamplingSettings | None = None,
        logits_callback: Callable[[int, np.ndarray], None] | None = None,

        normalize: bool = True,
        use_emotion: bool | None = None,
    ) -> SynthesisResult:
        """Generate speech and retain normalized text, prompt, IDs, and optional logits."""
        prepared = prepare_generation_text(
            text,
            tokenizer=self.tokenizer,
            generation_config=self.bundle.get("generation", {}),
            normalize=normalize,
            use_emotion=use_emotion,
        )
        normalized_text = prepared.processed_text
        if not normalized_text:
            raise ValueError("text must not be empty after normalization")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        voice = voice or self.bundle.get("default_voice") or next(iter(self.bundle["voices"]))
        speaker_embedding, reference_prefix, conditioning = self._voice(voice)
        target_text_ids = self.tokenizer.encode(normalized_text, add_special_tokens=False)
        prompt_ids = build_generation_prompt(
            self.token_map,
            target_text_ids,
            reference_prefix=reference_prefix,
            emotion_ids=prepared.emotion_ids,
        )
        settings = sampling_settings or sampling_settings_from_config(self.bundle, preset)
        generation: GenerationResult = self.lm.generate(
            prompt_ids,
            token_map=self.token_map,
            max_new_tokens=max_new_tokens,
            speaker_embedding=speaker_embedding,
            settings=settings,
            seed=seed,
            logits_callback=logits_callback,
        )
        if not generation.s3_ids:
            raise RuntimeError("LM generation produced no valid S3 speech tokens")

        waveform = self.decoder.decode(generation.s3_ids, conditioning, seed=seed)
        if waveform is None:
            raise RuntimeError("S3 decoder produced no waveform")
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim != 1:
            raise ValueError(f"decoder waveform must be mono, found shape {audio.shape}")
        return SynthesisResult(
            normalized_text=normalized_text,
            expression_detected=prepared.expression_detected,
            emotion_control_used=prepared.emotion_control_used,
            prompt_length=len(prompt_ids),
            prompt_ids=prompt_ids,
            raw_ids=generation.raw_ids,
            s3_ids=generation.s3_ids,
            waveform=audio,
        )
