"""Minimal ONNX Runtime driver for the exported KittenTTS 2 speech LM."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable

import numpy as np


Sampler = Callable[[np.ndarray], int]


def greedy_sampler(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


class Kitten2OnnxLM:
    """Drive the exported Qwen3 LM with explicit embeddings and KV cache.

    This class intentionally handles *only* LM token generation. It does not
    reproduce Kitten's custom logits processors or S3 waveform decoder yet.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        speaker_model: str | Path | None = None,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "lm.meta.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(self.model_dir / self.meta["embedding_file"], mmap_mode="r")
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.model_dir / "lm.onnx"),
            providers=providers,
        )
        self.speaker_session = (
            ort.InferenceSession(str(speaker_model), providers=providers)
            if speaker_model is not None
            else None
        )
        self.layers = int(self.meta["num_hidden_layers"])
        self.kv_heads = int(self.meta["num_key_value_heads"])
        self.head_dim = int(self.meta["head_dim"])

        # ONNX initial graph dtype can differ from embedding storage dtype (bf16
        # embeddings are stored as fp32). Infer the desired tensor type.
        first = self.session.get_inputs()[0]
        self.model_numpy_dtype = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
        }.get(first.type, np.float32)

    def _empty_cache(self) -> dict[str, np.ndarray]:
        shape = (1, self.kv_heads, 0, self.head_dim)
        return {
            name: np.zeros(shape, dtype=self.model_numpy_dtype)
            for i in range(self.layers)
            for name in (f"past_key_{i}", f"past_value_{i}")
        }

    def _project_speaker(self, speaker_embedding: np.ndarray) -> np.ndarray:
        if self.speaker_session is None:
            raise ValueError("speaker_model was not supplied")
        speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32)
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding[None, :]
        return self.speaker_session.run(
            ["speaker_hidden"],
            {"speaker_embedding": speaker_embedding},
        )[0]

    def _embed(
        self,
        token_ids: np.ndarray,
        *,
        speaker_embedding: np.ndarray | None = None,
        prefill: bool,
    ) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        values = np.asarray(self.embeddings[token_ids], dtype=self.model_numpy_dtype)
        if prefill and speaker_embedding is not None:
            speaker_hidden = self._project_speaker(speaker_embedding)
            values = values.copy()
            values[0, 0, :] = speaker_hidden[0].astype(values.dtype, copy=False)
        return values

    def step(
        self,
        token_ids: np.ndarray,
        *,
        past: dict[str, np.ndarray] | None = None,
        speaker_embedding: np.ndarray | None = None,
        position: int | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run prefill or one cached decode chunk.

        ``token_ids`` must have shape ``[1, S]``. If ``past`` is omitted this is
        treated as prefill and an empty cache is supplied.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("token_ids must have shape [1, S]")

        is_prefill = past is None
        past = self._empty_cache() if past is None else past
        past_len = next(iter(past.values())).shape[2]
        start = past_len if position is None else int(position)

        inputs = {
            "inputs_embeds": self._embed(
                token_ids,
                speaker_embedding=speaker_embedding,
                prefill=is_prefill,
            ),
            "attention_mask": np.ones((1, past_len + token_ids.shape[1]), dtype=np.int64),
            "position_ids": np.arange(
                start,
                start + token_ids.shape[1],
                dtype=np.int64,
            )[None, :],
            **past,
        }

        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        new_past: dict[str, np.ndarray] = {}
        for i in range(self.layers):
            new_past[f"past_key_{i}"] = outputs[1 + 2 * i]
            new_past[f"past_value_{i}"] = outputs[2 + 2 * i]
        return logits, new_past

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        speaker_embedding: np.ndarray | None = None,
        sampler: Sampler = greedy_sampler,
        stop_ids: set[int] | None = None,
    ) -> list[int]:
        """Greedy/custom-sampler LM loop.

        This does not yet include Kitten's temperature/top-k/top-p/min-p and
        repetition processors. Supply a custom ``sampler`` while experimenting.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids may not be empty")
        stop_ids = stop_ids or set()

        logits, past = self.step(
            np.asarray([prompt_ids], dtype=np.int64),
            speaker_embedding=speaker_embedding,
        )
        generated: list[int] = []

        for _ in range(int(max_new_tokens)):
            token = int(sampler(logits[0]))
            generated.append(token)
            if token in stop_ids:
                break
            logits, past = self.step(
                np.asarray([[token]], dtype=np.int64),
                past=past,
            )
        return generated
