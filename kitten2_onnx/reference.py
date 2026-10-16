"""Pinned PyTorch reference loading and built-in voice artifacts for validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from kitten2_onnx.prompt import build_generation_prompt, build_reference_prefix
from kitten2_onnx.tokens import TokenMap

UPSTREAM_CODE_REVISION = "36ef3090ca1e59122a81fc23be4a659a502b9d55"


@dataclass
class ReferenceArtifacts:
    repo_dir: Path
    config: dict[str, Any]
    model_revision: str | None
    token_map: TokenMap
    tokenizer: Any
    voice: str
    transcript: str
    speaker_embedding: np.ndarray
    reference_tokens: np.ndarray
    reference_prefix: list[int]

    def build_prompt(self, text: str, emotion_ids: list[int] | None = None) -> list[int]:
        target_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return build_generation_prompt(
            self.token_map,
            target_ids,
            reference_prefix=self.reference_prefix,
            emotion_ids=emotion_ids,
        )


@dataclass
class ReferenceModel:
    artifacts: ReferenceArtifacts
    model: nn.Module
    speaker_projection: nn.Module
    device: torch.device

    @property
    def input_embedding_table(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    def project_speaker(self) -> torch.Tensor:
        embedding = torch.as_tensor(
            self.artifacts.speaker_embedding,
            dtype=self.input_embedding_table.dtype,
            device=self.device,
        ).unsqueeze(0)
        return self.speaker_projection(embedding)


def load_reference_artifacts(repo_dir: str | Path, voice: str | None = None) -> ReferenceArtifacts:
    from transformers import AutoTokenizer

    repo_dir = Path(repo_dir)
    config_path = repo_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    token_map = TokenMap.from_config(config)
    voices_path = repo_dir / config.get("voices", "voices/voices.json")
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    voice = voice or config.get("default_voice")
    if voice not in voices:
        raise ValueError(f"Unknown voice {voice!r}. Available voices: {sorted(voices)}")

    spec = voices[voice]
    voice_dir = voices_path.parent
    artifact_path = voice_dir / spec["artifacts"]
    with np.load(artifact_path, allow_pickle=False) as archive:
        speaker_embedding = np.asarray(archive["embedding"], dtype=np.float32)
        reference_tokens = np.asarray(archive["reference_tokens"], dtype=np.int64)
    expected_speaker_dim = int(config["speaker_embedding_dim"])
    if speaker_embedding.shape != (expected_speaker_dim,):
        raise ValueError(
            f"Voice {voice!r} embedding has shape {speaker_embedding.shape}, "
            f"expected {(expected_speaker_dim,)}"
        )
    if reference_tokens.ndim != 1:
        raise ValueError(f"Voice {voice!r} reference tokens must be one-dimensional")
    if reference_tokens.size and (
        reference_tokens.min() < 0 or reference_tokens.max() >= token_map.num_audio_tokens
    ):
        raise ValueError(f"Voice {voice!r} reference tokens are outside the S3 vocabulary")

    transcript = spec["transcript"]
    if not isinstance(transcript, str):
        raise TypeError(f"Voice {voice!r} transcript must be text")
    lm_dir = repo_dir / config.get("lm_dir", "lm")
    tokenizer = AutoTokenizer.from_pretrained(str(lm_dir), local_files_only=True)
    transcript_ids = tokenizer.encode(transcript.strip(), add_special_tokens=False)
    reference_prefix = build_reference_prefix(token_map, transcript_ids, reference_tokens.tolist())

    manifest_path = repo_dir / "model_revision.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    model_revision = manifest.get("source_model_revision") if isinstance(manifest, dict) else None

    return ReferenceArtifacts(
        repo_dir=repo_dir,
        config=config,
        model_revision=model_revision,
        token_map=token_map,
        tokenizer=tokenizer,
        voice=voice,
        transcript=transcript,
        speaker_embedding=speaker_embedding,
        reference_tokens=reference_tokens,
        reference_prefix=reference_prefix,
    )


def _load_speaker_projection(
    repo_dir: Path,
    config: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Sequential:
    from safetensors import safe_open

    lm_dir = repo_dir / config.get("lm_dir", "lm")
    checkpoint = lm_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Plain LM checkpoint not found: {checkpoint}")

    speaker_state: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        for name in keys:
            if name.startswith("spk_proj."):
                speaker_state[name.removeprefix("spk_proj.")] = handle.get_tensor(name).to(dtype)
    if not speaker_state:
        raise ValueError(f"No spk_proj.* tensors found in {checkpoint}")
    linear_weight = speaker_state.get("0.weight")
    if linear_weight is None:
        raise ValueError(f"Expected spk_proj.0.weight in {checkpoint}")

    speaker_dim = int(config["speaker_embedding_dim"])
    hidden_size = int(
        json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))["hidden_size"]
    )
    if linear_weight.shape != (hidden_size, speaker_dim):
        raise ValueError(
            f"spk_proj.0.weight has shape {tuple(linear_weight.shape)}, "
            f"expected {(hidden_size, speaker_dim)}"
        )

    projection = nn.Sequential(
        nn.Linear(speaker_dim, hidden_size, bias=True),
        nn.LayerNorm(hidden_size),
    ).to(device=device, dtype=dtype)
    projection.load_state_dict(speaker_state, strict=True)
    return projection.eval()


def load_reference_model(
    artifacts: ReferenceArtifacts,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> ReferenceModel:
    from transformers import AutoModelForCausalLM

    device = torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16 if artifacts.config.get("dtype", "bf16") == "bf16" else torch.float32
    lm_dir = artifacts.repo_dir / artifacts.config.get("lm_dir", "lm")
    model = AutoModelForCausalLM.from_pretrained(
        str(lm_dir),
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device=device)
    model.eval()
    speaker_projection = _load_speaker_projection(
        artifacts.repo_dir, artifacts.config, dtype, device
    )
    return ReferenceModel(
        artifacts=artifacts,
        model=model,
        speaker_projection=speaker_projection,
        device=device,
    )
