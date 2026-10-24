"""Pinned PyTorch reference loading and built-in voice artifacts for validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from kitten2_onnx.prompt import build_generation_prompt, build_reference_prefix
from kitten2_onnx.text import PreparedGenerationText, prepare_generation_text
from kitten2_onnx.tokens import TokenMap

UPSTREAM_CODE_REVISION = "36ef3090ca1e59122a81fc23be4a659a502b9d55"


@dataclass
class ReferenceArtifacts:
    repo_dir: Path
    voice_artifacts_dir: Path
    voice_artifacts_source_model_revision: str | None
    config: dict[str, Any]
    model_revision: str | None
    token_map: TokenMap
    weight_variant: str
    source_weight_sha256: str | None
    source_weight_files: list[dict[str, Any]]
    speaker_projection_checkpoint: Path | None
    speaker_projection_source: str | None
    speaker_projection_weight_variant: str
    speaker_projection_source_model_revision: str | None
    speaker_projection_source_sha256: str | None
    tokenizer: Any
    voice: str
    transcript: str
    speaker_embedding: np.ndarray
    reference_tokens: np.ndarray
    reference_prefix: list[int]

    def prepare_text(
        self,
        text: str,
        *,
        normalize: bool = True,
        use_emotion: bool | None = None,
    ) -> PreparedGenerationText:
        return prepare_generation_text(
            text,
            tokenizer=self.tokenizer,
            generation_config=self.config.get("generation", {}),
            normalize=normalize,
            use_emotion=use_emotion,
        )

    def build_prompt_from_prepared(
        self,
        prepared: PreparedGenerationText,
        *,
        emotion_ids: list[int] | None = None,
    ) -> list[int]:
        target_ids = self.tokenizer.encode(prepared.processed_text, add_special_tokens=False)
        return build_generation_prompt(
            self.token_map,
            target_ids,
            reference_prefix=self.reference_prefix,
            emotion_ids=prepared.emotion_ids if emotion_ids is None else emotion_ids,
        )

    def build_prompt(
        self,
        text: str,
        emotion_ids: list[int] | None = None,
        *,
        normalize: bool = True,
        use_emotion: bool | None = None,
    ) -> list[int]:
        prepared = self.prepare_text(text, normalize=normalize, use_emotion=use_emotion)
        return self.build_prompt_from_prepared(prepared, emotion_ids=emotion_ids)

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


def _checkpoint_files(path: Path | None) -> list[Path]:
    if path is None:
        return []
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    plain = path / "model.safetensors"
    if plain.is_file():
        return [plain]
    return sorted(path.glob("model-*-of-*.safetensors"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_provenance(root: Path, checkpoint: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    files = _checkpoint_files(checkpoint)
    records = []
    for path in files:
        try:
            display_path = path.relative_to(root).as_posix()
        except ValueError:
            display_path = str(path)
        records.append({
            "path": display_path,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    if not records:
        return records, None
    if len(records) == 1:
        return records, records[0]["sha256"]
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return records, digest.hexdigest()


def _find_speaker_projection_checkpoint(checkpoint: Path | None) -> Path | None:
    from safetensors import safe_open

    for path in _checkpoint_files(checkpoint):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if any(name.startswith("spk_proj.") for name in handle.keys()):  # noqa: SIM118
                return path
    return None


def _model_revision_manifest(repo_dir: Path) -> dict[str, Any]:
    manifest_path = repo_dir / "model_revision.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"Reference revision manifest must be an object: {manifest_path}")
    return manifest


def load_reference_artifacts(
    repo_dir: str | Path,
    voice: str | None = None,
    *,
    weight_variant: str = "full",
    weight_model_dir: str | Path | None = None,
    speaker_projection_model_dir: str | Path | None = None,
) -> ReferenceArtifacts:
    from transformers import AutoTokenizer

    from kitten2_onnx.weights import resolve_weight_path

    if weight_variant not in {"full", "packed", "emb4"}:
        raise ValueError("weight_variant must be one of 'full', 'packed', or 'emb4'")
    voice_repo_dir = Path(repo_dir)
    weight_repo_dir = Path(weight_model_dir) if weight_model_dir is not None else voice_repo_dir
    config = json.loads((weight_repo_dir / "config.json").read_text(encoding="utf-8"))
    voice_config = (
        config
        if voice_repo_dir == weight_repo_dir
        else json.loads((voice_repo_dir / "config.json").read_text(encoding="utf-8"))
    )
    token_map = TokenMap.from_config(config)
    voices_path = voice_repo_dir / voice_config.get("voices", "voices/voices.json")
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    voice = voice or voice_config.get("default_voice")
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
    lm_dir = weight_repo_dir / config.get("lm_dir", "lm")
    tokenizer = AutoTokenizer.from_pretrained(str(lm_dir), local_files_only=True)
    transcript_ids = tokenizer.encode(transcript.strip(), add_special_tokens=False)
    reference_prefix = build_reference_prefix(token_map, transcript_ids, reference_tokens.tolist())

    weight_manifest = _model_revision_manifest(weight_repo_dir)
    voice_manifest = _model_revision_manifest(voice_repo_dir)
    declared_variant = weight_manifest.get("weight_variant")
    if declared_variant is not None and declared_variant != weight_variant:
        raise ValueError(
            f"Reference directory pins weight_variant={declared_variant!r}, "
            f"not requested {weight_variant!r}"
        )
    model_revision = weight_manifest.get("source_model_revision")
    voice_revision = voice_manifest.get("source_model_revision")
    if voice_repo_dir != weight_repo_dir and (
        not model_revision or voice_revision != model_revision
    ):
        raise ValueError(
            "Voice assets and selected weights must have the same pinned source revision: "
            f"{voice_revision!r} != {model_revision!r}"
        )
    model_revision = model_revision or voice_revision
    source_checkpoint = resolve_weight_path(weight_repo_dir, weight_variant)
    source_weight_files, source_weight_sha256 = _checkpoint_provenance(
        weight_repo_dir, source_checkpoint
    )
    speaker_checkpoint = _find_speaker_projection_checkpoint(source_checkpoint)
    speaker_repo = Path(speaker_projection_model_dir or voice_repo_dir)
    speaker_revision = model_revision
    speaker_variant = weight_variant
    if speaker_checkpoint is None:
        speaker_variant = "full"
        speaker_manifest = _model_revision_manifest(speaker_repo)
        speaker_revision = speaker_manifest.get("source_model_revision")
        speaker_source_checkpoint = resolve_weight_path(speaker_repo, "full")
        speaker_checkpoint = _find_speaker_projection_checkpoint(speaker_source_checkpoint)
        if speaker_checkpoint is None and speaker_projection_model_dir is not None:
            raise ValueError(f"No spk_proj.* tensors found in {speaker_repo}")
        if speaker_checkpoint is not None and (
            not model_revision or speaker_revision != model_revision
        ):
            raise ValueError(
                "Speaker projection reference must have the same pinned source revision "
                f"as {weight_variant}: {model_revision!r} != {speaker_revision!r}"
            )
    speaker_projection_sha256 = (
        _sha256_file(speaker_checkpoint) if speaker_checkpoint is not None else None
    )
    speaker_source = str(speaker_checkpoint) if speaker_checkpoint is not None else None

    return ReferenceArtifacts(
        repo_dir=weight_repo_dir,
        voice_artifacts_dir=voice_repo_dir,
        voice_artifacts_source_model_revision=voice_revision,
        config=config,
        model_revision=model_revision,
        token_map=token_map,
        weight_variant=weight_variant,
        source_weight_sha256=source_weight_sha256,
        source_weight_files=source_weight_files,
        speaker_projection_checkpoint=speaker_checkpoint,
        speaker_projection_source=speaker_source,
        speaker_projection_weight_variant=speaker_variant,
        speaker_projection_source_model_revision=speaker_revision,
        speaker_projection_source_sha256=speaker_projection_sha256,
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
    *,
    checkpoint_path: Path | None = None,
) -> nn.Sequential:
    from safetensors import safe_open

    if checkpoint_path is None:
        checkpoint_path = repo_dir / config.get("lm_dir", "lm")
    checkpoints = _checkpoint_files(checkpoint_path)
    if not checkpoints:
        raise FileNotFoundError(f"Plain LM checkpoint not found: {checkpoint_path}")

    speaker_state: dict[str, torch.Tensor] = {}
    for checkpoint in checkpoints:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118
                if name.startswith("spk_proj."):
                    if name in speaker_state:
                        raise ValueError(f"Duplicate speaker projection tensor {name!r}")
                    speaker_state[name.removeprefix("spk_proj.")] = handle.get_tensor(name).to(dtype)
    if not speaker_state:
        raise ValueError(f"No spk_proj.* tensors found in {checkpoint_path}")
    linear_weight = speaker_state.get("0.weight")
    if linear_weight is None:
        raise ValueError(f"Expected spk_proj.0.weight in {checkpoint_path}")

    speaker_dim = int(config["speaker_embedding_dim"])
    lm_dir = repo_dir / config.get("lm_dir", "lm")
    hidden_size = int(json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))["hidden_size"])
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
    device = torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16 if artifacts.config.get("dtype", "bf16") == "bf16" else torch.float32
    if artifacts.weight_variant == "full":
        from transformers import AutoModelForCausalLM

        lm_dir = artifacts.repo_dir / artifacts.config.get("lm_dir", "lm")
        model = AutoModelForCausalLM.from_pretrained(
            str(lm_dir),
            dtype=dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    else:
        from kitten2_onnx.export_lm import load_hf_lm

        model = load_hf_lm(
            artifacts.repo_dir,
            dtype,
            weight_variant=artifacts.weight_variant,
        )
    model.to(device=device)
    model.eval()
    if artifacts.speaker_projection_checkpoint is None:
        raise ValueError(
            f"No speaker projection is available for {artifacts.weight_variant}; "
            "provide a same-revision full reference with spk_proj.* tensors"
        )
    speaker_projection = _load_speaker_projection(
        artifacts.repo_dir,
        artifacts.config,
        dtype,
        device,
        checkpoint_path=artifacts.speaker_projection_checkpoint,
    )
    return ReferenceModel(
        artifacts=artifacts,
        model=model,
        speaker_projection=speaker_projection,
        device=device,
    )
