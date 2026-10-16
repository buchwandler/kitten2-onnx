"""Export the KittenTTS 2 Qwen3 speech LM to ONNX.

The graph intentionally starts at ``inputs_embeds``. Kitten's speaker projection
replaces embedding position 0 during prefill, so keeping token lookup outside the
graph is both simpler and closer to the upstream implementation.

One graph serves prefill and decode:

    inputs_embeds [B, S, H]
    attention_mask [B, P + S]
    position_ids [B, S]
    past_key_i / past_value_i [B, KV, P, D]
        ->
    logits [B, V]                       (last position only)
    present_key_i / present_value_i     [B, KV, P + S, D]

Prefill uses P=0; cached decode normally uses S=1.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ExportProfile:
    graph_dtype: torch.dtype
    graph_dtype_name: str
    io_dtype: str
    provider: str
    verify: bool


RUNTIME_PROFILES = {
    "dev-fp32": ExportProfile(torch.float32, "float32", "float32", "CPUExecutionProvider", True),
    "cuda-fp16": ExportProfile(torch.float16, "float16", "float16", "CUDAExecutionProvider", False),
    "cuda-bf16": ExportProfile(torch.bfloat16, "bfloat16", "bfloat16", "CUDAExecutionProvider", False),
}


class KittenQwen3ExportWrapper(nn.Module):
    """Flatten Hugging Face's cache object into stable ONNX tensor I/O."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.transformer = model.model
        # Kitten/Qwen3 uses tied embeddings. Keeping a direct reference avoids
        # depending on the high-level CausalLM output object during export.
        self.lm_head_weight = model.get_output_embeddings().weight
        self.num_layers = int(model.config.num_hidden_layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        from transformers.cache_utils import DynamicCache

        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(self.num_layers))
        cache = DynamicCache.from_legacy_cache(legacy)

        out = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        # Full prefill logits are enormous and generation only consumes the last
        # row. Project only the last hidden state.
        last_hidden = out.last_hidden_state[:, -1:, :]
        logits = torch.nn.functional.linear(last_hidden, self.lm_head_weight)
        logits = logits[:, 0, :].float()

        present = out.past_key_values.to_legacy_cache()
        flat = [tensor for layer in present for tensor in layer]
        return (logits, *flat)


def _io_names(num_layers: int) -> tuple[list[str], list[str]]:
    past = [name for i in range(num_layers) for name in (f"past_key_{i}", f"past_value_{i}")]
    present = [
        name for i in range(num_layers) for name in (f"present_key_{i}", f"present_value_{i}")
    ]
    return past, present


def _dummy_inputs(config, dtype: torch.dtype, seq: int = 4, past_len: int = 3):
    hidden = int(config.hidden_size)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    layers = int(config.num_hidden_layers)

    embeds = torch.randn(1, seq, hidden, dtype=dtype)
    mask = torch.ones(1, past_len + seq, dtype=torch.int64)
    pos = torch.arange(past_len, past_len + seq, dtype=torch.int64)[None, :]
    past = [torch.zeros(1, kv_heads, past_len, head_dim, dtype=dtype) for _ in range(2 * layers)]
    return embeds, mask, pos, past


def load_hf_lm(repo_dir: Path, dtype: torch.dtype):
    """Load and validate the plain Kitten Qwen3 checkpoint for ONNX export."""
    from transformers import AutoModelForCausalLM

    repo_dir = Path(repo_dir)
    lm_dir = repo_dir / "lm"
    config_path = lm_dir / "config.json"
    checkpoint_path = lm_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Kitten LM config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Plain Kitten LM checkpoint not found: {checkpoint_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError(f"Expected a Qwen3 checkpoint in {lm_dir}")

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        str(lm_dir),
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
        output_loading_info=True,
    )
    missing = set(loading_info["missing_keys"])
    if model.config.tie_word_embeddings:
        missing.discard("lm_head.weight")
    unexpected = {
        name for name in loading_info["unexpected_keys"] if not name.startswith("spk_proj.")
    }
    if missing:
        raise ValueError(f"Missing transformer checkpoint weights: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected checkpoint weights: {sorted(unexpected)}")

    model.tie_weights()
    if model.config.tie_word_embeddings:
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            raise ValueError("Qwen3 input and output embeddings are not tied")
    model.eval()
    return model


def save_embedding_table(model: nn.Module, output_dir: Path) -> Path:
    """Store token embeddings outside ONNX for speaker injection and token lookup."""
    table = model.get_input_embeddings().weight.detach().cpu()
    # NumPy has no broadly interoperable bf16 dtype. Store bf16 tables as fp32;
    # fp16 remains fp16.
    if table.dtype == torch.bfloat16:
        array = table.float().numpy()
    else:
        array = table.numpy()
    path = output_dir / "embed_tokens.npy"
    np.save(path, array, allow_pickle=False)
    return path


def _export_with_diagnostics(
    model: nn.Module,
    args: tuple,
    output_path: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    verify: bool = True,
    **export_kwargs,
) -> None:
    artifacts_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else (Path(__file__).resolve().parents[1] / "build" / "debug" / "lm-export")
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    try:
        torch.onnx.export(
            model,
            args,
            str(output_path),
            report=True,
            verify=verify,
            artifacts_dir=str(artifacts_dir),
            **export_kwargs,
        )
    except Exception as error:
        failure_path = artifacts_dir / "export-failure.txt"
        failure_path.write_text(
            f"Output: {output_path}\n{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        raise


def export_lm(
    repo_dir: str | Path,
    output_dir: str | Path,
    *,
    profile_name: str = "dev-fp32",
    opset: int = 18,
    diagnostics_dir: str | Path | None = None,
) -> Path:
    try:
        profile = RUNTIME_PROFILES[profile_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown export profile {profile_name!r}. Choose from {sorted(RUNTIME_PROFILES)}"
        ) from error

    repo_dir = Path(repo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = repo_dir / "lm"
    checkpoint_path = lm_dir / "model.safetensors"
    source_config = json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))
    source_dtype = str(source_config.get("dtype", source_config.get("torch_dtype", "unknown")))
    revision_path = repo_dir / "model_revision.json"
    revision = json.loads(revision_path.read_text(encoding="utf-8")) if revision_path.is_file() else {}

    model = load_hf_lm(repo_dir, profile.graph_dtype)
    wrapper = KittenQwen3ExportWrapper(model).eval()
    config = model.config
    past_names, present_names = _io_names(wrapper.num_layers)
    input_names = ["inputs_embeds", "attention_mask", "position_ids", *past_names]
    output_names = ["logits", *present_names]

    batch_dim = torch.export.Dim("batch", min=1)
    seq_dim = torch.export.Dim("seq", min=1)
    total_dim = torch.export.Dim("total", min=1)
    past_dim = torch.export.Dim("past", min=0)
    dynamic_shapes = {
        "inputs_embeds": {0: batch_dim, 1: seq_dim},
        "attention_mask": {0: batch_dim, 1: total_dim},
        "position_ids": {0: batch_dim, 1: seq_dim},
        "past": [{0: batch_dim, 2: past_dim} for _ in range(len(past_names))],
    }

    dummy = _dummy_inputs(config, dtype=profile.graph_dtype, seq=4, past_len=3)
    output_path = output_dir / "lm.onnx"
    _export_with_diagnostics(
        wrapper,
        dummy,
        output_path,
        artifacts_dir=diagnostics_dir,
        verify=profile.verify,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        dynamo=True,
        external_data=True,
    )

    embedding_path = save_embedding_table(model, output_dir)
    meta = {
        "format": "kitten2-onnx-lm-v1",
        "architecture": (getattr(config, "architectures", None) or [config.model_type])[0],
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "vocab_size": int(config.vocab_size),
        "source_dtype": source_dtype,
        "graph_dtype": profile.graph_dtype_name,
        "io_dtype": profile.io_dtype,
        "logits_dtype": "float32",
        "runtime_profile": profile_name,
        "runtime_provider": profile.provider,
        "verify_on_export": profile.verify,
        "source_model_path": str(repo_dir.resolve()),
        "source_model_revision": revision.get("source_model_revision"),
        "loading_path": str(checkpoint_path.resolve()),
        "attention_implementation": "eager",
        "external_data": True,
        "opset": int(opset),
        "embedding_file": embedding_path.name,
        "embedding_storage_dtype": (
            "float32" if profile.graph_dtype == torch.bfloat16 else profile.graph_dtype_name
        ),
        "input_names": input_names,
        "output_names": output_names,
    }
    (output_dir / "lm.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", type=Path, help="Local KittenML/kitten-tts-2 directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(RUNTIME_PROFILES), default="dev-fp32")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)

    path = export_lm(
        args.repo_dir,
        args.output,
        profile_name=args.profile,
        opset=args.opset,
    )
    print(path)


if __name__ == "__main__":
    main()
