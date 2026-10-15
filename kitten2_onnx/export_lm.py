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
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
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
    """Load the plain Hugging Face-compatible Kitten LM checkpoint."""
    from transformers import AutoModelForCausalLM

    lm_dir = repo_dir / "lm"
    if not lm_dir.is_dir():
        raise FileNotFoundError(f"Kitten LM directory not found: {lm_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        str(lm_dir),
        torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
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


def export_lm(
    repo_dir: str | Path,
    output_dir: str | Path,
    *,
    dtype_name: str = "bf16",
    opset: int = 18,
) -> Path:
    repo_dir = Path(repo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = DTYPES[dtype_name]
    model = load_hf_lm(repo_dir, dtype)
    wrapper = KittenQwen3ExportWrapper(model).eval()

    config = model.config
    past_names, present_names = _io_names(wrapper.num_layers)
    input_names = ["inputs_embeds", "attention_mask", "position_ids", *past_names]
    output_names = ["logits", *present_names]

    seq_dim = torch.export.Dim("seq", min=1)
    total_dim = torch.export.Dim("total", min=1)
    past_dim = torch.export.Dim("past", min=0)

    dynamic_shapes = {
        "inputs_embeds": {0: torch.export.Dim("batch", min=1), 1: seq_dim},
        "attention_mask": {0: torch.export.Dim("batch_mask", min=1), 1: total_dim},
        "position_ids": {0: torch.export.Dim("batch_pos", min=1), 1: seq_dim},
        "past": [
            {0: torch.export.Dim(f"batch_past_{i}", min=1), 2: past_dim}
            for i in range(len(past_names))
        ],
    }

    dummy = _dummy_inputs(config, dtype=dtype, seq=4, past_len=3)
    output_path = output_dir / "lm.onnx"

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
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
        "architecture": getattr(config, "architectures", [None])[0],
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "vocab_size": int(config.vocab_size),
        "dtype": dtype_name,
        "opset": int(opset),
        "embedding_file": embedding_path.name,
        "input_names": input_names,
        "output_names": output_names,
    }
    (output_dir / "lm.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", type=Path, help="Local KittenML/kitten-tts-2 directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)

    path = export_lm(
        args.repo_dir,
        args.output,
        dtype_name=args.dtype,
        opset=args.opset,
    )
    print(path)


if __name__ == "__main__":
    main()
