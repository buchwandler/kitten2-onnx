"""Export KittenTTS 2's speaker projection head to ONNX.

Upstream KittenTTS 2 stores ``spk_proj.*`` beside the Qwen3 LM weights even
though Qwen3 itself does not know about the head. The projection is:

    Linear(512 -> hidden_size)
    LayerNorm(hidden_size)

Its output replaces the first token embedding during prefill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Iterable

import torch
from safetensors.torch import load_file
from torch import nn


def _load_state(lm_dir: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    shards = sorted(lm_dir.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors files found under {lm_dir}")
    for shard in shards:
        state.update(load_file(str(shard)))
    return state


def build_speaker_projection(repo_dir: str | Path) -> nn.Sequential:
    repo_dir = Path(repo_dir)
    lm_dir = repo_dir / "lm"
    config = json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))
    hidden = int(config["hidden_size"])

    state = _load_state(lm_dir)
    prefix = "spk_proj."
    speaker_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not speaker_state:
        raise ValueError("No spk_proj.* tensors found. Use the full KittenTTS 2 LM checkpoint.")

    linear_weight = speaker_state.get("0.weight")
    if linear_weight is None:
        raise ValueError("Expected spk_proj.0.weight in checkpoint")
    speaker_dim = int(linear_weight.shape[1])

    module = nn.Sequential(
        nn.Linear(speaker_dim, hidden, bias=True),
        nn.LayerNorm(hidden),
    )
    module.load_state_dict(speaker_state, strict=True)
    module.eval()
    return module


def export_speaker(repo_dir: str | Path, output_dir: str | Path, *, opset: int = 18) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = build_speaker_projection(repo_dir)
    speaker_dim = int(module[0].in_features)
    hidden = int(module[0].out_features)
    dummy = torch.zeros(1, speaker_dim, dtype=torch.float32)

    # Export in fp32 for broad ORT compatibility. The runtime casts the resulting
    # hidden vector to the embedding table dtype before insertion.
    module = module.float()
    path = output_dir / "speaker.onnx"
    torch.onnx.export(
        module,
        (dummy,),
        str(path),
        input_names=["speaker_embedding"],
        output_names=["speaker_hidden"],
        dynamic_axes={
            "speaker_embedding": {0: "batch"},
            "speaker_hidden": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )

    meta = {
        "format": "kitten2-onnx-speaker-v1",
        "speaker_dim": speaker_dim,
        "hidden_size": hidden,
        "dtype": "fp32",
        "opset": int(opset),
    }
    (output_dir / "speaker.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    print(export_speaker(args.repo_dir, args.output, opset=args.opset))


if __name__ == "__main__":
    main()
