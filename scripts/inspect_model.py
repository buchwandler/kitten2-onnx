#!/usr/bin/env python3
"""Inspect a local KittenTTS 2 checkpoint or exported ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_checkpoint(repo_dir: Path) -> None:
    from safetensors import safe_open

    lm_dir = repo_dir / "lm"
    config = json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))
    keys: list[str] = []
    for shard in sorted(lm_dir.glob("model*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            keys.extend(handle.keys())

    info = {
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "vocab_size": config.get("vocab_size"),
        "tied_embeddings": config.get("tie_word_embeddings"),
        "safetensor_files": len(list(lm_dir.glob("model*.safetensors"))),
        "spk_proj_keys": sorted(k for k in keys if k.startswith("spk_proj.")),
    }
    print(json.dumps(info, indent=2))


def inspect_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    info = {
        "ir_version": model.ir_version,
        "opsets": {x.domain or "ai.onnx": x.version for x in model.opset_import},
        "inputs": [{"name": x.name, "type": str(x.type.tensor_type)} for x in model.graph.input],
        "outputs": [{"name": x.name, "type": str(x.type.tensor_type)} for x in model.graph.output],
        "initializers": len(model.graph.initializer),
        "nodes": len(model.graph.node),
    }
    print(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", nargs="?", type=Path)
    parser.add_argument("--onnx", type=Path)
    args = parser.parse_args()

    if bool(args.repo_dir) == bool(args.onnx):
        parser.error("give exactly one of repo_dir or --onnx")
    if args.onnx:
        inspect_onnx(args.onnx)
    else:
        inspect_checkpoint(args.repo_dir)


if __name__ == "__main__":
    main()
