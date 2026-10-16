"""Measure source-checkpoint bytes by parameter dtype and exact ternary structure."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open

from kitten2_onnx.ternary import GROUP_SIZE, is_ternary


def _checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    plain = path / "model.safetensors"
    if plain.is_file():
        return [plain]
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"No SafeTensors checkpoint found under {path}")


def audit_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Audit parameter bytes and identify tensors exactly representable as ternary."""
    checkpoint_path = Path(checkpoint)
    shards = _checkpoint_files(checkpoint_path)
    seen_names: set[str] = set()
    source_bytes_by_dtype: dict[str, int] = defaultdict(int)
    bf16_bytes_by_dtype: dict[str, int] = defaultdict(int)
    tensor_count = 0
    total_source_bytes = 0
    total_bf16_bytes = 0
    ternary_source_bytes = 0
    ternary_bf16_bytes = 0
    ternary_tensor_count = 0
    ternary_tensors: list[dict[str, Any]] = []

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                if name in seen_names:
                    raise ValueError(f"Duplicate checkpoint tensor across shards: {name}")
                seen_names.add(name)
                tensor = handle.get_tensor(name)
                source_bytes = tensor.numel() * tensor.element_size()
                bf16_bytes = tensor.numel() * 2
                dtype_name = str(tensor.dtype).removeprefix("torch.")
                tensor_count += 1
                total_source_bytes += source_bytes
                total_bf16_bytes += bf16_bytes
                source_bytes_by_dtype[dtype_name] += source_bytes
                bf16_bytes_by_dtype[dtype_name] += bf16_bytes

                if is_ternary(tensor, GROUP_SIZE):
                    ternary_tensor_count += 1
                    ternary_source_bytes += source_bytes
                    ternary_bf16_bytes += bf16_bytes
                    ternary_tensors.append(
                        {
                            "name": name,
                            "shape": list(tensor.shape),
                            "dtype": dtype_name,
                            "source_bytes": source_bytes,
                            "bf16_equivalent_bytes": bf16_bytes,
                        }
                    )

    nonternary_source_bytes = total_source_bytes - ternary_source_bytes
    nonternary_bf16_bytes = total_bf16_bytes - ternary_bf16_bytes
    return {
        "format": "kitten2-onnx-checkpoint-size-audit-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_files": [{"path": str(path), "bytes": path.stat().st_size} for path in shards],
        "group_size": GROUP_SIZE,
        "tensor_count": tensor_count,
        "ternary_tensor_count": ternary_tensor_count,
        "total_parameter_bytes_source_dtype": total_source_bytes,
        "ternary_parameter_bytes_source_dtype": ternary_source_bytes,
        "nonternary_parameter_bytes_source_dtype": nonternary_source_bytes,
        "source_bytes_by_dtype": dict(sorted(source_bytes_by_dtype.items())),
        "total_parameter_bytes_bf16": total_bf16_bytes,
        "ternary_parameter_bytes_bf16": ternary_bf16_bytes,
        "nonternary_parameter_bytes_bf16": nonternary_bf16_bytes,
        "bf16_equivalent_bytes_by_dtype": dict(sorted(bf16_bytes_by_dtype.items())),
        "ternary_fraction_by_bytes": (
            ternary_bf16_bytes / total_bf16_bytes if total_bf16_bytes else 0.0
        ),
        "ternary_fraction_by_source_bytes": (
            ternary_source_bytes / total_source_bytes if total_source_bytes else 0.0
        ),
        "ternary_tensors": sorted(ternary_tensors, key=lambda value: value["name"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_checkpoint(args.checkpoint)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
