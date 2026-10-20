"""Load plain and ternary-packed KittenTTS 2 language-model weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from kitten2_onnx.ternary import (
    TERNARY_FORMAT,
    signed_zero_bit_differences,
    tensor_bytes_equal,
    tensor_values_equal,
    unpack_state_dict,
)


def load_packed_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read and unpack one KittenML ternary SafeTensors checkpoint."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        names = handle.keys()
        tensors = {name: handle.get_tensor(name) for name in names}
    return unpack_state_dict(tensors, metadata)


def load_plain_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a plain SafeTensors file or an unsharded/sharded LM directory."""
    path = Path(path)
    if path.is_file():
        return load_file(str(path), device="cpu")
    if not path.is_dir():
        raise FileNotFoundError(path)

    plain_file = path / "model.safetensors"
    shards = [plain_file] if plain_file.is_file() else sorted(path.glob("model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No plain SafeTensors checkpoint found under {path}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        loaded = load_file(str(shard), device="cpu")
        overlap = state.keys() & loaded.keys()
        if overlap:
            raise ValueError(f"Duplicate tensors across SafeTensors shards: {sorted(overlap)}")
        state.update(loaded)
    return state


WeightVariant = Literal["full", "plain", "packed", "emb4"]
_WEIGHT_CONFIG_KEYS = {"packed": "lm_packed", "emb4": "lm_emb4"}


def resolve_weight_path(model_dir: str | Path, variant: WeightVariant) -> Path | None:
    """Resolve a configured source checkpoint path; ``plain`` aliases ``full``."""
    model_dir = Path(model_dir)
    if variant == "plain":
        variant = "full"
    if variant == "full":
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        name = config.get("lm_dir", "lm")
    elif variant in _WEIGHT_CONFIG_KEYS:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        name = config.get(_WEIGHT_CONFIG_KEYS[variant])
        if not isinstance(name, str) or not name:
            return None
    else:
        raise ValueError("variant must be one of 'full', 'packed', or 'emb4'")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid configured LM path for {variant!r}: {name!r}")
    path = (model_dir / name).resolve()
    if not path.is_relative_to(model_dir.resolve()):
        raise ValueError(f"Configured LM path escapes model directory: {name!r}")
    return path


def available_weight_variants(model_dir: str | Path) -> tuple[str, ...]:
    """Return declared local variants whose checkpoint payload is present."""
    available = []
    full_path = resolve_weight_path(model_dir, "full")
    if full_path is not None and (
        full_path.is_file()
        or full_path.is_dir()
        and (
            (full_path / "model.safetensors").is_file()
            or any(full_path.glob("model-*-of-*.safetensors"))
        )
    ):
        available.append("full")
    for variant in ("packed", "emb4"):
        path = resolve_weight_path(model_dir, variant)
        if path is not None and path.is_file():
            available.append(variant)
    return tuple(available)


def load_lm_state_dict(
    model_dir: str | Path, *, variant: WeightVariant = "packed"
) -> dict[str, torch.Tensor]:
    """Load one configured full, packed, or emb4 LM checkpoint."""
    model_dir = Path(model_dir)
    if variant == "plain":
        variant = "full"
    if variant not in {"full", "packed", "emb4"}:
        raise ValueError("variant must be one of 'full', 'packed', or 'emb4'")
    path = resolve_weight_path(model_dir, variant)
    if path is None:
        raise ValueError(f"No {variant!r} LM checkpoint is declared in {model_dir / 'config.json'}")
    if variant == "full":
        return load_plain_state_dict(path)
    if not path.is_file():
        raise FileNotFoundError(f"{variant!r} LM checkpoint not found: {path}")
    if variant == "packed":
        return load_packed_state_dict(path)
    from kitten2_onnx.tl2_emb4 import load_emb4_state_dict

    return load_emb4_state_dict(path)


def compare_packed_checkpoint(
    packed_path: str | Path,
    plain_path: str | Path,
) -> dict[str, Any]:
    """Compare reconstructed tensors bitwise and numerically to the plain checkpoint."""
    packed_path = Path(packed_path)
    plain_path = Path(plain_path)
    byte_mismatches: list[str] = []
    value_mismatches: list[str] = []
    structural_mismatches: list[str] = []
    signed_zero_differences = 0
    compared = 0

    with (
        safe_open(str(packed_path), framework="pt", device="cpu") as packed,
        safe_open(str(plain_path), framework="pt", device="cpu") as plain,
    ):
        metadata = packed.metadata() or {}
        if metadata.get("format") != TERNARY_FORMAT:
            raise ValueError(f"not a packed ternary checkpoint: {metadata.get('format')!r}")
        try:
            packed_specs = json.loads(metadata["packed"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("packed ternary metadata is incomplete or malformed") from error
        if not isinstance(packed_specs, dict):
            raise TypeError("packed ternary tensor metadata must be an object")

        packed_file_names = set(packed.keys())
        special_names = {name for name in packed_file_names if "::" in name}
        expected_special_names = {
            f"{name}::{suffix}" for name in packed_specs for suffix in ("codes", "scale")
        }
        if special_names != expected_special_names:
            structural_mismatches.append("packed code/scale tensor names differ from metadata")
        reconstructed_names = (packed_file_names - special_names) | set(packed_specs)
        plain_names = set(plain.keys())
        if reconstructed_names != plain_names:
            missing = sorted(plain_names - reconstructed_names)
            unexpected = sorted(reconstructed_names - plain_names)
            structural_mismatches.append(
                f"tensor names differ; missing={missing}, unexpected={unexpected}"
            )

        for name in sorted(reconstructed_names & plain_names):
            reference = plain.get_tensor(name)
            if name in packed_specs:
                tensors = {
                    f"{name}::codes": packed.get_tensor(f"{name}::codes"),
                    f"{name}::scale": packed.get_tensor(f"{name}::scale"),
                }
                single_metadata = {
                    "format": TERNARY_FORMAT,
                    "group_size": metadata.get("group_size", ""),
                    "packed": json.dumps({name: packed_specs[name]}),
                }
                actual = unpack_state_dict(tensors, single_metadata)[name]
            else:
                actual = packed.get_tensor(name)
            compared += 1
            values_match = tensor_values_equal(actual, reference)
            if not values_match:
                value_mismatches.append(name)
            if not tensor_bytes_equal(actual, reference):
                byte_mismatches.append(name)
                if values_match:
                    signed_zero_differences += signed_zero_bit_differences(actual, reference)

    return {
        "format": "kitten2-onnx-ternary-comparison-v1",
        "packed_checkpoint": str(packed_path),
        "plain_checkpoint": str(plain_path),
        "tensor_count": compared,
        "bitwise_exact": not structural_mismatches and not byte_mismatches,
        "numerically_exact": not structural_mismatches and not value_mismatches,
        "mismatches": structural_mismatches + value_mismatches,
        "byte_mismatches": byte_mismatches,
        "signed_zero_bit_differences": signed_zero_differences,
    }
