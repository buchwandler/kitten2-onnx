#!/usr/bin/env python3
"""Inspect a KittenTTS 2 checkpoint or exported ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CURRENT_CHECKPOINT_VOCAB_SIZE = 158241
TOKEN_FIELDS = (
    "audio_id_base",
    "num_audio_tokens",
    "speech_start_id",
    "speech_end_id",
    "text_start_id",
    "text_end_id",
    "start_id",
    "stop_id",
    "final_seg_id",
    "reference_speech_start_id",
    "reference_speech_end_id",
    "reference_text_start_id",
    "reference_text_end_id",
)


def _read_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"Cannot derive {label}: file is missing at {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read {label} at {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"Cannot derive {label}: expected a JSON object at {path}")
        return None
    return value


def _walk_objects(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _token_map(config: dict[str, Any] | None) -> dict[str, int]:
    if config is None:
        return {}
    for obj in _walk_objects(config):
        if all(name in obj for name in TOKEN_FIELDS):
            candidate = {name: obj[name] for name in TOKEN_FIELDS}
            if all(type(value) is int for value in candidate.values()):
                return candidate
    return {}


def _speaker_embedding_dim(config: dict[str, Any] | None) -> int | None:
    if config is None:
        return None
    for obj in _walk_objects(config):
        value = obj.get("speaker_embedding_dim")
        if type(value) is int:
            return value
    return None


def _revision(repo_dir: Path, *configs: dict[str, Any] | None) -> str | None:
    manifest_path = repo_dir / "model_revision.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if isinstance(manifest, dict):
            revision = manifest.get("source_model_revision") or manifest.get("revision")
            if isinstance(revision, str):
                return revision

    for config in configs:
        if config is None:
            continue
        for key in ("_commit_hash", "source_model_revision", "revision"):
            revision = config.get(key)
            if isinstance(revision, str):
                return revision
    return None


def _tensor_metadata(lm_dir: Path, files: list[Path], errors: list[str]):
    from safetensors import SafetensorError, safe_open

    tensors: dict[str, dict[str, Any]] = {}
    file_sizes = []
    for path in files:
        file_sizes.append({"path": str(path.relative_to(lm_dir.parent)), "size_bytes": path.stat().st_size})
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = handle.keys()
                for name in keys:
                    view = handle.get_slice(name)
                    tensors[name] = {
                        "shape": list(view.get_shape()),
                        "dtype": view.get_dtype(),
                        "file": str(path.relative_to(lm_dir.parent)),
                    }
        except (OSError, SafetensorError) as exc:
            errors.append(f"Cannot inspect safetensors metadata in {path}: {exc}")
    return tensors, file_sizes


def inspect_checkpoint(repo_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    top_config = _read_json(repo_dir / "config.json", "top-level Kitten config", errors)
    lm_config = _read_json(repo_dir / "lm" / "config.json", "LM Qwen config", errors)
    lm_dir = repo_dir / "lm"
    files = sorted(lm_dir.glob("model*.safetensors"))
    if not files:
        errors.append(f"No safetensors checkpoint files found under {lm_dir}")
    if not (lm_dir / "model.safetensors").is_file():
        errors.append(f"Plain LM checkpoint required for the first export is missing: {lm_dir / 'model.safetensors'}")

    tensors, file_sizes = _tensor_metadata(lm_dir, files, errors)
    token_map = _token_map(top_config)
    speaker_dim = _speaker_embedding_dim(top_config)
    if not token_map:
        errors.append("Cannot derive the complete token map from the top-level Kitten config")
    if speaker_dim is None:
        errors.append("Cannot derive speaker_embedding_dim from the top-level Kitten config")

    config_values = lm_config or {}
    hidden_size = config_values.get("hidden_size")
    num_layers = config_values.get("num_hidden_layers")
    num_heads = config_values.get("num_attention_heads")
    num_kv_heads = config_values.get("num_key_value_heads")
    vocab_size = config_values.get("vocab_size")
    head_dim = config_values.get("head_dim")
    if (
        head_dim is None
        and type(hidden_size) is int
        and type(num_heads) is int
        and num_heads
        and hidden_size % num_heads == 0
    ):
        head_dim = hidden_size // num_heads

    projection = {
        name: info
        for name, info in tensors.items()
        if name.startswith("spk_proj.")
    }
    speaker_weight = projection.get("spk_proj.0.weight")
    kv_projection = {
        name: info for name, info in tensors.items() if name.endswith("self_attn.k_proj.weight")
    }
    embedding_candidates = [
        (name, info)
        for name, info in tensors.items()
        if name.endswith("embed_tokens.weight")
    ]
    embedding_name, embedding = embedding_candidates[0] if embedding_candidates else (None, None)
    lm_head_present = any(name.endswith("lm_head.weight") for name in tensors)
    tied_embeddings = config_values.get("tie_word_embeddings")
    lm_head_contract_valid = lm_head_present or tied_embeddings is True

    invariants: dict[str, Any] = {
        "audio_range_ends_at_final_seg": None,
        "speaker_input_matches_configured_embedding_dim": None,
        "speaker_output_matches_hidden_size": None,
        "embedding_width_matches_hidden_size": None,
        "kv_head_dim_matches_head_dim": None,
        "vocab_size_matches_embedding_rows": None,
        "lm_head_present_or_tied": lm_head_contract_valid,
    }

    if token_map:
        invariants["audio_range_ends_at_final_seg"] = (
            token_map["audio_id_base"] + token_map["num_audio_tokens"] == token_map["final_seg_id"]
        )
    if speaker_weight is not None and speaker_dim is not None:
        invariants["speaker_input_matches_configured_embedding_dim"] = (
            len(speaker_weight["shape"]) == 2 and speaker_weight["shape"][1] == speaker_dim
        )
    if speaker_weight is not None and type(hidden_size) is int:
        invariants["speaker_output_matches_hidden_size"] = (
            len(speaker_weight["shape"]) == 2 and speaker_weight["shape"][0] == hidden_size
        )
    if embedding is not None and type(hidden_size) is int:
        invariants["embedding_width_matches_hidden_size"] = (
            len(embedding["shape"]) == 2 and embedding["shape"][1] == hidden_size
        )
    if (
        kv_projection
        and type(head_dim) is int
        and type(hidden_size) is int
        and type(num_kv_heads) is int
        and num_kv_heads > 0
    ):
        invariants["kv_head_dim_matches_head_dim"] = all(
            len(info["shape"]) == 2
            and info["shape"][0] == num_kv_heads * head_dim
            and info["shape"][1] == hidden_size
            for info in kv_projection.values()
        )
    if embedding is not None and type(vocab_size) is int:
        invariants["vocab_size_matches_embedding_rows"] = embedding["shape"][0] == vocab_size

    for name, value in {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "head_dim": head_dim,
    }.items():
        if type(value) is not int:
            errors.append(f"Cannot derive LM {name} from the Qwen config")

    for name, result in invariants.items():
        if result is False:
            errors.append(f"Checkpoint invariant failed: {name}")
        elif result is None:
            errors.append(f"Cannot derive checkpoint invariant: {name}")
    if vocab_size is not None and vocab_size != CURRENT_CHECKPOINT_VOCAB_SIZE:
        warnings.append(
            f"vocab_size is {vocab_size}, not {CURRENT_CHECKPOINT_VOCAB_SIZE}; using the value from this checkpoint config"
        )
    if not lm_head_contract_valid:
        errors.append("Checkpoint has no lm_head.weight and does not declare tied word embeddings")
    if speaker_weight is None:
        errors.append("Cannot derive speaker projection shape: spk_proj.0.weight is missing")
    if embedding is None:
        errors.append("Cannot derive embedding shape: embed_tokens.weight is missing")

    return {
        "model_directory": str(repo_dir),
        "model_revision": _revision(repo_dir, top_config, lm_config),
        "kitten_config": top_config,
        "lm_config": lm_config,
        "token_map": token_map,
        "speaker_embedding_dim": speaker_dim,
        "speaker_projection_tensors": projection,
        "kv_projection_tensors": kv_projection,
        "embedding_tensor": {"name": embedding_name, **embedding} if embedding is not None else None,
        "architecture": {
            "architectures": config_values.get("architectures"),
            "model_type": config_values.get("model_type"),
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
        },
        "lm_head": {
            "weight_present": lm_head_present,
            "tied_embeddings": tied_embeddings,
            "contract_valid": lm_head_contract_valid,
        },
        "source_safetensor_files": file_sizes,
        "invariants": invariants,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def inspect_onnx(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return {
        "ir_version": model.ir_version,
        "opsets": {entry.domain or "ai.onnx": entry.version for entry in model.opset_import},
        "inputs": [{"name": value.name, "type": str(value.type.tensor_type)} for value in model.graph.input],
        "outputs": [{"name": value.name, "type": str(value.type.tensor_type)} for value in model.graph.output],
        "initializers": len(model.graph.initializer),
        "nodes": len(model.graph.node),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", nargs="?", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args()

    if bool(args.repo_dir) == bool(args.onnx):
        parser.error("give exactly one of repo_dir or --onnx")
    if args.onnx:
        report = inspect_onnx(args.onnx)
        status = 0
    else:
        report = inspect_checkpoint(args.repo_dir)
        status = 0 if report["valid"] else 1
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
