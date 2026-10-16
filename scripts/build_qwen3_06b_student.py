from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from scripts.inspect_model import inspect_checkpoint

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
TEACHER_MODEL_REVISION = "fa8ce5c7547e322a065cad9b813d27a78fad9640"
STUDENT_SOURCE_REVISION = "student-qwen3-0.6b"
SOURCE_VOCAB_SIZE = 151936
KITTEN_TOKEN_START = 151669
STUDENT_VOCAB_SIZE = 158241
STUDENT_HIDDEN_SIZE = 1024
STUDENT_INTERMEDIATE_SIZE = 3072
STUDENT_NUM_LAYERS = 28
STUDENT_NUM_ATTENTION_HEADS = 16
STUDENT_NUM_KEY_VALUE_HEADS = 8
STUDENT_HEAD_DIM = 128
SPEAKER_EMBEDDING_DIM = 512

STUDENT_ARCHITECTURE = {
    "model_type": "qwen3",
    "hidden_size": STUDENT_HIDDEN_SIZE,
    "intermediate_size": STUDENT_INTERMEDIATE_SIZE,
    "num_hidden_layers": STUDENT_NUM_LAYERS,
    "num_attention_heads": STUDENT_NUM_ATTENTION_HEADS,
    "num_key_value_heads": STUDENT_NUM_KEY_VALUE_HEADS,
    "head_dim": STUDENT_HEAD_DIM,
    "vocab_size": STUDENT_VOCAB_SIZE,
    "tie_word_embeddings": True,
}

TEACHER_CONFIG_FIELDS = (
    "attention_bias",
    "attention_dropout",
    "hidden_act",
    "max_position_embeddings",
    "max_window_layers",
    "rms_norm_eps",
    "rope_theta",
    "sliding_window",
    "use_cache",
    "use_sliding_window",
)
TOKENIZER_FILES = (
    "added_tokens.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _validate_architecture(config: Any, expected: dict[str, Any], label: str) -> None:
    mismatches = {
        name: {"expected": expected_value, "actual": getattr(config, name, None)}
        for name, expected_value in expected.items()
        if getattr(config, name, None) != expected_value
    }
    if mismatches:
        raise ValueError(f"Unexpected {label} architecture: {json.dumps(mismatches, sort_keys=True)}")


def _apply_teacher_settings(student_config: Any, teacher_config: dict[str, Any]) -> None:
    for name in TEACHER_CONFIG_FIELDS:
        if name in teacher_config:
            setattr(student_config, name, teacher_config[name])


def _initialize_kitten_vocabulary(model: nn.Module) -> None:
    config = model.config
    embedding = model.get_input_embeddings().weight
    if embedding.shape != (SOURCE_VOCAB_SIZE, STUDENT_HIDDEN_SIZE):
        raise ValueError(
            f"Expected base embedding shape {(SOURCE_VOCAB_SIZE, STUDENT_HIDDEN_SIZE)}, "
            f"got {tuple(embedding.shape)}"
        )

    model.resize_token_embeddings(STUDENT_VOCAB_SIZE, mean_resizing=False)
    embedding = model.get_input_embeddings().weight
    with torch.no_grad():
        nn.init.normal_(
            embedding[KITTEN_TOKEN_START:],
            mean=0.0,
            std=float(config.initializer_range),
        )
    config.tie_word_embeddings = True
    model.tie_weights()
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ValueError("Student Qwen3 input and output embeddings are not tied")
    if tuple(model.get_input_embeddings().weight.shape) != (STUDENT_VOCAB_SIZE, STUDENT_HIDDEN_SIZE):
        raise ValueError("Student vocabulary resize produced an unexpected embedding shape")


def _copy_teacher_assets(teacher_dir: Path, output_dir: Path) -> None:
    for source in teacher_dir.iterdir():
        if source.name in {".cache", "lm", "model_revision.json"}:
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    target_lm_dir = output_dir / "lm"
    target_lm_dir.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_FILES:
        source = teacher_dir / "lm" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Kitten tokenizer asset is missing: {source}")
        shutil.copy2(source, target_lm_dir / filename)


def _validate_teacher(teacher_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(teacher_dir / "model_revision.json")
    if manifest.get("source_model_revision") != TEACHER_MODEL_REVISION:
        raise ValueError(
            "Teacher revision must be the pinned Kitten checkpoint "
            f"{TEACHER_MODEL_REVISION}, got {manifest.get('source_model_revision')!r}"
        )
    if manifest.get("upstream_code_revision") != UPSTREAM_CODE_REVISION:
        raise ValueError(
            "Teacher upstream_code_revision does not match the required "
            f"{UPSTREAM_CODE_REVISION}"
        )

    teacher_config = _read_json(teacher_dir / "lm" / "config.json")
    teacher_architecture = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": STUDENT_VOCAB_SIZE,
        "tie_word_embeddings": True,
    }
    mismatches = {
        name: {"expected": expected, "actual": teacher_config.get(name)}
        for name, expected in teacher_architecture.items()
        if teacher_config.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Unexpected pinned teacher architecture: {json.dumps(mismatches, sort_keys=True)}")

    top_config = _read_json(teacher_dir / "config.json")
    if top_config.get("speaker_embedding_dim") != SPEAKER_EMBEDDING_DIM:
        raise ValueError("Pinned teacher must use 512-dimensional speaker embeddings")
    return manifest, teacher_config


def _prepare_output(output_dir: Path, teacher_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_teacher = teacher_dir.resolve()
    if resolved_output == resolved_teacher or resolved_teacher in resolved_output.parents:
        raise ValueError("Output directory must not be the teacher directory or one of its descendants")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_student(
    base_model: str,
    teacher: str | Path,
    output: str | Path,
    *,
    experiment_id: str = "init",
    base_revision: str | None = None,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM

    teacher_dir = Path(teacher).resolve()
    output_dir = Path(output).resolve()
    teacher_manifest, teacher_config = _validate_teacher(teacher_dir)
    _prepare_output(output_dir, teacher_dir)

    base_config = AutoConfig.from_pretrained(base_model, revision=base_revision)
    base_architecture = {
        **STUDENT_ARCHITECTURE,
        "vocab_size": SOURCE_VOCAB_SIZE,
    }
    _validate_architecture(base_config, base_architecture, "Qwen3-0.6B base")
    _apply_teacher_settings(base_config, teacher_config)
    resolved_base_revision = base_revision or getattr(base_config, "_commit_hash", None)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=resolved_base_revision,
        config=base_config,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    _validate_architecture(model.config, base_architecture, "loaded Qwen3-0.6B base")
    _initialize_kitten_vocabulary(model)

    speaker_projection = nn.Sequential(
        nn.Linear(SPEAKER_EMBEDDING_DIM, STUDENT_HIDDEN_SIZE),
        nn.LayerNorm(STUDENT_HIDDEN_SIZE),
    ).to(dtype=model.get_input_embeddings().weight.dtype)
    model.spk_proj = speaker_projection
    model.config.architectures = ["Qwen3ForCausalLM"]
    model.config.tie_word_embeddings = True
    model.save_pretrained(
        output_dir / "lm",
        safe_serialization=True,
        max_shard_size="5GB",
    )
    checkpoint = output_dir / "lm" / "model.safetensors"
    if not checkpoint.is_file() or list((output_dir / "lm").glob("model-*.safetensors")):
        raise RuntimeError("Student checkpoint was not saved as one unsharded model.safetensors file")

    _copy_teacher_assets(teacher_dir, output_dir)
    provenance = {
        "format": "kitten2-student-init-v1",
        "source_model_revision": f"{STUDENT_SOURCE_REVISION}-{experiment_id}",
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "teacher_model_revision": teacher_manifest["source_model_revision"],
        "base_model": base_model,
        "base_model_revision": resolved_base_revision,
        "student_architecture": "qwen3-0.6b-kitten",
        "student_hidden_size": STUDENT_HIDDEN_SIZE,
        "student_intermediate_size": STUDENT_INTERMEDIATE_SIZE,
        "student_vocab_size": STUDENT_VOCAB_SIZE,
        "initialization_only": True,
    }
    (output_dir / "model_revision.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    report = inspect_checkpoint(output_dir)
    if not report["valid"]:
        raise RuntimeError("Initialized student failed inspection: " + "; ".join(report["errors"]))
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE_MODEL, help="Hugging Face Qwen3-0.6B PyTorch model")
    parser.add_argument("--base-revision", help="Optional immutable base model revision")
    parser.add_argument("--teacher", type=Path, required=True, help="Pinned local KittenTTS 2 model directory")
    parser.add_argument("--output", type=Path, required=True, help="New student model directory")
    parser.add_argument("--experiment-id", default="init", help="Short student initialization identifier")
    args = parser.parse_args(argv)
    report = build_student(
        args.base,
        args.teacher,
        args.output,
        experiment_id=args.experiment_id,
        base_revision=args.base_revision,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
