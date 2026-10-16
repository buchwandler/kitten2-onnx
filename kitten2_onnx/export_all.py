"""Build a self-contained ONNX Runtime bundle from a pinned KittenTTS 2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kitten2_onnx.export_flow import export_flow
from kitten2_onnx.export_lm import RUNTIME_PROFILES, export_lm
from kitten2_onnx.export_mel2wav import export_mel2wav
from kitten2_onnx.export_speaker import export_speaker
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from kitten2_onnx.s3gen_reference import S3GEN_REPOSITORY, S3GEN_REVISION
from kitten2_onnx.validate_bundle import validate_bundle
from scripts.capture_decoder_reference import capture_reference, write_capture
from scripts.inspect_model import inspect_checkpoint

TOKENIZER_SUFFIXES = {".json", ".txt", ".model", ".tiktoken", ".vocab"}
VERSION_PACKAGES = (
    "kittenml",
    "kitten-text-processing",
    "numpy",
    "onnx",
    "onnxruntime",
    "safetensors",
    "torch",
    "transformers",
)
MANAGED_DIRECTORIES = ("decoder", "lm", "speaker", "tokenizer", "voices")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tokenizer_assets(lm_dir: Path, tokenizer_dir: Path) -> list[str]:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in sorted(lm_dir.iterdir()):
        if source.is_file() and source.suffix.lower() in TOKENIZER_SUFFIXES:
            shutil.copy2(source, tokenizer_dir / source.name)
            copied.append(source.name)
    if not any(name in copied for name in ("tokenizer.json", "vocab.json", "tokenizer.model")):
        raise FileNotFoundError(f"No supported tokenizer assets found in {lm_dir}")
    return copied


def _bundle_files(bundle_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(bundle_dir).as_posix(): _sha256(path)
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file() and path.name != "bundle.json"
    }


def _package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in VERSION_PACKAGES
    }


def _check_onnx_graphs(graphs: list[Path]) -> list[str]:
    import onnx

    for graph in graphs:
        onnx.checker.check_model(str(graph))
    return [path.name for path in graphs]


def export_all(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    voice: str = "Bruno",
    profile_name: str = "dev-fp32",
    input_mode: str = "inputs_embeds",
    opset: int = 18,
    token_count: int = 32,
    validate: bool = False,
) -> Path:
    """Export graphs, package one built-in voice, and write bundle metadata."""
    if profile_name not in RUNTIME_PROFILES:
        raise ValueError(f"Unknown export profile {profile_name!r}; choose from {sorted(RUNTIME_PROFILES)}")

    if input_mode not in {"inputs_embeds", "input_ids"}:
        raise ValueError("input_mode must be 'inputs_embeds' or 'input_ids'")
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if (
        output_dir == model_dir
        or output_dir.is_relative_to(model_dir)
        or model_dir.is_relative_to(output_dir)
    ):
        raise ValueError("output_dir must not overlap the source model directory")

    inspection = inspect_checkpoint(model_dir)
    if not inspection["valid"]:
        errors = "\n".join(f"- {error}" for error in inspection["errors"])
        raise ValueError(f"Source checkpoint validation failed:\n{errors}")

    model_config = inspection["kitten_config"]
    revision_path = model_dir / "model_revision.json"
    if not revision_path.is_file():
        raise FileNotFoundError(f"Pinned model revision metadata is missing: {revision_path}")
    model_revision = json.loads(revision_path.read_text(encoding="utf-8"))
    source_revision = model_revision.get("source_model_revision")
    upstream_revision = model_revision.get("upstream_code_revision")
    if not source_revision:
        raise ValueError(f"No source_model_revision is recorded in {revision_path}")
    if upstream_revision != UPSTREAM_CODE_REVISION:
        raise ValueError(
            f"Checkpoint upstream revision {upstream_revision!r} does not match "
            f"the pinned exporter revision {UPSTREAM_CODE_REVISION!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in MANAGED_DIRECTORIES:
        managed = output_dir / name
        if managed.exists():
            shutil.rmtree(managed)

    speaker_dir = output_dir / "speaker"
    lm_dir = output_dir / "lm"
    voice_dir = output_dir / "voices"
    flow_dir = output_dir / "decoder" / "flow"
    mel2wav_dir = output_dir / "decoder" / "mel2wav"
    tokenizer_dir = output_dir / "tokenizer"
    voice_dir.mkdir(parents=True)

    export_speaker(model_dir, speaker_dir, opset=opset)
    export_lm(model_dir, lm_dir, profile_name=profile_name, input_mode=input_mode, opset=opset)

    voice_arrays, voice_metadata = capture_reference(model_dir, voice, token_count=token_count)
    voice_path = voice_dir / f"{voice}.npz"
    write_capture(voice_path, voice_arrays, voice_metadata)

    _, flow_metadata_path = export_flow(voice_path, flow_dir, opset=opset)
    _, mel2wav_metadata_path = export_mel2wav(voice_path, mel2wav_dir, opset=opset)
    tokenizer_assets = _copy_tokenizer_assets(model_dir / "lm", tokenizer_dir)

    flow_metadata = json.loads(flow_metadata_path.read_text(encoding="utf-8"))
    mel2wav_metadata = json.loads(mel2wav_metadata_path.read_text(encoding="utf-8"))
    if not flow_metadata["parity_passed"]:
        raise ValueError("Flow graph failed its export parity checks")
    if not mel2wav_metadata["parity_passed"]:
        raise ValueError("Mel-to-waveform graph failed its export parity checks")

    lm_metadata = json.loads((lm_dir / "lm.meta.json").read_text(encoding="utf-8"))
    speaker_metadata = json.loads((speaker_dir / "speaker.meta.json").read_text(encoding="utf-8"))
    lm_bundle_entry = {
        "model": "lm/lm.onnx",
        "metadata": "lm/lm.meta.json",
    }
    if "embedding_file" in lm_metadata:
        lm_bundle_entry["embeddings"] = f"lm/{lm_metadata['embedding_file']}"
    graphs = [
        lm_dir / "lm.onnx",
        speaker_dir / "speaker.onnx",
        flow_dir / flow_metadata["graph"],
        mel2wav_dir / mel2wav_metadata["graph"],
    ]

    bundle: dict[str, Any] = {
        "format": "kitten2-onnx-bundle-v1",
        "sample_rate": int(model_config["sample_rate"]),
        "batch_size": 1,
        "default_voice": model_config.get("default_voice", voice),
        "default_preset": model_config.get("default_preset"),
        "lm": lm_bundle_entry,
        "speaker": {
            "model": "speaker/speaker.onnx",
            "metadata": "speaker/speaker.meta.json",
        },
        "decoder": {
            "flow": f"decoder/flow/{flow_metadata['graph']}",
            "flow_metadata": "decoder/flow/flow.meta.json",
            "mel2wav": f"decoder/mel2wav/{mel2wav_metadata['graph']}",
            "mel2wav_metadata": "decoder/mel2wav/mel2wav.meta.json",
        },
        "tokenizer": "tokenizer",
        "tokenizer_assets": tokenizer_assets,
        "voices": {voice: f"voices/{voice}.npz"},
        "token_map": model_config["token_map"],
        "decode_presets": model_config.get("decode_presets", {}),
        "generation": model_config.get("generation", {}),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "revisions": {
            "source_model_repository": model_revision.get("repo_id"),
            "source_model_revision": source_revision,
            "upstream_code_revision": upstream_revision,
            "s3gen_repository": S3GEN_REPOSITORY,
            "s3gen_revision": S3GEN_REVISION,
        },
        "export": {
            "runtime_profile": profile_name,
            "opset": int(opset),
            "source_model_dtype": lm_metadata["source_dtype"],
            "graph_dtype": lm_metadata["graph_dtype"],
            "lm_precision": lm_metadata["graph_dtype"],
            "speaker_precision": speaker_metadata["dtype"],
            "speaker_output_cast": {
                "from": "float32",
                "to": lm_metadata["io_dtype"],
                "applied_by": "Kitten2OnnxLM",
            },
            "decoder_precision": {
                "flow": flow_metadata["precision"],
                "mel2wav": mel2wav_metadata["precision"],
            },
            "speaker_hidden_size": int(speaker_metadata["hidden_size"]),
        },
        "files": {},
    }
    bundle["files"] = _bundle_files(output_dir)
    bundle_path = output_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checked_graphs = _check_onnx_graphs(graphs)
    if len(checked_graphs) != 4:
        raise RuntimeError("Expected to validate all four ONNX graphs")
    if validate:
        validation_report = validate_bundle(output_dir)
        if not validation_report["valid"]:
            details = "\n".join(f"- {error}" for error in validation_report["errors"])
            raise ValueError(f"Exported bundle validation failed:\n{details}")
    return bundle_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Pinned local KittenTTS 2 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Bundle output directory")
    parser.add_argument("--voice", default="Bruno", help="Built-in voice to package")
    parser.add_argument("--profile", choices=sorted(RUNTIME_PROFILES), default="dev-fp32")
    parser.add_argument(
        "--input-mode", choices=("inputs_embeds", "input_ids"), default="inputs_embeds"
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--token-count", type=int, default=32)
    parser.add_argument("--validate", action="store_true", help="Run full bundle validation with CPU ONNX Runtime")
    args = parser.parse_args(argv)

    bundle_path = export_all(
        args.model_dir,
        args.output,
        voice=args.voice,
        profile_name=args.profile,
        input_mode=args.input_mode,
        opset=args.opset,
        token_count=args.token_count,
        validate=args.validate,
    )
    print(bundle_path)


if __name__ == "__main__":
    main()
