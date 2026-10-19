from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

from kitten2_onnx.reference import UPSTREAM_CODE_REVISION

BASE_FILES = (
    "config.json",
    "lm/config.json",
    "lm/generation_config.json",
    "lm/tokenizer.json",
    "lm/tokenizer_config.json",
    "lm/special_tokens_map.json",
    "lm/added_tokens.json",
    "lm/vocab.json",
    "lm/merges.txt",
)


def _repo_path(value: str, base: PurePosixPath | None = None) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    if base is None:
        base = PurePosixPath()
    if base.parts and path.parts and path.parts[0] == base.parts[0]:
        return path.as_posix()
    return (base / path).as_posix()


def voice_files(
    config: dict[str, Any],
    voice_index: dict[str, Any],
    voice_name: str,
    available_files: set[str],
) -> set[str]:
    if voice_name not in voice_index:
        names = ", ".join(sorted(voice_index))
        raise ValueError(f"Unknown voice {voice_name!r}. Available voices: {names}")

    index_path = _repo_path(str(config.get("voices", "voices/voices.json")))
    index_parent = PurePosixPath(index_path).parent
    selected = voice_index[voice_name]
    if not isinstance(selected, dict):
        raise TypeError(f"Voice entry {voice_name!r} must be a JSON object")

    files = {index_path}
    for key in ("reference", "artifacts"):
        value = selected.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Voice entry {voice_name!r} has no {key!r} path")
        candidate = PurePosixPath(value)
        if candidate.parts and candidate.parts[0] == index_parent.parts[0]:
            path = _repo_path(value)
        else:
            path = _repo_path(value, index_parent)
        files.add(path)

    default_decoder = config.get("default_decoder")
    decoders = config.get("decoders", {})
    decoder_path = decoders.get(default_decoder) if isinstance(decoders, dict) else None
    if isinstance(decoder_path, str):
        files.add(_repo_path(decoder_path))
    elif isinstance(default_decoder, str):
        if "/" in default_decoder:
            files.add(_repo_path(default_decoder))
        else:
            conventional_path = f"cpp/{default_decoder}/decoder.pt"
            if conventional_path in available_files:
                files.add(conventional_path)
    if not any(path.startswith(("cpp/", "decoders/")) for path in files):
        raise ValueError(f"Cannot derive the default decoder files for voice {voice_name!r}")

    missing = sorted(files - available_files)
    if missing:
        raise ValueError("Selected voice assets are missing from this revision: " + ", ".join(missing))
    return files

def all_voice_files(
    config: dict[str, Any],
    voice_index: dict[str, Any],
    available_files: set[str],
) -> set[str]:
    files: set[str] = set()
    for voice_name in sorted(voice_index):
        files.update(voice_files(config, voice_index, voice_name, available_files))
    return files



def download_model(
    repo_id: str,
    revision: str,
    output: Path,
    *,
    plain_lm: bool,
    voice: str | None,
    all_voices: bool = False,
) -> dict[str, Any]:
    if voice is not None and all_voices:
        raise ValueError("Select either one voice or all voices, not both")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    model_info = api.model_info(repo_id, revision=revision)
    resolved_revision = model_info.sha
    available_files = {sibling.rfilename for sibling in model_info.siblings or []}

    files = set(BASE_FILES)
    selected_voices: list[str] = []
    if plain_lm:
        files.add("lm/model.safetensors")

    if voice is not None or all_voices:
        config_path = hf_hub_download(repo_id, "config.json", revision=resolved_revision)
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        index_path = _repo_path(str(config.get("voices", "voices/voices.json")))
        index_local_path = hf_hub_download(repo_id, index_path, revision=resolved_revision)
        voice_index = json.loads(Path(index_local_path).read_text(encoding="utf-8"))
        selected_voices = sorted(voice_index) if all_voices else [voice]
        if all_voices:
            files.update(all_voice_files(config, voice_index, available_files))
        else:
            files.update(voice_files(config, voice_index, voice, available_files))

    missing = sorted(files - available_files)
    if missing:
        raise ValueError("Required files are missing from this revision: " + ", ".join(missing))

    output = output.resolve()
    snapshot_download(
        repo_id,
        revision=resolved_revision,
        local_dir=output,
        allow_patterns=sorted(files),
    )
    manifest = {
        "format": "kitten2-onnx-reference-model-v1",
        "repo_id": repo_id,
        "requested_revision": revision,
        "source_model_revision": resolved_revision,
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "downloaded_files": sorted(files),
        "voice": voice,
        "voices": selected_voices,
    }
    (output / "model_revision.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pinned KittenTTS 2 reference model.")
    parser.add_argument("repo_id", help="Hugging Face model repository, for example KittenML/kitten-tts-2")
    parser.add_argument("--revision", required=True, help="Pinned branch, tag, or commit revision")
    parser.add_argument("--output", type=Path, required=True, help="Local output directory")
    parser.add_argument("--plain-lm", action="store_true", help="Download lm/model.safetensors")
    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument("--voice", help="Also download one built-in voice and its selected decoder assets")
    voice_group.add_argument("--all-voices", action="store_true", help="Download every built-in voice reference and the shared decoder")
    args = parser.parse_args()

    manifest = download_model(
        args.repo_id,
        args.revision,
        args.output,
        plain_lm=args.plain_lm,
        voice=args.voice,
        all_voices=args.all_voices,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
