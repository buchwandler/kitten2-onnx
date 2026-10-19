"""Validate a packaged KittenTTS 2 ONNX bundle and emit a CI-friendly report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from kitten2_onnx.decoder_reference import load_decoder_reference
from kitten2_onnx.export_flow import INPUT_NAMES as FLOW_INPUT_NAMES
from kitten2_onnx.export_flow import OUTPUT_NAMES as FLOW_OUTPUT_NAMES
from kitten2_onnx.export_mel2wav import INPUT_NAMES as MEL2WAV_INPUT_NAMES
from kitten2_onnx.export_mel2wav import OUTPUT_NAMES as MEL2WAV_OUTPUT_NAMES
from kitten2_onnx.tokens import TokenMap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Bundle path must be relative and contained: {relative_path!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Bundle path escapes its root: {relative_path!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Bundle file is missing: {relative_path}")
    return path


def _check_shape(name: str, expected: list[Any], actual: list[Any], errors: list[str]) -> None:
    if len(expected) != len(actual):
        errors.append(f"{name} rank differs from metadata: expected {len(expected)}, found {len(actual)}")
        return
    for axis, (expected_dim, actual_dim) in enumerate(zip(expected, actual, strict=True)):
        if isinstance(expected_dim, int) and isinstance(actual_dim, int) and expected_dim != actual_dim:
            errors.append(
                f"{name} dimension {axis} differs from metadata: "
                f"expected {expected_dim}, found {actual_dim}"
            )


def _shape_matches_metadata(
    tensor_name: str,
    actual_shape: tuple[int, ...],
    metadata: dict[str, Any],
    *,
    output: bool = False,
) -> bool:
    dynamic_axes = metadata.get("dynamic_axes", {})
    if not isinstance(dynamic_axes, dict):
        return False
    if output:
        expected_shape = metadata.get("output_shape")
    else:
        input_shapes = metadata.get("input_shapes")
        if not isinstance(input_shapes, dict):
            return False
        expected_shape = input_shapes.get(tensor_name)
    tensor_axes = dynamic_axes.get(tensor_name, {})
    if not isinstance(expected_shape, list) or not isinstance(tensor_axes, dict):
        return False
    if len(expected_shape) != len(actual_shape):
        return False

    dynamic_dimensions = {int(axis) for axis in tensor_axes}
    if any(axis < 0 or axis >= len(actual_shape) for axis in dynamic_dimensions):
        raise ValueError(f"Invalid dynamic axis metadata for {tensor_name}")
    return all(
        axis in dynamic_dimensions
        or (type(expected_dim) is int and expected_dim == actual_dim)
        for axis, (expected_dim, actual_dim) in enumerate(zip(expected_shape, actual_shape, strict=True))
    )


def _check_graph(
    *,
    root: Path,
    label: str,
    graph_path: Path,
    metadata: dict[str, Any],
    providers: list[str],
    expected_inputs: list[str],
    expected_outputs: list[str],
    input_shapes: dict[str, list[Any]] | None,
    output_shape: list[Any] | None,
    errors: list[str],
) -> dict[str, Any] | None:
    import onnx
    import onnxruntime as ort

    try:
        onnx.checker.check_model(str(graph_path))
    except (onnx.checker.ValidationError, ValueError, OSError, RuntimeError) as error:
        errors.append(f"{label} ONNX checker failed: {error}")
        return None

    session_options = None
    if label == "lm":
        memory_profile = metadata.get("runtime_memory_profile", "default")
        if memory_profile == "low-memory-exact-ternary":
            if providers != ["CPUExecutionProvider"]:
                errors.append("Low-memory exact-ternary profile requires CPUExecutionProvider")
                return None
            from kitten2_onnx.runtime import _cpu_session_options

            session_options = _cpu_session_options(memory_profile)
        elif memory_profile != "default":
            errors.append(f"Unsupported runtime memory profile: {memory_profile!r}")
            return None

    try:
        if session_options is None:
            session = ort.InferenceSession(str(graph_path), providers=providers)
        else:
            session = ort.InferenceSession(
                str(graph_path), sess_options=session_options, providers=providers
            )
    except (ValueError, OSError, RuntimeError) as error:
        errors.append(f"{label} ONNX Runtime session failed: {error}")
        return None

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_names = [value.name for value in inputs]
    output_names = [value.name for value in outputs]
    if input_names != expected_inputs:
        errors.append(f"{label} graph inputs differ from metadata: {input_names} != {expected_inputs}")
    if output_names != expected_outputs:
        errors.append(f"{label} graph outputs differ from metadata: {output_names} != {expected_outputs}")
    if label == "lm":
        input_types = {value.name: value.type for value in inputs}
        expected_float_type = {
            "float32": "tensor(float)",
            "float16": "tensor(float16)",
            "bfloat16": "tensor(bfloat16)",
        }.get(metadata.get("io_dtype"))
        input_mode = metadata.get("input_mode", "inputs_embeds")
        if input_mode == "inputs_embeds":
            if input_types.get("inputs_embeds") != expected_float_type:
                errors.append("LM inputs_embeds type differs from io_dtype metadata")
        elif input_mode == "input_ids":
            if input_types.get("input_ids") != "tensor(int64)":
                errors.append("LM input_ids graph input must be int64")
            if input_types.get("speaker_hidden") != expected_float_type:
                errors.append("LM speaker_hidden type differs from io_dtype metadata")
        else:
            errors.append(f"Unsupported LM input mode: {input_mode!r}")

    for value in inputs:
        expected = (input_shapes or {}).get(value.name)
        if expected is not None:
            _check_shape(f"{label} input {value.name}", expected, value.shape, errors)
    for value in outputs:
        if output_shape is not None:
            _check_shape(f"{label} output {value.name}", output_shape, value.shape, errors)

    return {
        "graph": graph_path.relative_to(root).as_posix(),
        "provider": session.get_providers()[0],
        "inputs": input_names,
        "outputs": output_names,
    }


def validate_bundle(
    bundle_dir: str | Path,
    *,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Check file integrity, graph contracts, token/config invariants, and voice assets."""
    root = Path(bundle_dir).resolve()
    providers = providers or ["CPUExecutionProvider"]
    report: dict[str, Any] = {
        "format": "kitten2-onnx-bundle-validation-v1",
        "bundle_dir": str(root),
        "valid": False,
        "checks": [],
        "errors": [],
        "files": {"listed": 0, "verified": 0},
        "graphs": {},
        "voices": {},
    }
    errors: list[str] = report["errors"]
    bundle_path = root / "bundle.json"
    if not bundle_path.is_file():
        errors.append("bundle.json is missing")
        return report
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"Cannot read bundle.json: {error}")
        return report
    if not isinstance(bundle, dict) or bundle.get("format") != "kitten2-onnx-bundle-v1":
        errors.append("Unsupported or malformed bundle format")
        return report
    report["checks"].append("bundle format")

    files = bundle.get("files")
    if not isinstance(files, dict) or not files:
        errors.append("bundle.json files must be a non-empty path-to-SHA256 object")
        files = {}
    report["files"]["listed"] = len(files)
    for relative_path, expected_hash in files.items():
        try:
            path = _resolve_file(root, relative_path)
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"Invalid SHA256 entry for {relative_path!r}")
            if _sha256(path) != expected_hash:
                raise ValueError(f"SHA256 mismatch for {relative_path}")
            report["files"]["verified"] += 1
        except (OSError, TypeError, ValueError) as error:
            errors.append(str(error))
    if report["files"]["verified"] == len(files) and files:
        report["checks"].append("file hashes")

    def require(relative_path: str) -> Path | None:
        try:
            if relative_path not in files:
                raise ValueError(f"Referenced bundle file is absent from the SHA256 manifest: {relative_path}")
            return _resolve_file(root, relative_path)
        except (OSError, TypeError, ValueError) as error:
            errors.append(str(error))
            return None

    def read_metadata(relative_path: str) -> dict[str, Any] | None:
        path = require(relative_path)
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"Cannot read {relative_path}: {error}")
            return None
        if not isinstance(value, dict):
            errors.append(f"Metadata at {relative_path} must be a JSON object")
            return None
        return value

    try:
        token_map = TokenMap.from_config({"token_map": bundle["token_map"]})
        if token_map.num_audio_tokens <= 0 or token_map.audio_id_end != token_map.final_seg_id:
            raise ValueError("audio token range must be positive and end at final_seg_id")
        markers = [
            token_map.speech_start_id,
            token_map.speech_end_id,
            token_map.text_start_id,
            token_map.text_end_id,
            token_map.start_id,
            token_map.stop_id,
            token_map.final_seg_id,
        ]
        reference_markers = [
            token_map.reference_speech_start_id,
            token_map.reference_speech_end_id,
            token_map.reference_text_start_id,
            token_map.reference_text_end_id,
        ]
        if any(reference_markers) and not all(reference_markers):
            raise ValueError("reference prompt markers must be either all present or all absent")
        if all(reference_markers):
            markers.extend(reference_markers)
        if len(set(markers)) != len(markers):
            raise ValueError("prompt markers must be distinct")
        report["checks"].append("token map")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"Invalid token map: {error}")
        token_map = None

    revisions = bundle.get("revisions", {})
    if not isinstance(revisions, dict):
        errors.append("Bundle revisions must be an object")
        revisions = {}
    for name in ("source_model_revision", "upstream_code_revision", "s3gen_revision"):
        if not isinstance(revisions.get(name), str) or not revisions[name]:
            errors.append(f"Bundle revision metadata is missing {name}")
    versions = bundle.get("versions", {})
    if not isinstance(versions, dict):
        errors.append("Bundle versions must be an object")
        versions = {}
    packages = versions.get("packages")
    if (
        not isinstance(versions.get("python"), str)
        or not isinstance(versions.get("platform"), str)
        or not isinstance(packages, dict)
    ):
        errors.append("Bundle environment version metadata is incomplete")
    else:
        required_packages = {"kitten-text-processing", "onnx", "onnxruntime", "torch", "transformers"}
        missing_packages = required_packages - packages.keys()
        if missing_packages:
            errors.append(f"Bundle package versions are missing {sorted(missing_packages)}")
        else:
            report["checks"].append("revision and environment metadata")

    lm = bundle.get("lm", {})
    if not isinstance(lm, dict):
        errors.append("Bundle LM contract must be an object")
        lm = {}
    speaker = bundle.get("speaker", {})
    if not isinstance(speaker, dict):
        errors.append("Bundle speaker contract must be an object")
        speaker = {}
    decoder = bundle.get("decoder", {})
    if not isinstance(decoder, dict):
        errors.append("Bundle decoder contract must be an object")
        decoder = {}
    lm_meta = read_metadata(lm.get("metadata", ""))
    speaker_meta = read_metadata(speaker.get("metadata", ""))
    flow_meta = read_metadata(decoder.get("flow_metadata", ""))
    mel_meta = read_metadata(decoder.get("mel2wav_metadata", ""))
    export_meta = bundle.get("export", {})
    if not isinstance(export_meta, dict):
        errors.append("Bundle export metadata must be an object")
        export_meta = {}
    if export_meta.get("runtime_profile") == "cuda-fp16":
        if lm_meta is not None and (
            lm_meta.get("graph_dtype") != "float16"
            or lm_meta.get("io_dtype") != "float16"
            or lm_meta.get("runtime_provider") != "CUDAExecutionProvider"
        ):
            errors.append("cuda-fp16 LM metadata must use float16 and CUDAExecutionProvider")
        expected_cast = {
            "from": "float32",
            "to": "float16",
            "applied_by": "Kitten2OnnxLM",
        }
        if export_meta.get("speaker_output_cast") != expected_cast:
            errors.append(
                "cuda-fp16 bundle must cast FP32 speaker output to FP16 in the LM runtime"
            )
        if speaker_meta is not None and speaker_meta.get("dtype") != "fp32":
            errors.append("cuda-fp16 bundle speaker projection must remain FP32")
        expected_decoder_precision = {"flow": "fp32", "mel2wav": "fp32"}
        if export_meta.get("decoder_precision") != expected_decoder_precision:
            errors.append("cuda-fp16 bundle must document FP32 flow and mel2wav graphs")
        if flow_meta is not None and flow_meta.get("precision") != "fp32":
            errors.append("cuda-fp16 bundle flow graph must remain FP32")
        if mel_meta is not None and mel_meta.get("precision") != "fp32":
            errors.append("cuda-fp16 bundle mel2wav graph must remain FP32")
        if not any(error.startswith("cuda-fp16") for error in errors):
            report["checks"].append("cuda-fp16 precision and cast contract")
    tokenizer_dir = bundle.get("tokenizer")
    tokenizer_assets = bundle.get("tokenizer_assets")
    if not isinstance(tokenizer_dir, str) or not isinstance(tokenizer_assets, list) or not tokenizer_assets or not all(isinstance(asset, str) for asset in tokenizer_assets):
        errors.append("Bundle tokenizer directory and asset list are required")
    else:
        for asset in tokenizer_assets:
            require((Path(tokenizer_dir) / asset).as_posix())
        if not any(Path(asset).name in {"tokenizer.json", "vocab.json", "tokenizer.model"} for asset in tokenizer_assets):
            errors.append("Bundle does not list a supported tokenizer vocabulary asset")
        else:
            report["checks"].append("tokenizer assets")

    graph_specs = (
        ("lm", lm.get("model", ""), lm_meta),
        ("speaker", speaker.get("model", ""), speaker_meta),
        ("flow", decoder.get("flow", ""), flow_meta),
        ("mel2wav", decoder.get("mel2wav", ""), mel_meta),
    )
    for label, _, metadata in graph_specs:
        if metadata is None:
            continue
        if label in {"lm", "flow", "mel2wav"} and metadata.get("source_model_revision") != revisions.get("source_model_revision"):
            errors.append(f"{label} source model revision differs from the bundle")
        if metadata.get("upstream_code_revision") is not None and metadata["upstream_code_revision"] != revisions.get("upstream_code_revision"):
            errors.append(f"{label} upstream code revision differs from the bundle")
        if metadata.get("s3gen_revision") is not None and metadata["s3gen_revision"] != revisions.get("s3gen_revision"):
            errors.append(f"{label} S3Gen revision differs from the bundle")
    for label, relative_graph, metadata in graph_specs:
        graph_path = require(relative_graph)
        if graph_path is None or metadata is None:
            continue
        if metadata.get("graph") is not None and metadata["graph"] != graph_path.name:
            errors.append(f"{label} graph path differs from metadata")
        if label == "lm":
            inputs = metadata.get("input_names", [])
            outputs = metadata.get("output_names", [])
        elif label == "speaker":
            inputs = metadata.get("input_names", ["speaker_embedding"])
            outputs = metadata.get("output_names", ["speaker_hidden"])
        elif label == "flow":
            inputs = metadata.get("input_names", list(FLOW_INPUT_NAMES))
            outputs = metadata.get("output_names", list(FLOW_OUTPUT_NAMES))
        else:
            inputs = metadata.get("input_names", list(MEL2WAV_INPUT_NAMES))
            outputs = metadata.get("output_names", list(MEL2WAV_OUTPUT_NAMES))
        input_shapes = metadata.get("input_shapes")
        if label == "lm":
            input_shapes = None
        output_shape = metadata.get("output_shape")
        if label == "speaker":
            input_shapes = {"speaker_embedding": [1, metadata.get("speaker_dim")]}
            output_shape = [1, metadata.get("hidden_size")]
        if label == "lm":
            output_shape = None
        graph_report = _check_graph(
            root=root,
            label=label,
            graph_path=graph_path,
            metadata=metadata,
            providers=providers,
            expected_inputs=inputs,
            expected_outputs=outputs,
            input_shapes=input_shapes if isinstance(input_shapes, dict) else None,
            output_shape=output_shape if isinstance(output_shape, list) else None,
            errors=errors,
        )
        if graph_report is not None:
            report["graphs"][label] = graph_report
    if len(report["graphs"]) == 4:
        report["checks"].append("ONNX checker, Runtime sessions, and graph I/O")

    try:
        if lm_meta is None:
            raise ValueError("LM metadata is missing")
        input_mode = lm_meta.get("input_mode", "inputs_embeds")
        if input_mode == "inputs_embeds":
            embedding_file = lm_meta["embedding_file"]
            expected_embedding = (Path(lm["model"]).parent / embedding_file).as_posix()
            if lm.get("embeddings") != expected_embedding:
                raise ValueError("Bundle embedding path differs from LM metadata")
            embedding_path = require(lm["embeddings"])
            if embedding_path is not None:
                embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
                expected_shape = (int(lm_meta["vocab_size"]), int(lm_meta["hidden_size"]))
                if embeddings.shape != expected_shape:
                    raise ValueError(
                        f"Embedding shape {embeddings.shape} differs from LM config {expected_shape}"
                    )
                storage_dtype = lm_meta.get("embedding_storage_dtype", str(embeddings.dtype))
                if storage_dtype == "bfloat16-u16":
                    runtime_dtype = lm_meta.get("embedding_runtime_dtype")
                    if embeddings.dtype != np.uint16 or runtime_dtype != lm_meta.get("io_dtype"):
                        raise ValueError("bfloat16-u16 embedding storage metadata is inconsistent")
                elif str(embeddings.dtype) != storage_dtype:
                    raise ValueError("Embedding array dtype differs from LM metadata")
                report["checks"].append("LM embedding shape")
        elif input_mode == "input_ids":
            if "embedding_file" in lm_meta or lm.get("embeddings") is not None:
                raise ValueError("input_ids LM must not use a separate embedding table")
            input_names = set(lm_meta.get("input_names", []))
            if not {"input_ids", "speaker_hidden"}.issubset(input_names):
                raise ValueError("input_ids LM metadata must declare speaker_hidden input")
            report["checks"].append("LM input_ids and speaker injection contract")
        else:
            raise ValueError(f"Unsupported LM input mode: {input_mode!r}")
        if lm_meta is not None and speaker_meta is not None:
            if int(speaker_meta["hidden_size"]) != int(lm_meta["hidden_size"]):
                raise ValueError("Speaker output size differs from LM hidden size")
            report["checks"].append("speaker/LM hidden size")
        if mel_meta is not None and int(bundle["sample_rate"]) != int(mel_meta["sample_rate"]):
            raise ValueError("Bundle sample rate differs from mel2wav metadata")
    except (KeyError, OSError, TypeError, ValueError) as error:
        errors.append(f"Model contract validation failed: {error}")

    if token_map is not None and flow_meta is not None and mel_meta is not None:
        silence_count = int(flow_meta.get("silence_tail", {}).get("count", 0))
        token_mel_ratio = int(flow_meta.get("token_mel_ratio", 0))
        voices = bundle.get("voices", {})
        if not isinstance(voices, dict):
            errors.append("Bundle voices must be an object")
            voices = {}
        if not voices:
            errors.append("Bundle must contain at least one built-in voice")
        for voice_name, relative_voice in voices.items():
            voice_path = require(relative_voice)
            if voice_path is None:
                continue
            try:
                reference = load_decoder_reference(voice_path)
                arrays = reference.arrays
                for name, array in arrays.items():
                    if array.dtype.kind in "fc" and not np.isfinite(array).all():
                        raise ValueError(f"Voice {voice_name} array {name} contains non-finite values")
                prompt_tokens = arrays["conditioning_prompt_token"]
                prompt_lengths = arrays["conditioning_prompt_token_len"]
                prompt_feat = arrays["conditioning_prompt_feat"]
                embedding = arrays["conditioning_embedding"]
                expected_conditioning = {
                    "prompt_token": prompt_tokens,
                    "prompt_token_len": prompt_lengths,
                    "prompt_feat": prompt_feat,
                    "embedding": embedding,
                }
                for input_name, array in expected_conditioning.items():
                    if not _shape_matches_metadata(input_name, array.shape, flow_meta):
                        raise ValueError(
                            f"Voice {voice_name} conditioning shape mismatch for {input_name}"
                        )
                if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1 or prompt_lengths.shape != (1,):
                    raise ValueError(f"Voice {voice_name} prompt token tensors must have batch size one")
                if prompt_lengths.dtype.kind not in "iu":
                    raise ValueError(f"Voice {voice_name} prompt token length must be an integer")
                prompt_length = int(prompt_lengths[0])
                if prompt_length < 1 or prompt_length > prompt_tokens.shape[1]:
                    raise ValueError(f"Voice {voice_name} prompt token length is invalid")

                speech_tokens = arrays["speech_tokens"]
                speech_lengths = arrays["speech_token_lengths"]
                if speech_lengths.shape != (1,) or speech_lengths.dtype.kind not in "iu":
                    raise ValueError(f"Voice {voice_name} speech token lengths must be one integer")
                speech_token_count = int(speech_lengths[0])
                if speech_tokens.ndim != 2 or speech_tokens.shape[0] != 1:
                    raise ValueError(f"Voice {voice_name} speech tokens must have batch size one")
                if speech_token_count < 1 or speech_tokens.shape[1] != speech_token_count:
                    raise ValueError(f"Voice {voice_name} speech token length does not match its tokens")
                expected_padded_tokens = speech_token_count + silence_count
                if not _shape_matches_metadata(
                    "speech_tokens", (1, expected_padded_tokens), flow_meta
                ):
                    raise ValueError(f"Voice {voice_name} speech token shape differs from flow metadata")
                if not _shape_matches_metadata("speech_token_lengths", (1,), flow_meta):
                    raise ValueError(f"Voice {voice_name} speech length input must have batch size one")
                expected_noise_frames = token_mel_ratio * (prompt_length + expected_padded_tokens)
                if not _shape_matches_metadata("flow_noise", arrays["flow_noise"].shape, flow_meta):
                    raise ValueError(f"Voice {voice_name} flow noise shape differs from static flow dimensions")
                if arrays["flow_noise"].shape[2] != expected_noise_frames:
                    raise ValueError(f"Voice {voice_name} flow noise shape is inconsistent with conditioning")
                if not _shape_matches_metadata(
                    "mel", arrays["flow_mel"].shape, flow_meta, output=True
                ):
                    raise ValueError(f"Voice {voice_name} flow mel shape differs from flow metadata")
                if arrays["speaker_embedding"].shape != (int(speaker_meta["speaker_dim"]),):
                    raise ValueError(f"Voice {voice_name} speaker embedding shape differs from speaker metadata")
                audio_vocab_size = int(flow_meta.get("speech_vocab_size", token_map.num_audio_tokens))
                if audio_vocab_size != token_map.num_audio_tokens:
                    raise ValueError("Flow speech vocabulary differs from the bundle token map")
                for name in ("conditioning_prompt_token", "speech_tokens", "reference_tokens"):
                    tokens = arrays[name]
                    if tokens.dtype.kind not in "iu":
                        raise ValueError(f"Voice {voice_name} {name} must contain integer token IDs")
                    if tokens.size and (
                        tokens.min() < 0 or tokens.max() >= audio_vocab_size
                    ):
                        raise ValueError(f"Voice {voice_name} {name} is outside the audio vocabulary")
                    if name == "reference_tokens" and tokens.ndim != 1:
                        raise ValueError(f"Voice {voice_name} reference tokens must be one-dimensional")

                for input_name, array_name in (
                    ("mel", "mel2wav_input"),
                    ("phase_noise", "vocoder_phase_noise"),
                    ("sine_noise", "vocoder_sine_noise"),
                ):
                    if not _shape_matches_metadata(
                        input_name, arrays[array_name].shape, mel_meta
                    ):
                        raise ValueError(
                            f"Voice {voice_name} mel2wav conditioning shape mismatch for {input_name}"
                        )
                expected_samples = (
                    arrays["mel2wav_input"].shape[2]
                    * int(mel_meta["samples_per_mel_frame"])
                )
                if arrays["vocoder_sine_noise"].shape[2] != expected_samples:
                    raise ValueError(f"Voice {voice_name} vocoder noise length is inconsistent with mel frames")
                expected_waveform_shape = (1, expected_samples)
                if arrays["mel2wav_waveform"].shape != expected_waveform_shape:
                    raise ValueError(f"Voice {voice_name} mel2wav waveform length differs from mel frames")
                if arrays["waveform"].shape != expected_waveform_shape:
                    raise ValueError(f"Voice {voice_name} waveform length differs from mel2wav output")
                if reference.metadata.get("voice") != voice_name:
                    raise ValueError(f"Voice {voice_name} metadata identity differs from the manifest")
                if reference.metadata.get("source_model_revision") != revisions.get("source_model_revision"):
                    raise ValueError(f"Voice {voice_name} source model revision differs from the bundle")
                if reference.metadata.get("upstream_code_revision") != revisions.get("upstream_code_revision"):
                    raise ValueError(f"Voice {voice_name} upstream code revision differs from the bundle")
                if reference.metadata.get("s3gen_revision") != revisions.get("s3gen_revision"):
                    raise ValueError(f"Voice {voice_name} S3Gen revision differs from the bundle")
                report["voices"][voice_name] = {
                    "path": relative_voice,
                    "prompt_tokens": prompt_length,
                    "reference_tokens": int(arrays["reference_tokens"].size),
                    "speaker_embedding_shape": list(arrays["speaker_embedding"].shape),
                }
            except (KeyError, IndexError, OSError, TypeError, ValueError) as error:
                errors.append(f"Voice {voice_name} conditioning validation failed: {error}")
        if report["voices"] and len(report["voices"]) == len(voices):
            report["checks"].append("voice conditioning shapes")

    report["valid"] = not errors
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path, help="Directory containing bundle.json")
    parser.add_argument("--output", type=Path, help="Write the JSON validation report to this path")
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="ONNX Runtime provider to use (repeatable, defaults to CPUExecutionProvider)",
    )
    args = parser.parse_args(argv)

    report = validate_bundle(args.bundle_dir, providers=args.providers)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
