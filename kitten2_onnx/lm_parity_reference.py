from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_FORMAT = "kitten2-onnx-lm-parity-reference-v1"
LOGITS_RTOL = 1e-4
LOGITS_ATOL = 2e-4
CACHE_RTOL = 1e-4
CACHE_ATOL = 2e-3
_ARRAY_NAMES = (
    "prompt_ids",
    "prefill_inputs_embeds",
    "speaker_hidden",
    "generated_ids",
    "token_embeds",
    "logits",
    "cache_keys",
    "cache_values",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def source_identity(model_dir: Path, baseline_bundle: Path, fixture_path: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    baseline_bundle = baseline_bundle.resolve()
    fixture_path = fixture_path.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    lm_dir = model_dir / config.get("lm_dir", "lm")
    voices_path = model_dir / config.get("voices", "voices/voices.json")
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    voice_path = voices_path.parent / voices["Bruno"]["artifacts"]
    model_revision_path = model_dir / "model_revision.json"
    model_revision = (
        json.loads(model_revision_path.read_text(encoding="utf-8")).get("source_model_revision")
        if model_revision_path.is_file()
        else None
    )
    source_files = {
        "lm/model.safetensors": lm_dir / "model.safetensors",
        "lm/config.json": lm_dir / "config.json",
        "config.json": model_dir / "config.json",
        "voices/voices.json": voices_path,
        "voice/Bruno.npz": voice_path,
        "fixture/reference_bruno.npz": fixture_path,
        "tokenizer/added_tokens.json": lm_dir / "added_tokens.json",
        "tokenizer/merges.txt": lm_dir / "merges.txt",
        "tokenizer/special_tokens_map.json": lm_dir / "special_tokens_map.json",
        "tokenizer/tokenizer.json": lm_dir / "tokenizer.json",
        "tokenizer/tokenizer_config.json": lm_dir / "tokenizer_config.json",
        "tokenizer/vocab.json": lm_dir / "vocab.json",
    }
    source_hashes = {
        name: sha256_file(path) for name, path in source_files.items() if path.is_file()
    }
    if len(source_hashes) != len(source_files):
        missing = sorted(set(source_files) - set(source_hashes))
        raise FileNotFoundError(f"Missing source identity files: {missing}")

    bundle_manifest_path = baseline_bundle / "bundle.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    declared_hashes = bundle_manifest.get("files")
    if not isinstance(declared_hashes, dict) or not declared_hashes:
        raise ValueError(f"Baseline bundle has no file hashes: {bundle_manifest_path}")
    baseline_hashes: dict[str, str] = {}
    for relative_path, declared_hash in sorted(declared_hashes.items()):
        path = baseline_bundle / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Baseline bundle file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != declared_hash:
            raise ValueError(f"Baseline bundle file hash differs from its manifest: {relative_path}")
        baseline_hashes[relative_path] = actual_hash

    return {
        "model_revision": model_revision,
        "source_files_sha256": source_hashes,
        "baseline_bundle_files": baseline_hashes,
        "baseline_bundle_manifest_sha256": sha256_file(bundle_manifest_path),
    }


def pack_sha256(metadata: dict[str, Any]) -> str:
    identity = {key: value for key, value in metadata.items() if key != "pack_sha256"}
    return canonical_sha256(identity)


def compare_tensors(
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        return {
            "shape_mismatch": True,
            "actual_shape": list(actual_array.shape),
            "expected_shape": list(expected_array.shape),
            "outside_tolerance_count": max(actual_array.size, expected_array.size),
            "max_normalized_tolerance_violation": math.inf,
            "max_absolute_error": math.inf,
        }
    actual64 = actual_array.astype(np.float64, copy=False)
    expected64 = expected_array.astype(np.float64, copy=False)
    difference = np.abs(actual64 - expected64)
    allowed = atol + rtol * np.abs(expected64)
    outside = difference > allowed
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(difference, allowed, out=np.zeros_like(difference), where=allowed != 0)
    normalized[(allowed == 0) & (difference > 0)] = math.inf
    return {
        "shape_mismatch": False,
        "actual_shape": list(actual_array.shape),
        "expected_shape": list(expected_array.shape),
        "outside_tolerance_count": int(np.count_nonzero(outside)),
        "max_normalized_tolerance_violation": float(np.max(normalized, initial=0.0)),
        "max_absolute_error": float(np.max(difference, initial=0.0)),
    }


@dataclass
class LMParityReference:
    root: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def prompt_length(self) -> int:
        return int(self.metadata["prompt_length"])

    @property
    def layer_count(self) -> int:
        return int(self.metadata["num_hidden_layers"])

    def cache_tensors(self, sequence_length: int) -> list[np.ndarray]:
        if sequence_length < 0 or sequence_length > self.arrays["cache_keys"].shape[-2]:
            raise ValueError(f"Cache prefix length is out of range: {sequence_length}")
        keys = self.arrays["cache_keys"]
        values = self.arrays["cache_values"]
        return [
            tensor
            for layer in range(self.layer_count)
            for tensor in (keys[layer, ..., :sequence_length, :], values[layer, ..., :sequence_length, :])
        ]

    def prefill_inputs(self, input_mode: str) -> dict[str, np.ndarray]:
        if input_mode == "inputs_embeds":
            return {"inputs_embeds": self.arrays["prefill_inputs_embeds"][None]}
        if input_mode == "input_ids":
            return {
                "input_ids": self.arrays["prompt_ids"][None],
                "speaker_hidden": self.arrays["speaker_hidden"][None].astype(np.float32, copy=False),
            }
        raise ValueError(f"Unsupported LM input mode: {input_mode}")

    def decode_input(self, step: int, input_mode: str) -> dict[str, np.ndarray]:
        if not 0 <= step < len(self.arrays["generated_ids"]):
            raise ValueError(f"Decode step is out of range: {step}")
        if input_mode == "inputs_embeds":
            return {"inputs_embeds": self.arrays["token_embeds"][step][None, None, :]}
        if input_mode == "input_ids":
            return {
                "input_ids": self.arrays["generated_ids"][step : step + 1][None],
                "speaker_hidden": np.zeros(
                    (1, int(self.metadata["hidden_size"])), dtype=np.float32
                ),
            }
        raise ValueError(f"Unsupported LM input mode: {input_mode}")


def load_reference_pack(
    path: str | Path,
    *,
    expected_pack_sha256: str | None = None,
    verify_files: bool = True,
) -> LMParityReference:
    root = Path(path).resolve()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != REFERENCE_FORMAT:
        raise ValueError(f"Unsupported parity reference format: {metadata.get('format')}")
    actual_pack_hash = pack_sha256(metadata)
    if metadata.get("pack_sha256") != actual_pack_hash:
        raise ValueError("Parity reference metadata identity hash does not match")
    if expected_pack_sha256 is not None and actual_pack_hash != expected_pack_sha256:
        raise ValueError("Parity reference pack does not match the expected identity")
    arrays: dict[str, np.ndarray] = {}
    array_metadata = metadata.get("arrays")
    if not isinstance(array_metadata, dict):
        raise TypeError("Parity reference array metadata is missing")
    for name in _ARRAY_NAMES:
        record = array_metadata.get(name)
        if not isinstance(record, dict):
            raise TypeError(f"Parity reference array is missing: {name}")
        array_path = root / record["file"]
        if verify_files and sha256_file(array_path) != record["sha256"]:
            raise ValueError(f"Parity reference array hash differs: {name}")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"Parity reference array metadata differs: {name}")
        arrays[name] = array
    if len(arrays["generated_ids"]) != int(metadata["generated_token_count"]):
        raise ValueError("Generated token count does not match reference arrays")
    return LMParityReference(root=root, metadata=metadata, arrays=arrays)


def validate_source_identity(
    reference: LMParityReference,
    model_dir: str | Path,
    baseline_bundle: str | Path,
    fixture_path: str | Path,
) -> None:
    current = source_identity(Path(model_dir), Path(baseline_bundle), Path(fixture_path))
    if current != reference.metadata.get("source_identity"):
        raise ValueError("Parity reference source identity does not match the current model/baseline")


def _past_names(lm_metadata: dict[str, Any]) -> list[str]:
    return [name for name in lm_metadata["input_names"] if name.startswith("past_")]


def verify_baseline_against_reference(
    baseline_model: str | Path,
    reference: LMParityReference,
) -> dict[str, Any]:
    import onnxruntime as ort

    metadata = reference.metadata
    lm_metadata = metadata["lm_metadata"]
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(baseline_model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_mode = lm_metadata.get("input_mode", "inputs_embeds")
    past_names = _past_names(lm_metadata)
    kv_heads = int(metadata["num_key_value_heads"])
    head_dim = int(metadata["head_dim"])
    empty_cache = [
        np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32) for _ in past_names
    ]
    feed = {
        "attention_mask": np.ones((1, reference.prompt_length), dtype=np.int64),
        "position_ids": np.arange(reference.prompt_length, dtype=np.int64)[None],
        **dict(zip(past_names, empty_cache, strict=True)),
        **reference.prefill_inputs(input_mode),
    }
    failures: list[dict[str, Any]] = []
    values_outside = 0
    max_normalized = 0.0
    max_absolute = 0.0
    generated_ids: list[int] = []
    cache_outputs: list[np.ndarray] | None = None
    calls = 0

    for phase_index in range(int(metadata["generated_token_count"]) + 1):
        if phase_index > 0:
            step = phase_index - 1
            past_length = reference.prompt_length + step
            feed = {
                "attention_mask": np.ones((1, past_length + 1), dtype=np.int64),
                "position_ids": np.asarray([[past_length]], dtype=np.int64),
                **dict(zip(past_names, cache_outputs or [], strict=True)),
                **reference.decode_input(step, input_mode),
            }
        outputs = session.run(None, feed)
        cache_length = reference.prompt_length + phase_index
        expected_logits = reference.arrays["logits"][phase_index]
        comparisons = [
            (
                "logits",
                compare_tensors(outputs[0], expected_logits, atol=LOGITS_ATOL, rtol=LOGITS_RTOL),
            )
        ]
        expected_cache = reference.cache_tensors(cache_length)
        if len(outputs[1:]) != len(expected_cache):
            comparisons.append(
                (
                    "cache_count",
                    {
                        "outside_tolerance_count": max(len(outputs[1:]), len(expected_cache)),
                        "max_normalized_tolerance_violation": math.inf,
                        "max_absolute_error": math.inf,
                        "shape_mismatch": True,
                        "actual_shape": [len(outputs[1:])],
                        "expected_shape": [len(expected_cache)],
                    },
                )
            )
        else:
            comparisons.extend(
                (
                    f"cache[{index // 2}].{'key' if index % 2 == 0 else 'value'}",
                    compare_tensors(actual, expected, atol=CACHE_ATOL, rtol=CACHE_RTOL),
                )
                for index, (actual, expected) in enumerate(
                    zip(outputs[1:], expected_cache, strict=True)
                )
            )
        for name, comparison in comparisons:
            outside = comparison["outside_tolerance_count"]
            values_outside += outside
            if outside:
                max_normalized = max(
                    max_normalized, comparison["max_normalized_tolerance_violation"]
                )
                max_absolute = max(max_absolute, comparison["max_absolute_error"])
                failures.append(
                    {
                        "phase": "prefill" if phase_index == 0 else f"decode_step_{phase_index - 1}",
                        "output": name,
                        **comparison,
                    }
                )
        if phase_index < int(metadata["generated_token_count"]):
            predicted_id = int(np.argmax(outputs[0], axis=-1)[0])
            expected_id = int(reference.arrays["generated_ids"][phase_index])
            generated_ids.append(predicted_id)
            if predicted_id != expected_id:
                failures.append(
                    {
                        "phase": "prefill" if phase_index == 0 else f"decode_step_{phase_index - 1}",
                        "output": "token_argmax",
                        "expected_token": expected_id,
                        "candidate_token": predicted_id,
                        "outside_tolerance_count": 1,
                        "max_normalized_tolerance_violation": math.inf,
                        "max_absolute_error": float(abs(predicted_id - expected_id)),
                    }
                )
        cache_outputs = outputs[1:]
        calls += 1

    failed_tensor_count = sum(
        1 for row in failures if row["output"] != "token_argmax"
    )
    token_divergence_count = sum(row["output"] == "token_argmax" for row in failures)
    return {
        "diagnostic_score": [
            failed_tensor_count + token_divergence_count,
            values_outside,
            max_normalized,
            max_absolute,
        ],
        "failed_output_tensors": failed_tensor_count,
        "token_divergence_count": token_divergence_count,
        "total_values_outside_tolerance": values_outside,
        "max_normalized_tolerance_violation": max_normalized,
        "max_absolute_error": max_absolute,
        "first_divergence": failures[0] if failures else None,
        "generated_ids": generated_ids,
        "generated_ids_match": generated_ids == reference.arrays["generated_ids"].tolist(),
        "runtime_calls": calls,
    }
