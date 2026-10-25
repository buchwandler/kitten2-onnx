from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kitten2_onnx.export_lm import RUNTIME_PROFILES, load_hf_lm
from kitten2_onnx.reference import _load_speaker_projection, load_reference_artifacts
from kitten2_onnx.runtime import Kitten2OnnxLM

PROMPT = "Hello from Kitten."
LOGITS_RTOL = 1e-4
LOGITS_ATOL = 2e-4
CACHE_RTOL = 1e-4
CACHE_ATOL = 2e-3


def _cache_arrays(cache: Any) -> dict[str, np.ndarray]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return {
        name: tensor.detach().cpu().numpy()
        for layer, (key, value) in enumerate(cache)
        for name, tensor in ((f"past_key_{layer}", key), (f"past_value_{layer}", value))
    }


def _compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    difference = candidate - reference
    denominator = float(np.linalg.norm(reference.ravel()))
    return {
        "shape": list(reference.shape),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(np.linalg.norm(difference.ravel()) / denominator),
        "allclose": bool(np.allclose(candidate, reference, rtol=LOGITS_RTOL, atol=LOGITS_ATOL)),
        "argmax_match": bool(
            np.array_equal(np.argmax(candidate, axis=-1), np.argmax(reference, axis=-1))
        ),
    }


MAX_FAILURE_INDEX_SAMPLE = 10


def _sample_failing_indices(mask: np.ndarray) -> list[list[int]]:
    flattened = mask.ravel()
    samples: list[list[int]] = []
    chunk_size = 65_536
    for start in range(0, flattened.size, chunk_size):
        offsets = np.flatnonzero(flattened[start : start + chunk_size])
        for offset in offsets[: MAX_FAILURE_INDEX_SAMPLE - len(samples)]:
            samples.append(
                [int(index) for index in np.unravel_index(start + int(offset), mask.shape)]
            )
        if len(samples) == MAX_FAILURE_INDEX_SAMPLE:
            break
    return samples


def _compare_cache(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        return {"allclose": False, "error": "cache tensor names differ"}
    tensors = {name: _compare(reference[name], candidate[name]) for name in reference}
    failures = {}
    for name, reference_values in reference.items():
        candidate_values = candidate[name]
        difference = np.abs(candidate_values - reference_values)
        tolerance = CACHE_ATOL + CACHE_RTOL * np.abs(reference_values)
        failing_mask = difference > tolerance
        failing_count = int(np.count_nonzero(failing_mask))
        if failing_count:
            prefix, separator, layer_text = name.rpartition("_")
            layer = int(layer_text) if separator and layer_text.isdigit() else None
            cache_kind = prefix.removeprefix("past_") if layer is not None else name
            worst_index = tuple(
                int(index) for index in np.unravel_index(np.argmax(difference), difference.shape)
            )
            nonzero_reference = reference_values != 0
            max_relative_error = None
            if np.any(nonzero_reference):
                max_relative_error = float(
                    np.max(
                        difference[nonzero_reference] / np.abs(reference_values[nonzero_reference])
                    )
                )
            failing_indices = _sample_failing_indices(failing_mask)
            failures[name] = {
                "layer": layer,
                "cache_kind": cache_kind,
                "failing_elements": failing_count,
                "failing_indices": failing_indices,
                "failing_indices_truncated": failing_count > len(failing_indices),
                "max_absolute_error": tensors[name]["max_absolute_error"],
                "max_relative_error": max_relative_error,
                "worst_index": list(worst_index),
                "reference_value_at_worst_index": float(reference_values[worst_index]),
                "candidate_value_at_worst_index": float(candidate_values[worst_index]),
            }
    worst_tensor = max(tensors, key=lambda name: tensors[name]["max_absolute_error"])
    first_failure = None
    if failures:
        first_tensor = next(iter(failures))
        first_failure = {"tensor": first_tensor, **failures[first_tensor]}
    return {
        "allclose": not failures,
        "max_absolute_error": tensors[worst_tensor]["max_absolute_error"],
        "relative_l2_error": max(item["relative_l2_error"] for item in tensors.values()),
        "tensor_count": len(tensors),
        "failing_tensor_count": len(failures),
        "failing_element_count": sum(item["failing_elements"] for item in failures.values()),
        "worst_tensor": worst_tensor,
        "first_failure": first_failure,
        "failures": failures,
    }


def _first_cache_failure(
    prefill_cache: dict[str, Any], decode_cache_reports: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if prefill_cache.get("first_failure") is not None:
        return {"phase": "prefill", "step": 0, **prefill_cache["first_failure"]}
    for step, report in enumerate(decode_cache_reports):
        if report.get("first_failure") is not None:
            return {"phase": "decode", "step": step, **report["first_failure"]}
    return None


def benchmark_variant(
    variant_model: Path,
    reference_model: Path,
    onnx_bundle: Path,
    *,
    variant: str,
    decode_steps: int,
    materialization_cache_dir: Path | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort

    if variant not in {"packed", "emb4"}:
        raise ValueError("variant must be packed or emb4")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    profile = RUNTIME_PROFILES["dev-fp32"]
    model_load_started = time.perf_counter()
    model = load_hf_lm(variant_model, profile.graph_dtype, weight_variant=variant).eval()
    model_load_seconds = time.perf_counter() - model_load_started
    artifacts = load_reference_artifacts(reference_model, "Bruno")
    if (
        artifacts.model_revision
        != json.loads((onnx_bundle / "lm/lm.meta.json").read_text(encoding="utf-8"))[
            "source_model_revision"
        ]
    ):
        raise ValueError("ONNX bundle and reference model revisions differ")
    speaker_projection = _load_speaker_projection(
        reference_model, artifacts.config, profile.graph_dtype, torch.device("cpu")
    ).eval()
    speaker_embedding = torch.from_numpy(artifacts.speaker_embedding[None])
    with torch.inference_mode():
        speaker_hidden = speaker_projection(speaker_embedding).cpu().numpy()

    session_started = time.perf_counter()
    runtime = Kitten2OnnxLM(
        onnx_bundle / "lm",
        speaker_model=onnx_bundle / "speaker/speaker.onnx",
        providers=["CPUExecutionProvider"],
        materialization_cache_dir=materialization_cache_dir,
    )
    session_creation_seconds = time.perf_counter() - session_started
    prompt_ids = np.asarray(artifacts.build_prompt(PROMPT), dtype=np.int64)
    input_ids = torch.from_numpy(prompt_ids[None])
    with torch.inference_mode():
        input_embeddings = model.get_input_embeddings()(input_ids)
    prefill_embeddings = input_embeddings.cpu().numpy().copy()
    prefill_embeddings[0, 0] = speaker_hidden[0]
    attention_mask = torch.ones((1, len(prompt_ids)), dtype=torch.long)
    position_ids = torch.arange(len(prompt_ids), dtype=torch.long)[None, :]
    reference_started = time.perf_counter()
    with torch.inference_mode():
        reference_output = model(
            inputs_embeds=torch.from_numpy(prefill_embeddings),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    reference_prefill_seconds = time.perf_counter() - reference_started
    onnx_started = time.perf_counter()
    onnx_logits, onnx_cache = runtime.step(
        prompt_ids[None], speaker_embedding=artifacts.speaker_embedding
    )
    onnx_prefill_seconds = time.perf_counter() - onnx_started
    reference_logits = reference_output.logits.detach().cpu().numpy()
    onnx_logits = np.asarray(onnx_logits)
    if onnx_logits.ndim == 2:
        onnx_logits = onnx_logits[:, None, :]
    prefill_logits = _compare(reference_logits[:, -1:, :], onnx_logits)
    prefill_cache = _compare_cache(_cache_arrays(reference_output.past_key_values), onnx_cache)

    reference_cache = reference_output.past_key_values
    reference_next = reference_logits[:, -1, :].argmax(axis=-1)
    onnx_next = onnx_logits[:, -1, :].argmax(axis=-1)
    generated_reference: list[int] = []
    generated_onnx: list[int] = []
    decode_logit_reports = []
    decode_cache_reports = []
    reference_decode_seconds = 0.0
    onnx_decode_seconds = 0.0
    for step in range(decode_steps):
        reference_id = int(reference_next[0])
        onnx_id = int(onnx_next[0])
        generated_reference.append(reference_id)
        generated_onnx.append(onnx_id)
        token = torch.tensor([[reference_id]], dtype=torch.long)
        with torch.inference_mode():
            token_embedding = model.get_input_embeddings()(token)
            past_length = len(prompt_ids) + step
            reference_started = time.perf_counter()
            reference_output = model(
                inputs_embeds=token_embedding,
                attention_mask=torch.ones((1, past_length + 1), dtype=torch.long),
                position_ids=torch.tensor([[past_length]], dtype=torch.long),
                past_key_values=reference_cache,
                use_cache=True,
                return_dict=True,
            )
            reference_decode_seconds += time.perf_counter() - reference_started
        onnx_started = time.perf_counter()
        onnx_logits, onnx_cache = runtime.step(
            np.asarray([[reference_id]], dtype=np.int64), past=onnx_cache, position=past_length
        )
        onnx_decode_seconds += time.perf_counter() - onnx_started
        reference_logits = reference_output.logits.detach().cpu().numpy()
        onnx_logits = np.asarray(onnx_logits)
        if onnx_logits.ndim == 2:
            onnx_logits = onnx_logits[:, None, :]
        decode_logit_reports.append(_compare(reference_logits[:, -1:, :], onnx_logits))
        decode_cache_reports.append(
            _compare_cache(_cache_arrays(reference_output.past_key_values), onnx_cache)
        )
        reference_cache = reference_output.past_key_values
        reference_next = reference_logits[:, -1, :].argmax(axis=-1)
        onnx_next = onnx_logits[:, -1, :].argmax(axis=-1)

    meta = json.loads((onnx_bundle / "lm/lm.meta.json").read_text(encoding="utf-8"))
    return {
        "format": "kitten2-variant-lm-benchmark-v1",
        "variant": variant,
        "source_model_revision": meta["source_model_revision"],
        "source_weight_sha256": meta["source_weight_sha256"],
        "onnxruntime_version": ort.__version__,
        "execution_provider": "CPUExecutionProvider",
        "prompt": PROMPT,
        "prompt_token_count": len(prompt_ids),
        "decode_steps": decode_steps,
        "runtime_materialization": {
            "cache": runtime.materialization_provenance,
            "seconds": runtime.materialization_seconds,
        },
        "timing_seconds": {
            "reference_model_load": model_load_seconds,
            "onnx_session_creation": session_creation_seconds,
            "reference_prefill": reference_prefill_seconds,
            "onnx_prefill": onnx_prefill_seconds,
            "reference_decode_ms_per_token": reference_decode_seconds * 1000 / decode_steps,
            "onnx_decode_ms_per_token": onnx_decode_seconds * 1000 / decode_steps,
        },
        "prefill_logits": prefill_logits,
        "prefill_cache": prefill_cache,
        "first_cache_failure": _first_cache_failure(prefill_cache, decode_cache_reports),
        "decode_logits": {
            "allclose": all(item["allclose"] for item in decode_logit_reports),
            "argmax_match_steps": sum(item["argmax_match"] for item in decode_logit_reports),
            "max_absolute_error": max(item["max_absolute_error"] for item in decode_logit_reports),
            "relative_l2_error": max(item["relative_l2_error"] for item in decode_logit_reports),
        },
        "decode_cache": {
            "allclose": all(item["allclose"] for item in decode_cache_reports),
            "max_absolute_error": max(item["max_absolute_error"] for item in decode_cache_reports),
            "relative_l2_error": max(item["relative_l2_error"] for item in decode_cache_reports),
            "steps": decode_cache_reports,
        },
        "greedy_tokens_match": generated_reference == generated_onnx,
        "reference_generated_ids": generated_reference,
        "onnx_generated_ids": generated_onnx,
        "parity_passed": (
            prefill_logits["allclose"]
            and prefill_cache["allclose"]
            and all(item["allclose"] for item in decode_logit_reports)
            and all(item["allclose"] for item in decode_cache_reports)
            and generated_reference == generated_onnx
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a packed or emb4 q4 ONNX LM with its upstream loader."
    )
    parser.add_argument("variant_model", type=Path)
    parser.add_argument(
        "reference_model",
        type=Path,
        help="Full reference model directory for tokenizer and speaker assets",
    )
    parser.add_argument("onnx_bundle", type=Path)
    parser.add_argument("--variant", required=True, choices=("packed", "emb4"))
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--materialization-cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark_variant(
        args.variant_model,
        args.reference_model,
        args.onnx_bundle,
        variant=args.variant,
        decode_steps=args.decode_steps,
        materialization_cache_dir=args.materialization_cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
