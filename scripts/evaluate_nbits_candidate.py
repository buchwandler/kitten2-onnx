from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from kitten2_onnx.lm_parity_reference import (
    CACHE_ATOL,
    CACHE_RTOL,
    LOGITS_ATOL,
    LOGITS_RTOL,
    canonical_sha256,
    compare_tensors,
    load_reference_pack,
    sha256_file,
    validate_source_identity,
)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if os.uname().sysname == "Darwin" else peak * 1024)


def _write_json_atomic(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _first_bad_index(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float):
    if actual.shape != expected.shape:
        return None
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowed = atol + rtol * np.abs(expected.astype(np.float64))
    locations = np.argwhere(difference > allowed)
    return locations[0].tolist() if locations.size else None


def evaluate_candidate(
    model_path: Path,
    reference_path: Path,
    *,
    expected_source_identity_sha256: str | None = None,
    source_model_dir: Path | None = None,
    baseline_bundle: Path | None = None,
    fixture_path: Path | None = None,
    intra_op_threads: int = 4,
    inter_op_threads: int = 0,
    graph_optimization: str = "all",
    disable_memory_pattern: bool = False,
    disable_cpu_arena: bool = False,
    sequential: bool = False,
) -> dict[str, Any]:
    reference = load_reference_pack(reference_path)
    if not reference.metadata.get("trusted"):
        raise ValueError("Parity reference pack has not passed its FP32 baseline cross-check")
    source_identity_hash = canonical_sha256(reference.metadata["source_identity"])
    if (
        expected_source_identity_sha256 is not None
        and source_identity_hash != expected_source_identity_sha256
    ):
        raise ValueError("Parity reference source identity does not match the expected identity")
    if any((source_model_dir, baseline_bundle, fixture_path)):
        if not all((source_model_dir, baseline_bundle, fixture_path)):
            raise ValueError("Source model, baseline bundle, and fixture must be supplied together")
        validate_source_identity(reference, source_model_dir, baseline_bundle, fixture_path)

    lm_metadata = reference.metadata["lm_metadata"]
    input_mode = lm_metadata.get("input_mode", "inputs_embeds")
    past_names = [name for name in lm_metadata["input_names"] if name.startswith("past_")]
    graph_levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if graph_optimization not in graph_levels:
        raise ValueError(f"unsupported graph optimization level: {graph_optimization}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = intra_op_threads
    session_options.inter_op_num_threads = inter_op_threads
    session_options.enable_mem_pattern = not disable_memory_pattern
    session_options.enable_cpu_mem_arena = not disable_cpu_arena
    session_options.graph_optimization_level = graph_levels[graph_optimization]
    if sequential:
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    runtime_options = {
        "intra_op_threads": intra_op_threads,
        "inter_op_threads": inter_op_threads,
        "execution_mode": "sequential" if sequential else "parallel",
        "graph_optimization": graph_optimization,
        "enable_memory_pattern": not disable_memory_pattern,
        "enable_cpu_arena": not disable_cpu_arena,
    }
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    kv_heads = int(reference.metadata["num_key_value_heads"])
    head_dim = int(reference.metadata["head_dim"])
    empty_cache = [
        np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32) for _ in past_names
    ]
    feed: dict[str, np.ndarray] = {
        "attention_mask": np.ones((1, reference.prompt_length), dtype=np.int64),
        "position_ids": np.arange(reference.prompt_length, dtype=np.int64)[None],
        **dict(zip(past_names, empty_cache, strict=True)),
        **reference.prefill_inputs(input_mode),
    }
    cache_outputs: list[np.ndarray] | None = None
    failures: list[dict[str, Any]] = []
    output_failures: list[dict[str, Any]] = []
    token_divergence_steps: list[int] = []
    candidate_token_ids: list[int] = []
    values_outside = 0
    max_normalized = 0.0
    max_absolute = 0.0
    runtime_total_ms = 0.0
    comparison_count = 0
    runtime_calls = 0

    for phase_index in range(int(reference.metadata["generated_token_count"]) + 1):
        if phase_index > 0:
            step = phase_index - 1
            past_length = reference.prompt_length + step
            feed = {
                "attention_mask": np.ones((1, past_length + 1), dtype=np.int64),
                "position_ids": np.asarray([[past_length]], dtype=np.int64),
                **dict(zip(past_names, cache_outputs or [], strict=True)),
                **reference.decode_input(step, input_mode),
            }
        started = time.perf_counter()
        outputs = session.run(None, feed)
        runtime_total_ms += (time.perf_counter() - started) * 1000.0
        runtime_calls += 1
        phase = "prefill" if phase_index == 0 else f"decode_step_{phase_index - 1}"
        expected_logits = reference.arrays["logits"][phase_index]
        comparisons = [
            ("logits", outputs[0], expected_logits, LOGITS_ATOL, LOGITS_RTOL)
        ]
        expected_cache = reference.cache_tensors(reference.prompt_length + phase_index)
        if len(outputs[1:]) != len(expected_cache):
            comparisons.append(
                (
                    "cache_count",
                    np.zeros((len(outputs[1:]),), dtype=np.float32),
                    np.zeros((len(expected_cache),), dtype=np.float32),
                    0.0,
                    0.0,
                )
            )
        else:
            comparisons.extend(
                (
                    f"cache[{index // 2}].{'key' if index % 2 == 0 else 'value'}",
                    actual,
                    expected,
                    CACHE_ATOL,
                    CACHE_RTOL,
                )
                for index, (actual, expected) in enumerate(
                    zip(outputs[1:], expected_cache, strict=True)
                )
            )
        for output_name, actual, expected, atol, rtol in comparisons:
            comparison_count += 1
            comparison = compare_tensors(actual, expected, atol=atol, rtol=rtol)
            outside = comparison["outside_tolerance_count"]
            values_outside += outside
            if outside:
                max_normalized = max(
                    max_normalized, comparison["max_normalized_tolerance_violation"]
                )
                max_absolute = max(max_absolute, comparison["max_absolute_error"])
                detail = {
                    "phase": phase,
                    "output": output_name,
                    "first_divergent_index": _first_bad_index(
                        np.asarray(actual), np.asarray(expected), atol, rtol
                    ),
                    **comparison,
                }
                failures.append(detail)
                output_failures.append(detail)

        if phase_index < int(reference.metadata["generated_token_count"]):
            candidate_id = int(np.argmax(outputs[0], axis=-1).reshape(-1)[0])
            expected_id = int(reference.arrays["generated_ids"][phase_index])
            candidate_token_ids.append(candidate_id)
            if candidate_id != expected_id:
                token_divergence_steps.append(phase_index)
                failures.append(
                    {
                        "phase": phase,
                        "output": "token_argmax",
                        "first_divergent_index": None,
                        "candidate_token": candidate_id,
                        "expected_token": expected_id,
                    }
                )
        cache_outputs = outputs[1:]

    token_divergence_count = len(token_divergence_steps)
    score = [
        len(output_failures) + token_divergence_count,
        values_outside,
        max_normalized,
        max_absolute,
    ]
    diagnostic = {
        "runtime_options": runtime_options,
        "test_completed": True,
        "failure": None,
        "comparison_count": comparison_count,
        "failed_output_tensors": len(output_failures),
        "total_values_outside_tolerance": values_outside,
        "max_normalized_tolerance_violation": max_normalized,
        "max_absolute_error": max_absolute,
        "token_divergence_steps": token_divergence_steps,
        "diagnostic_score": score,
        "onnx_runtime_calls": runtime_calls,
        "onnx_runtime_total_ms": runtime_total_ms,
        "output_failures": output_failures,
    }
    return {
        "format": "kitten2-onnx-nbits-candidate-diagnostic-v1",
        "status": "diagnostic_complete",
        "model": str(model_path.resolve()),
        "runtime_options": runtime_options,
        "model_sha256": sha256_file(model_path),
        "reference": str(reference.root),
        "reference_pack_sha256": reference.metadata["pack_sha256"],
        "source_identity_sha256": source_identity_hash,
        "diagnostic": diagnostic,
        "diagnostic_score": score,
        "failed_output_tensors": len(output_failures),
        "total_values_outside_tolerance": values_outside,
        "max_normalized_tolerance_violation": max_normalized,
        "max_absolute_error": max_absolute,
        "first_divergence": failures[0] if failures else None,
        "candidate_token_ids": candidate_token_ids,
        "expected_token_ids": reference.arrays["generated_ids"].tolist(),
        "token_divergence_count": token_divergence_count,
        "onnx_runtime_calls": runtime_calls,
        "onnx_runtime_total_ms": runtime_total_ms,
        "elapsed_seconds": None,
        "peak_rss_bytes": _peak_rss_bytes(),
        "heavy_ml_imports_loaded": [
            name for name in ("torch", "transformers") if name in __import__("sys").modules
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ONNX-only NBits parity diagnostic.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-identity-sha256")
    parser.add_argument("--source-model-dir", type=Path)
    parser.add_argument("--baseline-bundle", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    parser.add_argument(
        "--graph-optimization", choices=("disable", "basic", "extended", "all"), default="all"
    )
    parser.add_argument("--disable-memory-pattern", action="store_true")
    parser.add_argument("--disable-cpu-arena", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    result = evaluate_candidate(
        args.model,
        args.reference,
        expected_source_identity_sha256=args.expected_source_identity_sha256,
        source_model_dir=args.source_model_dir,
        baseline_bundle=args.baseline_bundle,
        fixture_path=args.fixture,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        graph_optimization=args.graph_optimization,
        disable_memory_pattern=args.disable_memory_pattern,
        disable_cpu_arena=args.disable_cpu_arena,
        sequential=args.sequential,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json_atomic(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "diagnostic"}, indent=2))


if __name__ == "__main__":
    main()
