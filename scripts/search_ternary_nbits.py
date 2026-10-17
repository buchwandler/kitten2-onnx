"""Build isolated exact-NBits candidates and apply the strict real-model parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.optimize_bundle import optimize_bundle

_EXACT_CONFIGURATIONS = ((64, 1), (32, 1), (16, 1))
_HISTORICAL_BLOCK128_REFERENCE = {
    "configuration": {"bits": 4, "block_size": 128, "accuracy_level": 1},
    "status": "known_strict_parity_failure_on_prior_source_graph",
    "source_graph_sha256": "c08e6a4c1c78c9601a6c557b9dd71d3fd7e002019c14db50aa99b83a730e0e9a",
    "source_external_data_sha256": "190bb3f51ff3575447596450497d5da9ef14a178dd2b2189f4e971c11f100aa4",
    "first_failure": {
        "output": "prefill cache",
        "values_outside_tolerance": 2,
        "compared_values": 301056,
        "max_absolute_difference": 0.0038414,
    },
    "evidence": "ternary-nbits4-investigation-report.md",
}


def exact_configurations() -> list[dict[str, int]]:
    return [
        {"bits": 4, "block_size": block_size, "accuracy_level": accuracy_level}
        for block_size, accuracy_level in _EXACT_CONFIGURATIONS
    ]


def candidate_key(
    source_files: dict[str, str],
    inventory_sha256: str,
    configuration: dict[str, int],
    excluded_nodes: tuple[str, ...] | list[str] = (),
) -> str:
    identity = {
        "source_files": dict(sorted(source_files.items())),
        "inventory_sha256": inventory_sha256,
        "configuration": configuration,
        "excluded_nodes": sorted(set(excluded_nodes)),
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _strict_parity(
    candidate_bundle: Path,
    model_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "KITTEN2_MODEL_DIR": str(model_dir.resolve()),
            "KITTEN2_ONNX_DIR": str((candidate_bundle / "lm").resolve()),
            "KITTEN2_SPEAKER_ONNX": str((candidate_bundle / "speaker/speaker.onnx").resolve()),
        }
    )
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_real_lm_parity.py"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5400,
            check=False,
        )
        output = completed.stdout + completed.stderr
        log_path.write_text(output, encoding="utf-8")
        return {
            "status": "passed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "command": command,
            "log": str(log_path),
            "elapsed_seconds": time.perf_counter() - started,
            "output_tail": output[-12000:],
        }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        output = (stdout + stderr) if isinstance(stdout, str) else (stdout + stderr).decode(errors="replace")
        log_path.write_text(output, encoding="utf-8")
        return {
            "status": "timeout",
            "return_code": None,
            "command": command,
            "log": str(log_path),
            "elapsed_seconds": time.perf_counter() - started,
            "output_tail": output[-12000:],
        }


def _diagnose_candidate_parity(
    candidate_bundle: Path,
    model_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    import importlib.util
    import re

    project_root = Path(__file__).resolve().parents[1]
    test_path = project_root / "tests/test_real_lm_parity.py"
    spec = importlib.util.spec_from_file_location("_nbits_real_lm_parity", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load parity test module from {test_path}")
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    import onnxruntime as runtime

    original_inference_session = runtime.InferenceSession
    original_assert_outputs = test_module._assert_outputs
    original_environment = {
        name: os.environ.get(name)
        for name in ("KITTEN2_MODEL_DIR", "KITTEN2_ONNX_DIR", "KITTEN2_SPEAKER_ONNX")
    }
    os.environ.update(
        {
            "KITTEN2_MODEL_DIR": str(model_dir.resolve()),
            "KITTEN2_ONNX_DIR": str((candidate_bundle / "lm").resolve()),
            "KITTEN2_SPEAKER_ONNX": str(
                (candidate_bundle / "speaker/speaker.onnx").resolve()
            ),
        }
    )
    output_failures = []
    comparison_count = 0
    runtime_ms = 0.0
    runtime_calls = 0
    original_run = time.perf_counter

    class TimedSession:
        def __init__(self, *args, **kwargs):
            self._session = original_inference_session(*args, **kwargs)

        def run(self, *args, **kwargs):
            nonlocal runtime_ms, runtime_calls
            started = original_run()
            result = self._session.run(*args, **kwargs)
            runtime_ms += (original_run() - started) * 1000.0
            runtime_calls += 1
            return result

        def __getattr__(self, name):
            return getattr(self._session, name)

    def timed_session(*args, **kwargs):
        return TimedSession(*args, **kwargs)

    class DiagnosticMonkeyPatch:
        @staticmethod
        def setattr(target, name, value):
            setattr(target, name, value)

    def flatten_cache(cache):
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        return [tensor for layer in cache for tensor in layer]

    def compare_output(name, actual, expected, *, atol, rtol, phase):
        nonlocal comparison_count
        comparison_count += 1
        actual_array = np.asarray(actual)
        expected_array = (
            expected.detach().cpu().numpy() if hasattr(expected, "detach") else np.asarray(expected)
        )
        if actual_array.shape != expected_array.shape:
            output_failures.append(
                {
                    "phase": phase,
                    "output": name,
                    "shape_mismatch": True,
                    "actual_shape": list(actual_array.shape),
                    "expected_shape": list(expected_array.shape),
                    "outside_tolerance_count": max(actual_array.size, expected_array.size),
                    "max_normalized_tolerance_violation": float("inf"),
                    "max_absolute_error": float("inf"),
                }
            )
            return
        difference = np.abs(actual_array.astype(np.float64) - expected_array.astype(np.float64))
        allowed = atol + rtol * np.abs(expected_array.astype(np.float64))
        outside = difference > allowed
        outside_count = int(np.count_nonzero(outside))
        if outside_count:
            output_failures.append(
                {
                    "phase": phase,
                    "output": name,
                    "shape": list(actual_array.shape),
                    "outside_tolerance_count": outside_count,
                    "output_element_count": int(actual_array.size),
                    "max_normalized_tolerance_violation": float(np.max(difference / allowed)),
                    "max_absolute_error": float(np.max(difference)),
                }
            )

    call_index = 0

    def collect_outputs(actual, expected_logits, expected_cache):
        nonlocal call_index
        phase = "prefill" if call_index == 0 else f"decode_step_{call_index - 1}"
        compare_output("logits", actual[0], expected_logits, atol=2e-4, rtol=1e-4, phase=phase)
        expected = flatten_cache(expected_cache)
        for index, (actual_tensor, expected_tensor) in enumerate(
            zip(actual[1:], expected, strict=True)
        ):
            kind = "key" if index % 2 == 0 else "value"
            compare_output(
                f"cache[{index // 2}].{kind}",
                actual_tensor,
                expected_tensor,
                atol=2e-3,
                rtol=1e-4,
                phase=phase,
            )
        call_index += 1

    test_module._assert_outputs = collect_outputs
    runtime.InferenceSession = timed_session
    started = time.perf_counter()
    failure = None
    test_completed = False
    try:
        test_module.test_real_bruno_prompt_injection_and_32_step_lm_parity(
            DiagnosticMonkeyPatch()
        )
        test_completed = True
    except AssertionError as error:
        failure = str(error)
    finally:
        test_module._assert_outputs = original_assert_outputs
        runtime.InferenceSession = original_inference_session
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    token_match = re.search(r"generated token (\d+)", failure or "")
    token_divergence_steps = [int(token_match.group(1))] if token_match else []
    failed_output_tensors = len(output_failures)
    total_values_outside_tolerance = sum(
        row["outside_tolerance_count"] for row in output_failures
    )
    max_normalized = max(
        (row["max_normalized_tolerance_violation"] for row in output_failures), default=0.0
    )
    max_absolute = max((row["max_absolute_error"] for row in output_failures), default=0.0)
    report = {
        "diagnostic_only": True,
        "strict_acceptance_test": "tests/test_real_lm_parity.py",
        "test_completed": test_completed,
        "failure": failure[:2000] if failure else None,
        "comparison_count": comparison_count,
        "failed_output_tensors": failed_output_tensors,
        "total_values_outside_tolerance": total_values_outside_tolerance,
        "max_normalized_tolerance_violation": max_normalized,
        "max_absolute_error": max_absolute,
        "token_divergence_steps": token_divergence_steps,
        "diagnostic_score": [
            failed_output_tensors + len(token_divergence_steps),
            total_values_outside_tolerance,
            max_normalized,
            max_absolute,
        ],
        "onnx_runtime_calls": runtime_calls,
        "onnx_runtime_total_ms": runtime_ms,
        "elapsed_seconds": time.perf_counter() - started,
        "output_failures": output_failures,
    }
    _write_json(report_path, report)
    return report


def _load_inputs(
    baseline_bundle: Path,
    inventory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = json.loads((baseline_bundle / "bundle.json").read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["source_graph_sha256"] != bundle["files"]["lm/lm.onnx"]:
        raise ValueError("MatMul inventory does not describe the supplied baseline LM graph")
    if inventory["eligible_exact_ternary_matmul_count"] != 196:
        raise ValueError("Expected the fresh baseline inventory to contain 196 eligible MatMuls")
    if inventory["mapped_eligible_matmul_count"] != 196 or inventory["ambiguous_mapping_count"] != 0:
        raise ValueError("Baseline MatMul inventory is incomplete or ambiguous")
    return bundle, inventory


def run_exact_search(
    baseline_bundle: str | Path,
    model_dir: str | Path,
    inventory_path: str | Path,
    output_root: str | Path,
    reports_root: str | Path,
) -> dict[str, Any]:
    baseline = Path(baseline_bundle).resolve()
    model = Path(model_dir).resolve()
    inventory_file = Path(inventory_path).resolve()
    output = Path(output_root).resolve()
    reports = Path(reports_root).resolve()
    if not baseline.is_dir():
        raise FileNotFoundError(baseline)
    if output.exists() != reports.exists():
        raise FileExistsError("Candidate output and report roots must either both exist or both be new")
    if output == reports or output.is_relative_to(reports) or reports.is_relative_to(output):
        raise ValueError("Candidate output and report roots must not overlap")
    if (
        output == baseline
        or output.is_relative_to(baseline)
        or baseline.is_relative_to(output)
        or reports == baseline
        or reports.is_relative_to(baseline)
        or baseline.is_relative_to(reports)
    ):
        raise ValueError("Search output and report roots must not overlap the baseline")

    bundle, inventory = _load_inputs(baseline, inventory_file)
    results_path = reports / "results.json"
    if output.exists():
        if not results_path.is_file():
            raise FileExistsError(f"Existing report root has no search record: {results_path}")
        search_record = json.loads(results_path.read_text(encoding="utf-8"))
        if (
            search_record.get("baseline_bundle_files") != bundle["files"]
            or search_record.get("inventory_sha256") != inventory["inventory_sha256"]
        ):
            raise ValueError("Existing search record belongs to a different source or inventory")
        results = search_record["candidates"]
    else:
        output.mkdir(parents=True)
        reports.mkdir(parents=True)
        _write_json(reports / "inventory.json", inventory)
        results = []
        search_record = {
            "format": "kitten2-onnx-ternary-nbits-search-v1",
            "baseline_bundle": str(baseline),
            "baseline_bundle_files": bundle["files"],
            "baseline_bundle_bytes": sum(
                path.stat().st_size for path in baseline.rglob("*") if path.is_file()
            ),
            "inventory_sha256": inventory["inventory_sha256"],
            "historical_block128_reference": _HISTORICAL_BLOCK128_REFERENCE,
            "exact_configurations_in_order": exact_configurations(),
            "candidates": results,
            "decision": "exact_search_in_progress",
        }
        _write_json(results_path, search_record)

    results_by_name = {row["name"]: row for row in results}
    for configuration in exact_configurations():
        name = f"b4-bs{configuration['block_size']}-a{configuration['accuracy_level']}"
        candidate_bundle = output / name
        candidate_reports = reports / name
        key = candidate_key(bundle["files"], inventory["inventory_sha256"], configuration)
        row = results_by_name.get(name)
        if row is not None and row.get("candidate_key") != key:
            raise ValueError(f"Existing candidate key differs from the requested configuration: {name}")
        if row is not None and row.get("status") != "build_started":
            if row["status"] == "passed":
                search_record["decision"] = "exact_candidate_passed_strict_parity"
                break
            continue

        if row is None:
            if candidate_bundle.exists() or candidate_reports.exists():
                raise FileExistsError(f"Candidate paths exist without a search record: {name}")
            row = {
                "name": name,
                "candidate_key": key,
                "configuration": configuration,
                "excluded_nodes": [],
                "status": "build_started",
                "bundle": str(candidate_bundle),
                "reports": str(candidate_reports),
            }
            results.append(row)
            results_by_name[name] = row
            _write_json(results_path, search_record)

        try:
            if candidate_bundle.exists():
                optimization_path = candidate_reports / "optimization.json"
                if not optimization_path.is_file():
                    raise FileExistsError(f"Existing candidate has no optimization report: {name}")
                optimization = json.loads(optimization_path.read_text(encoding="utf-8"))
                quantization = optimization["quantization"]
                if (
                    optimization["lm_method"] != "ternary-nbits"
                    or quantization["bits"] != configuration["bits"]
                    or quantization["block_size"] != configuration["block_size"]
                    or quantization["accuracy_level"] != configuration["accuracy_level"]
                    or optimization["validation"]["status"] != "passed"
                ):
                    raise ValueError(f"Existing candidate does not match its search configuration: {name}")
            else:
                if candidate_reports.exists():
                    raise FileExistsError(f"Candidate report path exists without a bundle: {name}")
                optimization = optimize_bundle(
                    baseline,
                    candidate_bundle,
                    lm_method="ternary-nbits",
                    validate=True,
                    report_dir=candidate_reports,
                    nbits=configuration["bits"],
                    nbits_block_size=configuration["block_size"],
                    nbits_accuracy_level=configuration["accuracy_level"],
                )
            if optimization["validation"]["status"] != "passed":
                raise ValueError("candidate bundle validation did not pass")
            row["bundle_bytes"] = optimization["output_bundle_bytes"]
            row["size_report"] = optimization["size_report"]
            row["quantization"] = optimization["quantization"]
            row["validation"] = optimization["validation"]
            candidate_manifest = json.loads((candidate_bundle / "bundle.json").read_text(encoding="utf-8"))
            row["candidate_lm_graph_sha256"] = candidate_manifest["files"]["lm/lm.onnx"]
            external_paths = sorted(
                path
                for path in candidate_manifest["files"]
                if path.startswith("lm/") and path.endswith(".onnx.data")
            )
            if len(external_paths) != 1:
                raise ValueError(f"Expected one LM external-data file, found {external_paths}")
            row["candidate_lm_external_data"] = {
                path: candidate_manifest["files"][path] for path in external_paths
            }
            _write_json(results_path, search_record)
            parity = _strict_parity(
                candidate_bundle,
                model,
                candidate_reports / "strict-real-lm-parity.log",
            )
            row["strict_parity"] = parity
            row["status"] = "passed" if parity["status"] == "passed" else "strict_parity_failed"
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            row["status"] = "build_or_validation_failed"
            row["error"] = f"{type(error).__name__}: {error}"
        _write_json(candidate_reports / "search-result.json", row)
        _write_json(results_path, search_record)
        if row["status"] == "passed":
            search_record["decision"] = "exact_candidate_passed_strict_parity"
            break

    if (
        search_record["decision"] != "exact_candidate_passed_strict_parity"
        and "fallback_search" not in search_record
    ):
        search_record["decision"] = "all_exact_candidates_failed_strict_parity"
    _write_json(results_path, search_record)
    return search_record


def _semantic_fallback_groups(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    nodes_by_layer_module: dict[int, dict[str, list[str]]] = {}
    for row in inventory["nodes"]:
        if not row["eligible_exact_ternary"]:
            continue
        layer = row["layer"]
        module = row["module"]
        if layer is None or module is None:
            continue
        nodes_by_layer_module.setdefault(layer, {}).setdefault(module, []).append(row["node"])

    late_layers = range(23, 28)
    attention_modules = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    mlp_modules = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    groups = []
    for layer in late_layers:
        modules = nodes_by_layer_module.get(layer, {})
        if set(modules) != {*attention_modules, *mlp_modules}:
            raise ValueError(f"Inventory does not map all seven projections in layer {layer}")
        groups.append(
            {
                "name": f"layer-{layer}",
                "nodes": sorted(node for values in modules.values() for node in values),
            }
        )
    groups.append(
        {
            "name": "v-proj-layers-23-27",
            "nodes": sorted(
                node
                for layer in late_layers
                for node in nodes_by_layer_module[layer]["self_attn.v_proj"]
            ),
        }
    )
    for layer in late_layers:
        groups.append(
            {
                "name": f"attention-layer-{layer}",
                "nodes": sorted(
                    node
                    for module in attention_modules
                    for node in nodes_by_layer_module[layer][module]
                ),
            }
        )
    for layer in late_layers:
        groups.append(
            {
                "name": f"mlp-layer-{layer}",
                "nodes": sorted(
                    node
                    for module in mlp_modules
                    for node in nodes_by_layer_module[layer][module]
                ),
            }
        )
    for layer in late_layers:
        groups.append(
            {
                "name": f"v-proj-layer-{layer}",
                "nodes": sorted(nodes_by_layer_module[layer]["self_attn.v_proj"]),
            }
        )
    for layer in late_layers:
        for module in mlp_modules:
            groups.append(
                {
                    "name": f"{module.replace('.', '-')}-layer-{layer}",
                    "nodes": sorted(nodes_by_layer_module[layer][module]),
                }
            )
    groups.extend(
        [
            {
                "name": "attention-layers-23-27",
                "nodes": sorted(
                    node
                    for layer in late_layers
                    for module in attention_modules
                    for node in nodes_by_layer_module[layer][module]
                ),
            },
            {
                "name": "mlp-projections-layers-23-27",
                "nodes": sorted(
                    node
                    for layer in late_layers
                    for module in mlp_modules
                    for node in nodes_by_layer_module[layer][module]
                ),
            },
        ]
    )
    for layer in late_layers:
        for module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"):
            groups.append(
                {
                    "name": f"{module.replace('.', '-')}-layer-{layer}",
                    "nodes": sorted(nodes_by_layer_module[layer][module]),
                }
            )
    for layer in range(22, -1, -1):
        modules = nodes_by_layer_module.get(layer, {})
        if set(modules) != {*attention_modules, *mlp_modules}:
            raise ValueError(f"Inventory does not map all seven projections in layer {layer}")
        groups.append(
            {
                "name": f"layer-{layer}",
                "nodes": sorted(node for values in modules.values() for node in values),
                "phase": "earlier_layer",
            }
        )
    return groups


def _diagnostic_score(result: dict[str, Any]) -> tuple[int, int, float, float]:
    score = result["diagnostic_score"]
    return int(score[0]), int(score[1]), float(score[2]), float(score[3])


class FallbackCandidateLimitReached(RuntimeError):
    pass


def _diagnostic_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "test_completed",
            "failure",
            "comparison_count",
            "failed_output_tensors",
            "total_values_outside_tolerance",
            "max_normalized_tolerance_violation",
            "max_absolute_error",
            "token_divergence_steps",
            "diagnostic_score",
            "onnx_runtime_calls",
            "onnx_runtime_total_ms",
            "elapsed_seconds",
        )
    }


def run_fallback_search(
    baseline_bundle: str | Path,
    model_dir: str | Path,
    inventory_path: str | Path,
    output_root: str | Path,
    reports_root: str | Path,
    *,
    kernel_diagnostic_path: str | Path,
    max_fallback_candidates: int = 140,
) -> dict[str, Any]:
    baseline = Path(baseline_bundle).resolve()
    model = Path(model_dir).resolve()
    inventory_file = Path(inventory_path).resolve()
    output = Path(output_root).resolve()
    reports = Path(reports_root).resolve()
    kernel_report = json.loads(Path(kernel_diagnostic_path).read_text(encoding="utf-8"))
    bundle, inventory = _load_inputs(baseline, inventory_file)
    exact_results_path = reports / "results.json"
    exact_results = json.loads(exact_results_path.read_text(encoding="utf-8"))
    exact_candidates = exact_results["candidates"]
    if not exact_candidates or any(
        candidate.get("strict_parity", {}).get("status") != "failed"
        for candidate in exact_candidates
    ):
        raise ValueError("Fallback search requires every exact candidate to fail strict parity")
    if (
        exact_results["decision"] != "all_exact_candidates_failed_strict_parity"
        and "fallback_search" not in exact_results
    ):
        raise ValueError("Fallback search requires the exact sweep decision")
    if max_fallback_candidates <= 0:
        raise ValueError("max_fallback_candidates must be positive")

    candidates_by_block = {
        row["configuration"]["block_size"]: row for row in exact_results["candidates"]
    }
    kernel_scores = {}
    for block_size, candidate in candidates_by_block.items():
        if candidate["strict_parity"]["status"] != "failed":
            raise ValueError(f"Exact candidate {candidate['name']} has no strict parity failure")
        comparisons = [
            row
            for row in kernel_report["results"]
            if row["block_size"] == block_size and row["accuracy_level"] == 1
        ]
        if not comparisons:
            raise ValueError(f"Kernel diagnostic has no level-1 results for block {block_size}")
        kernel_scores[block_size] = {
            "failed_operator_cases": sum(
                row["outside_tolerance_count"] > 0 for row in comparisons
            ),
            "total_values_outside_tolerance": sum(
                row["outside_tolerance_count"] for row in comparisons
            ),
            "max_relative_difference": max(
                row["max_relative_difference"] for row in comparisons
            ),
            "max_absolute_difference": max(
                row["max_absolute_difference"] for row in comparisons
            ),
        }
    selected_block = min(
        candidates_by_block,
        key=lambda block: (
            kernel_scores[block]["failed_operator_cases"],
            kernel_scores[block]["total_values_outside_tolerance"],
            kernel_scores[block]["max_relative_difference"],
            kernel_scores[block]["max_absolute_difference"],
            candidates_by_block[block]["bundle_bytes"],
        ),
    )
    parent = candidates_by_block[selected_block]
    configuration = parent["configuration"]
    parent_bundle = Path(parent["bundle"])
    parent_reports = Path(parent["reports"])
    parent_diagnostic_path = parent_reports / "diagnostic-parity.json"
    if parent_diagnostic_path.is_file():
        parent_diagnostic = json.loads(parent_diagnostic_path.read_text(encoding="utf-8"))
    else:
        parent_diagnostic = _diagnose_candidate_parity(
            parent_bundle, model, parent_diagnostic_path
        )
    parent["parity_diagnostic"] = _diagnostic_summary(parent_diagnostic)
    _write_json(parent_reports / "search-result.json", parent)
    _write_json(exact_results_path, exact_results)
    parent_key = candidate_key(
        bundle["files"], inventory["inventory_sha256"], configuration, ()
    )
    parent_result = {
        "name": parent["name"],
        "candidate_key": parent_key,
        "configuration": configuration,
        "excluded_nodes": [],
        "status": "strict_parity_failed",
        "bundle": str(parent_bundle),
        "bundle_bytes": parent["bundle_bytes"],
        "fallback_storage": parent["quantization"]["fallback_storage"],
        "diagnostic": _diagnostic_summary(parent_diagnostic),
        "diagnostic_score": parent_diagnostic["diagnostic_score"],
        "strict_parity": parent["strict_parity"],
    }
    fallback_output = output / "fallback"
    fallback_reports = reports / "fallback"
    if fallback_output.exists() != fallback_reports.exists():
        raise FileExistsError("Fallback output and report roots must either both exist or both be new")
    fallback_results_path = fallback_reports / "results.json"
    if fallback_output.exists():
        if not fallback_results_path.is_file():
            raise FileExistsError(f"Existing fallback reports have no cache: {fallback_results_path}")
        search_record = json.loads(fallback_results_path.read_text(encoding="utf-8"))
        if (
            search_record["inventory_sha256"] != inventory["inventory_sha256"]
            or search_record["baseline_bundle_files"] != bundle["files"]
        ):
            raise ValueError("Existing fallback cache belongs to a different source or inventory")
        cache = search_record["cache"]
        group_evaluations = search_record["group_evaluations"]
    else:
        fallback_output.mkdir(parents=True)
        fallback_reports.mkdir(parents=True)
        cache = {parent_key: parent_result}
        group_evaluations = []
        baseline_bytes = exact_results["baseline_bundle_bytes"]
        search_record = {
            "format": "kitten2-onnx-ternary-nbits-fallback-search-v1",
            "baseline_bundle": str(baseline),
            "baseline_bundle_files": bundle["files"],
            "inventory_sha256": inventory["inventory_sha256"],
            "parent_candidate": parent["name"],
            "parent_configuration": configuration,
            "parent_selection": {
                "basis": "minimum same-activation diagnostic score, then smallest measured bundle",
                "kernel_diagnostic": str(Path(kernel_diagnostic_path).resolve()),
                "per_block_scores": kernel_scores,
                "selected_block_size": selected_block,
            },
            "fallback_size_budget_bytes": baseline_bytes // 2,
            "preferred_bundle_bytes": 4_000_000_000,
            "semantic_groups": _semantic_fallback_groups(inventory),
            "group_evaluations": group_evaluations,
            "cache": cache,
            "decision": "fallback_search_in_progress",
        }
        _write_json(fallback_reports / "inventory.json", inventory)
        _write_json(fallback_results_path, search_record)

    expected_groups = _semantic_fallback_groups(inventory)
    stored_groups = search_record["semantic_groups"]
    stored_by_name = {group["name"]: group for group in stored_groups}
    expected_by_name = {group["name"]: group for group in expected_groups}
    for name, group in stored_by_name.items():
        expected = expected_by_name.get(name)
        if expected is not None and group["nodes"] != expected["nodes"]:
            raise ValueError(f"Cached fallback group no longer matches inventory: {name}")
    search_record["semantic_groups"] = stored_groups + [
        group for group in expected_groups if group["name"] not in stored_by_name
    ]
    search_record["max_fallback_candidates"] = max_fallback_candidates
    _write_json(fallback_results_path, search_record)
    if not search_record["group_evaluations"]:
        search_record["group_evaluations"] = group_evaluations
    evaluation_by_name = {
        row["group"]: row for row in search_record["group_evaluations"]
    }
    best_passing: dict[str, Any] | None = None
    best_over_budget: dict[str, Any] | None = None

    def save_search() -> None:
        search_record["cache"] = cache
        _write_json(fallback_results_path, search_record)

    def evaluate(excluded_nodes: list[str] | tuple[str, ...], label: str) -> dict[str, Any]:
        normalized_nodes = tuple(sorted(set(excluded_nodes)))
        key = candidate_key(
            bundle["files"], inventory["inventory_sha256"], configuration, normalized_nodes
        )
        existing = cache.get(key)
        if existing is not None and existing["status"] != "build_started":
            return existing
        if existing is None and len(cache) - 1 >= max_fallback_candidates:
            raise FallbackCandidateLimitReached(
                f"Reached fallback candidate limit {max_fallback_candidates}"
            )
        name = f"b4-bs{configuration['block_size']}-a{configuration['accuracy_level']}-fallback-{key[:12]}"
        candidate_bundle = fallback_output / name
        candidate_reports = fallback_reports / name
        if existing is None:
            if candidate_bundle.exists() or candidate_reports.exists():
                raise FileExistsError(f"Fallback candidate paths exist without a cache record: {name}")
            row: dict[str, Any] = {
                "name": name,
                "candidate_key": key,
                "configuration": configuration,
                "excluded_nodes": list(normalized_nodes),
                "semantic_label": label,
                "bundle": str(candidate_bundle),
                "reports": str(candidate_reports),
                "status": "build_started",
            }
            cache[key] = row
            save_search()
        else:
            row = existing
            candidate_bundle = Path(row["bundle"])
            candidate_reports = Path(row["reports"])
        candidate_reports.mkdir(parents=True, exist_ok=True)
        try:
            optimization_path = candidate_reports / "optimization.json"
            if candidate_bundle.exists():
                if not optimization_path.is_file():
                    raise FileExistsError(f"Existing fallback bundle has no optimization report: {name}")
                optimization = json.loads(optimization_path.read_text(encoding="utf-8"))
                quantization = optimization["quantization"]
                if (
                    optimization["lm_method"] != "ternary-nbits"
                    or quantization["bits"] != configuration["bits"]
                    or quantization["block_size"] != configuration["block_size"]
                    or quantization["accuracy_level"] != configuration["accuracy_level"]
                    or quantization["exclude_nodes"] != list(normalized_nodes)
                    or optimization["validation"]["status"] != "passed"
                ):
                    raise ValueError(f"Existing fallback candidate does not match its key: {name}")
            else:
                optimization = optimize_bundle(
                    baseline,
                    candidate_bundle,
                    lm_method="ternary-nbits",
                    validate=True,
                    report_dir=candidate_reports,
                    nbits=configuration["bits"],
                    nbits_block_size=configuration["block_size"],
                    nbits_accuracy_level=configuration["accuracy_level"],
                    exclude_matmul_nodes=normalized_nodes,
                )
            if optimization["validation"]["status"] != "passed":
                raise ValueError("fallback candidate bundle validation did not pass")
            row["bundle_bytes"] = optimization["output_bundle_bytes"]
            row["size_report"] = optimization["size_report"]
            row["fallback_storage"] = optimization["quantization"]["fallback_storage"]
            row["transformed_matmul_count"] = optimization["quantization"][
                "transformed_matmul_count"
            ]
            row["validation"] = optimization["validation"]
            diagnostic_path = candidate_reports / "diagnostic-parity.json"
            if diagnostic_path.is_file():
                diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            else:
                diagnostic = _diagnose_candidate_parity(candidate_bundle, model, diagnostic_path)
            row["diagnostic"] = _diagnostic_summary(diagnostic)
            row["diagnostic_score"] = diagnostic["diagnostic_score"]
            row["size_budget_passed"] = (
                row["bundle_bytes"] <= search_record["fallback_size_budget_bytes"]
            )
            row["preferred_size_passed"] = row["bundle_bytes"] < search_record[
                "preferred_bundle_bytes"
            ]
            if diagnostic["diagnostic_score"][0] == 0 and diagnostic["test_completed"]:
                parity = _strict_parity(
                    candidate_bundle,
                    model,
                    candidate_reports / "strict-real-lm-parity.log",
                )
                row["strict_parity"] = parity
                row["status"] = (
                    "strict_parity_passed" if parity["status"] == "passed" else "strict_parity_failed"
                )
            else:
                row["strict_parity"] = {"status": "not_run", "reason": "diagnostic score is non-zero"}
                row["status"] = "diagnostic_complete"
        except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as error:
            row["status"] = "build_or_diagnostic_failed"
            row["error"] = f"{type(error).__name__}: {error}"
        _write_json(candidate_reports / "search-result.json", row)
        save_search()
        return row

    def test_group(label: str, nodes: list[str]) -> dict[str, Any]:
        result = evaluate(nodes, label)
        entry = evaluation_by_name.get(label)
        if entry is None:
            entry = {"group": label}
            search_record["group_evaluations"].append(entry)
            evaluation_by_name[label] = entry
        entry.update(
            {
                "candidate_key": result["candidate_key"],
                "nodes": sorted(set(nodes)),
                "diagnostic_score": result.get("diagnostic_score"),
                "status": result["status"],
            }
        )
        save_search()
        return result

    candidate_limit_reached = False
    try:
        late_groups = [
            group
            for group in search_record["semantic_groups"]
            if group.get("phase") not in {"earlier_layer", "earlier_projection"}
        ]
        for group in late_groups:
            result = test_group(group["name"], group["nodes"])
            if result["status"] == "strict_parity_passed":
                if result["size_budget_passed"]:
                    best_passing = result
                    break
                if best_over_budget is None or result["bundle_bytes"] < best_over_budget["bundle_bytes"]:
                    best_over_budget = result

        late_names = {group["name"] for group in late_groups}
        late_improved = any(
            entry["group"] in late_names
            and entry["diagnostic_score"] is not None
            and tuple(entry["diagnostic_score"]) < _diagnostic_score(parent_diagnostic)
            for entry in search_record["group_evaluations"]
        )
        early_improved_layers = []
        if best_passing is None and late_improved:
            for group in search_record["semantic_groups"]:
                if group.get("phase") != "earlier_layer":
                    continue
                result = test_group(group["name"], group["nodes"])
                if (
                    result.get("diagnostic_score") is not None
                    and _diagnostic_score(result) < _diagnostic_score(parent_diagnostic)
                ):
                    early_improved_layers.append(int(group["name"].split("-")[1]))
                if result["status"] == "strict_parity_passed":
                    if result["size_budget_passed"]:
                        best_passing = result
                        break
                    if best_over_budget is None or result["bundle_bytes"] < best_over_budget[
                        "bundle_bytes"
                    ]:
                        best_over_budget = result

        if best_passing is None:
            module_names = (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
            existing_group_names = {
                group["name"] for group in search_record["semantic_groups"]
            }
            for layer in sorted(set(early_improved_layers), reverse=True):
                for module in module_names:
                    nodes = sorted(
                        row["node"]
                        for row in inventory["nodes"]
                        if row["eligible_exact_ternary"]
                        and row["layer"] == layer
                        and row["module"] == module
                    )
                    name = f"{module.replace('.', '-')}-layer-{layer}"
                    group = {"name": name, "nodes": nodes, "phase": "earlier_projection"}
                    if name not in existing_group_names:
                        search_record["semantic_groups"].append(group)
                        existing_group_names.add(name)
                        save_search()
                    result = test_group(name, nodes)
                    if result["status"] == "strict_parity_passed":
                        if result["size_budget_passed"]:
                            best_passing = result
                            break
                        if best_over_budget is None or result["bundle_bytes"] < best_over_budget[
                            "bundle_bytes"
                        ]:
                            best_over_budget = result
                if best_passing is not None:
                    break

        def minimize_passing(seed: dict[str, Any]) -> dict[str, Any]:
            current = seed
            current_nodes = set(current["excluded_nodes"])
            changed = True
            while changed:
                changed = False
                for node in sorted(current_nodes):
                    trial_nodes = sorted(current_nodes - {node})
                    trial = evaluate(trial_nodes, "minimize-remove-one-node")
                    if trial["status"] == "strict_parity_passed":
                        current = trial
                        current_nodes = set(trial_nodes)
                        changed = True
            return current

        if best_passing is not None:
            best_passing = minimize_passing(best_passing)

        if best_passing is None:
            semantic_names = {group["name"] for group in search_record["semantic_groups"]}
            improved = [
                group
                for group in search_record["group_evaluations"]
                if group["group"] in semantic_names
                and group["diagnostic_score"] is not None
                and tuple(group["diagnostic_score"]) < _diagnostic_score(parent_diagnostic)
            ]
            combined_nodes = sorted({node for group in improved for node in group["nodes"]})
            if combined_nodes:
                combined = test_group("combined-improving-semantic-groups", combined_nodes)
                current_nodes = set(combined_nodes)
                current = combined
                if combined["status"] == "strict_parity_passed":
                    if combined["size_budget_passed"]:
                        best_passing = minimize_passing(combined)
                    else:
                        best_over_budget = combined
                elif combined.get("diagnostic_score") is not None:
                    changed = True
                    while changed:
                        changed = False
                        for node in sorted(current_nodes):
                            trial_nodes = sorted(current_nodes - {node})
                            trial = evaluate(trial_nodes, "minimize-diagnostic-improving-set")
                            if trial.get("diagnostic_score") is not None and _diagnostic_score(
                                trial
                            ) <= _diagnostic_score(current):
                                current = trial
                                current_nodes = set(trial_nodes)
                                changed = True
                            if trial["status"] == "strict_parity_passed":
                                if trial["size_budget_passed"]:
                                    best_passing = minimize_passing(trial)
                                elif best_over_budget is None or trial["bundle_bytes"] < best_over_budget[
                                    "bundle_bytes"
                                ]:
                                    best_over_budget = trial
                            if best_passing is not None:
                                break
                        if best_passing is not None:
                            break

        if best_passing is None and best_over_budget is not None:
            minimized = minimize_passing(best_over_budget)
            if minimized["size_budget_passed"]:
                best_passing = minimized
            else:
                best_over_budget = minimized
    except FallbackCandidateLimitReached:
        candidate_limit_reached = True
    candidates = [row for key, row in cache.items() if key != parent_key and row.get("bundle_bytes")]
    ranked = sorted(
        candidates,
        key=lambda row: (
            row.get("status") != "strict_parity_passed",
            row["bundle_bytes"],
            row.get("fallback_storage", {}).get("fallback_initializer_count", 0),
            row.get("diagnostic", {}).get("onnx_runtime_total_ms", float("inf")),
        ),
    )
    search_record["candidate_ranking"] = [row["candidate_key"] for row in ranked]
    diagnosed_candidates = [row for row in candidates if row.get("diagnostic_score") is not None]
    search_record["best_diagnostic_candidate"] = (
        min(diagnosed_candidates, key=_diagnostic_score)["candidate_key"]
        if diagnosed_candidates
        else None
    )
    search_record["pareto_frontier"] = [
        row["candidate_key"]
        for row in candidates
        if not any(
            other is not row
            and other["bundle_bytes"] <= row["bundle_bytes"]
            and other.get("fallback_storage", {}).get("fallback_initializer_count", 0)
            <= row.get("fallback_storage", {}).get("fallback_initializer_count", 0)
            and _diagnostic_score(other) <= _diagnostic_score(row)
            and (
                other["bundle_bytes"] < row["bundle_bytes"]
                or other.get("fallback_storage", {}).get("fallback_initializer_count", 0)
                < row.get("fallback_storage", {}).get("fallback_initializer_count", 0)
                or _diagnostic_score(other) < _diagnostic_score(row)
            )
            for other in candidates
            if other.get("diagnostic_score") is not None
            and row.get("diagnostic_score") is not None
        )
    ]
    if not candidate_limit_reached:
        search_record.pop("stop_condition", None)
    search_record["candidate_count"] = len(cache) - 1
    if candidate_limit_reached:
        search_record["decision"] = "fallback_search_candidate_limit_reached"
        search_record["stop_condition"] = {
            "reason": "Fallback search reached its declared candidate cap before completing semantic refinement.",
            "max_fallback_candidates": max_fallback_candidates,
            "minimality_proven": False,
            "strict_parity_tolerances_changed": False,
        }
        passing = best_passing or best_over_budget
        if passing is not None:
            search_record["best_parity_candidate_key"] = passing["candidate_key"]
    elif best_passing is not None:
        search_record["decision"] = "fallback_candidate_passed_within_size_budget"
        search_record["accepted_candidate_key"] = best_passing["candidate_key"]
        search_record["minimality_proven"] = True
    elif best_over_budget is not None:
        search_record["decision"] = "strict_parity_passed_over_size_budget"
        search_record["best_parity_candidate_key"] = best_over_budget["candidate_key"]
        search_record["minimality_proven"] = True
    else:
        search_record["decision"] = "no_fallback_candidate_passed_strict_parity_within_budget"
        search_record["stop_condition"] = {
            "reason": "No semantically searched fallback set passed unchanged strict parity within the declared 50-percent-reduction budget.",
            "strict_parity_tolerances_changed": False,
        }
        search_record["minimality_proven"] = False
    save_search()

    exact_results["fallback_search"] = {
        "report": str(fallback_results_path),
        "decision": search_record["decision"],
        "accepted_candidate_key": search_record.get("accepted_candidate_key"),
        "size_budget_bytes": search_record["fallback_size_budget_bytes"],
        "candidate_count": search_record["candidate_count"],
        "max_fallback_candidates": max_fallback_candidates,
        "best_parity_candidate_key": search_record.get("best_parity_candidate_key"),
    }
    exact_results["decision"] = search_record["decision"]
    _write_json(exact_results_path, exact_results)
    return search_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_bundle", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument(
        "--kernel-diagnostic",
        type=Path,
        default=Path("build/reports/nbits-kernel-diagnostic-level0.json"),
    )
    parser.add_argument("--max-fallback-candidates", type=int, default=140)
    args = parser.parse_args()
    report = run_exact_search(
        args.baseline_bundle,
        args.model_dir,
        args.inventory,
        args.output_root,
        args.reports_root,
    )
    fallback = None
    if report["decision"] == "all_exact_candidates_failed_strict_parity" or (
        report.get("fallback_search")
        and report["decision"] not in {
            "fallback_candidate_passed_within_size_budget",
            "strict_parity_passed_over_size_budget",
        }
    ):
        fallback = run_fallback_search(
            args.baseline_bundle,
            args.model_dir,
            args.inventory,
            args.output_root,
            args.reports_root,
            kernel_diagnostic_path=args.kernel_diagnostic,
            max_fallback_candidates=args.max_fallback_candidates,
        )
    summary = {
        "decision": fallback["decision"] if fallback else report["decision"],
        "inventory_sha256": report["inventory_sha256"],
        "candidates": [
            {
                key: candidate.get(key)
                for key in ("name", "status", "bundle_bytes", "candidate_key", "strict_parity")
            }
            for candidate in report["candidates"]
        ],
        "fallback_report": str(Path(args.reports_root) / "fallback/results.json")
        if fallback
        else report.get("fallback_search", {}).get("report"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
