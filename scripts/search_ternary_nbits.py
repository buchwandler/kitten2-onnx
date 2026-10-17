"""Build isolated exact-NBits candidates and apply the strict real-model parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kitten2_onnx.lm_parity_reference import (
    canonical_sha256,
    load_reference_pack,
    sha256_file,
    validate_source_identity,
)
from kitten2_onnx.nbits_fallback_delta import build_fallback_delta
from scripts.nbits_worker_process import run_candidate_worker
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


DIAGNOSTIC_VERSION = 2
ZERO_DIAGNOSTIC_SCORE = (0, 0, 0.0, 0.0)
_RETRYABLE_WORKER_STATUSES = {"worker_failed"}
_STOP_WORKER_STATUSES = {"worker_oom_guard", "disk_budget_exceeded", "worker_timeout"}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def delta_candidate_key(identity: dict[str, Any], excluded_nodes: list[str] | tuple[str, ...]) -> str:
    payload = {
        "identity": identity,
        "excluded_nodes": sorted(set(excluded_nodes)),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def import_legacy_cache(
    legacy_path: str | Path,
    *,
    baseline_bundle_files: dict[str, str],
    inventory_sha256: str,
) -> dict[str, Any]:
    path = Path(legacy_path)
    if not path.is_file():
        return {"source": str(path), "status": "missing", "rows": []}
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("baseline_bundle_files") != baseline_bundle_files
        or record.get("inventory_sha256") != inventory_sha256
    ):
        return {"source": str(path), "status": "identity_mismatch", "rows": []}
    rows = []
    for row in record.get("cache", {}).values():
        diagnostic = row.get("diagnostic") or {}
        score = row.get("diagnostic_score") or diagnostic.get("diagnostic_score")
        excluded = sorted(set(row.get("excluded_nodes", [])))
        if not isinstance(score, list) or len(score) != 4:
            continue
        rows.append(
            {
                "legacy_candidate_key": row.get("candidate_key"),
                "excluded_nodes": excluded,
                "semantic_label": row.get("semantic_label"),
                "diagnostic_score": score,
                "bundle_bytes": row.get("bundle_bytes"),
                "fallback_count": row.get("fallback_storage", {}).get(
                    "fallback_initializer_count"
                ),
                "status": row.get("status"),
                "artifact_state": row.get("artifact_state"),
                "reusable": False,
            }
        )
    rows.sort(
        key=lambda row: (
            _diagnostic_score(row),
            row.get("bundle_bytes") or float("inf"),
            row["legacy_candidate_key"] or "",
        )
    )
    return {
        "source": str(path),
        "status": "imported",
        "row_count": len(rows),
        "reusable_count": 0,
        "rows": rows,
        "group_evaluations": record.get("group_evaluations", []),
    }


def _legacy_seed_sets(legacy_import: dict[str, Any], width: int) -> list[list[str]]:
    seeds = []
    seen: set[tuple[str, ...]] = set()
    for row in legacy_import.get("rows", []):
        nodes = tuple(sorted(set(row.get("excluded_nodes", []))))
        if not nodes or nodes in seen:
            continue
        seeds.append(list(nodes))
        seen.add(nodes)
        if len(seeds) == width:
            break
    return seeds


def _frontier_objective(row: dict[str, Any]) -> tuple[Any, ...]:
    return (*_diagnostic_score(row), int(row.get("bundle_bytes_estimate", 0)), row["candidate_key"])


def bounded_frontier(rows: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("diagnostic_score") is not None
        and row.get("status") in {"diagnostic_complete", "strict_parity_passed"}
    ]
    return sorted(eligible, key=_frontier_objective)[:width]


def beam_expansions(
    frontier: list[dict[str, Any]],
    ranked_nodes: list[str],
    *,
    branch_factor: int = 2,
) -> list[list[str]]:
    expansions: dict[tuple[str, ...], None] = {}
    for state in frontier:
        existing = set(state["excluded_nodes"])
        additions = 0
        for node in ranked_nodes:
            if node in existing:
                continue
            nodes = tuple(sorted(existing | {node}))
            expansions[nodes] = None
            additions += 1
            if additions == branch_factor:
                break
    return [list(nodes) for nodes in sorted(expansions)]


def _known_parent_mismatch(row: dict[str, Any]) -> bool:
    divergence = row.get("first_divergence") or {}
    return (
        divergence.get("phase") == "prefill"
        and divergence.get("output") == "cache[24].value"
        and divergence.get("outside_tolerance_count") == 2
    )


class BoundedDeltaEvaluator:
    """Result-only cache with an absolute attempt budget and bounded retries."""

    def __init__(
        self,
        identity: dict[str, Any],
        cache: dict[str, dict[str, Any]],
        *,
        max_evaluations: int,
        max_retries: int,
        evaluate_one: Callable[[list[str], str], dict[str, Any]],
        on_update: Callable[[dict[str, dict[str, Any]], int], None],
        evaluation_count: int = 0,
    ) -> None:
        if max_evaluations <= 0 or max_retries < 0:
            raise ValueError("Evaluation budget must be positive and retries nonnegative")
        self.identity = identity
        self.cache = cache
        self.max_evaluations = max_evaluations
        self.max_retries = max_retries
        self.evaluate_one = evaluate_one
        self.on_update = on_update
        self.evaluation_count = evaluation_count
        self.stop_condition: dict[str, Any] | None = None

    def evaluate(self, excluded_nodes: list[str] | tuple[str, ...], label: str) -> dict[str, Any] | None:
        nodes = sorted(set(excluded_nodes))
        key = delta_candidate_key(self.identity, nodes)
        row = self.cache.get(key)
        if row and row.get("status") in {"diagnostic_complete", "strict_parity_passed", "strict_parity_failed", "strict_parity_worker_failed"}:
            return row
        if row is None:
            row = {
                "candidate_key": key,
                "excluded_nodes": nodes,
                "semantic_label": label,
                "status": "not_started",
                "attempts": 0,
            }
            self.cache[key] = row
        elif label and not row.get("semantic_label"):
            row["semantic_label"] = label

        for attempt in range(self.max_retries + 1):
            if self.evaluation_count >= self.max_evaluations:
                self.stop_condition = {
                    "reason": "evaluation_budget_exhausted",
                    "max_evaluations": self.max_evaluations,
                }
                self.on_update(self.cache, self.evaluation_count)
                return None
            self.evaluation_count += 1
            row["attempts"] = int(row.get("attempts", 0)) + 1
            row["status"] = "evaluation_started"
            try:
                result = self.evaluate_one(nodes, label)
            except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as error:
                result = {
                    "status": "delta_build_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            row.update(result)
            row["candidate_key"] = key
            row["excluded_nodes"] = nodes
            row["semantic_label"] = label
            row["attempts"] = int(row.get("attempts", 1))
            row["status"] = result.get("status", "worker_failed")
            self.on_update(self.cache, self.evaluation_count)
            if row["status"] in _STOP_WORKER_STATUSES:
                self.stop_condition = {
                    "reason": row["status"],
                    "candidate_key": key,
                    "attempt": row["attempts"],
                }
                return row
            if row["status"] not in _RETRYABLE_WORKER_STATUSES or attempt == self.max_retries:
                return row
        return row


def _evaluate_delta_worker(
    parent_model: Path,
    baseline_model: Path,
    inventory_path: Path,
    inventory: dict[str, Any],
    reference_path: Path,
    reference_identity_sha256: str,
    scratch_root: Path,
    reports_root: Path,
    parent_bundle_bytes: int,
    *,
    max_worker_rss_gib: float,
    max_search_disk_gib: float,
    worker_timeout_seconds: float,
) -> Callable[[list[str], str], dict[str, Any]]:
    scratch_root.mkdir(parents=True, exist_ok=True)

    def evaluate_one(nodes: list[str], label: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="delta-", dir=scratch_root) as temporary:
            work = Path(temporary)
            model_path = work / "lm.onnx"
            fallback_data = work / "fallback.data"
            result_path = work / "worker.json"
            delta = build_fallback_delta(
                parent_model,
                baseline_model,
                inventory,
                set(nodes),
                model_path,
                fallback_data,
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve().with_name("evaluate_nbits_candidate.py")),
                "--model",
                str(model_path),
                "--reference",
                str(reference_path),
                "--output",
                str(result_path),
                "--expected-source-identity-sha256",
                reference_identity_sha256,
            ]
            result = run_candidate_worker(
                command,
                output_path=result_path,
                search_root=reports_root,
                max_worker_rss_gib=max_worker_rss_gib,
                max_search_disk_gib=max_search_disk_gib,
                timeout_seconds=worker_timeout_seconds,
                scratch_dir=work,
                model_path=model_path,
                fallback_data_path=fallback_data,
                cwd=Path(__file__).resolve().parents[1],
            )
            return {
                "status": result.get("status", "worker_failed"),
                "diagnostic_score": result.get("diagnostic_score"),
                "first_divergence": result.get("first_divergence"),
                "diagnostic": result.get("diagnostic"),
                "fallback_storage": delta.fallback_storage,
                "bundle_bytes_estimate": parent_bundle_bytes
                + int(delta.fallback_storage["net_bundle_byte_cost"]),
                "model_sha256": result.get("model_sha256"),
                "peak_rss_bytes": result.get("peak_rss_bytes"),
                "scratch_bytes_peak": result.get("scratch_bytes_peak"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "worker_exit_code": result.get("worker_exit_code"),
                "worker_signal": result.get("worker_signal"),
                "worker_output_tail": result.get("worker_output_tail"),
                "fallback_data_bytes": result.get("fallback_data_bytes"),
                "semantic_label": label,
            }

    return evaluate_one


def _run_sensitivity_worker(
    parent_model: Path,
    baseline_model: Path,
    inventory_path: Path,
    reference_path: Path,
    identity: dict[str, Any],
    reports_root: Path,
    *,
    max_worker_rss_gib: float,
    max_search_disk_gib: float,
    worker_timeout_seconds: float,
) -> dict[str, Any]:
    identity_hash = canonical_sha256(identity)
    result_path = reports_root / "fallback" / f"sensitivity-{identity_hash[:16]}.json"
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "sensitivity_complete"
            and cached.get("parent_graph_sha256") == identity["parent_graph_sha256"]
            and cached.get("reference_pack_sha256") == identity["reference_pack_sha256"]
            and cached.get("baseline_graph_sha256") == identity["baseline_graph_sha256"]
        ):
            return cached

    scratch_root = reports_root / "fallback" / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sensitivity-", dir=scratch_root) as temporary:
        work = Path(temporary)
        instrumented_path = work / "instrumented.onnx"
        command = [
            sys.executable,
            str(Path(__file__).resolve().with_name("measure_nbits_sensitivity.py")),
            "--parent-model",
            str(parent_model),
            "--baseline-model",
            str(baseline_model),
            "--inventory",
            str(inventory_path),
            "--reference",
            str(reference_path),
            "--instrumented-model",
            str(instrumented_path),
            "--output",
            str(result_path),
        ]
        return run_candidate_worker(
            command,
            output_path=result_path,
            search_root=reports_root,
            max_worker_rss_gib=max_worker_rss_gib,
            max_search_disk_gib=max_search_disk_gib,
            timeout_seconds=worker_timeout_seconds,
            scratch_dir=work,
            model_path=parent_model,
            cwd=Path(__file__).resolve().parents[1],
        )


def _link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and os.path.samefile(source, destination):
            return
        raise FileExistsError(f"Candidate support file already exists: {destination}")
    os.link(source, destination)


def _run_delta_strict_parity(
    excluded_nodes: list[str],
    parent_bundle: Path,
    parent_model: Path,
    baseline_model: Path,
    inventory: dict[str, Any],
    model_dir: Path,
    reports_root: Path,
    *,
    max_worker_rss_gib: float,
    max_search_disk_gib: float,
    worker_timeout_seconds: float,
) -> dict[str, Any]:
    scratch_root = reports_root / "fallback" / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="strict-delta-", dir=scratch_root) as temporary:
        work = Path(temporary)
        lm_dir = work / "lm"
        model_path = lm_dir / "lm.onnx"
        fallback_data = lm_dir / "fallback.data"
        delta = build_fallback_delta(
            parent_model,
            baseline_model,
            inventory,
            set(excluded_nodes),
            model_path,
            fallback_data,
        )
        parent_lm = parent_bundle / "lm"
        metadata = json.loads((parent_lm / "lm.meta.json").read_text(encoding="utf-8"))
        metadata["external_data_files"] = sorted(
            set(metadata.get("external_data_files", [])) | {fallback_data.name}
        )
        (lm_dir / "lm.meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _link_file(parent_lm / metadata["embedding_file"], lm_dir / metadata["embedding_file"])
        speaker_model = work / "speaker" / "speaker.onnx"
        _link_file(parent_bundle / "speaker" / "speaker.onnx", speaker_model)
        result_path = work / "strict.json"
        log_path = work / "strict.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve().with_name("nbits_search_worker.py")),
            "strict-parity",
            "--model-dir",
            str(model_dir),
            "--lm-dir",
            str(lm_dir),
            "--speaker-model",
            str(speaker_model),
            "--output",
            str(result_path),
            "--log",
            str(log_path),
        ]
        result = run_candidate_worker(
            command,
            output_path=result_path,
            search_root=reports_root,
            max_worker_rss_gib=max_worker_rss_gib,
            max_search_disk_gib=max_search_disk_gib,
            timeout_seconds=worker_timeout_seconds,
            scratch_dir=work,
            model_path=model_path,
            fallback_data_path=fallback_data,
            cwd=Path(__file__).resolve().parents[1],
        )
        result["delta_manifest"] = delta.manifest_path.name
        return result


def finalist_eligible(row: dict[str, Any], size_budget_bytes: int) -> bool:
    return (
        tuple(_diagnostic_score(row)) == ZERO_DIAGNOSTIC_SCORE
        and row.get("strict_delta_parity", {}).get("status") == "passed"
        and int(row.get("bundle_bytes_estimate", 0)) <= size_budget_bytes
    )


def _strict_result_status(result: dict[str, Any]) -> str:
    if result.get("status") == "passed":
        return "strict_parity_passed"
    if result.get("status") == "failed":
        return "strict_parity_failed"
    return "strict_parity_worker_failed"


def _promote_finalist(
    row: dict[str, Any],
    baseline_bundle: Path,
    output_root: Path,
    reports_root: Path,
    model_dir: Path,
    *,
    max_worker_rss_gib: float,
    size_budget_bytes: int,
    max_search_disk_gib: float,
    worker_timeout_seconds: float,
) -> dict[str, Any]:
    if not finalist_eligible(row, size_budget_bytes):
        raise ValueError("Finalist must have zero diagnostic score, strict parity pass, and fit the size budget")
    candidate_id = row["candidate_key"][:12]
    output_bundle = output_root / f"b4-bs64-a1-finalist-{candidate_id}"
    report_dir = reports_root / "fallback" / f"finalist-{candidate_id}"
    if output_bundle.exists():
        raise FileExistsError(f"Finalist path already exists without a resumable record: {output_bundle}")
    report_dir.mkdir(parents=True, exist_ok=True)
    result_path = report_dir / "materialization.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("nbits_search_worker.py")),
        "build-finalist",
        "--input-bundle",
        str(baseline_bundle),
        "--output-bundle",
        str(output_bundle),
        "--report-dir",
        str(report_dir),
        "--model-dir",
        str(model_dir),
        "--output",
        str(result_path),
    ]
    for node in row["excluded_nodes"]:
        command.extend(["--exclude-node", node])
    build_result = run_candidate_worker(
        command,
        output_path=result_path,
        search_root=output_root,
        max_worker_rss_gib=max_worker_rss_gib,
        max_search_disk_gib=max_search_disk_gib,
        timeout_seconds=worker_timeout_seconds,
        scratch_dir=output_bundle,
        model_path=output_bundle / "lm" / "lm.onnx",
        cwd=Path(__file__).resolve().parents[1],
    )
    if build_result.get("status") != "materialized":
        shutil.rmtree(output_bundle, ignore_errors=True)
        return {"status": build_result.get("status", "materialization_failed"), "build": build_result}

    strict_path = report_dir / "strict-final.json"
    strict_log = report_dir / "strict-final.log"
    strict_command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("nbits_search_worker.py")),
        "strict-parity",
        "--model-dir",
        str(model_dir),
        "--lm-dir",
        str(output_bundle / "lm"),
        "--speaker-model",
        str(output_bundle / "speaker" / "speaker.onnx"),
        "--output",
        str(strict_path),
        "--log",
        str(strict_log),
    ]
    strict_result = run_candidate_worker(
        strict_command,
        output_path=strict_path,
        search_root=output_root,
        max_worker_rss_gib=max_worker_rss_gib,
        max_search_disk_gib=max_search_disk_gib,
        timeout_seconds=worker_timeout_seconds,
        scratch_dir=output_bundle,
        model_path=output_bundle / "lm" / "lm.onnx",
        cwd=Path(__file__).resolve().parents[1],
    )
    return {
        "status": "release_gates_passed"
        if strict_result.get("status") == "passed"
        else "final_strict_parity_failed",
        "output_bundle": str(output_bundle),
        "report_dir": str(report_dir),
        "build": build_result,
        "strict_parity": strict_result,
    }


def _run_beam_search(
    evaluator: BoundedDeltaEvaluator,
    seed_sets: list[list[str]],
    ranked_nodes: list[str],
    *,
    beam_width: int,
    strict_check: Callable[[dict[str, Any]], dict[str, Any]],
    max_depth: int = 8,
) -> dict[str, Any] | None:
    evaluated: dict[str, dict[str, Any]] = {}
    passing: dict[str, Any] | None = None

    def check(row: dict[str, Any] | None) -> None:
        nonlocal passing
        if row is None or row.get("diagnostic_score") is None:
            return
        evaluated[row["candidate_key"]] = row
        if row.get("status") == "strict_parity_passed":
            passing = row
            return
        if row.get("strict_delta_parity") is not None:
            if row.get("status") == "strict_parity_worker_failed":
                strict = strict_check(row)
                row["strict_delta_parity"] = strict
                row["status"] = _strict_result_status(strict)
                evaluator.on_update(evaluator.cache, evaluator.evaluation_count)
                if row["status"] == "strict_parity_passed":
                    passing = row
            return
        if tuple(_diagnostic_score(row)) != ZERO_DIAGNOSTIC_SCORE:
            return
        strict = strict_check(row)
        row["strict_delta_parity"] = strict
        row["status"] = _strict_result_status(strict)
        evaluator.on_update(evaluator.cache, evaluator.evaluation_count)
        if row["status"] == "strict_parity_passed":
            passing = row

    initial = []
    seen: set[tuple[str, ...]] = set()
    for nodes in [*seed_sets, *([node] for node in ranked_nodes[:beam_width])]:
        normalized = tuple(sorted(set(nodes)))
        if normalized and normalized not in seen:
            seen.add(normalized)
            initial.append(list(normalized))
    for nodes in initial:
        check(evaluator.evaluate(nodes, "legacy-seed" if nodes in seed_sets else "sensitivity-singleton"))
        if passing or evaluator.stop_condition:
            break

    frontier = bounded_frontier(list(evaluated.values()), beam_width)
    for depth in range(max_depth):
        if passing or evaluator.stop_condition or not frontier:
            break
        level_rows = []
        for nodes in beam_expansions(frontier, ranked_nodes):
            row = evaluator.evaluate(nodes, f"beam-depth-{depth + 1}")
            check(row)
            if row is not None and row.get("diagnostic_score") is not None:
                level_rows.append(row)
            if passing or evaluator.stop_condition:
                break
        frontier = bounded_frontier(level_rows, beam_width)

    if passing is None:
        return None

    current = passing
    current_nodes = set(current["excluded_nodes"])
    changed = True
    while changed and not evaluator.stop_condition:
        changed = False
        for node in sorted(current_nodes):
            trial_nodes = sorted(current_nodes - {node})
            trial = evaluator.evaluate(trial_nodes, "minimize-remove-one-node")
            if trial is None or trial.get("diagnostic_score") is None:
                if evaluator.stop_condition:
                    break
                continue
            if tuple(_diagnostic_score(trial)) == ZERO_DIAGNOSTIC_SCORE:
                if trial.get("strict_delta_parity") is None:
                    strict = strict_check(trial)
                    trial["strict_delta_parity"] = strict
                    trial["status"] = _strict_result_status(strict)
                    evaluator.on_update(evaluator.cache, evaluator.evaluation_count)
                if trial.get("status") == "strict_parity_passed":
                    current = trial
                    current_nodes = set(trial_nodes)
                    changed = True
                    break
        if evaluator.stop_condition:
            break
    return current


def _run_strict_delta_worker(
    row: dict[str, Any],
    parent_bundle: Path,
    parent_model: Path,
    baseline_model: Path,
    inventory: dict[str, Any],
    model_dir: Path,
    reports_root: Path,
    *,
    max_worker_rss_gib: float,
    max_search_disk_gib: float,
    worker_timeout_seconds: float,
) -> dict[str, Any]:
    return _run_delta_strict_parity(
        row["excluded_nodes"],
        parent_bundle,
        parent_model,
        baseline_model,
        inventory,
        model_dir,
        reports_root,
        max_worker_rss_gib=max_worker_rss_gib,
        max_search_disk_gib=max_search_disk_gib,
        worker_timeout_seconds=worker_timeout_seconds,
    )


def run_fallback_search(
    baseline_bundle: str | Path,
    model_dir: str | Path,
    inventory_path: str | Path,
    output_root: str | Path,
    reports_root: str | Path,
    *,
    kernel_diagnostic_path: str | Path | None = None,
    reference_path: str | Path = "build/reports/nbits-reference",
    fixture_path: str | Path = "tests/fixtures/reference_bruno.npz",
    max_fallback_candidates: int = 30,
    beam_width: int = 4,
    max_worker_rss_gib: float = 14.0,
    max_search_disk_gib: float = 30.0,
    worker_timeout_seconds: float = 5400.0,
    max_worker_retries: int = 1,
) -> dict[str, Any]:
    baseline = Path(baseline_bundle).resolve()
    model = Path(model_dir).resolve()
    inventory_file = Path(inventory_path).resolve()
    output = Path(output_root).resolve()
    reports = Path(reports_root).resolve()
    reference_root = Path(reference_path).resolve()
    fixture = Path(fixture_path).resolve()
    if max_fallback_candidates <= 0 or beam_width <= 0 or max_worker_retries < 0:
        raise ValueError("Evaluation budget and beam width must be positive, retries nonnegative")
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
        raise ValueError("Fallback search requires the completed exact-sweep failure decision")

    parent = next(
        (
            candidate
            for candidate in exact_candidates
            if candidate["configuration"]["block_size"] == 64
            and candidate["configuration"]["accuracy_level"] == 1
        ),
        None,
    )
    if parent is None:
        raise ValueError("The required immutable block-64 exact NBits parent is missing")
    parent_bundle = Path(parent["bundle"]).resolve()
    parent_model = parent_bundle / "lm" / "lm.onnx"
    baseline_model = baseline / "lm" / "lm.onnx"
    if not parent_model.is_file():
        raise FileNotFoundError(parent_model)

    reference = load_reference_pack(reference_root)
    if not reference.metadata.get("trusted"):
        raise ValueError("Fallback search requires an accepted trusted parity reference")
    validate_source_identity(reference, model, baseline, fixture)
    reference_identity_sha256 = canonical_sha256(reference.metadata["source_identity"])
    parent_graph_sha256 = parent.get("candidate_lm_graph_sha256") or sha256_file(parent_model)
    parent_external = parent.get("candidate_lm_external_data", {})
    if not parent_external:
        parent_manifest = json.loads((parent_bundle / "bundle.json").read_text(encoding="utf-8"))
        parent_external = {
            name: digest
            for name, digest in parent_manifest["files"].items()
            if name.startswith("lm/") and name.endswith(".onnx.data")
        }
    identity = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "diagnostic_format": "kitten2-onnx-nbits-candidate-diagnostic-v1",
        "reference_pack_sha256": reference.metadata["pack_sha256"],
        "reference_source_identity_sha256": reference_identity_sha256,
        "parent_graph_sha256": parent_graph_sha256,
        "parent_external_data_sha256": dict(sorted(parent_external.items())),
        "baseline_graph_sha256": inventory["source_graph_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
    }
    identity_hash = canonical_sha256(identity)
    fallback_reports = reports / "fallback"
    fallback_reports.mkdir(parents=True, exist_ok=True)
    legacy_import = import_legacy_cache(
        fallback_reports / "results.json",
        baseline_bundle_files=bundle["files"],
        inventory_sha256=inventory["inventory_sha256"],
    )
    result_path = fallback_reports / f"delta-search-{identity_hash[:16]}.json"
    if result_path.is_file():
        search_record = json.loads(result_path.read_text(encoding="utf-8"))
        if search_record.get("cache_identity") != identity:
            raise ValueError("Existing delta-search result has a different diagnostic cache identity")
        cache = search_record.get("evaluations", {})
        legacy_import = search_record.get("legacy_import", legacy_import)
    else:
        cache = {}
        search_record = {
            "format": "kitten2-onnx-ternary-nbits-delta-search-v2",
            "report_path": str(result_path),
            "cache_identity": identity,
            "legacy_import": legacy_import,
            "parent_candidate": parent["name"],
            "parent_bundle": str(parent_bundle),
            "parent_bundle_bytes": int(parent["bundle_bytes"]),
            "fallback_size_budget_bytes": int(exact_results["baseline_bundle_bytes"]) // 2,
            "beam_width": beam_width,
            "max_evaluations": max_fallback_candidates,
            "evaluations": cache,
            "evaluation_count": 0,
            "decision": "fallback_search_in_progress",
        }
    search_record["report_path"] = str(result_path)
    search_record["max_evaluations"] = max_fallback_candidates
    search_record["beam_width"] = beam_width
    search_record["historical_kernel_diagnostic"] = (
        str(Path(kernel_diagnostic_path).resolve()) if kernel_diagnostic_path is not None else None
    )

    def save(current_cache: dict[str, dict[str, Any]], evaluation_count: int) -> None:
        search_record["evaluations"] = {key: current_cache[key] for key in sorted(current_cache)}
        search_record["evaluation_count"] = evaluation_count
        search_record["candidate_count"] = sum(
            bool(row.get("excluded_nodes")) for row in current_cache.values()
        )
        _write_json_atomic(result_path, search_record)

    evaluator = BoundedDeltaEvaluator(
        identity,
        cache,
        max_evaluations=max_fallback_candidates,
        max_retries=max_worker_retries,
        evaluate_one=_evaluate_delta_worker(
            parent_model,
            baseline_model,
            inventory_file,
            inventory,
            reference_root,
            reference_identity_sha256,
            fallback_reports / ".scratch",
            reports,
            int(parent["bundle_bytes"]),
            max_worker_rss_gib=max_worker_rss_gib,
            max_search_disk_gib=max_search_disk_gib,
            worker_timeout_seconds=worker_timeout_seconds,
        ),
        on_update=save,
        evaluation_count=int(search_record.get("evaluation_count", 0)),
    )
    parent_row = evaluator.evaluate([], "exact-parent-preflight")
    if parent_row is None or parent_row.get("status") != "diagnostic_complete":
        search_record["decision"] = "parent_diagnostic_failed"
        search_record["stop_condition"] = evaluator.stop_condition or {
            "reason": parent_row.get("status") if parent_row else "evaluation_budget_exhausted"
        }
        save(evaluator.cache, evaluator.evaluation_count)
        return search_record
    if not _known_parent_mismatch(parent_row):
        search_record["decision"] = "diagnostic_equivalence_failed"
        search_record["stop_condition"] = {
            "reason": "block-64 diagnostic did not reproduce the established prefill cache[24].value two-value mismatch",
            "observed_first_divergence": parent_row.get("first_divergence"),
        }
        save(evaluator.cache, evaluator.evaluation_count)
        return search_record
    search_record["parent_diagnostic_equivalence"] = {
        "passed": True,
        "diagnostic_score": parent_row["diagnostic_score"],
        "first_divergence": parent_row["first_divergence"],
    }
    save(evaluator.cache, evaluator.evaluation_count)

    sensitivity = _run_sensitivity_worker(
        parent_model,
        baseline_model,
        inventory_file,
        reference_root,
        identity,
        reports,
        max_worker_rss_gib=max_worker_rss_gib,
        max_search_disk_gib=max_search_disk_gib,
        worker_timeout_seconds=worker_timeout_seconds,
    )
    search_record["sensitivity_report"] = str(
        fallback_reports / f"sensitivity-{identity_hash[:16]}.json"
    )
    if sensitivity.get("status") != "sensitivity_complete" or len(
        sensitivity.get("ranked_nodes", [])
    ) != inventory["eligible_exact_ternary_matmul_count"]:
        search_record["decision"] = "sensitivity_stage_failed"
        search_record["stop_condition"] = {
            "reason": sensitivity.get("status", "incomplete_sensitivity"),
            "worker_output_tail": sensitivity.get("worker_output_tail"),
        }
        save(evaluator.cache, evaluator.evaluation_count)
        return search_record
    search_record["sensitivity_node_count"] = sensitivity["measured_node_count"]
    ranked_nodes = [row["node"] for row in sensitivity["ranked_nodes"]]
    seed_sets = _legacy_seed_sets(legacy_import, beam_width)

    def strict_check(row: dict[str, Any]) -> dict[str, Any]:
        result = _run_strict_delta_worker(
            row,
            parent_bundle,
            parent_model,
            baseline_model,
            inventory,
            model,
            reports,
            max_worker_rss_gib=max_worker_rss_gib,
            max_search_disk_gib=max_search_disk_gib,
            worker_timeout_seconds=worker_timeout_seconds,
        )
        row["strict_delta_parity"] = result
        if result.get("status") not in {"passed", "failed"}:
            evaluator.stop_condition = {
                "reason": result.get("status", "strict_worker_failed"),
                "candidate_key": row["candidate_key"],
                "phase": "strict_delta_parity",
            }
        save(evaluator.cache, evaluator.evaluation_count)
        return result

    finalist = _run_beam_search(
        evaluator,
        seed_sets,
        ranked_nodes,
        beam_width=beam_width,
        strict_check=strict_check,
    )
    search_record["beam_frontier"] = [
        row["candidate_key"]
        for row in bounded_frontier(
            [
                row
                for row in evaluator.cache.values()
                if row.get("excluded_nodes") and row.get("diagnostic_score") is not None
            ],
            beam_width,
        )
    ]
    search_record["stop_condition"] = evaluator.stop_condition
    if finalist is not None:
        search_record["best_zero_diagnostic_candidate"] = finalist["candidate_key"]
        if finalist_eligible(finalist, search_record["fallback_size_budget_bytes"]):
            promotion = _promote_finalist(
                finalist,
                baseline,
                output,
                reports,
                model,
                size_budget_bytes=search_record["fallback_size_budget_bytes"],
                max_worker_rss_gib=max_worker_rss_gib,
                max_search_disk_gib=max_search_disk_gib,
                worker_timeout_seconds=worker_timeout_seconds,
            )
            finalist["promotion"] = promotion
            search_record["decision"] = promotion["status"]
        else:
            search_record["decision"] = "zero_diagnostic_strict_pass_over_size_budget"
    elif evaluator.stop_condition:
        search_record["decision"] = "fallback_search_stopped"
    else:
        search_record["decision"] = "no_zero_diagnostic_candidate"
        search_record["stop_condition"] = {
            "reason": "bounded beam found no zero-diagnostic candidate",
            "minimality_proven": False,
        }

    diagnosed = [
        row
        for row in evaluator.cache.values()
        if row.get("excluded_nodes") and row.get("diagnostic_score") is not None
    ]
    search_record["best_diagnostic_candidate"] = (
        min(diagnosed, key=_frontier_objective)["candidate_key"] if diagnosed else None
    )
    search_record["evaluation_count"] = evaluator.evaluation_count
    search_record["candidate_count"] = len(diagnosed)
    save(evaluator.cache, evaluator.evaluation_count)
    exact_results["fallback_search"] = {
        "report": str(result_path),
        "decision": search_record["decision"],
        "candidate_count": search_record["candidate_count"],
        "evaluation_count": search_record["evaluation_count"],
        "max_evaluations": max_fallback_candidates,
        "best_zero_diagnostic_candidate": search_record.get("best_zero_diagnostic_candidate"),
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
    parser.add_argument("--reference", type=Path, default=Path("build/reports/nbits-reference"))
    parser.add_argument("--fixture", type=Path, default=Path("tests/fixtures/reference_bruno.npz"))
    parser.add_argument("--max-fallback-candidates", type=int, default=30)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-worker-rss-gib", type=float, default=14.0)
    parser.add_argument("--max-search-disk-gib", type=float, default=30.0)
    parser.add_argument("--worker-timeout-seconds", type=float, default=5400.0)
    parser.add_argument("--max-worker-retries", type=int, default=1)
    args = parser.parse_args()
    report = run_exact_search(
        args.baseline_bundle,
        args.model_dir,
        args.inventory,
        args.output_root,
        args.reports_root,
    )
    fallback = None
    terminal_decisions = {
        "release_gates_passed",
        "final_strict_parity_failed",
        "zero_diagnostic_strict_pass_over_size_budget",
    }
    if report["decision"] == "all_exact_candidates_failed_strict_parity" or (
        report.get("fallback_search") and report["decision"] not in terminal_decisions
    ):
        fallback = run_fallback_search(
            args.baseline_bundle,
            args.model_dir,
            args.inventory,
            args.output_root,
            args.reports_root,
            kernel_diagnostic_path=args.kernel_diagnostic,
            reference_path=args.reference,
            fixture_path=args.fixture,
            max_fallback_candidates=args.max_fallback_candidates,
            beam_width=args.beam_width,
            max_worker_rss_gib=args.max_worker_rss_gib,
            max_search_disk_gib=args.max_search_disk_gib,
            worker_timeout_seconds=args.worker_timeout_seconds,
            max_worker_retries=args.max_worker_retries,
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
        "fallback_report": fallback.get("report_path")
        if fallback
        else report.get("fallback_search", {}).get("report"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
