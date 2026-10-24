"""Build a packed or emb4 ONNX release candidate through fail-closed local gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from kitten2_onnx.weights import resolve_weight_path
from scripts.resource_guard import run_bounded_process

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPRESSION_CORPUS = PROJECT_ROOT / "tests/fixtures/expression_cases.json"
MULTILINGUAL_CORPUS = PROJECT_ROOT / "tests/fixtures/multilingual_cases.json"


class ReleaseGateError(RuntimeError):
    def __init__(self, gate: str, message: str):
        super().__init__(message)
        self.gate = gate


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path, *, exclude: set[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    excluded = exclude or set()
    files = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        size = path.stat().st_size
        file_hash = _sha256_file(path)
        files.append({"path": relative, "bytes": size, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), files


def _revision_manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model_revision.json"
    if not path.is_file():
        raise FileNotFoundError(f"Pinned model revision manifest is missing: {path}")
    manifest = _read_json(path)
    if not manifest.get("source_model_revision") or not manifest.get("upstream_code_revision"):
        raise ValueError(f"Pinned model and upstream code revisions are required in {path}")
    return manifest


def _prepare_variant_source(
    model_dir: Path,
    variant_reference_dir: Path,
    variant: str,
    staged_model_dir: Path,
) -> Path:

    model_dir = model_dir.resolve()
    variant_reference_dir = variant_reference_dir.resolve()
    source_manifest = _revision_manifest(model_dir)
    weight_manifest = _revision_manifest(variant_reference_dir)
    source_revision = source_manifest["source_model_revision"]
    weight_revision = weight_manifest["source_model_revision"]
    source_code_revision = source_manifest["upstream_code_revision"]
    weight_code_revision = weight_manifest["upstream_code_revision"]
    if source_revision != weight_revision or source_code_revision != weight_code_revision:
        raise ValueError(
            f"Pinned source model and {variant} reference revisions do not match: "
            f"model={source_revision!r}/{weight_revision!r}, code={source_code_revision!r}/{weight_code_revision!r}"
        )
    declared_variant = weight_manifest.get("weight_variant")
    if declared_variant is not None and declared_variant != variant:
        raise ValueError(
            f"Variant reference declares {declared_variant!r} weights, not {variant!r}"
        )
    selected_weight = resolve_weight_path(variant_reference_dir, variant)
    if selected_weight is None or not selected_weight.is_file():
        raise FileNotFoundError(f"Pinned {variant} source weights are missing from {variant_reference_dir}")

    source_config = _read_json(model_dir / "config.json")
    source_lm_dir = (model_dir / source_config.get("lm_dir", "lm")).resolve()

    def ignore_lm_weight_files(directory: str, names: list[str]) -> set[str]:
        current = Path(directory).resolve()
        if current == source_lm_dir or current.is_relative_to(source_lm_dir):
            return {name for name in names if name.endswith(".safetensors")}
        return set()

    shutil.copytree(model_dir, staged_model_dir, ignore=ignore_lm_weight_files)
    destination_weight = resolve_weight_path(staged_model_dir, variant)
    if destination_weight is None:
        raise ValueError(f"Source config declares no {variant} checkpoint path")
    destination_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_weight, destination_weight)
    staged_manifest = dict(source_manifest)
    staged_manifest["weight_variant"] = variant
    staged_manifest["variant_weight_source_sha256"] = _sha256_file(selected_weight)
    _write_json(staged_model_dir / "model_revision.json", staged_manifest)
    return staged_model_dir


def _run_stage(
    name: str,
    command: Sequence[str],
    *,
    reports_dir: Path,
    max_rss_gib: float,
    min_available_gib: float,
    reserve_available_gib: float,
    timeout_seconds: float,
    command_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    result_path = reports_dir / "resources" / f"{name}.json"
    log_path = reports_dir / "logs" / f"{name}.log"
    result = command_runner(
        command,
        max_rss_gib=max_rss_gib,
        min_available_gib=min_available_gib,
        reserve_available_gib=reserve_available_gib,
        timeout_seconds=timeout_seconds,
        output_json=result_path,
        log_path=log_path,
        cwd=PROJECT_ROOT,
    )
    if result.get("status") != "ok":
        raise ReleaseGateError(name, f"{name} resource/process gate failed: {result}")
    resource_report = _require_report(result_path, name)
    if resource_report.get("status") != "ok":
        raise ReleaseGateError(name, f"{name} resource report did not record a successful guard result")
    return result


def _require_report(path: Path, gate: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseGateError(gate, f"{gate} did not produce its required report: {path}")
    return _read_json(path)


def _validate_e2e_report(
    path: Path,
    variant: str,
    corpus_path: Path,
    expected_bundle_dir: Path,
    *,
    source_model_revision: str,
    source_weight_sha256: str,
    upstream_code_revision: str,
) -> dict[str, Any]:
    report = _require_report(path, f"{variant}-e2e")
    bundle_provenance = report.get("onnx_bundle_provenance", {})
    if (
        report.get("weight_variant") != variant
        or report.get("validation_mode") != "exact_parity"
        or report.get("source_model_revision") != source_model_revision
        or report.get("source_weight_sha256") != source_weight_sha256
        or report.get("upstream_code_revision") != upstream_code_revision
        or report.get("onnx_bundle_revisions", {}).get("source_model_revision") != source_model_revision
        or report.get("requested_providers") != ["CPUExecutionProvider"]
        or report.get("actual_providers") != ["CPUExecutionProvider"]
        or bundle_provenance.get("path") != str(expected_bundle_dir)
        or bundle_provenance.get("status") != "available"
        or not isinstance(bundle_provenance.get("tree_sha256"), str)
        or len(bundle_provenance["tree_sha256"]) != 64
    ):
        raise ReleaseGateError(f"{variant}-e2e", "E2E report provenance or provider does not match the pinned source")
    cases = report.get("cases")
    expected_cases = _read_json(corpus_path).get("cases")
    if (
        not isinstance(cases, list)
        or not isinstance(expected_cases, list)
        or [case.get("name") for case in cases] != [case.get("name") for case in expected_cases]
        or report.get("first_divergence") is not None
        or report.get("sample_rate") != 24000
    ):
        raise ReleaseGateError(f"{variant}-e2e", "E2E corpus report is incomplete or has a token divergence")
    for case in cases:
        if (
            case.get("token_ids_match_reference") is not True
            or case.get("waveform_finite") is not True
            or case.get("waveform_samples", 0) <= 0
        ):
            raise ReleaseGateError(
                f"{variant}-e2e",
                f"E2E corpus case failed exact token or finite-audio checks: {case.get('name')}",
            )
    return report


def _tree_hash(root: Path) -> str:
    return _tree_manifest(root)[0]


def _release_manifest(
    candidate_dir: Path,
    reports_dir: Path,
    variant: str,
    exact_int8_tree_sha256: str,
    report_paths: dict[str, Path],
) -> dict[str, Any]:
    bundle = _read_json(candidate_dir / "bundle.json")
    lm_metadata = _read_json(candidate_dir / bundle["lm"]["metadata"])
    files_tree, files = _tree_manifest(candidate_dir, exclude={"release_manifest.json"})
    transformer = bundle["lm"].get("storage", {}).get("transformer", {})
    vocabulary = bundle["lm"].get("storage", {}).get("vocabulary", {})
    quantization = lm_metadata.get("quantization", {})
    report_refs = {
        name: {
            "path": str(path.relative_to(reports_dir)),
            "sha256": _sha256_file(path),
        }
        for name, path in report_paths.items()
        if path.is_file()
    }
    for path in sorted((reports_dir / "resources").glob("*.json")):
        report_refs[f"resource_{path.stem}"] = {
            "path": str(path.relative_to(reports_dir)),
            "sha256": _sha256_file(path),
        }
    return {
        "format": "kitten2-onnx-release-manifest-v1",
        "variant": variant,
        "bundle_format": bundle.get("format"),
        "source_model_revision": bundle.get("revisions", {}).get("source_model_revision"),
        "upstream_code_revision": bundle.get("revisions", {}).get("upstream_code_revision"),
        "source_weight_sha256": bundle["lm"].get("source_weight_sha256")
        or lm_metadata.get("source_weight_sha256"),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "execution_provider": "CPUExecutionProvider",
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "build_config": {
            "transformer_storage": transformer,
            "nbits": quantization.get("bits"),
            "block_size": quantization.get("block_size"),
            "accuracy_level": quantization.get("accuracy_level"),
            "vocabulary_storage": vocabulary,
            "quantization": quantization,
        },
        "source_exact_int8_tree_sha256": exact_int8_tree_sha256,
        "files_tree_sha256": files_tree,
        "files": {item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]} for item in files},
        "manifest_self_hash_excluded": True,
        "reports": report_refs,
        "reports_directory": str(reports_dir),
    }


def build_release(
    model_dir: str | Path,
    *,
    weights: str,
    output_dir: str | Path,
    reports_dir: str | Path,
    variant_reference_dir: str | Path | None = None,
    exact_int8_bundle_dir: str | Path = "build/baseline-hosted-onnx",
    max_rss_gib: float = 16.0,
    min_available_gib: float = 2.0,
    reserve_available_gib: float = 2.0,
    timeout_seconds: float = 21600.0,
    voice_smoke_max_new_tokens: int = 16,
    command_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if weights not in {"packed", "emb4"}:
        raise ValueError("weights must be 'packed' or 'emb4'")
    if voice_smoke_max_new_tokens < 1:
        raise ValueError("voice_smoke_max_new_tokens must be positive")
    model_dir = Path(model_dir).resolve()
    variant_reference_dir = Path(variant_reference_dir or model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    reports_dir = Path(reports_dir).resolve()
    exact_int8_bundle_dir = Path(exact_int8_bundle_dir).resolve()
    protected_roots = (model_dir, variant_reference_dir, exact_int8_bundle_dir)
    for label, artifact_path in (("Release output", output_dir), ("Release reports", reports_dir)):
        for protected_root in protected_roots:
            if (
                artifact_path == protected_root
                or artifact_path.is_relative_to(protected_root)
                or protected_root.is_relative_to(artifact_path)
            ):
                raise ValueError(f"{label} must not overlap protected input: {protected_root}")
        if any((parent / "bundle.json").is_file() for parent in (artifact_path, *artifact_path.parents)):
            raise ValueError(f"{label} must not be inside an existing bundle: {artifact_path}")
    if output_dir == reports_dir or output_dir.is_relative_to(reports_dir) or reports_dir.is_relative_to(output_dir):
        raise ValueError("Release output and reports directories must not overlap")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing release candidate: {output_dir}")
    if reports_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing report directory: {reports_dir}")
    reports_dir.mkdir(parents=True)
    command_runner = command_runner or run_bounded_process
    stage_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    stage_bundle = stage_root / "candidate"
    staged_model_dir = stage_root / "source-model"
    exported_bundle = stage_root / "exported"
    current_gate = "source-preflight"
    exact_int8_tree_before: str | None = None
    candidate_promoted = False
    report_paths: dict[str, Path] = {}
    completed_gates: list[str] = []

    try:
        source_manifest = _revision_manifest(model_dir)
        variant_manifest = _revision_manifest(variant_reference_dir)
        if source_manifest["source_model_revision"] != variant_manifest["source_model_revision"]:
            raise ValueError("Source model and variant reference revisions do not match")
        source_config = _read_json(model_dir / "config.json")
        voice_index_path = model_dir / source_config.get("voices", "voices/voices.json")
        voice_index = _read_json(voice_index_path)
        if not voice_index:
            raise ValueError(f"Pinned source voice index is empty: {voice_index_path}")
        expected_source_voices = sorted(voice_index)
        if not exact_int8_bundle_dir.is_dir() or not (exact_int8_bundle_dir / "bundle.json").is_file():
            raise FileNotFoundError(f"Existing exact-int8 bundle is missing: {exact_int8_bundle_dir}")
        exact_int8_tree_before = _tree_hash(exact_int8_bundle_dir)
        prepared_model_dir = _prepare_variant_source(
            model_dir, variant_reference_dir, weights, staged_model_dir
        )
        completed_gates.append("pinned-source-revision")

        current_gate = "export"
        _run_stage(
            current_gate,
            [
                sys.executable,
                "-m",
                "kitten2_onnx.export_all",
                str(prepared_model_dir),
                "--weights",
                weights,
                "--all-voices",
                "--output",
                str(exported_bundle),
                "--validate",
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        completed_gates.append(current_gate)

        current_gate = "q4-transform"
        _run_stage(
            current_gate,
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/optimize_bundle.py"),
                str(exported_bundle),
                str(stage_bundle),
                "--lm",
                "ternary-nbits4",
                "--validate",
                "--report-dir",
                str(reports_dir / "optimization"),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        report_paths["optimization"] = reports_dir / "optimization/optimization.json"
        optimization_report = _require_report(report_paths["optimization"], current_gate)
        if optimization_report.get("validation", {}).get("status") != "passed":
            raise ReleaseGateError(current_gate, "q4 optimizer validation report did not pass")
        optimized_bundle = _read_json(stage_bundle / "bundle.json")
        optimized_lm = optimized_bundle["lm"]
        optimized_lm_metadata = _read_json(stage_bundle / optimized_lm["metadata"])
        candidate_source_weight_sha256 = optimized_lm.get("source_weight_sha256") or optimized_lm_metadata.get(
            "source_weight_sha256"
        )
        if (
            optimized_lm.get("weight_variant") != weights
            or optimized_bundle.get("revisions", {}).get("source_model_revision")
            != source_manifest["source_model_revision"]
            or optimized_bundle.get("revisions", {}).get("upstream_code_revision")
            != source_manifest["upstream_code_revision"]
            or not candidate_source_weight_sha256
        ):
            raise ReleaseGateError(current_gate, "Optimized bundle variant or pinned source provenance is invalid")
        completed_gates.append(current_gate)

        current_gate = "bundle-validation"
        validation_path = reports_dir / "bundle-validation.json"
        _run_stage(
            current_gate,
            [
                sys.executable,
                "-m",
                "kitten2_onnx.validate_bundle",
                str(stage_bundle),
                "--output",
                str(validation_path),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        report_paths["validation"] = validation_path
        validation = _require_report(validation_path, current_gate)
        if validation.get("valid") is not True or validation.get("bundle_dir") != str(stage_bundle):
            raise ReleaseGateError(current_gate, "Candidate bundle validation reported invalid")
        completed_gates.append(current_gate)

        current_gate = "size-audit"
        size_path = reports_dir / "size.json"
        _run_stage(
            current_gate,
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/audit_bundle_size.py"),
                str(stage_bundle),
                "--output",
                str(size_path),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        report_paths["size"] = size_path
        size_report = _require_report(size_path, current_gate)
        if (
            int(size_report.get("bundle_bytes", 0)) <= 0
            or size_report.get("bundle_dir") != str(stage_bundle)
        ):
            raise ReleaseGateError(current_gate, "Size audit did not report positive candidate bytes or the expected bundle")
        completed_gates.append(current_gate)

        current_gate = "lm-parity"
        lm_parity_path = reports_dir / "lm-parity.json"
        _run_stage(
            current_gate,
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/benchmark_variant_lm.py"),
                str(variant_reference_dir),
                str(model_dir),
                str(stage_bundle),
                "--variant",
                weights,
                "--decode-steps",
                "32",
                "--output",
                str(lm_parity_path),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        report_paths["lm_parity"] = lm_parity_path
        lm_report = _require_report(lm_parity_path, current_gate)
        if lm_report.get("parity_passed") is not True:
            raise ReleaseGateError(current_gate, "Strict 32-step LM logits/cache/token parity failed")
        if (
            lm_report.get("variant") != weights
            or lm_report.get("source_model_revision") != source_manifest["source_model_revision"]
            or lm_report.get("source_weight_sha256") != candidate_source_weight_sha256
        ):
            raise ReleaseGateError(current_gate, "LM parity report provenance does not match the candidate source")
        completed_gates.append(current_gate)

        for corpus_name, corpus_path in (
            ("expression", EXPRESSION_CORPUS),
            ("multilingual", MULTILINGUAL_CORPUS),
        ):
            current_gate = f"{corpus_name}-e2e"
            e2e_path = reports_dir / f"e2e-{corpus_name}.json"
            _run_stage(
                current_gate,
                [
                    sys.executable,
                    "-m",
                    "scripts.validate_e2e_corpus",
                    str(model_dir),
                    str(stage_bundle),
                    "--weights",
                    weights,
                    "--weight-model",
                    str(variant_reference_dir),
                    "--corpus",
                    str(corpus_path),
                    "--provider",
                    "CPUExecutionProvider",
                    "--output",
                    str(e2e_path),
                    "--lm-parity-report",
                    str(lm_parity_path),
                ],
                reports_dir=reports_dir,
                max_rss_gib=max_rss_gib,
                min_available_gib=min_available_gib,
                reserve_available_gib=reserve_available_gib,
                timeout_seconds=timeout_seconds,
                command_runner=command_runner,
            )
            report_paths[corpus_name] = e2e_path
            _validate_e2e_report(
                e2e_path,
                weights,
                corpus_path,
                stage_bundle,
                source_model_revision=source_manifest["source_model_revision"],
                source_weight_sha256=candidate_source_weight_sha256,
                upstream_code_revision=source_manifest["upstream_code_revision"],
            )
            completed_gates.append(current_gate)

        current_gate = "all-voice-smoke"
        voice_smoke_path = reports_dir / "voice-smoke.json"
        _run_stage(
            current_gate,
            [
                sys.executable,
                "-m",
                "scripts.smoke_variant_bundle",
                str(stage_bundle),
                "--output",
                str(voice_smoke_path),
                "--max-new-tokens",
                str(voice_smoke_max_new_tokens),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        report_paths["voices"] = voice_smoke_path
        voice_report = _require_report(voice_smoke_path, current_gate)
        voice_cases = voice_report.get("cases")
        expected_voices = voice_report.get("expected_voices")
        candidate_voices = optimized_bundle.get("voices")
        if (
            voice_report.get("status") != "passed"
            or voice_report.get("bundle") != str(stage_bundle)
            or not isinstance(voice_report.get("bundle_tree_sha256"), str)
            or len(voice_report["bundle_tree_sha256"]) != 64
            or not isinstance(expected_voices, list)
            or not expected_voices
            or expected_voices != expected_source_voices
            or not isinstance(candidate_voices, dict)
            or sorted(candidate_voices) != expected_source_voices
            or not isinstance(voice_cases, list)
            or len(voice_cases) != len(expected_voices)
            or {case.get("voice") for case in voice_cases} != set(expected_voices)
            or any(
                case.get("status") != "passed"
                or case.get("sample_rate") != 24000
                or case.get("waveform_finite") is not True
                or int(case.get("waveform_samples", 0)) <= 0
                for case in voice_cases
            )
        ):
            raise ReleaseGateError(current_gate, "All-voice smoke did not pass every candidate voice")
        completed_gates.append(current_gate)

        current_gate = "exact-int8-regression"
        exact_validation_path = reports_dir / "exact-int8-validation.json"
        _run_stage(
            current_gate,
            [
                sys.executable,
                "-m",
                "kitten2_onnx.validate_bundle",
                str(exact_int8_bundle_dir),
                "--output",
                str(exact_validation_path),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        exact_validation = _require_report(exact_validation_path, current_gate)
        if (
            exact_validation.get("valid") is not True
            or exact_validation.get("bundle_dir") != str(exact_int8_bundle_dir)
        ):
            raise ReleaseGateError(current_gate, "Existing exact-int8 bundle validation failed")
        exact_smoke_path = reports_dir / "exact-int8-smoke.json"
        _run_stage(
            f"{current_gate}-smoke",
            [
                sys.executable,
                "-m",
                "scripts.smoke_variant_bundle",
                str(exact_int8_bundle_dir),
                "--output",
                str(exact_smoke_path),
                "--voice",
                "Bruno",
                "--max-new-tokens",
                str(voice_smoke_max_new_tokens),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        exact_smoke = _require_report(exact_smoke_path, current_gate)
        exact_cases = exact_smoke.get("cases")
        if (
            exact_smoke.get("status") != "passed"
            or exact_smoke.get("bundle") != str(exact_int8_bundle_dir)
            or not isinstance(exact_smoke.get("bundle_tree_sha256"), str)
            or len(exact_smoke["bundle_tree_sha256"]) != 64
            or exact_smoke.get("expected_voices") != ["Bruno"]
            or not isinstance(exact_cases, list)
            or len(exact_cases) != 1
            or exact_cases[0].get("voice") != "Bruno"
            or exact_cases[0].get("status") != "passed"
            or exact_cases[0].get("sample_rate") != 24000
            or exact_cases[0].get("waveform_finite") is not True
            or int(exact_cases[0].get("waveform_samples", 0)) <= 0
        ):
            raise ReleaseGateError(current_gate, "Existing exact-int8 Bruno smoke failed")
        report_paths["exact_int8_validation"] = exact_validation_path
        report_paths["exact_int8_smoke"] = exact_smoke_path
        completed_gates.append(current_gate)

        current_gate = "release-manifest"
        manifest = _release_manifest(
            stage_bundle,
            reports_dir,
            weights,
            exact_int8_tree_before,
            report_paths,
        )
        manifest_path = stage_bundle / "release_manifest.json"
        _write_json(manifest_path, manifest)
        shutil.copy2(manifest_path, reports_dir / "release_manifest.json")
        report_paths["release_manifest"] = reports_dir / "release_manifest.json"

        current_gate = "final-validation"
        final_validation_path = reports_dir / "final-validation.json"
        _run_stage(
            current_gate,
            [
                sys.executable,
                "-m",
                "kitten2_onnx.validate_bundle",
                str(stage_bundle),
                "--output",
                str(final_validation_path),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        final_validation = _require_report(final_validation_path, current_gate)
        if (
            final_validation.get("valid") is not True
            or final_validation.get("bundle_dir") != str(stage_bundle)
        ):
            raise ReleaseGateError(current_gate, "Final bundle validation failed after manifest generation")
        final_size_path = reports_dir / "final-size.json"
        current_gate = "final-size-audit"
        _run_stage(
            current_gate,
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/audit_bundle_size.py"),
                str(stage_bundle),
                "--output",
                str(final_size_path),
            ],
            reports_dir=reports_dir,
            max_rss_gib=max_rss_gib,
            min_available_gib=min_available_gib,
            reserve_available_gib=reserve_available_gib,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        final_size = _require_report(final_size_path, current_gate)
        if (
            int(final_size.get("bundle_bytes", 0)) <= 0
            or final_size.get("bundle_dir") != str(stage_bundle)
        ):
            raise ReleaseGateError(current_gate, "Final size audit report is invalid or targets another bundle")
        completed_gates.extend(("release-manifest", "final-validation", "final-size-audit"))

        current_gate = "exact-int8-immutability"
        if _tree_hash(exact_int8_bundle_dir) != exact_int8_tree_before:
            raise ReleaseGateError(current_gate, "Existing exact-int8 bundle changed during qualification")
        completed_gates.append(current_gate)

        current_gate = "atomic-promotion"
        if output_dir.exists():
            raise FileExistsError(f"Refusing to replace an existing release candidate: {output_dir}")
        _, staged_files = _tree_manifest(stage_bundle, exclude={"release_manifest.json"})
        staged_tree_sha256 = _tree_hash(stage_bundle)
        staged_manifest_path = stage_bundle / "release_manifest.json"
        manifest_sha256 = _sha256_file(staged_manifest_path)
        os.replace(stage_bundle, output_dir)
        candidate_promoted = True
        status = {
            "status": "verified_local_candidate",
            "candidate_promoted": True,
            "variant": weights,
            "candidate": str(output_dir),
            "candidate_bytes": int(final_size["bundle_bytes"]),
            "candidate_tree_sha256": staged_tree_sha256,
            "release_manifest": str(output_dir / "release_manifest.json"),
            "release_manifest_sha256": manifest_sha256,
            "payload_file_count": len(staged_files),
            "reports": str(reports_dir),
            "completed_gates": completed_gates,
        }
        _write_json(reports_dir / "release_status.json", status)
        return status
    except Exception as error:
        failure = {
            "status": "failed",
            "candidate_promoted": candidate_promoted,
            "variant": weights,
            "candidate": str(output_dir),
            "reports": str(reports_dir),
            "failed_gate": getattr(error, "gate", current_gate),
            "error": f"{type(error).__name__}: {error}",
            "completed_gates": completed_gates,
        }
        _write_json(reports_dir / "release_status.json", failure)
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--weights", choices=("packed", "emb4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--variant-reference", type=Path)
    parser.add_argument("--exact-int8-bundle", type=Path, default=Path("build/baseline-hosted-onnx"))
    parser.add_argument("--max-rss-gib", type=float, default=16.0)
    parser.add_argument("--min-available-gib", type=float, default=2.0)
    parser.add_argument("--reserve-available-gib", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--voice-smoke-max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    result = build_release(
        args.model_dir,
        weights=args.weights,
        output_dir=args.output,
        reports_dir=args.reports,
        variant_reference_dir=args.variant_reference,
        exact_int8_bundle_dir=args.exact_int8_bundle,
        max_rss_gib=args.max_rss_gib,
        min_available_gib=args.min_available_gib,
        reserve_available_gib=args.reserve_available_gib,
        timeout_seconds=args.timeout_seconds,
        voice_smoke_max_new_tokens=args.voice_smoke_max_new_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
