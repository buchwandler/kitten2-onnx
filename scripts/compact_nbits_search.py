from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any

_COMPLETED_STATUSES = {"diagnostic_complete", "strict_parity_failed", "strict_parity_passed"}
_METRIC_FIELDS = (
    "bundle_bytes",
    "diagnostic",
    "diagnostic_score",
    "fallback_storage",
    "size_budget_passed",
    "preferred_size_passed",
    "strict_parity",
    "status",
    "transformed_matmul_count",
    "validation",
)
_REQUIRED_DIAGNOSTIC_FIELDS = (
    "test_completed",
    "failed_output_tensors",
    "total_values_outside_tolerance",
    "max_normalized_tolerance_violation",
    "max_absolute_error",
    "diagnostic_score",
)


class SearchDiskBudgetExceeded(RuntimeError):
    pass


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except FileNotFoundError:
            continue
    return total


def check_search_disk_budget(search_root: Path, max_bytes: int) -> int:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    used = directory_size_bytes(search_root)
    if used > max_bytes:
        raise SearchDiskBudgetExceeded(
            f"Search root uses {used} bytes, exceeding the {max_bytes}-byte budget"
        )
    return used


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _merge_metrics(row: dict[str, Any], report: dict[str, Any] | None) -> None:
    if report is None:
        return
    for key in _METRIC_FIELDS:
        if row.get(key) is None and report.get(key) is not None:
            row[key] = report[key]
    if isinstance(row.get("diagnostic"), dict) and isinstance(report.get("diagnostic"), dict):
        for key, value in report["diagnostic"].items():
            row["diagnostic"].setdefault(key, value)
    if isinstance(row.get("fallback_storage"), dict) and isinstance(
        report.get("fallback_storage"), dict
    ):
        for key, value in report["fallback_storage"].items():
            row["fallback_storage"].setdefault(key, value)


def _validate_metrics(candidate_key: str, row: dict[str, Any]) -> None:
    errors: list[str] = []
    if not isinstance(row.get("bundle_bytes"), int) or row["bundle_bytes"] < 0:
        errors.append("bundle_bytes is missing or invalid")
    fallback_storage = row.get("fallback_storage")
    if not isinstance(fallback_storage, dict) or not isinstance(
        fallback_storage.get("fallback_initializer_count"), int
    ):
        errors.append("fallback_storage.fallback_initializer_count is missing or invalid")
    score = row.get("diagnostic_score")
    if (
        not isinstance(score, list)
        or len(score) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in score
        )
    ):
        errors.append("diagnostic_score must contain four finite numbers")
    diagnostic = row.get("diagnostic")
    if not isinstance(diagnostic, dict):
        errors.append("diagnostic metrics are missing")
    else:
        for key in _REQUIRED_DIAGNOSTIC_FIELDS:
            if key not in diagnostic:
                errors.append(f"diagnostic.{key} is missing")
    if errors:
        raise ValueError(f"Completed cache row {candidate_key} is incomplete: " + "; ".join(errors))


def _pid_is_live(row: dict[str, Any]) -> bool:
    pid = row.get("worker_pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    worker_host = row.get("worker_host")
    if worker_host and worker_host not in {socket.gethostname(), "localhost"}:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _safe_candidate_path(value: str, fallback_root: Path, *, allow_missing: bool = True) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = fallback_root / candidate
    resolved = candidate.resolve()
    if resolved.parent != fallback_root.resolve():
        raise ValueError(f"Candidate path is not a direct child of {fallback_root}: {value}")
    if candidate.is_symlink():
        raise ValueError(f"Refusing to operate on symlink candidate path: {candidate}")
    if not allow_missing and not resolved.is_dir():
        raise ValueError(f"Candidate directory does not exist: {resolved}")
    return resolved


def _safe_scratch_path(value: str, search_root: Path) -> Path:
    scratch_root = (search_root / "_scratch").resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = search_root / candidate
    resolved = candidate.resolve()
    if not _is_within(resolved, scratch_root) or resolved == scratch_root:
        raise ValueError(f"Refusing to remove non-scratch path: {value}")
    if candidate.is_symlink():
        raise ValueError(f"Refusing to remove symlink scratch path: {candidate}")
    return resolved


def compact_search(
    search_root: Path,
    reports_root: Path,
    *,
    apply: bool = False,
    keep: tuple[str, ...] = (),
) -> dict[str, Any]:
    search_root = search_root.resolve()
    reports_root = reports_root.resolve()
    fallback_root = search_root / "fallback"
    fallback_reports = reports_root / "fallback"
    results_path = fallback_reports / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Fallback search results not found: {results_path}")
    results = _read_json(results_path)
    cache = results.get("cache")
    if not isinstance(cache, dict):
        raise TypeError(f"Fallback search cache is missing or invalid: {results_path}")

    keep_values = set(keep)
    protected_keys = {
        results.get("accepted_candidate_key"),
        results.get("best_parity_candidate_key"),
    }
    protected_names = {results.get("parent_candidate")}
    protected_names.discard(None)
    deletion_entries: list[dict[str, Any]] = []
    stale_entries: list[dict[str, Any]] = []

    for candidate_key, row in cache.items():
        if not isinstance(row, dict):
            raise TypeError(f"Cache row {candidate_key} is not an object")
        status = row.get("status")
        if status in {"build_started", "worker_started"}:
            if _pid_is_live(row):
                stale_entries.append(
                    {"candidate_key": candidate_key, "action": "worker_live", "status": status}
                )
                continue
            row["status"] = "stale_recovered"
            row["artifact_state"] = "stale_recovered"
            scratch_value = row.get("scratch_dir") or row.get("scratch_path")
            if scratch_value:
                scratch_path = _safe_scratch_path(str(scratch_value), search_root)
                stale_entries.append(
                    {
                        "candidate_key": candidate_key,
                        "action": "remove_scratch" if apply else "would_remove_scratch",
                        "path": str(scratch_path),
                    }
                )
                if apply:
                    shutil.rmtree(scratch_path, ignore_errors=False)
            else:
                stale_entries.append(
                    {"candidate_key": candidate_key, "action": "mark_stale_recovered"}
                )
            continue
        if status not in _COMPLETED_STATUSES:
            continue

        bundle_value = row.get("bundle")
        if row.get("name") in protected_names:
            row["artifact_state"] = "protected"
            continue
        if bundle_value:
            candidate_path = Path(str(bundle_value))
            if not candidate_path.is_absolute():
                candidate_path = fallback_root / candidate_path
            if not _is_within(candidate_path.resolve(), fallback_root):
                row["artifact_state"] = "protected"
                continue
        report_path = Path(row.get("reports", fallback_reports / str(row.get("name", ""))))
        if report_path.is_dir():
            report_path = report_path / "search-result.json"
        if report_path.is_file():
            _merge_metrics(row, _read_json(report_path))
        _validate_metrics(candidate_key, row)
        row["artifact_state"] = "prunable"
        row["diagnostic_reproducible"] = bool(row["diagnostic"].get("test_completed"))

        bundle_value = row.get("bundle")
        if not bundle_value:
            continue
        bundle_path = _safe_candidate_path(str(bundle_value), fallback_root)
        if (
            candidate_key in protected_keys
            or row.get("name") in protected_names
            or row.get("status") == "strict_parity_passed"
            or candidate_key in keep_values
            or row.get("name") in keep_values
            or str(bundle_path) in keep_values
        ):
            row["artifact_state"] = "protected"
            continue
        exists = bundle_path.is_dir()
        deletion_entries.append(
            {
                "candidate_key": candidate_key,
                "name": row.get("name"),
                "path": str(bundle_path),
                "exists": exists,
                "bytes": directory_size_bytes(bundle_path) if exists else 0,
                "action": "delete" if apply and exists else "would_delete" if exists else "already_missing",
            }
        )
        if apply and exists:
            shutil.rmtree(bundle_path)
            row["artifact_state"] = "pruned"
        elif not exists:
            row["artifact_state"] = "pruned"

    manifest = {
        "format": "kitten2-onnx-nbits-search-deletion-manifest-v1",
        "search_root": str(search_root),
        "reports_root": str(reports_root),
        "mode": "apply" if apply else "dry_run",
        "deletions": deletion_entries,
        "stale_rows": stale_entries,
        "bytes_marked_for_deletion": sum(item["bytes"] for item in deletion_entries),
    }
    manifest_path = fallback_reports / "deletion-manifest.json"
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(results_path, results)
    return {**manifest, "manifest_path": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact old NBits fallback artifacts while retaining resumable result rows."
    )
    parser.add_argument("search_root", type=Path)
    parser.add_argument("reports_root", type=Path)
    parser.add_argument("--apply", action="store_true", help="Delete eligible candidate directories.")
    parser.add_argument(
        "--keep", action="append", default=[], help="Candidate name, key, or path to retain."
    )
    args = parser.parse_args()
    result = compact_search(
        args.search_root, args.reports_root, apply=args.apply, keep=tuple(args.keep)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
