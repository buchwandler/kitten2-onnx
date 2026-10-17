from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compact_nbits_search import (
    SearchDiskBudgetExceeded,
    check_search_disk_budget,
    compact_search,
)


def _completed_row(name: str, bundle: Path, reports: Path, *, status: str = "diagnostic_complete"):
    diagnostic = {
        "test_completed": True,
        "failed_output_tensors": 2,
        "total_values_outside_tolerance": 4,
        "max_normalized_tolerance_violation": 1.25,
        "max_absolute_error": 0.003,
        "diagnostic_score": [2, 4, 1.25, 0.003],
    }
    report = {
        "name": name,
        "status": status,
        "bundle": str(bundle),
        "reports": str(reports),
        "bundle_bytes": 100,
        "fallback_storage": {"fallback_initializer_count": 1},
        "diagnostic": diagnostic,
        "diagnostic_score": diagnostic["diagnostic_score"],
    }
    return report


def _seed_search(tmp_path: Path, *, statuses: tuple[str, ...] = ("diagnostic_complete",)):
    search_root = tmp_path / "experiments"
    reports_root = tmp_path / "reports"
    fallback_root = search_root / "fallback"
    fallback_reports = reports_root / "fallback"
    fallback_root.mkdir(parents=True)
    fallback_reports.mkdir(parents=True)
    cache = {}
    for index, status in enumerate(statuses):
        name = f"candidate-{index}"
        bundle = fallback_root / name
        bundle.mkdir()
        (bundle / "payload.bin").write_bytes(b"candidate")
        candidate_reports = fallback_reports / name
        candidate_reports.mkdir()
        row = _completed_row(name, bundle, candidate_reports, status=status)
        row["diagnostic"] = None
        row["diagnostic_score"] = None
        row["fallback_storage"] = None
        (candidate_reports / "search-result.json").write_text(
            json.dumps(_completed_row(name, bundle, candidate_reports, status=status)),
            encoding="utf-8",
        )
        cache[f"key-{index}"] = row
    (fallback_reports / "results.json").write_text(
        json.dumps({"cache": cache}), encoding="utf-8"
    )
    return search_root, reports_root, cache


def test_dry_run_merges_metrics_and_preserves_bundles_and_reports(tmp_path):
    search_root, reports_root, _ = _seed_search(tmp_path)
    bundle = search_root / "fallback/candidate-0"

    result = compact_search(search_root, reports_root)

    assert bundle.is_dir()
    assert (reports_root / "fallback/candidate-0/search-result.json").is_file()
    assert result["mode"] == "dry_run"
    assert result["deletions"][0]["action"] == "would_delete"
    stored = json.loads((reports_root / "fallback/results.json").read_text())
    row = stored["cache"]["key-0"]
    assert row["diagnostic_score"] == [2, 4, 1.25, 0.003]
    assert row["artifact_state"] == "prunable"
    assert row["diagnostic_reproducible"] is True


def test_apply_deletes_only_registered_failed_bundle_and_keeps_reports(tmp_path):
    search_root, reports_root, _ = _seed_search(
        tmp_path, statuses=("diagnostic_complete", "strict_parity_passed")
    )
    failed_bundle = search_root / "fallback/candidate-0"
    passing_bundle = search_root / "fallback/candidate-1"
    failed_report = reports_root / "fallback/candidate-0/search-result.json"

    result = compact_search(search_root, reports_root, apply=True)

    assert not failed_bundle.exists()
    assert passing_bundle.is_dir()
    assert failed_report.is_file()
    assert result["deletions"][0]["action"] == "delete"
    stored = json.loads((reports_root / "fallback/results.json").read_text())
    assert stored["cache"]["key-0"]["artifact_state"] == "pruned"
    assert stored["cache"]["key-1"]["artifact_state"] == "protected"


def test_stale_worker_recovery_removes_only_its_scratch_directory(tmp_path):
    search_root, reports_root, _ = _seed_search(tmp_path, statuses=())
    fallback_root = search_root / "fallback"
    scratch = search_root / "_scratch/worker-1"
    scratch.mkdir(parents=True)
    (scratch / "partial.onnx").write_bytes(b"scratch")
    leftover_bundle = fallback_root / "partial-candidate"
    leftover_bundle.mkdir()
    (leftover_bundle / "partial.bin").write_bytes(b"bundle")
    results_path = reports_root / "fallback/results.json"
    results_path.write_text(
        json.dumps(
            {
                "cache": {
                    "stale": {
                        "status": "worker_started",
                        "worker_pid": 2_147_483_647,
                        "worker_host": "localhost",
                        "scratch_dir": str(scratch),
                        "bundle": str(leftover_bundle),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = compact_search(search_root, reports_root, apply=True)

    assert not scratch.exists()
    assert leftover_bundle.is_dir()
    assert result["stale_rows"][0]["action"] == "remove_scratch"
    stored = json.loads(results_path.read_text())
    assert stored["cache"]["stale"]["status"] == "stale_recovered"


def test_completed_row_without_required_metrics_blocks_pruning(tmp_path):
    search_root, reports_root, _ = _seed_search(tmp_path)
    bundle = search_root / "fallback/candidate-0"
    results_path = reports_root / "fallback/results.json"
    results = json.loads(results_path.read_text())
    results["cache"]["key-0"]["reports"] = str(reports_root / "missing-report")
    (reports_root / "fallback/candidate-0/search-result.json").unlink()
    results["cache"]["key-0"].pop("diagnostic_score")
    results_path.write_text(json.dumps(results), encoding="utf-8")

    with pytest.raises(ValueError, match="Completed cache row key-0 is incomplete"):
        compact_search(search_root, reports_root, apply=True)
    assert bundle.is_dir()


def test_disk_budget_reports_usage_and_rejects_over_budget(tmp_path):
    search_root = tmp_path / "search"
    search_root.mkdir()
    (search_root / "candidate.bin").write_bytes(b"12345")

    assert check_search_disk_budget(search_root, 5) == 5
    with pytest.raises(SearchDiskBudgetExceeded, match="exceeding"):
        check_search_disk_budget(search_root, 4)
