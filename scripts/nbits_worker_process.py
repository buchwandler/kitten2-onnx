from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.compact_nbits_search import (
    SearchDiskBudgetExceeded,
    check_search_disk_budget,
    directory_size_bytes,
)
from scripts.resource_guard import (
    _available_memory_bytes,
    _process_tree_rss_bytes,
    run_bounded_process,
)

_GIB = 1024**3


def run_candidate_worker(
    command: Sequence[str],
    *,
    output_path: Path,
    search_root: Path,
    max_worker_rss_gib: float = 14.0,
    max_search_disk_gib: float = 30.0,
    timeout_seconds: float = 5400.0,
    scratch_dir: Path | None = None,
    model_path: Path | None = None,
    fallback_data_path: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    if max_worker_rss_gib <= 0 or max_search_disk_gib <= 0 or timeout_seconds <= 0:
        raise ValueError("Worker RSS, disk, and timeout limits must be positive")
    output_path = output_path.resolve()
    search_root = search_root.resolve()
    log_path = output_path.with_suffix(output_path.suffix + ".log")
    available_before = _available_memory_bytes()
    disk_budget_bytes = int(max_search_disk_gib * _GIB)
    try:
        disk_bytes_before = check_search_disk_budget(search_root, disk_budget_bytes)
    except SearchDiskBudgetExceeded as error:
        result = {
            "status": "disk_budget_exceeded",
            "error": str(error),
            "available_memory_before_bytes": available_before,
            "available_memory_after_bytes": _available_memory_bytes(),
            "worker_exit_code": None,
            "worker_signal": None,
            "peak_rss_bytes": 0,
            "oom_guard_triggered": False,
            "disk_budget_triggered": True,
            "disk_budget_error": str(error),
            "scratch_bytes_peak": 0,
            "search_disk_bytes_before": directory_size_bytes(search_root),
        }
        from scripts.resource_guard import _write_json_atomic

        _write_json_atomic(output_path, result)
        return result

    scratch_peak = 0
    last_disk_check = time.perf_counter()
    disk_error: str | None = None

    def monitor_disk() -> tuple[str, dict[str, Any]] | None:
        nonlocal scratch_peak, last_disk_check, disk_error
        if scratch_dir is not None:
            scratch_peak = max(scratch_peak, directory_size_bytes(scratch_dir))
        now = time.perf_counter()
        if now - last_disk_check < 1.0:
            return None
        last_disk_check = now
        try:
            check_search_disk_budget(search_root, disk_budget_bytes)
        except SearchDiskBudgetExceeded as error:
            disk_error = str(error)
            return "disk_budget_exceeded", {"disk_budget_error": disk_error}
        return None

    guard_result = run_bounded_process(
        command,
        max_rss_gib=max_worker_rss_gib,
        min_available_gib=0,
        reserve_available_gib=0,
        timeout_seconds=timeout_seconds,
        output_json=output_path,
        log_path=log_path,
        cwd=cwd,
        max_additional_swap_gib=None,
        monitor_callback=monitor_disk,
        rss_reader=_process_tree_rss_bytes,
    )
    child_result = guard_result.get("worker_result")
    status = guard_result["status"]
    if status == "rss_guard":
        status = "worker_oom_guard"
    elif status == "timeout":
        status = "worker_timeout"
    elif status == "disk_budget_exceeded":
        status = "disk_budget_exceeded"
    elif status == "failed" or child_result is None:
        status = "worker_failed"
    else:
        status = child_result.get("status", "worker_failed")

    try:
        disk_bytes_after = check_search_disk_budget(search_root, disk_budget_bytes)
    except SearchDiskBudgetExceeded as error:
        disk_bytes_after = directory_size_bytes(search_root)
        disk_error = str(error)
        status = "disk_budget_exceeded"

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    result = dict(child_result or {})
    result.update(
        {
            "status": status,
            "worker_exit_code": guard_result["worker_exit_code"],
            "worker_signal": guard_result["worker_signal"],
            "peak_rss_bytes": guard_result["peak_rss_bytes"],
            "available_memory_before_bytes": guard_result["available_memory_before_bytes"],
            "available_memory_after_bytes": guard_result["available_memory_after_bytes"],
            "oom_guard_triggered": guard_result["status"] == "rss_guard",
            "timeout_triggered": guard_result["status"] == "timeout",
            "disk_budget_triggered": status == "disk_budget_exceeded",
            "disk_budget_error": disk_error,
            "elapsed_seconds": guard_result["elapsed_seconds"],
            "scratch_bytes_peak": scratch_peak,
            "search_disk_bytes_before": disk_bytes_before,
            "search_disk_bytes_after": disk_bytes_after,
            "graph_bytes": model_path.stat().st_size if model_path and model_path.is_file() else 0,
            "fallback_data_bytes": (
                fallback_data_path.stat().st_size
                if fallback_data_path and fallback_data_path.is_file()
                else 0
            ),
            "worker_log": str(log_path),
            "worker_output_tail": log_text[-12000:],
        }
    )
    from scripts.resource_guard import _write_json_atomic

    _write_json_atomic(output_path, result)
    return result
