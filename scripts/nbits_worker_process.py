from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.compact_nbits_search import (
    SearchDiskBudgetExceeded,
    check_search_disk_budget,
    directory_size_bytes,
)

_GIB = 1024**3


def _available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _process_tree(pid: int) -> list[int]:
    found: list[int] = []
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.append(current)
        children = Path(f"/proc/{current}/task/{current}/children")
        try:
            pending.extend(int(value) for value in children.read_text().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return found


def _process_tree_rss_bytes(pid: int) -> int:
    total = 0
    for process_id in _process_tree(pid):
        status = Path(f"/proc/{process_id}/status")
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return total


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _read_child_result(output_path: Path) -> dict[str, Any] | None:
    if not output_path.is_file():
        return None
    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) else None


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
    try:
        disk_bytes_before = check_search_disk_budget(
            search_root, int(max_search_disk_gib * _GIB)
        )
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
            "scratch_bytes_peak": 0,
            "search_disk_bytes_before": directory_size_bytes(search_root),
        }
        _write_json_atomic(output_path, result)
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    started = time.perf_counter()
    peak_rss = 0
    scratch_peak = 0
    oom_guard_triggered = False
    timeout_triggered = False
    disk_budget_triggered = False
    disk_budget_error = None
    last_disk_check = started
    disk_budget_bytes = int(max_search_disk_gib * _GIB)
    child_result: dict[str, Any] | None = None
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            peak_rss = max(peak_rss, _process_tree_rss_bytes(process.pid))
            if scratch_dir is not None:
                scratch_peak = max(scratch_peak, directory_size_bytes(scratch_dir))
            if peak_rss > int(max_worker_rss_gib * _GIB):
                oom_guard_triggered = True
                _terminate_process_group(process)
                break
            if time.perf_counter() - started > timeout_seconds:
                timeout_triggered = True
                _terminate_process_group(process)
                break
            now = time.perf_counter()
            if now - last_disk_check >= 1.0:
                last_disk_check = now
                try:
                    check_search_disk_budget(search_root, disk_budget_bytes)
                except SearchDiskBudgetExceeded as error:
                    disk_budget_triggered = True
                    disk_budget_error = str(error)
                    _terminate_process_group(process)
                    break
            time.sleep(0.1)
        return_code = process.wait()
    peak_rss = max(peak_rss, int((_read_child_result(output_path) or {}).get("peak_rss_bytes", 0)))
    child_result = _read_child_result(output_path)
    if not disk_budget_triggered:
        try:
            check_search_disk_budget(search_root, disk_budget_bytes)
        except SearchDiskBudgetExceeded as error:
            disk_budget_triggered = True
            disk_budget_error = str(error)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    worker_signal = signal.Signals(-return_code).name if return_code < 0 else None
    if oom_guard_triggered:
        status = "worker_oom_guard"
    elif timeout_triggered:
        status = "worker_timeout"
    elif disk_budget_triggered:
        status = "disk_budget_exceeded"
    elif return_code != 0 or child_result is None:
        status = "worker_failed"
    else:
        status = child_result.get("status", "worker_failed")
    result = dict(child_result or {})
    result.update(
        {
            "status": status,
            "worker_exit_code": return_code,
            "worker_signal": worker_signal,
            "peak_rss_bytes": peak_rss,
            "available_memory_before_bytes": available_before,
            "available_memory_after_bytes": _available_memory_bytes(),
            "oom_guard_triggered": oom_guard_triggered,
            "timeout_triggered": timeout_triggered,
            "disk_budget_triggered": disk_budget_triggered,
            "disk_budget_error": disk_budget_error,
            "elapsed_seconds": time.perf_counter() - started,
            "scratch_bytes_peak": scratch_peak,
            "search_disk_bytes_before": disk_bytes_before,
            "search_disk_bytes_after": directory_size_bytes(search_root),
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
    _write_json_atomic(output_path, result)
    return result
