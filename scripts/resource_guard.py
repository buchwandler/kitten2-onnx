from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_GIB = 1024**3


def _memory_info() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key in {"MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = int(value.split()[0]) * 1024
    return values


def _available_memory_bytes() -> int | None:
    return _memory_info().get("MemAvailable")


def _swap_used_bytes() -> int | None:
    info = _memory_info()
    if "SwapTotal" not in info or "SwapFree" not in info:
        return None
    return info["SwapTotal"] - info["SwapFree"]


def _process_tree(pid: int) -> list[int]:
    found: list[int] = []
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.append(current)
        task_root = Path(f"/proc/{current}/task")
        for children in task_root.glob("*/children"):
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


def _last_stage(progress_jsonl: Path | None) -> str | None:
    if progress_jsonl is None or not progress_jsonl.is_file():
        return None
    try:
        lines = progress_jsonl.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if line:
                event = json.loads(line)
                return event.get("stage") if isinstance(event, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_bounded_process(
    command: Sequence[str],
    *,
    max_rss_gib: float,
    min_available_gib: float,
    reserve_available_gib: float,
    timeout_seconds: float,
    output_json: str | Path,
    log_path: str | Path,
    cwd: str | Path | None = None,
    max_additional_swap_gib: float | None = 2.0,
    poll_interval_seconds: float = 0.1,
    progress_jsonl: str | Path | None = None,
    monitor_callback: Callable[[], tuple[str, dict[str, Any]] | None] | None = None,
    rss_reader: Callable[[int], int] | None = None,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command must not be empty")
    if max_rss_gib <= 0 or timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("RSS, timeout, and poll interval limits must be positive")
    if min_available_gib < 0 or reserve_available_gib < 0:
        raise ValueError("available-memory limits must not be negative")
    if max_additional_swap_gib is not None and max_additional_swap_gib < 0:
        raise ValueError("maximum additional swap must not be negative")

    result_path = Path(output_json).resolve()
    log_file = Path(log_path).resolve()
    progress_file = Path(progress_jsonl).resolve() if progress_jsonl is not None else None
    working_directory = Path(cwd).resolve() if cwd is not None else None
    available_before = _available_memory_bytes()
    swap_before = _swap_used_bytes()
    result: dict[str, Any] = {
        "status": "failed",
        "command": list(command),
        "peak_rss_bytes": 0,
        "available_memory_before_bytes": available_before,
        "minimum_available_memory_bytes": available_before,
        "available_memory_after_bytes": available_before,
        "swap_used_before_bytes": swap_before,
        "swap_used_peak_bytes": swap_before,
        "worker_exit_code": None,
        "worker_signal": None,
        "elapsed_seconds": 0.0,
        "last_stage": _last_stage(progress_file),
        "log": str(log_file),
    }
    available_minimum = int(min_available_gib * _GIB)
    reserve = int(reserve_available_gib * _GIB)
    if available_before is not None and available_before < max(available_minimum, reserve):
        result["status"] = "memory_reserve_guard"
        result["error"] = "available memory is below the configured preflight threshold"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch()
        _write_json_atomic(result_path, result)
        return result

    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    started = time.perf_counter()
    peak_rss = 0
    minimum_available = available_before
    swap_peak = swap_before
    stop_status: str | None = None
    stop_details: dict[str, Any] = {}
    get_tree_rss = rss_reader or _process_tree_rss_bytes
    try:
        with log_file.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                list(command),
                cwd=working_directory,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            while process.poll() is None:
                peak_rss = max(peak_rss, get_tree_rss(process.pid))
                available_now = _available_memory_bytes()
                if available_now is not None:
                    minimum_available = (
                        available_now
                        if minimum_available is None
                        else min(minimum_available, available_now)
                    )
                swap_now = _swap_used_bytes()
                if swap_now is not None:
                    swap_peak = swap_now if swap_peak is None else max(swap_peak, swap_now)
                if peak_rss > int(max_rss_gib * _GIB):
                    stop_status = "rss_guard"
                elif available_now is not None and available_now < reserve:
                    stop_status = "memory_reserve_guard"
                elif (
                    max_additional_swap_gib is not None
                    and swap_before is not None
                    and swap_now is not None
                    and swap_peak - swap_before > int(max_additional_swap_gib * _GIB)
                ):
                    stop_status = "swap_guard"
                elif time.perf_counter() - started > timeout_seconds:
                    stop_status = "timeout"
                elif monitor_callback is not None:
                    callback_result = monitor_callback()
                    if callback_result is not None:
                        stop_status, stop_details = callback_result
                if stop_status is not None:
                    _terminate_process_group(process)
                    break
                time.sleep(poll_interval_seconds)
            worker_exit_code = process.wait()
            peak_rss = max(peak_rss, get_tree_rss(process.pid))
    except OSError as error:
        result.update(
            {
                "status": "failed",
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _write_json_atomic(result_path, result)
        return result

    worker_result = _read_json(result_path)
    worker_signal = signal.Signals(-worker_exit_code).name if worker_exit_code < 0 else None
    status = stop_status or ("ok" if worker_exit_code == 0 else "failed")
    result.update(
        {
            "status": status,
            "peak_rss_bytes": max(
                peak_rss,
                int((worker_result or {}).get("peak_rss_bytes", 0)),
            ),
            "minimum_available_memory_bytes": minimum_available,
            "available_memory_after_bytes": _available_memory_bytes(),
            "swap_used_peak_bytes": swap_peak,
            "worker_exit_code": worker_exit_code,
            "worker_signal": worker_signal,
            "elapsed_seconds": time.perf_counter() - started,
            "last_stage": _last_stage(progress_file),
            "worker_result": worker_result,
            **stop_details,
        }
    )
    _write_json_atomic(result_path, result)
    return result
