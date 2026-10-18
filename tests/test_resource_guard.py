from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts import resource_guard


def _run(command, tmp_path, **overrides):
    options = {
        "max_rss_gib": 1,
        "min_available_gib": 0,
        "reserve_available_gib": 0,
        "timeout_seconds": 10,
        "output_json": tmp_path / "result.json",
        "log_path": tmp_path / "worker.log",
        "max_additional_swap_gib": None,
        "poll_interval_seconds": 0.01,
    }
    options.update(overrides)
    return resource_guard.run_bounded_process(command, **options)


def test_bounded_runner_records_success_and_log(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_guard, "_available_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(resource_guard, "_swap_used_bytes", lambda: 0)

    progress = tmp_path / "build-progress.jsonl"
    progress.write_text('{"stage":"encode_weight_002_of_196"}\n', encoding="utf-8")
    result = _run(
        [sys.executable, "-c", "print('child output')"],
        tmp_path,
        progress_jsonl=progress,
    )

    assert result["status"] == "ok"
    assert result["last_stage"] == "encode_weight_002_of_196"
    assert result["worker_exit_code"] == 0
    assert result["available_memory_before_bytes"] == 16 * 1024**3
    assert "child output" in Path(result["log"]).read_text(encoding="utf-8")
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == result


def test_bounded_runner_kills_rss_excess_and_keeps_parent_alive(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_guard, "_available_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(resource_guard, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(resource_guard, "_process_tree_rss_bytes", lambda _pid: 2 * 1024**3)

    result = _run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path)

    assert result["status"] == "rss_guard"
    assert result["worker_exit_code"] != 0
    assert os.kill(os.getpid(), 0) is None
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "worker.log").is_file()


def test_bounded_runner_refuses_low_memory_preflight(tmp_path, monkeypatch):
    sentinel = tmp_path / "started"
    monkeypatch.setattr(resource_guard, "_available_memory_bytes", lambda: 1024)
    monkeypatch.setattr(resource_guard, "_swap_used_bytes", lambda: 0)

    result = _run(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(sentinel)!r}).touch()"],
        tmp_path,
        min_available_gib=2,
    )

    assert result["status"] == "memory_reserve_guard"
    assert result["worker_exit_code"] is None
    assert not sentinel.exists()
    assert (tmp_path / "worker.log").is_file()


def test_bounded_runner_stops_on_additional_swap(tmp_path, monkeypatch):
    swap_values = iter([0, 3 * 1024**3])
    monkeypatch.setattr(resource_guard, "_available_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(resource_guard, "_swap_used_bytes", lambda: next(swap_values, 3 * 1024**3))

    result = _run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        max_additional_swap_gib=2,
    )

    assert result["status"] == "swap_guard"
    assert result["swap_used_before_bytes"] == 0
    assert result["swap_used_peak_bytes"] == 3 * 1024**3


def test_process_tree_includes_grandchild(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    code = f"""
import pathlib
import subprocess
import sys
import time
import threading

def launch_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
    time.sleep(10)

threading.Thread(target=launch_child, daemon=True).start()
time.sleep(10)
"""
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        grandchild = int(pid_file.read_text(encoding="utf-8"))
        assert grandchild in resource_guard._process_tree(process.pid)
        assert resource_guard._process_tree_rss_bytes(process.pid) > 0
    finally:
        resource_guard._terminate_process_group(process)


def test_run_bounded_cli_executes_command_and_writes_json(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    result_path = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_bounded.py",
            "--max-rss-gib",
            "1",
            "--min-available-gib",
            "0",
            "--reserve-available-gib",
            "0",
            "--timeout-seconds",
            "10",
            "--result",
            str(result_path),
            "--log",
            str(tmp_path / "worker.log"),
            "--",
            sys.executable,
            "-c",
            "print('bounded cli child')",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["status"] == "ok"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "ok"
