from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.resource_guard import run_bounded_process


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one command under process-tree resource limits.")
    parser.add_argument("--max-rss-gib", type=float, required=True)
    parser.add_argument("--min-available-gib", type=float, required=True)
    parser.add_argument("--reserve-available-gib", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--max-additional-swap-gib", type=float, default=2.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    result = run_bounded_process(
        command,
        max_rss_gib=args.max_rss_gib,
        min_available_gib=args.min_available_gib,
        reserve_available_gib=args.reserve_available_gib,
        timeout_seconds=args.timeout_seconds,
        output_json=args.result,
        log_path=args.log,
        cwd=args.cwd,
        max_additional_swap_gib=args.max_additional_swap_gib,
        poll_interval_seconds=args.poll_interval_seconds,
        progress_jsonl=args.progress_jsonl,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "ok":
        return
    if result["status"] == "failed" and result["worker_exit_code"]:
        sys.exit(result["worker_exit_code"])
    sys.exit(1)


if __name__ == "__main__":
    main()
