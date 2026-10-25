"""Convert a source LM graph to a streaming tritpack diagnostic candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kitten2_onnx.ternary_tritpack_storage import (
    compress_ternary_matmuls_to_tritpack_streaming,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_graph", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output_graph", type=Path)
    parser.add_argument("output_data", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("progress", type=Path)
    parser.add_argument("--column-chunk", type=int, default=64)
    args = parser.parse_args()

    for path in (args.report, args.progress):
        if path.exists():
            raise FileExistsError(path)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    expected_count = int(inventory.get("eligible_exact_ternary_matmul_count", -1))
    if expected_count != 196:
        raise ValueError(f"packed diagnostic requires 196 eligible MatMuls, found {expected_count}")
    args.progress.parent.mkdir(parents=True, exist_ok=True)

    def record_progress(stage: str, details: dict[str, Any]) -> None:
        with args.progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, **details}, sort_keys=True) + "\n")
            stream.flush()

    report = compress_ternary_matmuls_to_tritpack_streaming(
        args.source_graph,
        args.output_graph,
        args.output_data,
        inventory=inventory,
        column_chunk=args.column_chunk,
        stage_callback=record_progress,
    )
    if report["transformed_matmul_count"] != expected_count:
        raise ValueError(
            "packed diagnostic conversion count differs from its exact ternary inventory"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Converted {report['transformed_matmul_count']} MatMuls; "
        f"packed bytes={report['code_bytes']}, scale bytes={report['scale_bytes']}"
    )


if __name__ == "__main__":
    main()
