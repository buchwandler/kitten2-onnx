"""Inventory exact group-local ternary MatMuls in a source ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kitten2_onnx.ternary_tritpack_storage import build_ternary_tritpack_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_graph", type=Path)
    parser.add_argument("--expected-eligible-matmul-count", type=int, default=196)
    parser.add_argument("--column-chunk", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/reports/ternary-tritpack-source-inventory.json"),
    )
    args = parser.parse_args()
    report = build_ternary_tritpack_inventory(
        args.source_graph,
        expected_eligible_matmul_count=args.expected_eligible_matmul_count,
        column_chunk=args.column_chunk,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.output}: {report['eligible_exact_ternary_matmul_count']} eligible MatMuls, "
        f"inventory SHA-256 {report['inventory_sha256']}"
    )


if __name__ == "__main__":
    main()
