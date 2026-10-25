"""Prebuild the local FP32 cache for a tritpack LM bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from kitten2_onnx.materialized_weights import (
    default_materialization_cache_root,
    materialize_tritpack_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle_path = args.bundle.resolve()
    output_path = args.output.resolve()
    cache_root = (args.cache_dir or default_materialization_cache_root()).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if output_path.is_relative_to(bundle_path):
        raise ValueError("materialization report must remain outside the model bundle")
    manifest_path = bundle_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    storage = manifest["lm"]["storage"]["transformer"]
    model_path = bundle_path / manifest["lm"]["model"]
    started = time.perf_counter()
    result = materialize_tritpack_model(
        model_path,
        cache_root,
        storage_metadata=storage,
    )
    report: dict[str, Any] = {
        "format": "kitten2-onnx-tritpack-materialization-report-v1",
        "bundle": str(bundle_path),
        "bundle_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "materialization_seconds": time.perf_counter() - started,
        "cache": result.provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
