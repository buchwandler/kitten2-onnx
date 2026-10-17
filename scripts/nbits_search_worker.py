"""Disposable workers for conditional NBits strict checks and finalist builds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def run_strict_parity(
    model_dir: Path,
    lm_dir: Path,
    speaker_model: Path,
    output: Path,
    log_path: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "KITTEN2_MODEL_DIR": str(model_dir.resolve()),
            "KITTEN2_ONNX_DIR": str(lm_dir.resolve()),
            "KITTEN2_SPEAKER_ONNX": str(speaker_model.resolve()),
        }
    )
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_real_lm_parity.py"]
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    text = completed.stdout + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    result = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "command": command,
        "log": str(log_path),
        "output_tail": text[-12000:],
    }
    _write_json_atomic(output, result)
    return result


def build_finalist(
    input_bundle: Path,
    output_bundle: Path,
    report_dir: Path,
    model_dir: Path,
    output: Path,
    excluded_nodes: list[str],
) -> dict[str, Any]:
    from scripts.optimize_bundle import optimize_bundle

    report = optimize_bundle(
        input_bundle,
        output_bundle,
        lm_method="ternary-nbits",
        validate=True,
        report_dir=report_dir,
        nbits=4,
        nbits_block_size=64,
        nbits_accuracy_level=1,
        exclude_matmul_nodes=excluded_nodes,
    )
    result = {
        "status": "materialized" if report["validation"]["status"] == "passed" else "validation_failed",
        "output_bundle": str(output_bundle.resolve()),
        "optimization": report,
    }
    _write_json_atomic(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    strict = subparsers.add_parser("strict-parity")
    strict.add_argument("--model-dir", type=Path, required=True)
    strict.add_argument("--lm-dir", type=Path, required=True)
    strict.add_argument("--speaker-model", type=Path, required=True)
    strict.add_argument("--output", type=Path, required=True)
    strict.add_argument("--log", type=Path, required=True)
    finalist = subparsers.add_parser("build-finalist")
    finalist.add_argument("--input-bundle", type=Path, required=True)
    finalist.add_argument("--output-bundle", type=Path, required=True)
    finalist.add_argument("--report-dir", type=Path, required=True)
    finalist.add_argument("--model-dir", type=Path, required=True)
    finalist.add_argument("--output", type=Path, required=True)
    finalist.add_argument("--exclude-node", action="append", default=[])
    args = parser.parse_args()
    if args.operation == "strict-parity":
        result = run_strict_parity(
            args.model_dir,
            args.lm_dir,
            args.speaker_model,
            args.output,
            args.log,
        )
    else:
        result = build_finalist(
            args.input_bundle,
            args.output_bundle,
            args.report_dir,
            args.model_dir,
            args.output,
            args.exclude_node,
        )
    print(json.dumps({key: value for key, value in result.items() if key not in {"optimization", "output_tail"}}, indent=2))


if __name__ == "__main__":
    main()
