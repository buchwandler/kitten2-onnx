"""Optimize selected graphs in a KittenTTS 2 ONNX bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_files(bundle_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(bundle_dir).as_posix(): _sha256(path)
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file() and path.name != "bundle.json"
    }


def _bundle_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"Bundle paths must be relative: {relative}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Bundle path escapes its root: {relative}") from error
    return resolved


def _external_locations(model: Any) -> set[str]:
    locations = set()
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                if Path(entry.value).is_absolute():
                    raise ValueError("ONNX external-data locations must be relative")
                locations.add(entry.value)
    return locations


def _quantize_lm(bundle_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    lm_entry = bundle["lm"]
    lm_meta_path = _bundle_path(bundle_dir, lm_entry["metadata"])
    lm_model_path = _bundle_path(bundle_dir, lm_entry["model"])
    metadata = json.loads(lm_meta_path.read_text(encoding="utf-8"))
    if metadata.get("runtime_provider") != "CPUExecutionProvider":
        raise ValueError("INT8 dynamic LM optimization requires a CPU deployment bundle")
    if metadata.get("io_dtype") != "float32":
        raise ValueError("INT8 dynamic LM optimization currently requires float32 graph I/O")

    source_hash = _sha256(lm_model_path)
    source_graph = onnx.load(str(lm_model_path), load_external_data=False)
    source_locations = _external_locations(source_graph)
    temporary_graph = lm_model_path.with_name(f"{lm_model_path.stem}.quantized.onnx")
    quantize_dynamic(
        str(lm_model_path),
        str(temporary_graph),
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
        extra_options={"MatMulConstBOnly": True},
    )

    quantized_graph = onnx.load(str(temporary_graph), load_external_data=False)
    quantized_locations = _external_locations(quantized_graph)
    for location in quantized_locations:
        external_path = _bundle_path(lm_model_path.parent, location)
        if not external_path.is_file():
            raise FileNotFoundError(f"Quantized ONNX external data is missing: {external_path}")
    staged_graph = lm_model_path.with_name(f"{lm_model_path.name}.tmp")
    onnx.save_model(quantized_graph, str(staged_graph))
    os.replace(staged_graph, lm_model_path)
    temporary_graph.unlink(missing_ok=True)

    for location in source_locations - quantized_locations:
        _bundle_path(lm_model_path.parent, location).unlink(missing_ok=True)

    onnx.checker.check_model(str(lm_model_path))
    import onnxruntime as ort

    ort.InferenceSession(str(lm_model_path), providers=["CPUExecutionProvider"])

    metadata["external_data"] = bool(quantized_locations)
    metadata["external_data_files"] = sorted(quantized_locations)
    metadata["quantization"] = {
        "method": "int8-dynamic",
        "weight_type": "QInt8",
        "op_types": ["MatMul"],
        "per_channel": True,
        "matmul_const_b_only": True,
        "source_graph_sha256": source_hash,
        "graph_sha256": _sha256(lm_model_path),
        "external_data_sha256": {
            location: _sha256(_bundle_path(lm_model_path.parent, location))
            for location in sorted(quantized_locations)
        },
    }
    lm_meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata["quantization"]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def optimize_bundle(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    lm_method: str,
    validate: bool = False,
    model_dir: str | Path | None = None,
    corpus_path: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    from kitten2_onnx.validate_bundle import validate_bundle
    from scripts.audit_bundle_size import audit_bundle

    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if (
        output_root == input_root
        or output_root.is_relative_to(input_root)
        or input_root.is_relative_to(output_root)
    ):
        raise ValueError("Input and output bundle directories must not overlap")
    if output_root.exists():
        raise FileExistsError(output_root)
    report_root = (
        Path(report_dir).resolve()
        if report_dir is not None
        else output_root.with_name(f"{output_root.name}-reports")
    )
    report_root.mkdir(parents=True, exist_ok=True)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if lm_method != "int8-dynamic":
        raise ValueError(f"Unsupported LM optimization method: {lm_method!r}")

    input_bundle_path = input_root / "bundle.json"
    bundle = json.loads(input_bundle_path.read_text(encoding="utf-8"))
    if bundle.get("format") != "kitten2-onnx-bundle-v1":
        raise ValueError("Unsupported bundle format")
    source_size = sum(path.stat().st_size for path in input_root.rglob("*") if path.is_file())
    shutil.copytree(input_root, output_root)
    output_bundle_path = output_root / "bundle.json"
    bundle = json.loads(output_bundle_path.read_text(encoding="utf-8"))

    quantization = _quantize_lm(output_root, bundle)
    export_metadata = bundle.get("export")
    if not isinstance(export_metadata, dict):
        raise TypeError("Bundle export metadata must be an object")
    export_metadata["lm_optimization"] = lm_method
    bundle["files"] = _bundle_files(output_root)
    output_bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    size_report = audit_bundle(output_root)
    size_path = report_root / "size-report.json"
    _write_json(size_path, size_report)
    report: dict[str, Any] = {
        "format": "kitten2-onnx-optimization-report-v1",
        "lm_method": lm_method,
        "source_bundle": str(input_root),
        "output_bundle": str(output_root),
        "input_bundle_bytes": source_size,
        "output_bundle_bytes": int(size_report["bundle_bytes"]),
        "quantization": quantization,
        "size_report": str(size_path),
        "validation": {"status": "not_run"},
        "parity": {"status": "not_run"},
    }

    if validate:
        output_validation = validate_bundle(output_root)
        output_validation["status"] = "passed" if output_validation["valid"] else "failed"
        report["validation"] = {
            "status": output_validation["status"],
            "report": str(report_root / "bundle-validation.json"),
        }
        _write_json(report_root / "bundle-validation.json", output_validation)
        if not output_validation["valid"]:
            _write_json(report_root / "optimization.json", report)
            raise ValueError("Optimized bundle failed validation")

    if model_dir is not None:
        from scripts.validate_e2e_corpus import validate_corpus

        corpus = (
            corpus_path
            or Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "e2e_corpus.json"
        )
        parity_path = report_root / "e2e-parity.json"
        try:
            parity_report = validate_corpus(model_dir, output_root, corpus, parity_path)
        except AssertionError as error:
            report["parity"] = {"status": "failed", "error": str(error), "report": str(parity_path)}
            _write_json(report_root / "optimization.json", report)
            raise
        report["parity"] = {
            "status": "passed",
            "first_divergence": parity_report["first_divergence"],
            "report": str(parity_path),
        }
    else:
        report["parity"] = {"status": "not_run"}

    _write_json(report_root / "optimization.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_bundle", type=Path)
    parser.add_argument("output_bundle", type=Path)
    parser.add_argument("--lm", choices=("int8-dynamic",), required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    report = optimize_bundle(
        args.input_bundle,
        args.output_bundle,
        lm_method=args.lm,
        validate=args.validate,
        model_dir=args.model_dir,
        corpus_path=args.corpus,
        report_dir=args.report_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
