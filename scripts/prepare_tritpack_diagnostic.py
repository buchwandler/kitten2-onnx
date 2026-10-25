"""Update the copied packed bundle manifest for a tritpack diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("external_data", type=Path)
    parser.add_argument("conversion_report", type=Path)
    args = parser.parse_args()

    bundle_path = args.bundle.resolve()
    manifest_path = bundle_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(args.conversion_report.read_text(encoding="utf-8"))
    graph_relative = args.graph.resolve().relative_to(bundle_path).as_posix()
    data_relative = args.external_data.resolve().relative_to(bundle_path).as_posix()
    payload_relative = args.external_data.resolve().relative_to(args.graph.resolve().parent).as_posix()
    graph_model = onnx.load(str(args.graph), load_external_data=False)
    matmuls_by_name = {
        node.name: node for node in graph_model.graph.node if node.op_type == "MatMul"
    }
    weights = []
    for entry in report["initializer_reports"]:
        first_node = matmuls_by_name[entry["transformed_nodes"][0]]
        weight_value = first_node.input[1]
        if entry.get("weight_value", weight_value) != weight_value:
            raise ValueError(f"conversion report weight value differs for {entry['source_initializer']}")
        weight = {
            key: entry[key]
            for key in (
                "source_initializer",
                "source_shape",
                "source_sha256",
                "code_initializer",
                "scale_initializer",
                "transformed_nodes",
            )
        }
        weight["weight_value"] = weight_value
        weights.append(weight)
    manifest["lm"]["storage"]["transformer"] = {
        "method": report["method"],
        "format_version": report["format_version"],
        "source_group_size": report["group_size"],
        "trits_per_byte": report["trits_per_byte"],
        "packed_bytes_per_group": report["packed_bytes_per_group"],
        "scale_dtype": "float32",
        "execution_op": "MatMul",
        "execution_semantics": "standard-fp32-matmul",
        "lossy": False,
        "numeric_reconstruction_exact": report["numeric_reconstruction_exact"],
        "signed_zero_preserved": report["signed_zero_preserved"],
        "source_graph_sha256": report["source_graph_sha256"],
        "source_external_data_sha256": report["source_external_data_sha256"],
        "graph_sha256": report["graph_sha256"],
        "payload_file": payload_relative,
        "payload_sha256": report["external_data_sha256"],
        "packed_code_bytes": report["packed_code_bytes"],
        "scale_bytes": report["scale_bytes"],
        "logical_fp32_operand_bytes": report["logical_fp32_operand_bytes"],
        "transformed_matmul_count": report["transformed_matmul_count"],
        "transformed_initializer_count": report["transformed_initializer_count"],
        "runtime_materialization_required": True,
        "runtime_memory_profile": "low-memory-exact-ternary",
        "materialized_dtype": "float32",
        "materialized_external_bytes": report["logical_fp32_operand_bytes"],
        "cache_key_inputs": ["graph_sha256", "payload_sha256", "format_version"],
        "weights": weights,
    }
    for relative_path in (graph_relative, data_relative):
        path = bundle_path / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest["files"][relative_path] = _sha256_file(path)
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(
        f"Updated {manifest_path}: {report['transformed_matmul_count']} tritpack MatMuls, "
        f"payload SHA-256 {report['external_data_sha256']}"
    )


if __name__ == "__main__":
    main()
