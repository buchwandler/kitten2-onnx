"""Optimize selected graphs in a KittenTTS 2 ONNX bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
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



def _all_external_locations(model: Any) -> set[str]:
    import onnx

    locations: set[str] = set()

    def add_tensor(tensor: Any) -> None:
        external = {entry.key: entry.value for entry in tensor.external_data}
        location = external.get("location")
        if tensor.data_location == onnx.TensorProto.EXTERNAL and not location:
            raise ValueError("External ONNX tensor has no location")
        if location:
            if Path(location).is_absolute():
                raise ValueError("ONNX external-data locations must be relative")
            locations.add(location)

    def visit_node(node: Any) -> None:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                add_tensor(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                for tensor in attribute.tensors:
                    add_tensor(tensor)
            elif attribute.type == onnx.AttributeProto.GRAPH:
                visit_graph(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    visit_graph(graph)
            elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                add_tensor(attribute.sparse_tensor.values)
                add_tensor(attribute.sparse_tensor.indices)
            elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
                for sparse_tensor in attribute.sparse_tensors:
                    add_tensor(sparse_tensor.values)
                    add_tensor(sparse_tensor.indices)

    def visit_graph(graph: Any) -> None:
        for tensor in graph.initializer:
            add_tensor(tensor)
        for node in graph.node:
            visit_node(node)

    visit_graph(model.graph)
    for function in model.functions:
        for node in function.node:
            visit_node(node)
    for training_info in model.training_info:
        visit_graph(training_info.initialization)
        visit_graph(training_info.algorithm)
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




def _bundle_external_paths(bundle_dir: Path) -> set[Path]:
    import onnx

    root = bundle_dir.resolve()
    paths: set[Path] = set()
    for graph_path in sorted(root.rglob("*.onnx")):
        model = onnx.load(str(graph_path), load_external_data=False)
        for location in _all_external_locations(model):
            external_path = (graph_path.parent / location).resolve()
            if not external_path.is_relative_to(root):
                raise ValueError(f"ONNX external-data path escapes bundle: {graph_path}: {location}")
            paths.add(external_path)
    return paths


def _chunked_external_location(
    lm_model_path: Path,
    referenced_paths: set[Path],
) -> tuple[str, Path, list[Path]]:
    stale_candidates: list[Path] = []
    for index in range(1000):
        location = "lm.int8.onnx.data" if index == 0 else f"lm.int8-{index}.onnx.data"
        path = _bundle_path(lm_model_path.parent, location)
        if path in referenced_paths:
            continue
        if path.exists():
            stale_candidates.append(path)
            continue
        return location, path, stale_candidates
    raise FileExistsError("Could not allocate an unreferenced chunked external-data path")




def _ternary_external_location(
    lm_model_path: Path,
    referenced_paths: set[Path],
) -> tuple[str, Path, list[Path]]:
    stale_candidates: list[Path] = []
    for index in range(1000):
        location = "lm.nbits4.onnx.data" if index == 0 else f"lm.nbits4-{index}.onnx.data"
        path = _bundle_path(lm_model_path.parent, location)
        if path in referenced_paths:
            continue
        if path.exists():
            stale_candidates.append(path)
            continue
        return location, path, stale_candidates
    raise FileExistsError("Could not allocate an unreferenced NBits4 external-data path")



def _exact_ternary_external_location(
    lm_model_path: Path,
    referenced_paths: set[Path],
) -> tuple[str, Path, list[Path]]:
    stale_candidates: list[Path] = []
    for index in range(1000):
        location = (
            "lm.ternary-exact-int8.onnx.data"
            if index == 0
            else f"lm.ternary-exact-int8-{index}.onnx.data"
        )
        path = _bundle_path(lm_model_path.parent, location)
        if path in referenced_paths:
            continue
        if path.exists():
            stale_candidates.append(path)
            continue
        return location, path, stale_candidates
    raise FileExistsError("Could not allocate exact ternary INT8 external-data path")


def _tritpack_external_location(
    lm_model_path: Path,
    referenced_paths: set[Path],
) -> tuple[str, Path, list[Path]]:
    stale_candidates: list[Path] = []
    for index in range(1000):
        location = (
            "lm.tritpack-exact.onnx.data"
            if index == 0
            else f"lm.tritpack-exact-{index}.onnx.data"
        )
        path = _bundle_path(lm_model_path.parent, location)
        if path in referenced_paths:
            continue
        if path.exists():
            stale_candidates.append(path)
            continue
        return location, path, stale_candidates
    raise FileExistsError("Could not allocate tritpack external-data path")
def _quantize_lm_chunked(
    bundle_dir: Path,
    bundle: dict[str, Any],
    *,
    exclude_nodes: Sequence[str] = (),
    exclude_initializers: Sequence[str] = (),
    chunk_rows: int = 64,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    from kitten2_onnx.quantize_int8_streaming import quantize_dynamic_matmul_streaming

    lm_entry = bundle["lm"]
    lm_meta_path = _bundle_path(bundle_dir, lm_entry["metadata"])
    lm_model_path = _bundle_path(bundle_dir, lm_entry["model"])
    metadata = json.loads(lm_meta_path.read_text(encoding="utf-8"))
    if metadata.get("runtime_provider") != "CPUExecutionProvider":
        raise ValueError("Chunked INT8 LM optimization requires a CPU deployment bundle")
    if metadata.get("io_dtype") != "float32":
        raise ValueError("Chunked INT8 LM optimization currently requires float32 graph I/O")

    source_hash = _sha256(lm_model_path)
    source_graph = onnx.load(str(lm_model_path), load_external_data=False)
    source_locations = _all_external_locations(source_graph)
    source_paths = {
        _bundle_path(lm_model_path.parent, location) for location in source_locations
    }
    source_external_hashes = {
        location: _sha256(_bundle_path(lm_model_path.parent, location))
        for location in sorted(source_locations)
    }
    referenced_before = _bundle_external_paths(bundle_dir)
    external_location, external_path, stale_candidates = _chunked_external_location(
        lm_model_path,
        referenced_before,
    )
    temporary_graph = lm_model_path.with_name(
        f".{lm_model_path.stem}.chunked-{uuid.uuid4().hex}.tmp.onnx"
    )

    try:
        transformation = quantize_dynamic_matmul_streaming(
            lm_model_path,
            temporary_graph,
            output_external_data=external_location,
            chunk_rows=chunk_rows,
            exclude_nodes=exclude_nodes,
            exclude_initializers=exclude_initializers,
        )
        quantized_graph = onnx.load(str(temporary_graph), load_external_data=False)
        quantized_locations = _all_external_locations(quantized_graph)
        for location in quantized_locations:
            external_file = _bundle_path(lm_model_path.parent, location)
            if not external_file.is_file():
                raise FileNotFoundError(f"Chunked ONNX external data is missing: {external_file}")
        onnx.checker.check_model(str(temporary_graph))
        ort.InferenceSession(str(temporary_graph), providers=["CPUExecutionProvider"])
        os.replace(temporary_graph, lm_model_path)
    except BaseException:
        temporary_graph.unlink(missing_ok=True)
        external_path.unlink(missing_ok=True)
        raise

    referenced_after = _bundle_external_paths(bundle_dir)
    stale_paths = (source_paths | set(stale_candidates)) - referenced_after
    for stale_path in sorted(stale_paths):
        if stale_path.is_file():
            stale_path.unlink()

    metadata["external_data"] = bool(quantized_locations)
    metadata["external_data_files"] = sorted(quantized_locations)
    metadata["quantization"] = {
        "method": "int8-dynamic-chunked",
        "external_data_files": sorted(quantized_locations),
        "backend": "streaming",
        "weight_type": "QInt8",
        "op_types": ["MatMul"],
        "per_channel": True,
        "matmul_const_b_only": True,
        "chunk_rows": chunk_rows,
        "exclude_nodes": sorted(exclude_nodes),
        "exclude_initializers": sorted(exclude_initializers),
        "source_graph_sha256": source_hash,
        "source_external_data_sha256": source_external_hashes,
        "graph_sha256": _sha256(lm_model_path),
        "external_data_sha256": {
            location: _sha256(_bundle_path(lm_model_path.parent, location))
            for location in sorted(quantized_locations)
        },
        **transformation,
    }
    lm_meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return metadata["quantization"]


def _quantize_lm_ternary_nbits4(
    bundle_dir: Path,
    bundle: dict[str, Any],
    *,
    bits: int,
    block_size: int,
    accuracy_level: int,
    exclude_nodes: Sequence[str] = (),
    exclude_initializers: Sequence[str] = (),
) -> dict[str, Any]:
    import platform
    import subprocess

    import onnx
    import onnxruntime as ort

    from kitten2_onnx.quantize_ternary_nbits import (
        TernaryNBitsConfig,
        quantize_ternary_matmuls_streaming,
    )

    lm_entry = bundle["lm"]
    lm_meta_path = _bundle_path(bundle_dir, lm_entry["metadata"])
    lm_model_path = _bundle_path(bundle_dir, lm_entry["model"])
    metadata = json.loads(lm_meta_path.read_text(encoding="utf-8"))
    if metadata.get("runtime_provider") != "CPUExecutionProvider":
        raise ValueError("Ternary NBits4 LM optimization requires a CPU deployment bundle")
    if metadata.get("io_dtype") != "float32":
        raise ValueError("Ternary NBits4 LM optimization currently requires float32 graph I/O")

    source_hash = _sha256(lm_model_path)
    source_graph = onnx.load(str(lm_model_path), load_external_data=False)
    source_locations = _all_external_locations(source_graph)
    source_paths = {
        _bundle_path(lm_model_path.parent, location) for location in source_locations
    }
    source_external_hashes = {
        location: _sha256(_bundle_path(lm_model_path.parent, location))
        for location in sorted(source_locations)
    }
    referenced_before = _bundle_external_paths(bundle_dir)
    external_location, external_path, stale_candidates = _ternary_external_location(
        lm_model_path,
        referenced_before,
    )
    temporary_graph = lm_model_path.with_name(
        f".lm.nbits4-{uuid.uuid4().hex}.tmp.onnx"
    )

    try:
        transformation = quantize_ternary_matmuls_streaming(
            lm_model_path,
            temporary_graph,
            output_external_data=external_location,
            config=TernaryNBitsConfig(
                bits=bits,
                block_size=block_size,
                accuracy_level=accuracy_level,
            ),
            expected_source_ternary_matmul_count=196,
            exclude_nodes=exclude_nodes,
            exclude_initializers=exclude_initializers,
        )
        quantized_graph = onnx.load(str(temporary_graph), load_external_data=False)
        quantized_locations = _all_external_locations(quantized_graph)
        for location in quantized_locations:
            external_file = _bundle_path(lm_model_path.parent, location)
            if not external_file.is_file():
                raise FileNotFoundError(f"NBits4 ONNX external data is missing: {external_file}")
        onnx.checker.check_model(str(temporary_graph))
        ort.InferenceSession(str(temporary_graph), providers=["CPUExecutionProvider"])
        os.replace(temporary_graph, lm_model_path)
    except BaseException:
        temporary_graph.unlink(missing_ok=True)
        external_path.unlink(missing_ok=True)
        raise

    referenced_after = _bundle_external_paths(bundle_dir)
    stale_paths = (source_paths | set(stale_candidates)) - referenced_after
    for stale_path in sorted(stale_paths):
        if stale_path.is_file():
            stale_path.unlink()

    metadata["external_data"] = bool(quantized_locations)
    metadata["external_data_files"] = sorted(quantized_locations)
    repository_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provenance = {
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "python_version": platform.python_version(),
        "cpu_architecture": platform.machine(),
        "source_model_revision": bundle.get("revisions", {}).get(
            "source_model_revision", metadata.get("source_model_revision")
        ),
        "repository_commit": repository_commit,
    }
    metadata["quantization"] = {
        **transformation,
        "method": "ternary-nbits",
        "backend": "streaming",
        "provenance": provenance,
        "external_data_files": sorted(quantized_locations),
        "source_graph_sha256": source_hash,
        "source_external_data_sha256": source_external_hashes,
        "graph_sha256": _sha256(lm_model_path),
        "external_data_sha256": {
            location: _sha256(_bundle_path(lm_model_path.parent, location))
            for location in sorted(quantized_locations)
        },
    }
    lm_meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata["quantization"]

def _quantize_lm_ternary_exact_int8_to_nbits4(
    bundle_dir: Path,
    bundle: dict[str, Any],
    *,
    block_size: int,
    accuracy_level: int,
    exclude_nodes: Sequence[str] = (),
    exclude_initializers: Sequence[str] = (),
) -> dict[str, Any]:
    import platform
    import subprocess

    import onnx
    import onnxruntime as ort

    from kitten2_onnx.quantize_ternary_nbits import TernaryNBitsConfig
    from kitten2_onnx.ternary_exact_storage import (
        transcode_exact_ternary_matmuls_to_nbits4_streaming,
    )

    lm_entry = bundle["lm"]
    lm_meta_path = _bundle_path(bundle_dir, lm_entry["metadata"])
    lm_model_path = _bundle_path(bundle_dir, lm_entry["model"])
    metadata = json.loads(lm_meta_path.read_text(encoding="utf-8"))
    exact_storage_metadata = metadata.get("quantization")
    if not isinstance(exact_storage_metadata, dict):
        raise TypeError("LM metadata has no exact ternary storage record")
    if metadata.get("runtime_provider") != "CPUExecutionProvider":
        raise ValueError("Exact ternary INT8 to q4 conversion requires a CPU deployment bundle")
    if metadata.get("io_dtype") != "float32":
        raise ValueError("Exact ternary INT8 to q4 conversion requires float32 graph I/O")

    source_graph = onnx.load(str(lm_model_path), load_external_data=False)
    source_locations = _all_external_locations(source_graph)
    source_paths = {_bundle_path(lm_model_path.parent, location) for location in source_locations}
    referenced_before = _bundle_external_paths(bundle_dir)
    _, external_path, stale_candidates = _ternary_external_location(
        lm_model_path,
        referenced_before,
    )
    temporary_graph = lm_model_path.with_name(f".lm.nbits4-{uuid.uuid4().hex}.tmp.onnx")

    try:
        transformation = transcode_exact_ternary_matmuls_to_nbits4_streaming(
            lm_model_path,
            temporary_graph,
            external_path,
            exact_storage_metadata=exact_storage_metadata,
            config=TernaryNBitsConfig(
                bits=4,
                block_size=block_size,
                accuracy_level=accuracy_level,
            ),
            exclude_nodes=set(exclude_nodes),
            exclude_initializers=set(exclude_initializers),
        )
        candidate_graph = onnx.load(str(temporary_graph), load_external_data=False)
        quantized_locations = _all_external_locations(candidate_graph)
        for location in quantized_locations:
            external_file = _bundle_path(lm_model_path.parent, location)
            if not external_file.is_file():
                raise FileNotFoundError(f"q4 ONNX external data is missing: {external_file}")
        onnx.checker.check_model(str(temporary_graph))
        ort.InferenceSession(str(temporary_graph), providers=["CPUExecutionProvider"])
        os.replace(temporary_graph, lm_model_path)
    except BaseException:
        temporary_graph.unlink(missing_ok=True)
        external_path.unlink(missing_ok=True)
        raise

    referenced_after = _bundle_external_paths(bundle_dir)
    stale_paths = (source_paths | set(stale_candidates)) - referenced_after
    for stale_path in sorted(stale_paths):
        if stale_path.is_file():
            stale_path.unlink()

    project_root = Path(__file__).resolve().parents[1]
    repository_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata["external_data"] = bool(quantized_locations)
    metadata["external_data_files"] = sorted(quantized_locations)
    metadata["quantization"] = {
        **transformation,
        "method": "ternary-exact-int8-to-nbits4",
        "backend": "streaming",
        "provenance": {
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "source_model_revision": exact_storage_metadata.get("provenance", {}).get(
                "source_model_revision", metadata.get("source_model_revision")
            ),
            "repository_commit": repository_commit,
        },
    }
    lm_meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata["quantization"]


def _quantize_lm_ternary_exact_int8(
    bundle_dir: Path,
    bundle: dict[str, Any],
    source_bundle_dir: Path | None = None,
    *,
    inventory_path: str | Path | None = None,
    exclude_nodes: Sequence[str] = (),
    exclude_initializers: Sequence[str] = (),
    include_nodes: Sequence[str] = (),
    progress_path: Path | None = None,
) -> dict[str, Any]:
    import platform
    import subprocess

    import onnx

    from kitten2_onnx.ternary_exact_storage import compress_ternary_matmuls_streaming

    project_root = Path(__file__).resolve().parents[1]
    inventory_file = (
        Path(inventory_path).resolve()
        if inventory_path is not None
        else project_root / "build/reports/dev-fp32-current-ternary-nbits-inventory.json"
    )
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    inventory_rows = {
        row["node"]: row
        for row in inventory.get("nodes", [])
        if row.get("eligible_exact_ternary")
    }
    if len(inventory_rows) != int(inventory.get("eligible_exact_ternary_matmul_count", -1)):
        raise ValueError("ternary inventory eligible count does not match its node rows")
    included = set(include_nodes)
    if included and (exclude_nodes or exclude_initializers):
        raise ValueError("MatMul inclusion cannot be combined with exclusions")
    unknown_includes = included - inventory_rows.keys()
    if unknown_includes:
        raise ValueError(f"unknown eligible MatMul inclusions: {sorted(unknown_includes)}")
    excluded = set(inventory_rows) - included if included else set(exclude_nodes)
    initializer_exclusions = set(exclude_initializers)
    if initializer_exclusions:
        known_initializers = {
            row["initializer"] for row in inventory_rows.values() if row.get("initializer")
        }
        unknown_initializers = initializer_exclusions - known_initializers
        if unknown_initializers:
            raise ValueError(f"unknown eligible initializer exclusions: {sorted(unknown_initializers)}")
        excluded.update(
            name
            for name, row in inventory_rows.items()
            if row.get("initializer") in initializer_exclusions
        )
    unknown_nodes = excluded - inventory_rows.keys()
    if unknown_nodes:
        raise ValueError(f"unknown eligible MatMul exclusions: {sorted(unknown_nodes)}")

    lm_entry = bundle["lm"]
    lm_meta_path = _bundle_path(bundle_dir, lm_entry["metadata"])
    lm_model_path = _bundle_path(bundle_dir, lm_entry["model"])
    metadata = json.loads(lm_meta_path.read_text(encoding="utf-8"))
    if metadata.get("runtime_provider") != "CPUExecutionProvider":
        raise ValueError("Exact ternary INT8 LM optimization requires a CPU deployment bundle")
    if metadata.get("io_dtype") != "float32":
        raise ValueError("Exact ternary INT8 LM optimization requires float32 graph I/O")

    source_root = source_bundle_dir or bundle_dir
    source_lm_model_path = _bundle_path(source_root, lm_entry["model"])
    source_hash = _sha256(source_lm_model_path)
    source_graph = onnx.load(str(source_lm_model_path), load_external_data=False)
    source_locations = _all_external_locations(source_graph)
    source_paths = {
        _bundle_path(source_lm_model_path.parent, location) for location in source_locations
    }
    source_external_hashes = {
        location: _sha256(_bundle_path(source_lm_model_path.parent, location))
        for location in sorted(source_locations)
    }
    referenced_before = _bundle_external_paths(bundle_dir)
    _, external_path, stale_candidates = _exact_ternary_external_location(
        lm_model_path,
        referenced_before,
    )
    candidate_graph = lm_model_path.with_name(
        f".lm.ternary-exact-int8-{uuid.uuid4().hex}.tmp.onnx"
    )

    def record_stage(stage: str, details: dict[str, Any]) -> None:
        if progress_path is not None:
            _append_build_progress(progress_path, stage, **details)

    try:
        transformation = compress_ternary_matmuls_streaming(
            source_lm_model_path,
            candidate_graph,
            external_path,
            inventory=inventory,
            exclude_nodes=excluded,
            stage_callback=record_stage,
        )
        candidate_model = onnx.load(str(candidate_graph), load_external_data=False)
        external_locations = _all_external_locations(candidate_model)
        for location in external_locations:
            external_file = _bundle_path(lm_model_path.parent, location)
            if not external_file.is_file():
                raise FileNotFoundError(f"Exact ternary INT8 external data is missing: {external_file}")
        onnx.checker.check_model(str(candidate_graph))
        os.replace(candidate_graph, lm_model_path)
    except BaseException:
        candidate_graph.unlink(missing_ok=True)
        external_path.unlink(missing_ok=True)
        raise

    referenced_after = _bundle_external_paths(bundle_dir)
    stale_paths = {
        path
        for path in source_paths | set(stale_candidates)
        if path.is_relative_to(bundle_dir.resolve())
    } - referenced_after
    for stale_path in sorted(stale_paths):
        if stale_path.is_file():
            stale_path.unlink()

    repository_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    code_bytes = transformation["code_bytes"]
    scale_bytes = transformation["scale_bytes"]
    source_fp32_bytes = transformation["source_fp32_bytes_replaced"]
    stored_bytes = code_bytes + scale_bytes
    metadata["external_data"] = bool(external_locations)
    metadata["external_data_files"] = sorted(external_locations)
    runtime_options = {
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "execution_mode": "sequential",
        "enable_mem_pattern": False,
        "enable_cpu_mem_arena": False,
        "graph_optimization_level": "ORT_DISABLE_ALL",
    }
    metadata["runtime_memory_profile"] = "low-memory-exact-ternary"
    metadata["runtime_options"] = runtime_options
    metadata["quantization"] = {
        "method": "ternary-exact-int8",
        "runtime_memory_profile": "low-memory-exact-ternary",
        "runtime_options": runtime_options,
        "backend": "streaming",
        "group_size": 128,
        "code_dtype": "int8",
        "scale_dtype": "float32",
        "encoding_version": transformation["encoding_version"],
        "code_values": transformation["code_values"],
        "negative_zero_code": transformation["negative_zero_code"],
        "negative_zero_count": transformation["negative_zero_count"],
        "negative_zero_preserved": transformation["negative_zero_preserved"],
        "transformed_matmul_count": transformation["transformed_matmul_count"],
        "transformed_initializer_count": len(transformation["source_hashes"]),
        "transformed_nodes": transformation["transformed_nodes"],
        "exact_reconstruction": transformation["exact_reconstruction"],
        "source_initializer_hashes": transformation["source_hashes"],
        "code_bytes": code_bytes,
        "scale_bytes": scale_bytes,
        "source_fp32_bytes_replaced": source_fp32_bytes,
        "operand_storage_reduction_bytes": source_fp32_bytes - stored_bytes,
        "exact_storage_ratio": stored_bytes / source_fp32_bytes if source_fp32_bytes else None,
        "inventory_sha256": transformation["inventory_sha256"],
        "source_graph_sha256": source_hash,
        "source_external_data_sha256": source_external_hashes,
        "graph_sha256": _sha256(lm_model_path),
        "compact_external_data_sha256": transformation[
            "compact_external_data_sha256"
        ],
        "external_data_files": sorted(external_locations),
        "external_data_sha256": {
            location: _sha256(_bundle_path(lm_model_path.parent, location))
            for location in sorted(external_locations)
        },
        "excluded_nodes": sorted(excluded),
        "initializer_reports": transformation["initializer_reports"],
        "provenance": {
            "onnx_version": onnx.__version__,
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "repository_commit": repository_commit,
        },
    }
    if progress_path is not None:
        _append_build_progress(progress_path, "write_metadata")
    lm_meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata["quantization"]




def _quantize_lm_ternary_tritpack_exact(
    bundle_dir: Path,
    bundle: dict[str, Any],
    *,
    column_chunk: int,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    import platform
    import subprocess

    import onnx

    from kitten2_onnx.ternary_tritpack_storage import (
        build_ternary_tritpack_inventory,
        compress_ternary_matmuls_to_tritpack_streaming,
    )

    lm_entry = bundle["lm"]
    lm_meta_path = _bundle_path(bundle_dir, lm_entry["metadata"])
    lm_model_path = _bundle_path(bundle_dir, lm_entry["model"])
    metadata = json.loads(lm_meta_path.read_text(encoding="utf-8"))
    if metadata.get("runtime_provider") != "CPUExecutionProvider":
        raise ValueError("Tritpack LM optimization requires a CPU deployment bundle")
    if metadata.get("io_dtype") != "float32":
        raise ValueError("Tritpack LM optimization requires float32 graph I/O")

    source_hash = _sha256(lm_model_path)
    source_model = onnx.load(str(lm_model_path), load_external_data=False)
    source_locations = _all_external_locations(source_model)
    source_paths = {
        _bundle_path(lm_model_path.parent, location) for location in source_locations
    }
    source_external_hashes = {
        location: _sha256(_bundle_path(lm_model_path.parent, location))
        for location in sorted(source_locations)
    }
    referenced_before = _bundle_external_paths(bundle_dir)
    external_location, external_path, stale_candidates = _tritpack_external_location(
        lm_model_path,
        referenced_before,
    )
    candidate_graph = lm_model_path.with_name(
        f".lm.tritpack-{uuid.uuid4().hex}.tmp.onnx"
    )

    def record_stage(stage: str, details: dict[str, Any]) -> None:
        if progress_path is not None:
            _append_build_progress(progress_path, stage, **details)

    try:
        inventory = build_ternary_tritpack_inventory(
            lm_model_path,
            expected_eligible_matmul_count=196,
            column_chunk=column_chunk,
        )
        transformation = compress_ternary_matmuls_to_tritpack_streaming(
            lm_model_path,
            candidate_graph,
            external_path,
            inventory=inventory,
            column_chunk=column_chunk,
            stage_callback=record_stage,
        )
        candidate_model = onnx.load(str(candidate_graph), load_external_data=False)
        external_locations = _all_external_locations(candidate_model)
        if external_location not in external_locations:
            raise ValueError("Tritpack payload is not referenced by the transformed graph")
        onnx.checker.check_model(str(candidate_graph))
        os.replace(candidate_graph, lm_model_path)
    except BaseException:
        candidate_graph.unlink(missing_ok=True)
        external_path.unlink(missing_ok=True)
        raise

    referenced_after = _bundle_external_paths(bundle_dir)
    stale_paths = {
        path
        for path in source_paths | set(stale_candidates)
        if path.is_relative_to(bundle_dir.resolve())
    } - referenced_after
    for stale_path in sorted(stale_paths):
        if stale_path.is_file():
            stale_path.unlink()

    transformed_model = onnx.load(str(lm_model_path), load_external_data=False)
    nodes_by_name = {node.name: node for node in transformed_model.graph.node if node.name}
    weights = []
    for entry in transformation["initializer_reports"]:
        first_node = nodes_by_name[entry["transformed_nodes"][0]]
        weight_value = entry.get("weight_value", first_node.input[1])
        if first_node.input[1] != weight_value:
            raise ValueError(f"Tritpack weight value differs for {entry['source_initializer']}")
        weights.append(
            {
                "source_initializer": entry["source_initializer"],
                "source_shape": entry["source_shape"],
                "source_sha256": entry["source_sha256"],
                "code_initializer": entry["code_initializer"],
                "scale_initializer": entry["scale_initializer"],
                "transformed_nodes": entry["transformed_nodes"],
                "weight_value": weight_value,
            }
        )

    project_root = Path(__file__).resolve().parents[1]
    repository_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_options = {
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "execution_mode": "sequential",
        "enable_mem_pattern": False,
        "enable_cpu_mem_arena": False,
        "graph_optimization_level": "ORT_DISABLE_ALL",
    }
    runtime_memory_profile = "low-memory-exact-ternary"
    payload_sha256 = _sha256(external_path)
    materialized_bytes = transformation["logical_fp32_operand_bytes"]
    metadata["external_data"] = bool(external_locations)
    metadata["external_data_files"] = sorted(external_locations)
    metadata["runtime_memory_profile"] = runtime_memory_profile
    metadata["runtime_options"] = runtime_options
    metadata["quantization"] = {
        **transformation,
        "method": "ternary-tritpack-exact",
        "backend": "streaming",
        "runtime_materialization_required": True,
        "runtime_memory_profile": runtime_memory_profile,
        "runtime_options": runtime_options,
        "materialized_dtype": "float32",
        "materialized_external_bytes": materialized_bytes,
        "cache_key_inputs": ["graph_sha256", "payload_sha256", "format_version"],
        "payload_file": external_location,
        "payload_sha256": payload_sha256,
        "provenance": {
            "onnx_version": onnx.__version__,
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "repository_commit": repository_commit,
            "source_model_revision": bundle.get("revisions", {}).get(
                "source_model_revision", metadata.get("source_model_revision")
            ),
            "source_weight_variant": bundle["lm"].get("weight_variant"),
        },
    }
    metadata["quantization"].update(
        {
            "source_graph_sha256": source_hash,
            "source_external_data_sha256": source_external_hashes,
            "graph_sha256": _sha256(lm_model_path),
            "external_data_sha256": {
                location: _sha256(_bundle_path(lm_model_path.parent, location))
                for location in sorted(external_locations)
            },
            "initializer_reports": transformation["initializer_reports"],
            "weights": weights,
        }
    )
    lm_meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata["quantization"]
def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")



def _append_build_progress(path: Path, stage: str, **details: Any) -> None:
    event = {"timestamp": datetime.now(UTC).isoformat(), "stage": stage, **details}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")

def _copy_bundle_for_exact_ternary(
    input_root: Path, output_root: Path, bundle: dict[str, Any]
) -> None:
    import onnx

    root = input_root.resolve()
    lm_model_path = _bundle_path(root, bundle["lm"]["model"])
    lm_model = onnx.load(str(lm_model_path), load_external_data=False)
    lm_external = {
        (lm_model_path.parent / location).resolve()
        for location in _all_external_locations(lm_model)
    }
    other_external: set[Path] = set()
    for graph_path in root.rglob("*.onnx"):
        if graph_path.resolve() == lm_model_path:
            continue
        graph = onnx.load(str(graph_path), load_external_data=False)
        other_external.update(
            (graph_path.parent / location).resolve()
            for location in _all_external_locations(graph)
        )
    omitted = {lm_model_path} | (lm_external - other_external)
    if any(not path.is_relative_to(root) for path in other_external):
        raise ValueError("Exact-ternary bundle references external data outside the input bundle")
    if any(not path.is_relative_to(root) for path in omitted):
        raise ValueError("Exact-ternary source files must remain inside the input bundle")

    def ignore(directory: str, names: list[str]) -> list[str]:
        parent = Path(directory).resolve()
        return [name for name in names if (parent / name).resolve() in omitted]

    shutil.copytree(root, output_root, ignore=ignore)


def _optimize_bundle_build(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    lm_method: str,
    validate: bool = False,
    model_dir: str | Path | None = None,
    corpus_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    chunk_rows: int = 64,
    nbits: int = 4,
    nbits_block_size: int = 128,
    nbits_accuracy_level: int = 1,
    exclude_matmul_nodes: Sequence[str] = (),
    exclude_matmul_initializers: Sequence[str] = (),
    ternary_inventory_path: str | Path | None = None,
    include_matmul_nodes: Sequence[str] = (),
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
    progress_path = (
        report_root / "build-progress.jsonl"
        if lm_method in {"ternary-exact-int8", "ternary-tritpack-exact"}
        else None
    )
    if progress_path is not None and progress_path.exists():
        raise FileExistsError(progress_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if lm_method not in (
        "int8-dynamic",
        "int8-dynamic-chunked",
        "ternary-nbits",
        "ternary-nbits4",
        "ternary-tritpack-exact",
        "ternary-exact-int8",
        "ternary-exact-int8-to-nbits4",
    ):
        raise ValueError(f"Unsupported LM optimization method: {lm_method!r}")
    if (
        lm_method not in (
            "int8-dynamic-chunked",
            "ternary-nbits",
            "ternary-nbits4",
            "ternary-tritpack-exact",
            "ternary-exact-int8",
            "ternary-exact-int8-to-nbits4",
        )
        and (exclude_matmul_nodes or exclude_matmul_initializers)
    ):
        raise ValueError("MatMul exclusions require a streaming optimization method")
    if include_matmul_nodes and lm_method != "ternary-exact-int8":
        raise ValueError("MatMul inclusion requires the ternary-exact-int8 method")
    if include_matmul_nodes and (exclude_matmul_nodes or exclude_matmul_initializers):
        raise ValueError("MatMul inclusion cannot be combined with exclusions")
    if lm_method == "ternary-tritpack-exact" and (
        exclude_matmul_nodes or exclude_matmul_initializers or include_matmul_nodes
    ):
        raise ValueError("Tritpack optimization requires all eligible MatMuls")

    input_bundle_path = input_root / "bundle.json"
    bundle = json.loads(input_bundle_path.read_text(encoding="utf-8"))
    if bundle.get("format") not in {"kitten2-onnx-bundle-v1", "kitten2-onnx-bundle-v2"}:
        raise ValueError("Unsupported bundle format")
    if lm_method == "ternary-tritpack-exact" and bundle["format"] != "kitten2-onnx-bundle-v2":
        raise ValueError("Exact tritpack storage is supported only for bundle v2")
    if bundle["format"] == "kitten2-onnx-bundle-v2":
        if bundle.get("lm", {}).get("weight_variant") not in {"packed", "emb4"}:
            raise ValueError("Bundle v2 requires a packed or emb4 LM weight variant")
        if lm_method not in {"ternary-nbits4", "ternary-tritpack-exact"} or (
            lm_method == "ternary-nbits4" and nbits != 4
        ):
            raise ValueError("Packed and emb4 bundle v2 requires q4 diagnostics or exact tritpack storage")
        if exclude_matmul_nodes or exclude_matmul_initializers:
            raise ValueError("Packed and emb4 bundle v2 requires all transformer MatMuls to be transformed")
    source_size = sum(path.stat().st_size for path in input_root.rglob("*") if path.is_file())
    if lm_method == "ternary-exact-int8":
        _copy_bundle_for_exact_ternary(input_root, output_root, bundle)
    else:
        shutil.copytree(input_root, output_root)
    output_bundle_path = output_root / "bundle.json"
    bundle = json.loads(output_bundle_path.read_text(encoding="utf-8"))

    if lm_method == "int8-dynamic":
        quantization = _quantize_lm(output_root, bundle)
    elif lm_method == "int8-dynamic-chunked":
        quantization = _quantize_lm_chunked(
            output_root,
            bundle,
            exclude_nodes=exclude_matmul_nodes,
            exclude_initializers=exclude_matmul_initializers,
            chunk_rows=chunk_rows,
        )
    elif lm_method == "ternary-exact-int8":
        quantization = _quantize_lm_ternary_exact_int8(
            output_root,
            bundle,
            source_bundle_dir=input_root,
            inventory_path=ternary_inventory_path,
            exclude_nodes=exclude_matmul_nodes,
            exclude_initializers=exclude_matmul_initializers,
            include_nodes=include_matmul_nodes,
            progress_path=progress_path,
        )
    elif lm_method == "ternary-tritpack-exact":
        quantization = _quantize_lm_ternary_tritpack_exact(
            output_root,
            bundle,
            column_chunk=chunk_rows,
            progress_path=progress_path,
        )
    elif lm_method == "ternary-exact-int8-to-nbits4":
        if nbits != 4:
            raise ValueError("ternary-exact-int8-to-nbits4 requires --nbits 4")
        quantization = _quantize_lm_ternary_exact_int8_to_nbits4(
            output_root,
            bundle,
            block_size=nbits_block_size,
            accuracy_level=nbits_accuracy_level,
            exclude_nodes=exclude_matmul_nodes,
            exclude_initializers=exclude_matmul_initializers,
        )
    else:
        quantization = _quantize_lm_ternary_nbits4(
            output_root,
            bundle,
            bits=nbits,
            block_size=nbits_block_size,
            accuracy_level=nbits_accuracy_level,
            exclude_nodes=exclude_matmul_nodes,
            exclude_initializers=exclude_matmul_initializers,
        )
    if bundle["format"] == "kitten2-onnx-bundle-v2":
        lm_meta_path = _bundle_path(output_root, bundle["lm"]["metadata"])
        lm_metadata = json.loads(lm_meta_path.read_text(encoding="utf-8"))
        quantization["source_weight_sha256"] = bundle["lm"]["source_weight_sha256"]
        quantization["provenance"]["source_weight_sha256"] = bundle["lm"][
            "source_weight_sha256"
        ]
        quantization["provenance"]["source_weight_variant"] = bundle["lm"]["weight_variant"]
        lm_metadata["quantization"] = quantization
        lm_meta_path.write_text(
            json.dumps(lm_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if lm_method == "ternary-tritpack-exact":
            bundle["lm"]["storage"]["transformer"] = {
                "method": "ternary-tritpack-exact",
                "format_version": quantization["format_version"],
                "source_group_size": quantization["group_size"],
                "trits_per_byte": quantization["trits_per_byte"],
                "packed_bytes_per_group": quantization["packed_bytes_per_group"],
                "scale_dtype": "float32",
                "execution_op": "MatMul",
                "execution_semantics": "standard-fp32-matmul",
                "lossy": False,
                "numeric_reconstruction_exact": quantization[
                    "numeric_reconstruction_exact"
                ],
                "signed_zero_preserved": quantization["signed_zero_preserved"],
                "source_weight_sha256": bundle["lm"]["source_weight_sha256"],
                "source_graph_sha256": quantization["source_graph_sha256"],
                "source_external_data_sha256": quantization[
                    "source_external_data_sha256"
                ],
                "graph_sha256": quantization["graph_sha256"],
                "payload_file": quantization["payload_file"],
                "payload_sha256": quantization["payload_sha256"],
                "packed_code_bytes": quantization["packed_code_bytes"],
                "scale_bytes": quantization["scale_bytes"],
                "logical_fp32_operand_bytes": quantization[
                    "logical_fp32_operand_bytes"
                ],
                "transformed_matmul_count": quantization[
                    "transformed_matmul_count"
                ],
                "transformed_initializer_count": quantization[
                    "transformed_initializer_count"
                ],
                "runtime_materialization_required": True,
                "runtime_memory_profile": quantization["runtime_memory_profile"],
                "materialized_dtype": "float32",
                "materialized_external_bytes": quantization[
                    "materialized_external_bytes"
                ],
                "cache_key_inputs": quantization["cache_key_inputs"],
                "weights": quantization["weights"],
            }
        else:
            bundle["lm"]["storage"]["transformer"] = {
                "method": "matmul-nbits",
                "bits": 4,
                "block_size": quantization["block_size"],
                "source_group_size": quantization.get("source_group_size", 128),
                "zero_point": 8,
                "source_weight_sha256": bundle["lm"]["source_weight_sha256"],
                "source_graph_sha256": quantization["source_graph_sha256"],
                "graph_sha256": quantization["graph_sha256"],
            }
    export_metadata = bundle.get("export")
    if not isinstance(export_metadata, dict):
        raise TypeError("Bundle export metadata must be an object")
    export_metadata["lm_optimization"] = lm_method
    bundle["files"] = _bundle_files(output_root)
    if progress_path is not None:
        _append_build_progress(progress_path, "write_bundle_metadata")
    output_bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if progress_path is not None:
        _append_build_progress(progress_path, "size_audit")
    size_report = audit_bundle(output_root)
    size_path = report_root / "size-report.json"
    _write_json(size_path, size_report)
    report: dict[str, Any] = {
        "format": "kitten2-onnx-optimization-report-v1",
        "lm_method": lm_method,
        "source_bundle": str(input_root),
        "output_bundle": str(output_root),
        "build_progress": str(progress_path) if progress_path is not None else None,
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
    if progress_path is not None:
        _append_build_progress(progress_path, "complete")
    return report


def optimize_bundle(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    lm_method: str,
    validate: bool = False,
    model_dir: str | Path | None = None,
    corpus_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    chunk_rows: int = 64,
    nbits: int = 4,
    nbits_block_size: int = 128,
    nbits_accuracy_level: int = 1,
    exclude_matmul_nodes: Sequence[str] = (),
    exclude_matmul_initializers: Sequence[str] = (),
    ternary_inventory_path: str | Path | None = None,
    include_matmul_nodes: Sequence[str] = (),
) -> dict[str, Any]:
    input_root = Path(input_dir).resolve()
    final_output_root = Path(output_dir).resolve()
    if (
        final_output_root == input_root
        or final_output_root.is_relative_to(input_root)
        or input_root.is_relative_to(final_output_root)
    ):
        raise ValueError("Input and output bundle directories must not overlap")
    if final_output_root.exists():
        raise FileExistsError(final_output_root)

    report_root = (
        Path(report_dir).resolve()
        if report_dir is not None
        else final_output_root.with_name(f"{final_output_root.name}-reports")
    )
    if report_root == final_output_root or report_root.is_relative_to(final_output_root):
        raise ValueError("Report directory must be outside the output bundle")
    final_output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = final_output_root.with_name(
        f".{final_output_root.name}.{uuid.uuid4().hex}.partial"
    )
    staging_root.mkdir()
    marker_path = staging_root / "BUILDING.json"
    _write_json(
        marker_path,
        {
            "status": "building",
            "pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
            "lm_method": lm_method,
        },
    )

    try:
        staged_output = staging_root / "bundle"
        report = _optimize_bundle_build(
            input_root,
            staged_output,
            lm_method=lm_method,
            validate=validate,
            model_dir=model_dir,
            corpus_path=corpus_path,
            report_dir=report_root,
            chunk_rows=chunk_rows,
            nbits=nbits,
            nbits_block_size=nbits_block_size,
            nbits_accuracy_level=nbits_accuracy_level,
            exclude_matmul_nodes=exclude_matmul_nodes,
            exclude_matmul_initializers=exclude_matmul_initializers,
            ternary_inventory_path=ternary_inventory_path,
            include_matmul_nodes=include_matmul_nodes,
        )
        report["output_bundle"] = str(final_output_root)
        _write_json(report_root / "optimization.json", report)
        marker_path.unlink()
        if final_output_root.exists():
            raise FileExistsError(final_output_root)
        os.replace(staged_output, final_output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    shutil.rmtree(staging_root, ignore_errors=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_bundle", type=Path)
    parser.add_argument("output_bundle", type=Path)
    parser.add_argument(
        "--lm",
        "--lm-method",
        dest="lm",
        choices=(
            "int8-dynamic",
            "int8-dynamic-chunked",
            "ternary-nbits",
            "ternary-nbits4",
            "ternary-tritpack-exact",
            "ternary-exact-int8",
            "ternary-exact-int8-to-nbits4",
        ),
        required=True,
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--ternary-inventory", type=Path)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--nbits-block-size", type=int, default=128)
    parser.add_argument("--nbits-accuracy-level", type=int, default=1)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--exclude-matmul-node", action="append", default=[])
    parser.add_argument("--exclude-matmul-initializer", action="append", default=[])
    parser.add_argument("--include-matmul-node", action="append", default=[])
    args = parser.parse_args()
    report = optimize_bundle(
        args.input_bundle,
        args.output_bundle,
        lm_method=args.lm,
        validate=args.validate,
        model_dir=args.model_dir,
        corpus_path=args.corpus,
        report_dir=args.report_dir,
        chunk_rows=args.chunk_rows,
        nbits=args.nbits,
        nbits_block_size=args.nbits_block_size,
        nbits_accuracy_level=args.nbits_accuracy_level,
        exclude_matmul_nodes=args.exclude_matmul_node,
        exclude_matmul_initializers=args.exclude_matmul_initializer,
        ternary_inventory_path=args.ternary_inventory,
        include_matmul_nodes=args.include_matmul_node,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
