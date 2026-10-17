"""Report exact bundle bytes, ONNX initializer storage, and source tensor sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from kitten2_onnx.external_data import logical_tensor_bytes, resolve_external_data_range
from scripts.audit_checkpoint_weights import audit_checkpoint

_OPERATORS_OF_INTEREST = {"MatMul", "Gemm", "Gather", "Conv"}


def _dtype_name(tensor: TensorProto) -> str:
    return TensorProto.DataType.Name(tensor.data_type)


def _logical_tensor_bytes(tensor: TensorProto) -> int:
    return logical_tensor_bytes(tensor)

def _external_data(tensor: TensorProto) -> dict[str, str]:
    return {item.key: item.value for item in tensor.external_data}


def _storage_info(tensor: TensorProto, graph_dir: Path, logical_bytes: int) -> dict[str, Any]:
    external = _external_data(tensor)
    if not external:
        raw_bytes = len(tensor.raw_data)
        return {
            "kind": "inline",
            "bytes": raw_bytes or logical_bytes,
        }

    resolved = resolve_external_data_range(
        graph_dir, tensor, logical_bytes=logical_bytes
    )
    location = resolved.location
    offset = resolved.offset
    length = resolved.length
    path = resolved.path
    return {
        "kind": "external",
        "location": location,
        "offset": offset,
        "bytes": length,
        "path": path,
    }


def _update_external_hash(digest: Any, path: Path, offset: int, length: int) -> None:
    remaining = length
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"External tensor data ended early: {path}")
            digest.update(block)
            remaining -= len(block)


def _tensor_hash(
    tensor: TensorProto,
    graph_dir: Path,
    logical_bytes: int,
    storage: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(_dtype_name(tensor).encode("ascii"))
    digest.update(json.dumps(list(tensor.dims), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    if storage["kind"] == "external":
        _update_external_hash(digest, storage["path"], storage["offset"], storage["bytes"])
    elif tensor.raw_data:
        digest.update(tensor.raw_data)
    elif tensor.data_type == TensorProto.STRING:
        for value in tensor.string_data:
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    else:
        array = np.ascontiguousarray(onnx.numpy_helper.to_array(tensor))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()




def _matmul_nbits4_size_report(
    graph: Any,
    initializer_details: list[dict[str, Any]],
) -> dict[str, Any]:
    initializers = {item["name"]: item for item in initializer_details}
    roles: dict[str, set[str]] = {
        "packed_codes": set(),
        "scales": set(),
        "zero_points": set(),
    }
    nodes = []
    for node_index, node in enumerate(graph.node):
        if node.op_type != "MatMulNBits":
            continue
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if attributes.get("bits") != 4:
            continue
        input_names = {
            "packed_codes": node.input[1] if len(node.input) > 1 else "",
            "scales": node.input[2] if len(node.input) > 2 else "",
            "zero_points": node.input[3] if len(node.input) > 3 else "",
        }
        for role, name in input_names.items():
            if name in initializers:
                roles[role].add(name)
        nodes.append(
            {
                "node": node.name or f"MatMulNBits#{node_index}",
                "K": attributes.get("K"),
                "N": attributes.get("N"),
                "bits": attributes["bits"],
                "block_size": attributes.get("block_size"),
                "initializers": input_names,
            }
        )

    initializer_roles = {
        role: [
            {
                "name": name,
                "dtype": initializers[name]["dtype"],
                "shape": initializers[name]["shape"],
                "logical_bytes": initializers[name]["logical_bytes"],
                "storage_bytes": initializers[name]["storage_bytes"],
            }
            for name in sorted(names)
        ]
        for role, names in roles.items()
    }
    storage_bytes_by_role = {
        role: sum(initializers[name]["storage_bytes"] for name in names)
        for role, names in roles.items()
    }
    all_names = set().union(*roles.values())
    return {
        "node_count": len(nodes),
        "initializer_count": len(all_names),
        "storage_bytes_by_role": storage_bytes_by_role,
        "total_storage_bytes": sum(initializers[name]["storage_bytes"] for name in all_names),
        "initializer_roles": initializer_roles,
        "nodes": nodes,
    }
def _graph_size_report(graph_path: Path) -> dict[str, Any]:
    model = onnx.load(str(graph_path), load_external_data=False)
    graph = model.graph
    initializers = list(graph.initializer)
    initializer_names = {tensor.name for tensor in initializers}
    dtype_bytes: dict[str, int] = defaultdict(int)
    bytes_by_name: dict[str, int] = {}
    initializer_details: list[dict[str, Any]] = []
    hash_groups: dict[str, list[str]] = defaultdict(list)
    external_files: dict[str, int] = {}
    consumer_rows: dict[str, list[dict[str, Any]]] = {
        operator: [] for operator in sorted(_OPERATORS_OF_INTEREST)
    }

    for tensor in initializers:
        logical_bytes = _logical_tensor_bytes(tensor)
        dtype = _dtype_name(tensor)
        storage = _storage_info(tensor, graph_path.parent, logical_bytes)
        digest = _tensor_hash(tensor, graph_path.parent, logical_bytes, storage)
        dtype_bytes[dtype] += logical_bytes
        bytes_by_name[tensor.name] = logical_bytes
        hash_groups[digest].append(tensor.name)
        detail: dict[str, Any] = {
            "name": tensor.name,
            "dtype": dtype,
            "shape": list(tensor.dims),
            "logical_bytes": logical_bytes,
            "storage_bytes": storage["bytes"],
            "sha256": digest,
        }
        if storage["kind"] == "external":
            detail["external_data"] = {
                "location": storage["location"],
                "offset": storage["offset"],
                "bytes": storage["bytes"],
            }
            external_files[storage["location"]] = storage["path"].stat().st_size
        initializer_details.append(detail)

    node_type_counts = Counter(node.op_type for node in graph.node)
    for node_index, node in enumerate(graph.node):
        if node.op_type not in consumer_rows:
            continue
        node_name = node.name or f"{node.op_type}#{node_index}"
        for input_index, name in enumerate(node.input):
            if name in initializer_names:
                consumer_rows[node.op_type].append(
                    {
                        "initializer": name,
                        "node": node_name,
                        "input_index": input_index,
                    }
                )

    duplicate_groups = [
        {
            "sha256": digest,
            "names": sorted(names),
            "logical_bytes": bytes_by_name[names[0]],
        }
        for digest, names in sorted(hash_groups.items())
        if len(names) > 1
    ]
    report: dict[str, Any] = {
        "graph": graph_path.name,
        "graph_bytes": graph_path.stat().st_size,
        "external_data_bytes": sum(external_files.values()),
        "external_data_files": dict(sorted(external_files.items())),
        "initializer_count": len(initializers),
        "initializer_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
        "initializer_bytes_by_name": dict(sorted(bytes_by_name.items())),
        "top_20_initializers": sorted(
            initializer_details,
            key=lambda item: (-item["logical_bytes"], item["name"]),
        )[:20],
        "duplicate_initializer_groups": duplicate_groups,
        "node_type_counts": dict(sorted(node_type_counts.items())),

        "matmul_nbits4": _matmul_nbits4_size_report(graph, initializer_details),
        "initializer_consumers": consumer_rows,
    }

    if graph_path.name == "lm.onnx":
        metadata_path = graph_path.parent / "lm.meta.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
        vocab_size = metadata.get("vocab_size")
        hidden_size = metadata.get("hidden_size")
        candidates: list[dict[str, Any]] = []
        if isinstance(vocab_size, int) and isinstance(hidden_size, int):
            valid_shapes = {
                (vocab_size, hidden_size),
                (hidden_size, vocab_size),
            }
            consumers_by_name: dict[str, list[str]] = defaultdict(list)
            for operator, rows in consumer_rows.items():
                for row in rows:
                    consumers_by_name[row["initializer"]].append(operator)
            candidates = [
                {
                    "name": tensor.name,
                    "shape": list(tensor.dims),
                    "dtype": _dtype_name(tensor),
                    "logical_bytes": bytes_by_name[tensor.name],
                    "storage_bytes": next(
                        item["storage_bytes"]
                        for item in initializer_details
                        if item["name"] == tensor.name
                    ),
                    "consumers": sorted(consumers_by_name[tensor.name]),
                }
                for tensor in initializers
                if tuple(tensor.dims) in valid_shapes
            ]

        embedding_name = metadata.get("embedding_file", "embed_tokens.npy")
        embedding_path = graph_path.parent / embedding_name
        embedding: dict[str, Any] | None = None
        if embedding_path.is_file():
            array = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            embedding = {
                "file": embedding_name,
                "file_bytes": embedding_path.stat().st_size,
                "logical_bytes": int(array.nbytes),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

        vocabulary_initializer_bytes = sum(item["logical_bytes"] for item in candidates)
        vocabulary_initializer_storage_bytes = sum(item["storage_bytes"] for item in candidates)
        report["vocabulary"] = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "initializer_candidates": candidates,
            "initializer_count": len(candidates),
            "initializer_logical_bytes": vocabulary_initializer_bytes,
            "initializer_storage_bytes": vocabulary_initializer_storage_bytes,
            "tied_weight_storage": (
                "not_found" if not candidates else "once" if len(candidates) == 1 else "multiple"
            ),
            "embedding_file": embedding,
            "total_bytes_attributable_to_vocabulary": vocabulary_initializer_storage_bytes
            + (embedding["file_bytes"] if embedding else 0),
        }

    return report


def audit_bundle(
    bundle_dir: str | Path,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Audit all bundle files and ONNX graphs without loading external weights."""
    root = Path(bundle_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    files = {path.relative_to(root).as_posix(): path.stat().st_size for path in paths}
    directories = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    directory_bytes = {
        "." if path == root else path.relative_to(root).as_posix(): sum(
            size
            for name, size in files.items()
            if path == root or name.startswith(path.relative_to(root).as_posix() + "/")
        )
        for path in directories
    }
    bundle_manifest_path = root / "bundle.json"
    bundle_manifest = (
        json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        if bundle_manifest_path.is_file()
        else {}
    )
    graphs: dict[str, Any] = {}
    for graph_path in sorted(root.rglob("*.onnx")):
        relative = graph_path.relative_to(root).as_posix()
        graph_key = {
            "lm/lm.onnx": "lm",
            "speaker/speaker.onnx": "speaker",
            "decoder/flow/flow.onnx": "flow",
            "decoder/mel2wav/mel2wav.onnx": "mel2wav",
        }.get(relative, relative.removesuffix(".onnx"))
        graphs[graph_key] = _graph_size_report(graph_path)

    report: dict[str, Any] = {
        "format": "kitten2-onnx-size-audit-v1",
        "bundle_dir": str(root.resolve()),
        "runtime_profile": bundle_manifest.get("export", {}).get("runtime_profile"),
        "bundle_bytes": sum(files.values()),
        "directories": directory_bytes,
        "files": files,
        "onnx": graphs,
    }
    if checkpoint is not None:
        report["checkpoint"] = audit_checkpoint(checkpoint)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_bundle(args.bundle_dir, args.checkpoint)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}: {report['bundle_bytes']} bundle bytes")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
