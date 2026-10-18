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




def _matmul_nbits_size_report(
    graph: Any,
    initializer_details: list[dict[str, Any]],
) -> dict[str, Any]:
    initializers = {item["name"]: item for item in initializer_details}
    role_names: dict[str, set[str]] = {
        "packed_codes": set(),
        "scales": set(),
        "zero_points": set(),
    }
    grouped_nodes: dict[tuple[int, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    grouped_roles: dict[tuple[int, int, int | None], dict[str, set[str]]] = defaultdict(
        lambda: {role: set() for role in role_names}
    )
    nodes = []
    for node_index, node in enumerate(graph.node):
        if node.op_type != "MatMulNBits":
            continue
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        bits = attributes.get("bits")
        block_size = attributes.get("block_size")
        if bits is None or block_size is None:
            continue
        bits = int(bits)
        block_size = int(block_size)
        accuracy_level = attributes.get("accuracy_level")
        if accuracy_level is not None:
            accuracy_level = int(accuracy_level)
        configuration = (bits, block_size, accuracy_level)
        input_names = {
            "packed_codes": node.input[1] if len(node.input) > 1 else "",
            "scales": node.input[2] if len(node.input) > 2 else "",
            "zero_points": node.input[3] if len(node.input) > 3 else "",
        }
        operand_bytes = {}
        for role, name in input_names.items():
            if name in initializers:
                role_names[role].add(name)
                grouped_roles[configuration][role].add(name)
                operand_bytes[role] = initializers[name]["storage_bytes"]
            else:
                operand_bytes[role] = 0
        row = {
            "node": node.name or f"MatMulNBits#{node_index}",
            "K": attributes.get("K"),
            "N": attributes.get("N"),
            "bits": bits,
            "block_size": block_size,
            "accuracy_level": accuracy_level,
            "packed_code_bytes": operand_bytes["packed_codes"],
            "scale_bytes": operand_bytes["scales"],
            "zero_point_bytes": operand_bytes["zero_points"],
            "initializers": input_names,
        }
        nodes.append(row)
        grouped_nodes[configuration].append(row)

    def role_details(names_by_role: dict[str, set[str]]) -> dict[str, list[dict[str, Any]]]:
        return {
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
            for role, names in names_by_role.items()
        }

    storage_bytes_by_role = {
        role: sum(initializers[name]["storage_bytes"] for name in names)
        for role, names in role_names.items()
    }
    all_names = set().union(*role_names.values())
    configurations = []
    for bits, block_size, accuracy_level in sorted(
        grouped_nodes,
        key=lambda key: (key[0], key[1], -1 if key[2] is None else key[2]),
    ):
        config_roles = grouped_roles[(bits, block_size, accuracy_level)]
        config_storage = {
            role: sum(initializers[name]["storage_bytes"] for name in names)
            for role, names in config_roles.items()
        }
        config_names = set().union(*config_roles.values())
        configurations.append(
            {
                "bits": bits,
                "block_size": block_size,
                "accuracy_level": accuracy_level,
                "node_count": len(grouped_nodes[(bits, block_size, accuracy_level)]),
                "initializer_count": len(config_names),
                "storage_bytes_by_role": config_storage,
                "total_storage_bytes": sum(config_storage.values()),
                "initializer_roles": role_details(config_roles),
            }
        )

    return {
        "node_count": len(nodes),
        "initializer_count": len(all_names),
        "storage_bytes_by_role": storage_bytes_by_role,
        "total_storage_bytes": sum(initializers[name]["storage_bytes"] for name in all_names),
        "initializer_roles": role_details(role_names),
        "configurations": configurations,
        "nodes": nodes,
    }

def _ternary_exact_int8_size_report(
    graph: Any,
    initializer_details: list[dict[str, Any]],
) -> dict[str, Any]:
    initializers = {item["name"]: item for item in initializer_details}
    scalar_values: dict[str, Any] = {}
    for tensor in graph.initializer:
        if tensor.dims or tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
            continue
        if tensor.data_type in {TensorProto.INT8, TensorProto.FLOAT}:
            value = np.asarray(onnx.numpy_helper.to_array(tensor))
            if value.size == 1:
                scalar_values[tensor.name] = value.item()

    producers = {
        output: node
        for node in graph.node
        for output in node.output
        if output
    }

    def casts_to_float(node: Any) -> bool:
        return any(
            attribute.name == "to"
            and onnx.helper.get_attribute_value(attribute) == TensorProto.FLOAT
            for attribute in node.attribute
        )

    code_casts: dict[str, str] = {}
    float_cast_inputs: dict[str, str] = {}
    for node in graph.node:
        if node.op_type != "Cast" or not node.input or not node.output or not casts_to_float(node):
            continue
        source = initializers.get(node.input[0])
        if (
            source
            and source["dtype"] == "INT8"
            and len(source["shape"]) == 3
            and source["shape"][1] == 128
        ):
            code_casts[node.output[0]] = node.input[0]
        else:
            float_cast_inputs[node.output[0]] = node.input[0]

    def scale_for(inputs: list[str], code_name: str) -> str | None:
        code_shape = initializers[code_name]["shape"]
        for name in inputs:
            scale = initializers.get(name)
            if (
                scale
                and scale["dtype"] == "FLOAT"
                and len(scale["shape"]) == 3
                and scale["shape"][1] == 1
                and code_shape[0] == scale["shape"][0]
                and code_shape[2] == scale["shape"][2]
            ):
                return name
        return None

    def is_marker_equal(output: str, code_name: str) -> bool:
        node = producers.get(output)
        if node is None or node.op_type != "Equal" or len(node.input) != 2:
            return False
        return any(
            name == code_name and scalar_values.get(other) == -2
            for name, other in ((node.input[0], node.input[1]), (node.input[1], node.input[0]))
        )

    reconstructed: dict[str, tuple[str, str, int, str]] = {}
    for node in graph.node:
        if node.op_type != "Mul" or not node.output:
            continue
        code_name = next((code_casts[name] for name in node.input if name in code_casts), None)
        if code_name is None:
            continue
        scale_name = scale_for(list(node.input), code_name)
        if scale_name is not None:
            reconstructed[node.output[0]] = (code_name, scale_name, 1, "direct")

    for node in graph.node:
        if node.op_type != "Where" or len(node.input) != 3 or not node.output:
            continue
        pair = reconstructed.get(node.input[2])
        if pair is None:
            continue
        code_name, scale_name, _, _ = pair
        if (
            is_marker_equal(node.input[0], code_name)
            and node.input[1] in scalar_values
            and float(scalar_values[node.input[1]]) == 0.0
            and np.signbit(np.float32(scalar_values[node.input[1]]))
        ):
            reconstructed[node.output[0]] = (code_name, scale_name, 2, "where")

    for node in graph.node:
        if node.op_type != "Mul" or len(node.input) != 2 or not node.output:
            continue
        for scaled_output, sign_output in (node.input, node.input[::-1]):
            scaled = producers.get(scaled_output)
            sign = producers.get(sign_output)
            if scaled is None or scaled.op_type != "Mul" or sign is None or sign.op_type != "Sub":
                continue
            add_output = next(
                (name for name in scaled.input if name not in initializers), None
            )
            add = producers.get(add_output) if add_output else None
            if add is None or add.op_type != "Add":
                continue
            code_float = next((name for name in add.input if name in code_casts), None)
            if code_float is None:
                continue
            code_name = code_casts[code_float]
            scale_name = scale_for(list(scaled.input), code_name)
            if scale_name is None:
                continue
            offset_output = next(
                (name for name in add.input if name != code_float), None
            )
            offset = producers.get(offset_output) if offset_output else None
            if offset is None or offset.op_type != "Mul" or len(offset.input) != 2:
                continue
            mask_float = next(
                (name for name in offset.input if name in float_cast_inputs), None
            )
            if mask_float is None or scalar_values.get(next(
                (name for name in offset.input if name != mask_float), ""
            )) != 2.0:
                continue
            mask_output = float_cast_inputs[mask_float]
            sign_uses_offset = offset_output in sign.input
            sign_uses_one = any(
                name != offset_output and scalar_values.get(name) == 1.0 for name in sign.input
            )
            if (
                sign_uses_offset
                and sign_uses_one
                and is_marker_equal(mask_output, code_name)
            ):
                reconstructed[node.output[0]] = (code_name, scale_name, 2, "arithmetic")
                break

    reshaped: dict[str, tuple[str, str, int, str]] = {}
    for node in graph.node:
        if node.op_type == "Reshape" and node.input and node.output:
            pair = reconstructed.get(node.input[0])
            if pair is not None:
                reshaped[node.output[0]] = pair

    matched_nodes: list[dict[str, Any]] = []
    code_names: set[str] = set()
    scale_names: set[str] = set()
    for index, node in enumerate(graph.node):
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        pair = reshaped.get(node.input[1])
        if pair is None:
            continue
        code_name, scale_name, encoding_version, reconstruction = pair
        code_names.add(code_name)
        scale_names.add(scale_name)
        matched_nodes.append(
            {
                "node": node.name or f"MatMul#{index}",
                "code_initializer": code_name,
                "scale_initializer": scale_name,
                "encoding_version": encoding_version,
                "negative_zero_preserved": encoding_version == 2,
                "reconstruction": reconstruction,
            }
        )

    code_bytes = sum(initializers[name]["storage_bytes"] for name in code_names)
    scale_bytes = sum(initializers[name]["storage_bytes"] for name in scale_names)
    source_fp32_bytes = sum(
        int(np.prod(initializers[name]["shape"], dtype=np.int64)) * 4 for name in code_names
    )
    stored_bytes = code_bytes + scale_bytes
    encoding_versions = sorted({row["encoding_version"] for row in matched_nodes})
    reconstructions = sorted({row["reconstruction"] for row in matched_nodes})
    return {
        "group_size": 128,
        "code_dtype": "int8",
        "scale_dtype": "float32",
        "encoding_versions": encoding_versions,
        "negative_zero_code": -2 if 2 in encoding_versions else None,
        "negative_zero_preserved": 2 in encoding_versions,
        "negative_zero_reconstruction": (
            reconstructions[0] if len(reconstructions) == 1 and 2 in encoding_versions
            else "mixed" if 2 in encoding_versions
            else None
        ),
        "node_count": len(matched_nodes),
        "initializer_count": len(code_names | scale_names),
        "storage_bytes_by_role": {
            "int8_codes": code_bytes,
            "float32_scales": scale_bytes,
        },
        "total_storage_bytes": stored_bytes,
        "source_fp32_bytes_replaced": source_fp32_bytes,
        "storage_reduction_bytes": source_fp32_bytes - stored_bytes,
        "exact_storage_ratio": stored_bytes / source_fp32_bytes if source_fp32_bytes else None,
        "nodes": matched_nodes,
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
    matmul_nbits = _matmul_nbits_size_report(graph, initializer_details)
    ternary_exact_int8 = _ternary_exact_int8_size_report(graph, initializer_details)
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
        "matmul_nbits": matmul_nbits,
        "matmul_nbits4": matmul_nbits,
        "ternary_exact_int8": ternary_exact_int8,
        "initializer_consumers": consumer_rows,
    }

    if graph_path.name == "lm.onnx":
        metadata_path = graph_path.parent / "lm.meta.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
        quantization = metadata.get("quantization", {})
        report["fallback_storage"] = quantization.get("fallback_storage", {})
        report["quantization_provenance"] = quantization.get("provenance", {})
        if quantization.get("method") == "ternary-exact-int8":
            exact_storage = report["ternary_exact_int8"]
            source_fp32_bytes = int(
                quantization.get(
                    "source_fp32_bytes_replaced",
                    exact_storage["source_fp32_bytes_replaced"],
                )
            )
            exact_storage["source_fp32_bytes_replaced"] = source_fp32_bytes
            exact_storage["storage_reduction_bytes"] = (
                source_fp32_bytes - exact_storage["total_storage_bytes"]
            )
            exact_storage["exact_storage_ratio"] = (
                exact_storage["total_storage_bytes"] / source_fp32_bytes
                if source_fp32_bytes
                else None
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

    bundle_bytes = sum(files.values())
    lm_bytes = directory_bytes.get("lm", 0)
    vocabulary = graphs.get("lm", {}).get("vocabulary", {})
    embedding = vocabulary.get("embedding_file") or {}
    report: dict[str, Any] = {
        "format": "kitten2-onnx-size-audit-v1",
        "bundle_dir": str(root.resolve()),
        "runtime_profile": bundle_manifest.get("export", {}).get("runtime_profile"),
        "bundle_bytes": bundle_bytes,
        "directories": directory_bytes,
        "files": files,
        "onnx": graphs,
        "storage_breakdown": {
            "lm_bytes": lm_bytes,
            "non_lm_bytes": bundle_bytes - lm_bytes,
            "decoder_bytes": directory_bytes.get("decoder", 0),
            "vocabulary_initializer_storage_bytes": vocabulary.get(
                "initializer_storage_bytes", 0
            ),
            "embedding_file_bytes": embedding.get("file_bytes", 0),
            "vocabulary_total_bytes": vocabulary.get(
                "total_bytes_attributable_to_vocabulary", 0
            ),
            "fallback_storage": graphs.get("lm", {}).get("fallback_storage", {}),
            "ternary_exact_int8": graphs.get("lm", {}).get("ternary_exact_int8", {}),
        },
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
