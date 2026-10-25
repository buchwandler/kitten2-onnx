"""Build and validate local standard-MatMul materializations of tritpack graphs."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from kitten2_onnx.external_data import (
    logical_tensor_bytes,
    make_external_tensor,
    resolve_external_data_range,
)
from kitten2_onnx.quantize_int8_streaming import ExternalDataWriter
from kitten2_onnx.ternary_tritpack_storage import (
    TERNARY_TRITPACK_FORMAT_VERSION,
    TernaryTritpackConfig,
    unpack_group,
)


@dataclass(frozen=True)
class MaterializedWeights:
    model_path: Path
    data_path: Path
    cache_dir: Path
    cache_key: str
    cache_hit: bool
    provenance: dict[str, Any]



def default_materialization_cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / (
        "kitten2-onnx/tritpack-materialized"
    )

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"external-data range ended early: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _set_external_reference(
    tensor: onnx.TensorProto, location: str, offset: int, length: int
) -> None:
    entries = {
        entry.key: entry.value
        for entry in tensor.external_data
        if entry.key not in {"location", "offset", "length", "basepath"}
    }
    entries.update({"location": location, "offset": str(offset), "length": str(length)})
    del tensor.external_data[:]
    for key, value in entries.items():
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    tensor.data_location = TensorProto.EXTERNAL


def _cache_identity(
    model_path: Path, storage_metadata: Mapping[str, Any]
) -> tuple[str, str, str, Path]:
    graph_sha256 = _sha256_file(model_path)
    expected_graph_hash = storage_metadata.get("graph_sha256")
    if graph_sha256 != expected_graph_hash:
        raise ValueError("tritpack graph hash differs from storage metadata")
    payload_name = storage_metadata.get("payload_file")
    if not isinstance(payload_name, str) or Path(payload_name).is_absolute():
        raise ValueError("tritpack metadata must name a relative payload file")
    payload_path = (model_path.parent / payload_name).resolve()
    if not payload_path.is_relative_to(model_path.parent.resolve()):
        raise ValueError("tritpack payload path escapes the model directory")
    payload_sha256 = _sha256_file(payload_path)
    if payload_sha256 != storage_metadata.get("payload_sha256"):
        raise ValueError("tritpack payload hash differs from storage metadata")
    identity = {
        "graph_sha256": graph_sha256,
        "payload_sha256": payload_sha256,
        "format_version": storage_metadata.get("format_version"),
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return cache_key, graph_sha256, payload_sha256, payload_path


def _trace_decode_chain(
    weight_value: str,
    expected_codes: str,
    expected_scales: str,
    producers: Mapping[str, onnx.NodeProto],
    initializers: Mapping[str, onnx.TensorProto],
) -> tuple[list[onnx.NodeProto], set[str]]:
    chain: list[onnx.NodeProto] = []

    def producer(value: str, expected_op: str) -> onnx.NodeProto:
        node = producers.get(value)
        if node is None or node.op_type != expected_op:
            raise ValueError(f"tritpack decode chain expected {expected_op} for {value!r}")
        chain.append(node)
        return node

    final_reshape = producer(weight_value, "Reshape")
    if "tritpack_restore_shape" not in final_reshape.name or len(final_reshape.input) != 2:
        raise ValueError(f"MatMul weight is not a tritpack reconstruction: {weight_value}")
    multiply = producer(final_reshape.input[0], "Mul")
    if len(multiply.input) != 2 or multiply.input[1] != expected_scales:
        raise ValueError("tritpack scale input differs from storage metadata")
    float_cast = producer(multiply.input[0], "Cast")
    transpose = producer(float_cast.input[0], "Transpose")
    slice_node = producer(transpose.input[0], "Slice")
    flatten = producer(slice_node.input[0], "Reshape")
    gather = producer(flatten.input[0], "Gather")
    code_cast = producer(gather.input[1], "Cast")
    if len(code_cast.input) != 1 or code_cast.input[0] != expected_codes:
        raise ValueError("tritpack code input differs from storage metadata")

    initializer_inputs = {
        name for node in chain for name in node.input if name in initializers
    }
    return chain, initializer_inputs


def _validate_cache(
    cache_dir: Path,
    cache_key: str,
    graph_sha256: str,
    payload_sha256: str,
    storage_metadata: Mapping[str, Any],
    cache_hit: bool = True,
) -> MaterializedWeights:
    model_path = cache_dir / "lm.onnx"
    data_path = cache_dir / "lm.onnx.data"
    manifest_path = cache_dir / "materialization.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"materialized cache manifest is missing or corrupt: {cache_dir}") from error
    if (
        manifest.get("format") != "kitten2-onnx-materialized-weights-v1"
        or manifest.get("cache_key") != cache_key
        or manifest.get("source_graph_sha256") != graph_sha256
        or manifest.get("source_payload_sha256") != payload_sha256
        or manifest.get("tritpack_format_version") != storage_metadata.get("format_version")
    ):
        raise ValueError(f"materialized cache provenance differs: {cache_dir}")
    if (
        not model_path.is_file()
        or not data_path.is_file()
        or _sha256_file(model_path) != manifest.get("graph_sha256")
        or _sha256_file(data_path) != manifest.get("external_data_sha256")
    ):
        raise ValueError(f"materialized cache files are corrupt: {cache_dir}")

    model = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(str(model_path))
    if any(
        node.op_type == "MatMulNBits" or "tritpack" in node.name
        for node in model.graph.node
    ):
        raise ValueError(f"materialized cache retains tritpack execution nodes: {cache_dir}")
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    nodes = {node.name: node for node in model.graph.node}
    weights = storage_metadata.get("weights", [])
    for weight in weights:
        tensor = initializers.get(weight["source_initializer"])
        if (
            tensor is None
            or tensor.data_type != TensorProto.FLOAT
            or list(tensor.dims) != weight["source_shape"]
            or tensor.data_location != TensorProto.EXTERNAL
        ):
            raise ValueError(f"materialized cache weight is missing: {weight['source_initializer']}")
        for node_name in weight["transformed_nodes"]:
            node = nodes.get(node_name)
            if node is None or node.op_type != "MatMul" or node.input[1] != tensor.name:
                raise ValueError(f"materialized cache MatMul is invalid: {node_name}")

    if (
        manifest.get("materialized_initializer_count") != len(weights)
        or manifest.get("transformed_matmul_count")
        != storage_metadata.get("transformed_matmul_count")
    ):
        raise ValueError(f"materialized cache counts differ: {cache_dir}")
    provenance = {
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "cache_hit": cache_hit,
        "source_graph_sha256": graph_sha256,
        "source_payload_sha256": payload_sha256,
        "materialized_graph_sha256": manifest["graph_sha256"],
        "materialized_external_data_sha256": manifest["external_data_sha256"],
        "materialized_initializer_count": manifest["materialized_initializer_count"],
        "transformed_matmul_count": manifest["transformed_matmul_count"],
        "materialized_external_bytes": manifest["materialized_external_bytes"],
    }
    return MaterializedWeights(
        model_path, data_path, cache_dir, cache_key, cache_hit, provenance
    )


def _build_cache(
    source_model_path: Path,
    payload_path: Path,
    temporary_dir: Path,
    cache_key: str,
    graph_sha256: str,
    payload_sha256: str,
    storage_metadata: Mapping[str, Any],
) -> None:
    model = onnx.load(str(source_model_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    nodes_by_name = {node.name: node for node in model.graph.node if node.name}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for value in node.input:
            if value:
                consumers.setdefault(value, []).append(node)

    weights = storage_metadata.get("weights")
    if not isinstance(weights, list) or not weights:
        raise ValueError("materialization requires tritpack weight metadata")
    expected_matmul_count = int(storage_metadata.get("transformed_matmul_count", -1))
    if expected_matmul_count <= 0:
        raise ValueError("materialization requires a positive tritpack MatMul count")

    target_nodes: dict[str, str] = {}
    decode_node_names: set[str] = set()
    decode_initializer_names: set[str] = set()
    specs_by_value: dict[str, Mapping[str, Any]] = {}
    for weight in weights:
        weight_value = weight["weight_value"]
        if weight_value in specs_by_value:
            raise ValueError(f"duplicate tritpack weight value: {weight_value}")
        source_name = weight["source_initializer"]
        shape = [int(value) for value in weight["source_shape"]]
        code_name = weight["code_initializer"]
        scale_name = weight["scale_initializer"]
        if len(shape) != 2 or shape[0] <= 0 or shape[0] % 128 or shape[1] <= 0:
            raise ValueError(f"invalid tritpack source shape: {source_name}")
        if source_name in initializers or code_name not in initializers or scale_name not in initializers:
            raise ValueError(f"tritpack initializer metadata is invalid: {source_name}")
        chain, chain_initializers = _trace_decode_chain(
            weight_value, code_name, scale_name, producers, initializers
        )
        chain_names = {node.name for node in chain}
        if None in chain_names or len(chain_names) != len(chain):
            raise ValueError(f"tritpack decode node names are missing or duplicated: {source_name}")
        decode_node_names.update(chain_names)
        decode_initializer_names.update(chain_initializers)
        specs_by_value[weight_value] = weight
        for node_name in weight["transformed_nodes"]:
            node = nodes_by_name.get(node_name)
            if (
                node is None
                or node.op_type != "MatMul"
                or len(node.input) != 2
                or node.input[1] != weight_value
                or node_name in target_nodes
            ):
                raise ValueError(f"tritpack MatMul metadata is invalid: {node_name}")
            target_nodes[node_name] = source_name

    if len(target_nodes) != expected_matmul_count:
        raise ValueError("tritpack MatMul metadata does not account for all 196 nodes")
    for weight_value, weight in specs_by_value.items():
        expected_consumers = set(weight["transformed_nodes"])
        actual_consumers = {node.name for node in consumers.get(weight_value, [])}
        if actual_consumers != expected_consumers:
            raise ValueError(f"tritpack weight has unexpected consumers: {weight_value}")
    for node_name in decode_node_names:
        node = nodes_by_name[node_name]
        for output in node.output:
            outside = {
                consumer.name
                for consumer in consumers.get(output, [])
                if consumer.name not in decode_node_names and consumer.name not in target_nodes
            }
            if outside:
                raise ValueError(f"tritpack decode output has unexpected consumers: {sorted(outside)}")

    for initializer_name in decode_initializer_names:
        outside = {
            node.name
            for node in consumers.get(initializer_name, [])
            if node.name not in decode_node_names
        }
        if outside:
            raise ValueError(f"tritpack initializer has unexpected consumers: {initializer_name}")

    config = TernaryTritpackConfig()
    data_path = temporary_dir / "lm.onnx.data"
    model_path = temporary_dir / "lm.onnx"
    initializer_reports: list[dict[str, Any]] = []
    with ExternalDataWriter(data_path) as writer:
        new_initializers: list[onnx.TensorProto] = []
        for tensor in model.graph.initializer:
            if tensor.name in decode_initializer_names:
                continue
            if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
                tensor_slice = resolve_external_data_range(
                    source_model_path.parent,
                    tensor,
                    logical_bytes=logical_tensor_bytes(tensor),
                    require_exact_length=True,
                )
                offset, length = writer.copy_range(tensor_slice)
                copied = copy.deepcopy(tensor)
                _set_external_reference(copied, data_path.name, offset, length)
                new_initializers.append(copied)
            else:
                new_initializers.append(copy.deepcopy(tensor))

        for weight in weights:
            source_name = weight["source_initializer"]
            shape = tuple(int(value) for value in weight["source_shape"])
            k, n = shape
            group_count = k // config.group_size
            codes_tensor = initializers[weight["code_initializer"]]
            scales_tensor = initializers[weight["scale_initializer"]]
            expected_code_shape = (group_count, n, config.packed_bytes_per_group)
            expected_scale_shape = (group_count, 1, n)
            if (
                codes_tensor.data_type != TensorProto.UINT8
                or tuple(codes_tensor.dims) != expected_code_shape
                or scales_tensor.data_type != TensorProto.FLOAT
                or tuple(scales_tensor.dims) != expected_scale_shape
            ):
                raise ValueError(f"tritpack payload shape or type is invalid: {source_name}")
            code_slice = resolve_external_data_range(
                source_model_path.parent,
                codes_tensor,
                logical_bytes=logical_tensor_bytes(codes_tensor),
                require_exact_length=True,
            )
            scale_slice = resolve_external_data_range(
                source_model_path.parent,
                scales_tensor,
                logical_bytes=logical_tensor_bytes(scales_tensor),
                require_exact_length=True,
            )
            if code_slice.path.resolve() != payload_path or scale_slice.path.resolve() != payload_path:
                raise ValueError(f"tritpack tensors do not use the declared payload: {source_name}")
            codes = np.memmap(
                code_slice.path,
                mode="r",
                dtype=np.uint8,
                offset=code_slice.offset,
                shape=expected_code_shape,
                order="C",
            )
            scales = np.memmap(
                scale_slice.path,
                mode="r",
                dtype="<f4",
                offset=scale_slice.offset,
                shape=expected_scale_shape,
                order="C",
            )
            offset = writer.start_tensor()
            digest = hashlib.sha256()
            try:
                for group_index in range(group_count):
                    weight_block = np.ascontiguousarray(
                        unpack_group(
                            np.asarray(codes[group_index]),
                            np.asarray(scales[group_index, 0]),
                            config=config,
                        )
                    )
                    payload = memoryview(weight_block).cast("B")
                    writer.write(payload)
                    digest.update(payload)
                offset, length = writer.finish_tensor(offset)
            finally:
                codes._mmap.close()
                scales._mmap.close()
            expected_length = k * n * np.dtype(np.float32).itemsize
            if length != expected_length:
                raise ValueError(f"materialized weight length differs: {source_name}")
            new_initializers.append(
                make_external_tensor(
                    source_name,
                    TensorProto.FLOAT,
                    shape,
                    data_path.name,
                    offset,
                    length,
                )
            )
            initializer_reports.append(
                {
                    "initializer": source_name,
                    "shape": list(shape),
                    "sha256": digest.hexdigest(),
                    "bytes": length,
                }
            )

    used_names = {
        name
        for node in model.graph.node
        if node.name not in decode_node_names
        for name in node.input
        if name
    }
    used_names.update(value.name for value in (*model.graph.input, *model.graph.output))
    new_initializers = [
        tensor
        for tensor in new_initializers
        if tensor.name not in decode_initializer_names or tensor.name in used_names
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_initializers)
    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in decode_node_names:
            continue
        copied = copy.deepcopy(node)
        if copied.name in target_nodes:
            copied.input[1] = target_nodes[copied.name]
        rewritten_nodes.append(copied)
    model.graph.ClearField("node")
    model.graph.node.extend(rewritten_nodes)
    onnx.save_model(model, str(model_path), save_as_external_data=False)
    onnx.checker.check_model(str(model_path))

    materialized_bytes = sum(item["bytes"] for item in initializer_reports)
    if len(initializer_reports) != len(weights):
        raise ValueError("materialized initializer count differs from tritpack metadata")
    manifest = {
        "format": "kitten2-onnx-materialized-weights-v1",
        "cache_key": cache_key,
        "source_graph_sha256": graph_sha256,
        "source_payload_sha256": payload_sha256,
        "tritpack_format_version": storage_metadata["format_version"],
        "graph_sha256": _sha256_file(model_path),
        "external_data_sha256": _sha256_file(data_path),
        "materialized_initializer_count": len(initializer_reports),
        "transformed_matmul_count": expected_matmul_count,
        "materialized_external_bytes": materialized_bytes,
        "weights": initializer_reports,
    }
    (temporary_dir / "materialization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def materialize_tritpack_model(
    source_model: str | Path,
    cache_root: str | Path,
    *,
    storage_metadata: Mapping[str, Any],
) -> MaterializedWeights:
    """Stream tritpack weights into an external-data cache for standard MatMul."""
    if storage_metadata.get("method") != "ternary-tritpack-exact":
        raise ValueError("materialization requires ternary-tritpack-exact storage metadata")
    if storage_metadata.get("format_version") != TERNARY_TRITPACK_FORMAT_VERSION:
        raise ValueError("unsupported tritpack format version")
    if storage_metadata.get("runtime_materialization_required") is not True:
        raise ValueError("storage metadata does not select runtime materialization")
    config = TernaryTritpackConfig()
    if (
        storage_metadata.get("source_group_size") != config.group_size
        or storage_metadata.get("trits_per_byte") != config.trits_per_byte
        or storage_metadata.get("packed_bytes_per_group") != config.packed_bytes_per_group
    ):
        raise ValueError("tritpack storage configuration is inconsistent")
    weights = storage_metadata.get("weights")
    if storage_metadata.get("transformed_initializer_count") != len(weights or []):
        raise ValueError("tritpack initializer metadata count is inconsistent")
    if storage_metadata.get("materialized_external_bytes") != storage_metadata.get(
        "logical_fp32_operand_bytes"
    ):
        raise ValueError("materialized FP32 byte count differs from logical tritpack weights")

    source_model_path = Path(source_model).resolve()
    cache_root_path = Path(cache_root).resolve()
    bundle_root = (
        source_model_path.parent.parent
        if source_model_path.parent.name == "lm"
        else source_model_path.parent
    )
    if cache_root_path.is_relative_to(bundle_root):
        raise ValueError("materialization cache must remain outside the model bundle")
    cache_key, graph_sha256, payload_sha256, payload_path = _cache_identity(
        source_model_path, storage_metadata
    )
    cache_root_path.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root_path / cache_key
    lock_path = cache_root_path / f".{cache_key}.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if cache_dir.exists():
            return _validate_cache(
                cache_dir,
                cache_key,
                graph_sha256,
                payload_sha256,
                storage_metadata,
            )
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{cache_key}.tmp-", dir=cache_root_path)
        )
        try:
            _build_cache(
                source_model_path,
                payload_path,
                temporary_dir,
                cache_key,
                graph_sha256,
                payload_sha256,
                storage_metadata,
            )
            os.replace(temporary_dir, cache_dir)
        except BaseException:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        return _validate_cache(
            cache_dir,
            cache_key,
            graph_sha256,
            payload_sha256,
            storage_metadata,
            cache_hit=False,
        )
