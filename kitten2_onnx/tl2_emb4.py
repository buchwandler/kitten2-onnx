"""Read the upstream TL2 body and lossy emb4 vocabulary SafeTensors format."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

GROUP = 128
TL2_ROWS = 32
TL2_GROUP_BYTES = 928
EMB4_SCHEMAS = frozenset({"kitten_ternary_tl2_rl_v1", "kitten_ternary_tl2_emb4_v1"})
_EMB4_PREFIX = "emb4/"


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_embedding_metadata(metadata: dict[str, str]) -> dict[str, Any]:
    raw = metadata.get("embedding_metadata")
    if raw is None:
        raise ValueError("emb4 metadata is missing embedding_metadata")
    if not isinstance(raw, str):
        raise TypeError("emb4 embedding_metadata must be a JSON string")
    try:
        parsed = json.loads(raw)
        vocab_size = int(parsed["V"])
        hidden_size = int(parsed["D"])
        codebook_sizes = json.loads(parsed["Ks"]) if isinstance(parsed["Ks"], str) else parsed["Ks"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("emb4 embedding_metadata is malformed") from error
    if vocab_size <= 0 or hidden_size <= 0 or hidden_size & (hidden_size - 1):
        raise ValueError("emb4 vocabulary dimensions must be positive with power-of-two width")
    if not isinstance(codebook_sizes, list) or not codebook_sizes:
        raise ValueError("emb4 Ks must declare at least one tier codebook")
    if any(not isinstance(size, int) or size < 2 or size > 256 for size in codebook_sizes):
        raise ValueError("emb4 codebook sizes must be integers in [2, 256]")
    return {"V": vocab_size, "D": hidden_size, "Ks": codebook_sizes}


def _metadata_for_handle(
    handle, path: Path, *, embedding_only: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(handle.metadata() or {})
    schema = metadata.get("schema")
    if schema not in EMB4_SCHEMAS:
        raise ValueError(f"unexpected emb4 schema {schema!r} in {path}")
    embedding_metadata = _parse_embedding_metadata(metadata)
    keys = set(handle.keys())
    vocab_size = embedding_metadata["V"]
    hidden_size = embedding_metadata["D"]
    codebook_sizes = embedding_metadata["Ks"]
    required = {_EMB4_PREFIX + "tier_map"}
    for tier in range(len(codebook_sizes)):
        required.update(
            f"{_EMB4_PREFIX}t{tier}_{suffix}" for suffix in ("codes", "centers", "s", "t")
        )
    missing = required - keys
    if missing:
        raise ValueError(f"emb4 payload is missing tensors: {sorted(missing)}")
    unexpected_emb4 = {name for name in keys if name.startswith(_EMB4_PREFIX)} - required
    if unexpected_emb4:
        raise ValueError(f"unknown emb4 tensor names: {sorted(unexpected_emb4)}")
    unexpected_prefixes = {
        name for name in keys if not name.startswith(("tl2/", "fp/", _EMB4_PREFIX))
    }
    if unexpected_prefixes:
        raise ValueError(f"unknown emb4 tensor names: {sorted(unexpected_prefixes)}")
    state_names = [
        name[4:] if name.startswith("tl2/") else name[3:]
        for name in keys
        if name.startswith(("tl2/", "fp/"))
    ]
    if len(state_names) != len(set(state_names)):
        raise ValueError("emb4 payload contains duplicate reconstructed tensor names")
    if "model.embed_tokens.weight" in state_names:
        raise ValueError("emb4 payload must not contain a duplicate dense embedding tensor")
    if not embedding_only and not any(name.startswith("tl2/") for name in keys):
        raise ValueError("emb4 payload contains no TL2 transformer tensors")

    tier_map = _as_numpy(handle.get_tensor(_EMB4_PREFIX + "tier_map"))
    if tier_map.shape != (vocab_size,) or not np.issubdtype(tier_map.dtype, np.integer):
        raise ValueError("emb4 tier_map shape or dtype differs from metadata")
    if tier_map.size and (int(tier_map.min()) < 0 or int(tier_map.max()) >= len(codebook_sizes)):
        raise ValueError("emb4 tier_map references an undeclared tier")
    tier_counts = np.bincount(tier_map.astype(np.int64), minlength=len(codebook_sizes))
    for name in keys:
        if name.startswith("tl2/"):
            tensor_slice = handle.get_slice(name)
            shape = tensor_slice.get_shape()
            if (
                len(shape) != 3
                or shape[0] <= 0
                or shape[1] <= 0
                or shape[2] != TL2_GROUP_BYTES
                or tensor_slice.get_dtype() != "U8"
            ):
                raise ValueError(f"TL2 tensor {name!r} has invalid geometry or dtype")
    for tier, (codebook_size, row_count) in enumerate(
        zip(codebook_sizes, tier_counts, strict=True)
    ):
        code_name = f"{_EMB4_PREFIX}t{tier}_codes"
        code_slice = handle.get_slice(code_name)
        bits = math.ceil(math.log2(codebook_size))
        per_byte = 8 // bits
        if hidden_size % per_byte:
            raise ValueError("emb4 hidden size is incompatible with code packing")
        expected_bytes = math.ceil(int(row_count) * hidden_size / per_byte)
        if code_slice.get_shape() != [expected_bytes] or code_slice.get_dtype() != "U8":
            raise ValueError(f"emb4 tier {tier} code payload has invalid shape or dtype")
        expected_shapes = {
            "centers": [codebook_size, hidden_size],
            "s": [int(row_count)],
            "t": [hidden_size],
        }
        for suffix, shape in expected_shapes.items():
            tensor_slice = handle.get_slice(f"{_EMB4_PREFIX}t{tier}_{suffix}")
            if tensor_slice.get_shape() != shape:
                raise ValueError(f"emb4 tier {tier} {suffix} shape differs from metadata")
            if tensor_slice.get_dtype() not in {"BF16", "F16", "F32"}:
                raise ValueError(f"emb4 tier {tier} {suffix} has unsupported dtype")
    return metadata, {
        "schema": schema,
        "embedding_metadata": embedding_metadata,
        "tensor_count": len(keys),
        "tensor_names": sorted(keys),
    }


def inspect_emb4(path: str | Path) -> dict[str, Any]:
    """Validate an emb4 checkpoint header, tensor inventory, and declared dimensions."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        _, summary = _metadata_for_handle(handle, path)
    return summary


EMB4_RECONSTRUCTION_VERSION = "kitten2-tl2-emb4-v1"


def inspect_emb4_embedding_asset(path: str | Path) -> dict[str, Any]:
    """Validate a compact SafeTensors asset containing only the emb4 vocabulary."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        _, summary = _metadata_for_handle(handle, path, embedding_only=True)
    return summary


def write_emb4_embedding_asset(source: str | Path, destination: str | Path) -> Path:
    """Copy only the emb4 vocabulary tensors into a compact runtime asset."""
    from safetensors.torch import save_file

    source = Path(source)
    destination = Path(destination)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata, _ = _metadata_for_handle(handle, source)
        tensor_names = handle.keys()
        tensors = {
            name: handle.get_tensor(name)
            for name in tensor_names
            if name.startswith(_EMB4_PREFIX)
        }
    save_file(
        tensors,
        str(destination),
        metadata={"schema": metadata["schema"], "embedding_metadata": metadata["embedding_metadata"]},
    )
    inspect_emb4_embedding_asset(destination)
    return destination


def read_emb4_embedding_asset(path: str | Path, *, chunk: int = 1 << 12) -> torch.Tensor:
    """Reconstruct the vocabulary from a compact emb4 asset in bounded chunks."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        _, summary = _metadata_for_handle(handle, path, embedding_only=True)
        return dequant_embedding(
            handle.get_tensor,
            summary["embedding_metadata"],
            chunk=chunk,
            get_slice=handle.get_slice,
        )


def unpack_tl2_tensor(buffer: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Decode one upstream TL2 tensor to its exact bf16 ternary values."""
    buf = _as_numpy(buffer)
    if buf.dtype != np.uint8 or buf.ndim != 3 or buf.shape[2] != TL2_GROUP_BYTES:
        raise ValueError("TL2 buffer must be uint8 with shape [row_blocks, input_groups, 928]")
    row_blocks, groups, _ = buf.shape
    if row_blocks <= 0 or groups <= 0:
        raise ValueError("TL2 buffer dimensions must be nonzero")

    gamma = buf[:, :, : TL2_ROWS * 2].copy().view(np.float16).transpose(0, 2, 1)
    trits = np.empty((row_blocks, TL2_ROWS, groups, GROUP), dtype=np.int8)
    cursor = TL2_ROWS * 2
    for sub in range(4):
        segment = buf[:, :, cursor : cursor + 200].reshape(row_blocks, groups, 10, 20)
        cursor += 200
        packed = np.ascontiguousarray(segment[..., :16].transpose(0, 3, 1, 2))
        magnitude = np.empty((row_blocks, TL2_ROWS, groups, 10), dtype=np.uint8)
        magnitude[:, 0::2] = packed & 0x0F
        magnitude[:, 1::2] = packed >> 4
        signs = np.ascontiguousarray(segment[..., 16:].transpose(0, 3, 1, 2))
        negative = np.unpackbits(signs, axis=1, count=TL2_ROWS, bitorder="little").astype(bool)
        balanced = magnitude.astype(np.int16)
        np.negative(balanced, out=balanced, where=negative)
        if np.any(np.abs(balanced) > 13):
            raise ValueError("TL2 packed triple exceeds balanced base-3 range")
        encoded = balanced + 13
        block = trits[:, :, :, sub * 32 : (sub + 1) * 32]
        triple = np.empty((row_blocks, TL2_ROWS, groups, 10, 3), dtype=np.int8)
        triple[..., 0] = encoded // 9 - 1
        triple[..., 1] = (encoded // 3) % 3 - 1
        triple[..., 2] = encoded % 3 - 1
        block[..., :30] = triple.reshape(row_blocks, TL2_ROWS, groups, 30)

        tail_bytes = np.ascontiguousarray(buf[:, :, cursor : cursor + 16].transpose(0, 2, 1))
        cursor += 16
        tail = np.empty((row_blocks, TL2_ROWS, groups), dtype=np.uint8)
        tail[:, 0::2] = tail_bytes & 0x0F
        tail[:, 1::2] = tail_bytes >> 4
        first = tail & 0x03
        second = (tail >> 2) & 0x03
        if np.any(first > 2) or np.any(second > 2):
            raise ValueError("TL2 packed tail contains an invalid trit")
        block[..., 30] = first.astype(np.int8) - 1
        block[..., 31] = second.astype(np.int8) - 1

    values = torch.from_numpy(trits.reshape(row_blocks * TL2_ROWS, groups, GROUP)).to(
        torch.bfloat16
    )
    scales = torch.from_numpy(
        np.ascontiguousarray(gamma.reshape(row_blocks * TL2_ROWS, groups))
    ).to(torch.bfloat16)
    return (values * scales.unsqueeze(-1)).reshape(row_blocks * TL2_ROWS, groups * GROUP)


def _hadamard(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError("emb4 Hadamard width must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    return matrix / math.sqrt(size)


def _unpack_codes(buffer: np.ndarray, codebook_size: int, count: int) -> np.ndarray:
    bits = math.ceil(math.log2(codebook_size))
    per_byte = 8 // bits
    unpacked = np.empty(buffer.size * per_byte, dtype=np.uint8)
    for index in range(per_byte):
        unpacked[index::per_byte] = (buffer >> (bits * index)) & ((1 << bits) - 1)
    codes = unpacked[:count]
    if np.any(codes >= codebook_size):
        raise ValueError("emb4 code index exceeds its tier codebook")
    return codes


def dequant_embedding(
    get_tensor: Callable[[str], torch.Tensor],
    metadata: dict[str, Any],
    chunk: int = 1 << 12,
    *,
    get_slice: Callable[[str], Any] | None = None,
) -> torch.Tensor:
    """Reconstruct the bf16 emb4 table with bounded row chunks."""
    vocab_size = int(metadata["V"])
    hidden_size = int(metadata["D"])
    codebook_sizes = metadata["Ks"]
    if isinstance(codebook_sizes, str):
        codebook_sizes = json.loads(codebook_sizes)
    if chunk <= 0:
        raise ValueError("emb4 reconstruction chunk must be positive")
    tier_map = _as_numpy(get_tensor(_EMB4_PREFIX + "tier_map")).astype(np.int64, copy=False)
    if tier_map.shape != (vocab_size,):
        raise ValueError("emb4 tier_map shape differs from metadata")
    if tier_map.size and (tier_map.min() < 0 or tier_map.max() >= len(codebook_sizes)):
        raise ValueError("emb4 tier_map references an undeclared tier")

    hadamard = _hadamard(hidden_size)
    weights = torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16)
    for tier, codebook_size in enumerate(codebook_sizes):
        rows = np.flatnonzero(tier_map == tier)
        if rows.size == 0:
            continue
        code_name = f"{_EMB4_PREFIX}t{tier}_codes"
        bits = math.ceil(math.log2(codebook_size))
        per_byte = 8 // bits
        if hidden_size % per_byte:
            raise ValueError("emb4 hidden size is incompatible with code packing")
        centers = _as_numpy(get_tensor(f"{_EMB4_PREFIX}t{tier}_centers").to(torch.float32))
        row_scale = _as_numpy(get_tensor(f"{_EMB4_PREFIX}t{tier}_s").to(torch.float32))
        col_scale = torch.from_numpy(
            _as_numpy(get_tensor(f"{_EMB4_PREFIX}t{tier}_t").to(torch.float32))
        )
        if centers.shape != (codebook_size, hidden_size):
            raise ValueError(f"emb4 tier {tier} centers shape differs from metadata")
        if row_scale.shape != (rows.size,) or col_scale.shape != (hidden_size,):
            raise ValueError(f"emb4 tier {tier} scale shape differs from metadata")
        centers_t = np.ascontiguousarray(centers.T)
        code_buffer = None
        code_slice = get_slice(code_name) if get_slice is not None else None
        if code_slice is None:
            code_buffer = _as_numpy(get_tensor(code_name))
        for start in range(0, rows.size, chunk):
            stop = min(start + chunk, rows.size)
            count = (stop - start) * hidden_size
            if code_slice is None:
                packed = code_buffer[
                    start * hidden_size // per_byte : math.ceil(stop * hidden_size / per_byte)
                ]
            else:
                byte_start = start * hidden_size // per_byte
                byte_stop = math.ceil(stop * hidden_size / per_byte)
                packed = _as_numpy(code_slice[byte_start:byte_stop])
            codes = _unpack_codes(packed, codebook_size, count).reshape(stop - start, hidden_size)
            rotated = np.take_along_axis(centers_t, codes.T, axis=1).T
            restored = torch.from_numpy(np.ascontiguousarray(rotated)) @ hadamard.T
            restored *= col_scale[None, :]
            restored *= torch.from_numpy(row_scale[start:stop])[:, None]
            weights[rows[start:stop]] = restored.to(torch.bfloat16)
    return weights


def read_emb4_embedding_payload(path: str | Path, *, chunk: int = 1 << 12) -> torch.Tensor:
    """Read and reconstruct only the emb4 vocabulary payload."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        _, summary = _metadata_for_handle(handle, path)
        return dequant_embedding(
            handle.get_tensor,
            summary["embedding_metadata"],
            chunk=chunk,
            get_slice=handle.get_slice,
        )


def iter_emb4_tensors(
    path: str | Path, *, embedding_chunk: int = 1 << 12
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield reconstructed tensors without retaining the full language-model state."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        _, summary = _metadata_for_handle(handle, path)
        for name in summary["tensor_names"]:
            if name.startswith("tl2/"):
                tensor = unpack_tl2_tensor(handle.get_tensor(name))
                yield name[4:], tensor
            elif name.startswith("fp/"):
                yield name[3:], handle.get_tensor(name)
        embedding = dequant_embedding(
            handle.get_tensor,
            summary["embedding_metadata"],
            chunk=embedding_chunk,
            get_slice=handle.get_slice,
        )
        yield "model.embed_tokens.weight", embedding


def load_emb4_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Materialize a full state dict for callers that explicitly need one."""
    return dict(iter_emb4_tensors(path))
