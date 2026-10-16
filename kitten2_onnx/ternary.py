"""Value-preserving five-trit-per-byte packing for KittenTTS 2 ternary matrices."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import torch

GROUP_SIZE = 128
TRITS_PER_BYTE = 5
TERNARY_FORMAT = "kittenml-ternary-1.58"
_POWERS = torch.tensor([3**index for index in range(TRITS_PER_BYTE)], dtype=torch.int32)


def is_ternary(tensor: torch.Tensor, group_size: int = GROUP_SIZE) -> bool:
    """Return whether every group is exactly in ``{-scale, 0, +scale}``."""
    if (
        tensor.ndim != 2
        or not tensor.is_floating_point()
        or group_size < 1
        or tensor.shape[1] % group_size
    ):
        return False
    grouped = tensor.float().reshape(tensor.shape[0], -1, group_size)
    magnitude = grouped.abs()
    scale = magnitude.amax(dim=2, keepdim=True)
    nonzero = grouped != 0
    return bool(torch.all(~nonzero | (magnitude == scale)))


def _pack_trits(trits: torch.Tensor) -> torch.Tensor:
    padding = (-trits.numel()) % TRITS_PER_BYTE
    if padding:
        trits = torch.cat((trits, torch.zeros(padding, dtype=trits.dtype, device=trits.device)))
    digits = trits.to(torch.int32).reshape(-1, TRITS_PER_BYTE)
    powers = _POWERS.to(device=trits.device)
    return (digits * powers).sum(dim=1).to(torch.uint8).contiguous()


def _unpack_trits(packed: torch.Tensor) -> torch.Tensor:
    values = packed.to(torch.int32).unsqueeze(1)
    powers = _POWERS.to(device=packed.device)
    return (torch.div(values, powers, rounding_mode="floor") % 3).reshape(-1)


def pack_tensor(
    tensor: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one ternary matrix as packed codes and per-group scales."""
    if not is_ternary(tensor, group_size):
        raise ValueError("tensor is not an exactly ternary matrix for this group size")
    grouped = tensor.reshape(tensor.shape[0], -1, group_size)
    scale = grouped.abs().amax(dim=2, keepdim=True)
    trits = (grouped.sign().to(torch.int8) + 1).reshape(-1)
    return _pack_trits(trits), scale.squeeze(2).contiguous()


def unpack_tensor(
    codes: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...] | list[int],
    dtype: torch.dtype,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Decode values; the format cannot retain the sign bit of zero values."""
    shape = tuple(shape)
    if len(shape) != 2 or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError(f"ternary matrix shape must contain two positive dimensions, got {shape}")
    if group_size < 1 or shape[1] % group_size:
        raise ValueError(f"matrix width must be divisible by group_size={group_size}")
    if codes.dtype != torch.uint8:
        raise TypeError(f"ternary codes must have dtype uint8, found {codes.dtype}")
    expected_codes = (shape[0] * shape[1] + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
    if codes.numel() != expected_codes:
        raise ValueError(f"expected {expected_codes} ternary code bytes, found {codes.numel()}")
    expected_scale_shape = (shape[0], shape[1] // group_size)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(f"expected scale shape {expected_scale_shape}, found {tuple(scales.shape)}")
    if not scales.is_floating_point() or scales.dtype != dtype:
        raise TypeError(f"scale dtype must match floating weight dtype {dtype}, found {scales.dtype}")
    if codes.numel() and int(codes.max()) >= 3**TRITS_PER_BYTE:
        raise ValueError("ternary code byte is outside the valid base-3 range")

    numel = shape[0] * shape[1]
    trits = _unpack_trits(codes)
    if torch.any(trits[numel:] != 0):
        raise ValueError("unused ternary padding digits must be zero")
    signs = (trits[:numel].to(torch.int8) - 1).reshape(shape[0], -1, group_size)
    result = (signs.to(scales.dtype) * scales.reshape(shape[0], -1, 1)).reshape(shape)
    return result.to(dtype).contiguous()


def pack_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    group_size: int = GROUP_SIZE,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Pack ternary matrices while preserving all other tensors unchanged."""
    tensors: dict[str, torch.Tensor] = {}
    packed_keys: dict[str, dict[str, Any]] = {}
    for name, tensor in state_dict.items():
        if "::" in name:
            raise ValueError(f"state dictionary tensor names may not contain '::': {name!r}")
        if not is_ternary(tensor, group_size):
            tensors[name] = tensor.contiguous()
            continue
        codes, scales = pack_tensor(tensor, group_size)
        tensors[f"{name}::codes"] = codes
        tensors[f"{name}::scale"] = scales
        packed_keys[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}

    metadata = {
        "format": TERNARY_FORMAT,
        "group_size": str(group_size),
        "packed": json.dumps(packed_keys, sort_keys=True),
    }
    return tensors, metadata


def unpack_state_dict(
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    """Reconstruct a state dictionary from the packed checkpoint representation."""
    if metadata.get("format") != TERNARY_FORMAT:
        raise ValueError(f"not a packed ternary checkpoint: {metadata.get('format')!r}")
    try:
        group_size = int(metadata["group_size"])
        packed_keys = json.loads(metadata["packed"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("packed ternary metadata is incomplete or malformed") from error
    if group_size < 1 or not isinstance(packed_keys, dict):
        raise ValueError("packed ternary metadata has an invalid group size or tensor map")

    expected_packed_names = {
        f"{name}::{suffix}" for name in packed_keys for suffix in ("codes", "scale")
    }
    actual_packed_names = {name for name in tensors if "::" in name}
    if actual_packed_names != expected_packed_names:
        missing = sorted(expected_packed_names - actual_packed_names)
        unexpected = sorted(actual_packed_names - expected_packed_names)
        raise ValueError(f"packed tensor entries differ from metadata; missing={missing}, unexpected={unexpected}")

    state = {name: tensor for name, tensor in tensors.items() if "::" not in name}
    for name, spec in packed_keys.items():
        if not isinstance(name, str) or name in state or not isinstance(spec, dict):
            raise ValueError(f"invalid packed tensor metadata for {name!r}")
        shape = spec.get("shape")
        dtype_name = spec.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype_name, str):
            raise TypeError(f"invalid packed tensor shape or dtype for {name!r}")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"unsupported packed tensor dtype {dtype_name!r} for {name!r}")
        state[name] = unpack_tensor(
            tensors[f"{name}::codes"],
            tensors[f"{name}::scale"],
            shape,
            dtype,
            group_size,
        )
    return state


def tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor shape, dtype, and storage bytes without numeric coercion."""
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def tensor_values_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare numerical values, treating IEEE signed zeros as equivalent."""
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)


def signed_zero_bit_differences(left: torch.Tensor, right: torch.Tensor) -> int:
    """Count only zero values whose IEEE sign bit differs between tensors."""
    if left.shape != right.shape or left.dtype != right.dtype or not left.is_floating_point():
        return 0
    zero_mask = (left == 0) & (right == 0)
    return int(torch.count_nonzero(zero_mask & (torch.signbit(left) != torch.signbit(right))))
