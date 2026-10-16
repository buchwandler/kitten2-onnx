import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kitten2_onnx.ternary import (
    is_ternary,
    pack_state_dict,
    signed_zero_bit_differences,
    tensor_bytes_equal,
    unpack_state_dict,
    unpack_tensor,
)
from kitten2_onnx.weights import compare_packed_checkpoint, load_lm_state_dict


def _sample_state():
    ternary = torch.zeros((3, 128), dtype=torch.bfloat16)
    for row, scale in enumerate((0.5, 1.25, 2.0)):
        ternary[row, ::3] = scale
        ternary[row, 1::4] = -scale
    dense = torch.tensor([[0.25, -1.5], [2.0, 0.125]], dtype=torch.float32)
    vector = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
    return {"layer.weight": ternary, "dense.weight": dense, "norm.weight": vector}


def test_pack_unpack_preserves_exact_tensor_bytes():
    state = _sample_state()
    assert is_ternary(state["layer.weight"])
    assert not is_ternary(state["dense.weight"])

    packed, metadata = pack_state_dict(state)
    restored = unpack_state_dict(packed, metadata)

    assert "layer.weight::codes" in packed
    assert "layer.weight::scale" in packed
    assert "dense.weight" in packed
    assert set(restored) == set(state)
    for name, tensor in state.items():
        assert tensor_bytes_equal(restored[name], tensor)


def test_packed_safetensors_loader_and_streaming_comparison(tmp_path):
    state = _sample_state()
    packed, metadata = pack_state_dict(state)
    model_dir = tmp_path / "model"
    lm_dir = model_dir / "lm"
    lm_dir.mkdir(parents=True)
    save_file(state, str(lm_dir / "model.safetensors"))
    save_file(packed, str(lm_dir / "model-ternary.safetensors"), metadata=metadata)
    (model_dir / "config.json").write_text(
        json.dumps({"lm_dir": "lm", "lm_packed": "lm/model-ternary.safetensors"}),
        encoding="utf-8",
    )

    loaded_plain = load_lm_state_dict(model_dir, variant="plain")
    loaded_packed = load_lm_state_dict(model_dir, variant="packed")
    comparison = compare_packed_checkpoint(
        lm_dir / "model-ternary.safetensors",
        lm_dir / "model.safetensors",
    )

    assert comparison["bitwise_exact"]
    assert comparison["numerically_exact"]
    assert comparison["tensor_count"] == len(state)
    for name in state:
        assert tensor_bytes_equal(loaded_packed[name], loaded_plain[name])


def test_unpacked_codes_reject_invalid_base3_and_padding():
    state = _sample_state()["layer.weight"]
    packed, _ = pack_state_dict({"weight": state})
    codes = packed["weight::codes"]
    scales = packed["weight::scale"]
    invalid_code = codes.clone()
    invalid_code[-1] = 243
    with pytest.raises(ValueError, match="base-3 range"):
        unpack_tensor(invalid_code, scales, state.shape, state.dtype)

    invalid_padding = codes.clone()
    invalid_padding[-1] |= 81
    with pytest.raises(ValueError, match="padding digits"):
        unpack_tensor(invalid_padding, scales, state.shape, state.dtype)


def test_real_plain_checkpoint_tensor_round_trip_when_available():
    model_dir = Path(os.environ.get("KITTEN2_MODEL_DIR", ".models/kitten-tts-2"))
    plain_path = model_dir / "lm" / "model.safetensors"
    if not plain_path.is_file():
        pytest.skip("plain KittenTTS 2 checkpoint is not available")

    with safe_open(str(plain_path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        ternary_name = next(
            name for name in keys if name.endswith("model.layers.0.self_attn.q_proj.weight")
        )
        dense_name = "spk_proj.0.weight"
        assert dense_name in keys
        sample = {
            ternary_name: checkpoint.get_tensor(ternary_name),
            dense_name: checkpoint.get_tensor(dense_name),
        }

    assert is_ternary(sample[ternary_name])
    packed, metadata = pack_state_dict(sample)
    restored = unpack_state_dict(packed, metadata)
    assert f"{ternary_name}::codes" in packed
    for name, tensor in sample.items():
        assert torch.equal(restored[name], tensor)
    assert tensor_bytes_equal(restored[dense_name], sample[dense_name])
    assert not tensor_bytes_equal(restored[ternary_name], sample[ternary_name])
    assert signed_zero_bit_differences(restored[ternary_name], sample[ternary_name]) > 0


def test_available_official_packed_checkpoint_matches_all_plain_tensor_values():
    model_dir = Path(os.environ.get("KITTEN2_MODEL_DIR", ".models/kitten-tts-2"))
    packed_path = model_dir / "lm" / "model-ternary.safetensors"
    plain_path = model_dir / "lm" / "model.safetensors"
    if not packed_path.is_file() or not plain_path.is_file():
        pytest.skip("official packed and plain checkpoints are not both available")

    report = compare_packed_checkpoint(packed_path, plain_path)

    assert report["numerically_exact"], report["mismatches"]
    assert report["bitwise_exact"] or report["signed_zero_bit_differences"] > 0
    assert report["tensor_count"] > 0
