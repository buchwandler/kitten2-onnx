from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from kitten2_onnx.tl2_emb4 import (
    EMB4_SCHEMAS,
    dequant_embedding,
    inspect_emb4,
    inspect_emb4_embedding_asset,
    iter_emb4_tensors,
    load_emb4_state_dict,
    read_emb4_embedding_asset,
    read_emb4_embedding_payload,
    unpack_tl2_tensor,
    write_emb4_embedding_asset,
)
from kitten2_onnx.weights import available_weight_variants, load_lm_state_dict, resolve_weight_path


def _pack_tl2(trits: np.ndarray, scales: np.ndarray) -> np.ndarray:
    buffer = np.zeros((1, 1, 928), dtype=np.uint8)
    buffer[0, 0, :64] = np.asarray(scales, dtype=np.float16).view(np.uint8)
    cursor = 64
    for sub in range(4):
        segment = buffer[0, 0, cursor : cursor + 200].reshape(10, 20)
        for index in range(10):
            triple = trits[:, sub * 32 + index * 3 : sub * 32 + index * 3 + 3]
            encoded = (triple[:, 0] + 1) * 9 + (triple[:, 1] + 1) * 3 + triple[:, 2] + 1
            balanced = encoded - 13
            magnitude = np.abs(balanced).astype(np.uint8)
            negative = balanced < 0
            segment[index, :16] = magnitude[0::2] | (magnitude[1::2] << 4)
            segment[index, 16:20] = np.packbits(negative, bitorder="little")
        cursor += 200
        for pair in range(16):
            even = trits[2 * pair, sub * 32 + 30 : sub * 32 + 32]
            odd = trits[2 * pair + 1, sub * 32 + 30 : sub * 32 + 32]
            even_nibble = (even[0] + 1) | ((even[1] + 1) << 2)
            odd_nibble = (odd[0] + 1) | ((odd[1] + 1) << 2)
            buffer[0, 0, cursor + pair] = even_nibble | (odd_nibble << 4)
        cursor += 16
    return buffer


def _pack_codes(codes: np.ndarray, codebook_size: int) -> torch.Tensor:
    bits = math.ceil(math.log2(codebook_size))
    per_byte = 8 // bits
    packed = np.zeros(math.ceil(codes.size / per_byte), dtype=np.uint8)
    for index, code in enumerate(codes.reshape(-1)):
        packed[index // per_byte] |= int(code) << (bits * (index % per_byte))
    return torch.from_numpy(packed)


def _upstream_embedding_reference(tier_map, codebooks, row_scales, col_scales):
    hidden_size = col_scales[0].numel()
    hadamard = torch.ones((1, 1), dtype=torch.float32)
    while hadamard.shape[0] < hidden_size:
        hadamard = torch.cat(
            [torch.cat([hadamard, hadamard], 1), torch.cat([hadamard, -hadamard], 1)], 0
        )
    hadamard /= math.sqrt(hidden_size)
    result = torch.empty((tier_map.numel(), hidden_size), dtype=torch.bfloat16)
    for tier, centers in enumerate(codebooks):
        rows = np.flatnonzero(tier_map.numpy() == tier)
        if not rows.size:
            continue
        codes = centers["codes"].numpy().astype(np.int64)
        selected = np.take_along_axis(centers["centers"].numpy().T, codes.T, axis=1).T
        rotated = torch.from_numpy(np.ascontiguousarray(selected)) @ hadamard.T
        restored = rotated * col_scales[tier][None, :] * row_scales[tier][:, None]
        result[rows] = restored.to(torch.bfloat16)
    return result


def _write_emb4_fixture(path, *, schema="kitten_ternary_tl2_rl_v1", duplicate_embedding=False):
    rng = np.random.default_rng(73)
    vocab_size, hidden_size = 5, 8
    tier_map = torch.tensor([0, 1, 0, 1, 0], dtype=torch.uint8)
    codebook_sizes = [4, 2]
    tl2_trits = rng.choice(np.asarray([-1, 0, 1], dtype=np.int8), (32, 128))
    tl2_scales = np.linspace(0.25, 1.0, 32, dtype=np.float16)
    tensors = {
        "tl2/layer.weight": torch.from_numpy(_pack_tl2(tl2_trits, tl2_scales)),
        "fp/norm.weight": torch.tensor([0.5, 1.0], dtype=torch.bfloat16),
        "emb4/tier_map": tier_map,
    }
    if duplicate_embedding:
        tensors["fp/model.embed_tokens.weight"] = torch.zeros(
            (vocab_size, hidden_size), dtype=torch.bfloat16
        )
    codebooks = []
    row_scales = []
    col_scales = []
    for tier, codebook_size in enumerate(codebook_sizes):
        rows = int((tier_map == tier).sum())
        codes = rng.integers(0, codebook_size, size=(rows, hidden_size), dtype=np.uint8)
        centers = torch.from_numpy(rng.normal(size=(codebook_size, hidden_size)).astype(np.float32))
        row_scale = torch.from_numpy(rng.uniform(0.5, 1.5, size=rows).astype(np.float32))
        col_scale = torch.from_numpy(rng.uniform(0.5, 1.5, size=hidden_size).astype(np.float32))
        tensors[f"emb4/t{tier}_codes"] = _pack_codes(codes, codebook_size)
        tensors[f"emb4/t{tier}_centers"] = centers
        tensors[f"emb4/t{tier}_s"] = row_scale
        tensors[f"emb4/t{tier}_t"] = col_scale
        codebooks.append({"codes": torch.from_numpy(codes), "centers": centers})
        row_scales.append(row_scale)
        col_scales.append(col_scale)
    metadata = {
        "schema": schema,
        "embedding_metadata": json.dumps(
            {"V": vocab_size, "D": hidden_size, "Ks": json.dumps(codebook_sizes)}
        ),
    }
    save_file(tensors, str(path), metadata=metadata)
    expected_embedding = _upstream_embedding_reference(tier_map, codebooks, row_scales, col_scales)
    expected_values = torch.from_numpy(tl2_trits.reshape(32, 1, 128)).to(torch.bfloat16)
    expected_scales = torch.from_numpy(tl2_scales.reshape(32, 1)).to(torch.bfloat16)
    expected_tl2 = (expected_values * expected_scales.unsqueeze(-1)).reshape(32, 128)
    return tensors, expected_embedding, expected_tl2


def test_tl2_tensor_matches_upstream_packed_fixture(tmp_path):
    path = tmp_path / "emb4.safetensors"
    tensors, _, expected = _write_emb4_fixture(path)

    actual = unpack_tl2_tensor(tensors["tl2/layer.weight"])

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("schema", sorted(EMB4_SCHEMAS))
def test_emb4_embedding_reconstruction_and_streamed_state(schema, tmp_path):
    path = tmp_path / "emb4.safetensors"
    tensors, expected_embedding, expected_tl2 = _write_emb4_fixture(path, schema=schema)

    metadata = inspect_emb4(path)
    actual_embedding = read_emb4_embedding_payload(path, chunk=1)
    streamed = dict(iter_emb4_tensors(path, embedding_chunk=2))
    full_state = load_emb4_state_dict(path)

    assert metadata["schema"] == schema
    assert metadata["embedding_metadata"] == {"V": 5, "D": 8, "Ks": [4, 2]}
    assert torch.equal(actual_embedding, expected_embedding)
    assert torch.equal(streamed["model.embed_tokens.weight"], expected_embedding)
    assert torch.equal(streamed["layer.weight"], expected_tl2)
    assert torch.equal(streamed["norm.weight"], tensors["fp/norm.weight"])
    assert set(full_state) == {"model.embed_tokens.weight", "layer.weight", "norm.weight"}
    assert torch.equal(full_state["model.embed_tokens.weight"], expected_embedding)



def test_compact_emb4_asset_contains_only_vocabulary_and_reconstructs(tmp_path):
    source = tmp_path / "full-emb4.safetensors"
    _, expected_embedding, _ = _write_emb4_fixture(source)
    asset = tmp_path / "vocabulary.emb4.safetensors"

    write_emb4_embedding_asset(source, asset)

    summary = inspect_emb4_embedding_asset(asset)
    actual = read_emb4_embedding_asset(asset, chunk=1)
    assert summary["tensor_names"] == [
        "emb4/t0_centers",
        "emb4/t0_codes",
        "emb4/t0_s",
        "emb4/t0_t",
        "emb4/t1_centers",
        "emb4/t1_codes",
        "emb4/t1_s",
        "emb4/t1_t",
        "emb4/tier_map",
    ]
    assert torch.equal(actual, expected_embedding)

def test_emb4_variant_discovery_and_weight_loading(tmp_path):
    model_dir = tmp_path
    lm_dir = model_dir / "lm"
    lm_dir.mkdir()
    path = lm_dir / "model-tl2-emb4.safetensors"
    _, expected_embedding, expected_tl2 = _write_emb4_fixture(path)
    (model_dir / "config.json").write_text(
        json.dumps({"lm_dir": "lm", "lm_emb4": "lm/model-tl2-emb4.safetensors"}),
        encoding="utf-8",
    )

    assert available_weight_variants(model_dir) == ("emb4",)
    assert resolve_weight_path(model_dir, "emb4") == path.resolve()
    loaded = load_lm_state_dict(model_dir, variant="emb4")
    assert torch.equal(loaded["model.embed_tokens.weight"], expected_embedding)
    assert torch.equal(loaded["layer.weight"], expected_tl2)


def test_emb4_rejects_unknown_schema(tmp_path):
    path = tmp_path / "unknown.safetensors"
    _write_emb4_fixture(path, schema="unknown")
    with pytest.raises(ValueError, match="unexpected emb4 schema"):
        inspect_emb4(path)


def test_emb4_rejects_duplicate_embedding(tmp_path):
    path = tmp_path / "duplicate.safetensors"
    _write_emb4_fixture(path, duplicate_embedding=True)
    with pytest.raises(ValueError, match="duplicate dense embedding"):
        inspect_emb4(path)


def test_dequant_embedding_rejects_invalid_code_indices():
    metadata = {"V": 1, "D": 8, "Ks": [3]}
    tensors = {
        "emb4/tier_map": torch.tensor([0], dtype=torch.uint8),
        "emb4/t0_codes": torch.full((4,), 255, dtype=torch.uint8),
        "emb4/t0_centers": torch.zeros((3, 8)),
        "emb4/t0_s": torch.ones(1),
        "emb4/t0_t": torch.ones(8),
    }
    with pytest.raises(ValueError, match="code index exceeds"):
        dequant_embedding(lambda name: tensors[name], metadata, chunk=1)
