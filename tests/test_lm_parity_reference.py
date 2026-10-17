from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from kitten2_onnx.lm_parity_reference import (
    REFERENCE_FORMAT,
    LMParityReference,
    compare_tensors,
    load_reference_pack,
    pack_sha256,
    sha256_file,
    source_identity,
    validate_source_identity,
    verify_baseline_against_reference,
)


def _write_pack(root: Path) -> LMParityReference:
    root.mkdir(parents=True)
    arrays = {
        "prompt_ids": np.asarray([10, 11, 12], dtype=np.int64),
        "prefill_inputs_embeds": np.zeros((3, 4), dtype=np.float32),
        "speaker_hidden": np.zeros(4, dtype=np.float32),
        "generated_ids": np.asarray([2, 1], dtype=np.int64),
        "token_embeds": np.zeros((2, 4), dtype=np.float32),
        "logits": np.zeros((3, 1, 5), dtype=np.float32),
        "cache_keys": np.arange(20, dtype=np.float32).reshape(2, 1, 1, 5, 2),
        "cache_values": np.arange(20, 40, dtype=np.float32).reshape(2, 1, 1, 5, 2),
    }
    arrays["logits"][0, 0, 2] = 5.0
    arrays["logits"][1, 0, 1] = 5.0
    arrays["logits"][2, 0, 4] = 5.0
    records = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        path = root / filename
        np.save(path, array, allow_pickle=False)
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        records[name] = {
            "file": filename,
            "shape": list(loaded.shape),
            "dtype": str(loaded.dtype),
            "sha256": sha256_file(path),
        }
    metadata = {
        "format": REFERENCE_FORMAT,
        "source_identity": {"model_revision": "test-revision"},
        "lm_metadata": {
            "input_mode": "inputs_embeds",
            "input_names": [
                "inputs_embeds",
                "attention_mask",
                "position_ids",
                "past_key_0",
                "past_value_0",
                "past_key_1",
                "past_value_1",
            ],
        },
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "prompt_length": 3,
        "generated_token_count": 2,
        "arrays": records,
    }
    metadata["pack_sha256"] = pack_sha256(metadata)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return load_reference_pack(root)


def test_reference_arrays_are_memory_mapped_and_cache_prefixes_are_views(tmp_path):
    reference = _write_pack(tmp_path / "reference")

    prefix = reference.cache_tensors(4)

    assert isinstance(reference.arrays["cache_keys"], np.memmap)
    assert len(prefix) == 4
    assert all(array.shape == (1, 1, 4, 2) for array in prefix)
    np.testing.assert_array_equal(prefix[0], reference.arrays["cache_keys"][0, ..., :4, :])
    with pytest.raises(ValueError, match="out of range"):
        reference.cache_tensors(6)


def test_pack_identity_and_array_hashes_are_validated(tmp_path):
    reference = _write_pack(tmp_path / "reference")
    assert load_reference_pack(reference.root, expected_pack_sha256=reference.metadata["pack_sha256"])

    with (reference.root / "generated_ids.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="array hash differs"):
        load_reference_pack(reference.root)


def test_comparator_uses_parity_tolerance_formula_and_shape_errors():
    comparison = compare_tensors(
        np.asarray([1.0, 2.2]), np.asarray([1.0, 2.0]), atol=0.1, rtol=0.0
    )
    assert comparison["outside_tolerance_count"] == 1
    assert comparison["max_normalized_tolerance_violation"] == pytest.approx(2.0)
    assert comparison["max_absolute_error"] == pytest.approx(0.2)

    mismatch = compare_tensors(np.zeros((2,)), np.zeros((3,)), atol=0.1, rtol=0.0)
    assert mismatch["shape_mismatch"] is True
    assert mismatch["outside_tolerance_count"] == 3


def test_source_identity_binds_model_reference_and_baseline_files(tmp_path):
    model = tmp_path / "model"
    baseline = tmp_path / "baseline"
    fixture = tmp_path / "fixture.npz"
    for relative in (
        "lm/model.safetensors",
        "lm/config.json",
        "config.json",
        "voices/voices.json",
        "voices/bruno.npz",
        "lm/added_tokens.json",
        "lm/merges.txt",
        "lm/special_tokens_map.json",
        "lm/tokenizer.json",
        "lm/tokenizer_config.json",
        "lm/vocab.json",
    ):
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    (model / "config.json").write_text(
        json.dumps({"lm_dir": "lm", "voices": "voices/voices.json"}), encoding="utf-8"
    )
    (model / "voices/voices.json").write_text(
        json.dumps({"Bruno": {"artifacts": "bruno.npz"}}), encoding="utf-8"
    )
    (model / "model_revision.json").write_text(
        json.dumps({"source_model_revision": "rev-1"}), encoding="utf-8"
    )
    fixture.write_bytes(b"fixture")
    baseline_files = {}
    for relative in ("lm/lm.onnx", "lm/lm.onnx.data"):
        path = baseline / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        baseline_files[relative] = sha256_file(path)
    (baseline / "bundle.json").write_text(json.dumps({"files": baseline_files}), encoding="utf-8")
    identity = source_identity(model, baseline, fixture)
    reference = LMParityReference(tmp_path, {"source_identity": identity}, {})

    validate_source_identity(reference, model, baseline, fixture)
    (baseline / "lm/lm.onnx.data").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        validate_source_identity(reference, model, baseline, fixture)


def test_baseline_verifier_replays_all_logits_cache_prefixes_and_tokens(tmp_path, monkeypatch):
    reference = _write_pack(tmp_path / "reference")

    class SessionOptions:
        intra_op_num_threads = 0

    class Session:
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def run(self, _output_names, _feed):
            phase = self.calls
            self.calls += 1
            return [
                reference.arrays["logits"][phase],
                *reference.cache_tensors(reference.prompt_length + phase),
            ]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=Session),
    )

    result = verify_baseline_against_reference(tmp_path / "baseline.onnx", reference)

    assert result["diagnostic_score"] == [0, 0, 0.0, 0.0]
    assert result["generated_ids_match"] is True
    assert result["runtime_calls"] == 3
