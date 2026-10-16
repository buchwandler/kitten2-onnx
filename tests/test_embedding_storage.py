from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

from kitten2_onnx.export_lm import save_embedding_table
from kitten2_onnx.runtime import Kitten2OnnxLM


class _EmbeddingModel:
    def __init__(self, weight):
        self.weight = weight

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)


def test_bfloat16_embedding_table_round_trips_through_uint16_bits(tmp_path):
    torch.manual_seed(11)
    weight = torch.randn((32, 16), dtype=torch.bfloat16)
    path, storage_dtype = save_embedding_table(
        _EmbeddingModel(weight), tmp_path, source_dtype="bfloat16"
    )
    table = np.load(path, mmap_mode="r", allow_pickle=False)

    assert path.name == "embed_tokens.bf16.npy"
    assert storage_dtype == "bfloat16-u16"
    assert table.dtype == np.uint16
    assert table.nbytes == weight.numel() * 2
    row_ids = np.random.default_rng(13).choice(weight.shape[0], size=8, replace=False)
    packed_rows = np.asarray(table[row_ids], dtype=np.uint32) << 16
    restored = packed_rows.view(np.float32)
    legacy_fp32_table = weight.float().numpy()
    np.testing.assert_array_equal(restored, legacy_fp32_table[row_ids])


def test_runtime_expands_only_selected_bfloat16_embedding_rows(tmp_path, monkeypatch):
    class FakeSession:
        def __init__(self, path, providers):
            self.providers = providers

        def get_inputs(self):
            return [SimpleNamespace(name="inputs_embeds", type="tensor(float)")]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))
    torch.manual_seed(12)
    weight = torch.randn((16, 8), dtype=torch.bfloat16)
    path, storage_dtype = save_embedding_table(
        _EmbeddingModel(weight), tmp_path, source_dtype="bfloat16"
    )
    (tmp_path / "lm.meta.json").write_text(
        json.dumps(
            {
                "embedding_file": path.name,
                "embedding_storage_dtype": storage_dtype,
                "embedding_runtime_dtype": "float32",
                "runtime_profile": "dev-fp32",
                "runtime_provider": "CPUExecutionProvider",
                "io_dtype": "float32",
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "head_dim": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "lm.onnx").write_bytes(b"fake graph")

    runtime = Kitten2OnnxLM(tmp_path)
    token_ids = np.asarray([[12, 2, 7]], dtype=np.int64)
    selected = runtime._embed(token_ids, prefill=False)

    assert runtime.embeddings.dtype == np.uint16
    np.testing.assert_array_equal(selected, weight[token_ids].float().numpy())
