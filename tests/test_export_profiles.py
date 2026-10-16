from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import kitten2_onnx.export_lm as export_lm_module
from kitten2_onnx.runtime import Kitten2OnnxLM


class FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embedding = nn.Embedding(7, 4)
        self.lm_head = nn.Linear(4, 7, bias=False)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


def _make_repo(path: Path):
    lm_dir = path / "lm"
    lm_dir.mkdir(parents=True)
    (lm_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "dtype": "bfloat16"}),
        encoding="utf-8",
    )
    (lm_dir / "model.safetensors").write_bytes(b"plain-checkpoint-placeholder")
    (path / "model_revision.json").write_text(
        json.dumps({"source_model_revision": "pinned-revision"}),
        encoding="utf-8",
    )
    return lm_dir


@pytest.mark.parametrize(
    ("profile_name", "dtype", "graph_dtype", "io_dtype", "provider", "verify"),
    [
        ("dev-fp32", torch.float32, "float32", "float32", "CPUExecutionProvider", True),
        ("cuda-fp16", torch.float16, "float16", "float16", "CUDAExecutionProvider", False),
        ("cuda-bf16", torch.bfloat16, "bfloat16", "bfloat16", "CUDAExecutionProvider", False),
    ],
)
def test_export_lm_records_precision_profile(
    tmp_path,
    monkeypatch,
    tiny_qwen3,
    profile_name,
    dtype,
    graph_dtype,
    io_dtype,
    provider,
    verify,
):
    repo_dir = tmp_path / "source"
    lm_dir = _make_repo(repo_dir)
    output_dir = tmp_path / "exported"
    seen = {}

    def fake_load_hf_lm(path, requested_dtype):
        seen["loaded_dtype"] = requested_dtype
        return tiny_qwen3.to(requested_dtype)

    def fake_export(model, args, output_path, **kwargs):
        seen["export_kwargs"] = kwargs
        Path(output_path).write_bytes(b"test-graph")

    monkeypatch.setattr(export_lm_module, "load_hf_lm", fake_load_hf_lm)
    monkeypatch.setattr(export_lm_module, "_export_with_diagnostics", fake_export)

    export_lm_module.export_lm(
        repo_dir,
        output_dir,
        profile_name=profile_name,
        diagnostics_dir=tmp_path / "diagnostics",
    )

    metadata = json.loads((output_dir / "lm.meta.json").read_text(encoding="utf-8"))
    assert seen["loaded_dtype"] == dtype
    assert seen["export_kwargs"]["verify"] is verify
    assert seen["export_kwargs"]["external_data"] is True
    assert metadata["source_dtype"] == "bfloat16"
    assert metadata["graph_dtype"] == graph_dtype
    assert metadata["io_dtype"] == io_dtype
    assert metadata["runtime_profile"] == profile_name
    assert metadata["runtime_provider"] == provider
    assert metadata["verify_on_export"] is verify
    assert metadata["source_model_revision"] == "pinned-revision"
    assert metadata["loading_path"] == str((lm_dir / "model.safetensors").resolve())
    assert metadata["input_mode"] == "inputs_embeds"
    expected_storage_dtype = "float16" if dtype == torch.float16 else "bfloat16-u16"
    assert metadata["embedding_storage_dtype"] == expected_storage_dtype
    assert metadata["embedding_runtime_dtype"] == io_dtype
    embedding = np.load(output_dir / metadata["embedding_file"], allow_pickle=False)
    assert str(embedding.dtype) == ("float16" if dtype == torch.float16 else "uint16")


def test_load_hf_lm_tolerates_speaker_head_but_verifies_transformer_weights(tmp_path, monkeypatch):
    import transformers

    repo_dir = tmp_path / "source"
    lm_dir = _make_repo(repo_dir)
    model = FakeCausalLM()
    calls = {}

    def fake_from_pretrained(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return model, {
            "missing_keys": [],
            "unexpected_keys": ["spk_proj.0.weight", "spk_proj.0.bias"],
        }

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    loaded = export_lm_module.load_hf_lm(repo_dir, torch.float32)

    assert loaded is model
    assert not loaded.training
    assert loaded.get_input_embeddings().weight.data_ptr() == loaded.get_output_embeddings().weight.data_ptr()
    assert calls["path"] == str(lm_dir)
    assert calls["kwargs"]["dtype"] == torch.float32
    assert calls["kwargs"]["attn_implementation"] == "eager"
    assert calls["kwargs"]["output_loading_info"] is True
    assert calls["kwargs"]["local_files_only"] is True


def test_load_hf_lm_rejects_missing_transformer_weight(tmp_path, monkeypatch):
    import transformers

    repo_dir = tmp_path / "source"
    _make_repo(repo_dir)
    model = FakeCausalLM()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(
            lambda *args, **kwargs: (
                model,
                {"missing_keys": ["model.layers.0.mlp.down_proj.weight"], "unexpected_keys": []},
            )
        ),
    )

    with pytest.raises(ValueError, match="Missing transformer checkpoint weights"):
        export_lm_module.load_hf_lm(repo_dir, torch.float32)


def test_runtime_uses_profile_provider_and_refuses_bfloat16_numpy(tmp_path, monkeypatch):
    import sys

    ort = SimpleNamespace()

    class FakeSession:
        def __init__(self, path, providers):
            self.providers = providers
            self.path = path

        def get_inputs(self):
            return [SimpleNamespace(name="inputs_embeds", type="tensor(float16)")]

    ort.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    model_dir = tmp_path / "fp16"
    model_dir.mkdir()
    np.save(model_dir / "embeddings.npy", np.zeros((7, 4), dtype=np.float16))
    (model_dir / "lm.meta.json").write_text(
        json.dumps(
            {
                "embedding_file": "embeddings.npy",
                "runtime_profile": "cuda-fp16",
                "runtime_provider": "CUDAExecutionProvider",
                "io_dtype": "float16",
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
            }
        ),
        encoding="utf-8",
    )

    runtime = Kitten2OnnxLM(model_dir)

    with pytest.raises(ValueError, match="requires CUDAExecutionProvider"):
        Kitten2OnnxLM(model_dir, providers=["CPUExecutionProvider"])

    assert runtime.session.providers == ["CUDAExecutionProvider"]
    assert runtime.model_numpy_dtype is np.float16

    (model_dir / "lm.meta.json").write_text(
        json.dumps(
            {
                "embedding_file": "embeddings.npy",
                "runtime_profile": "cuda-bf16",
                "runtime_provider": "CUDAExecutionProvider",
                "io_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="I/O binding"):
        Kitten2OnnxLM(model_dir)


@pytest.mark.parametrize(
    ("profile_name", "tensor_type"),
    [
        ("dev-fp32", "float32"),
        ("cuda-fp16", "float16"),
        ("cuda-bf16", "bfloat16"),
    ],
)
def test_tiny_qwen3_graph_io_types_match_profile(tmp_path, tiny_qwen3, profile_name, tensor_type):
    onnx = pytest.importorskip("onnx")
    profile = export_lm_module.RUNTIME_PROFILES[profile_name]
    model = tiny_qwen3.to(profile.graph_dtype)
    wrapper = export_lm_module.KittenQwen3ExportWrapper(model).eval()
    num_layers = int(model.config.num_hidden_layers)
    past_names, present_names = export_lm_module._io_names(num_layers)
    path = tmp_path / f"{profile_name}.onnx"
    inputs = (
        torch.randn(1, 2, model.config.hidden_size, dtype=profile.graph_dtype),
        torch.ones(1, 5, dtype=torch.long),
        torch.arange(3, 5, dtype=torch.long)[None, :],
        [
            torch.zeros(
                1,
                model.config.num_key_value_heads,
                3,
                model.config.head_dim,
                dtype=profile.graph_dtype,
            )
            for _ in range(2 * num_layers)
        ],
    )
    names = {
        torch.float32: onnx.TensorProto.FLOAT,
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.bfloat16: onnx.TensorProto.BFLOAT16,
    }
    diagnostics_dir = tmp_path / "diagnostics"
    export_lm_module._export_with_diagnostics(
        wrapper,
        inputs,
        path,
        artifacts_dir=diagnostics_dir,
        verify=profile.verify,
        input_names=["inputs_embeds", "attention_mask", "position_ids", *past_names],
        output_names=["logits", *present_names],
        opset_version=18,
        dynamo=True,
        external_data=True,
    )
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(str(path))
    graph_inputs = {value.name: value for value in graph.graph.input}
    graph_outputs = {value.name: value for value in graph.graph.output}
    expected_type = names[profile.graph_dtype]

    assert graph_inputs["inputs_embeds"].type.tensor_type.elem_type == expected_type
    assert all(graph_inputs[name].type.tensor_type.elem_type == expected_type for name in past_names)
    assert graph_outputs["logits"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert all(graph_outputs[name].type.tensor_type.elem_type == expected_type for name in present_names)
    assert profile.graph_dtype_name == tensor_type
