from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from kitten2_onnx.export_lm import (
    KittenQwen3ExportWrapper,
    _export_with_diagnostics,
    _io_names,
    export_lm,
)


def _empty_cache(model: torch.nn.Module) -> list[torch.Tensor]:
    return [
        torch.zeros(
            1,
            model.config.num_key_value_heads,
            0,
            model.config.head_dim,
            dtype=model.get_input_embeddings().weight.dtype,
        )
        for _ in range(2 * model.config.num_hidden_layers)
    ]


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _feed(embeds, attention_mask, position_ids, past):
    past_names, _ = _io_names(len(past) // 2)
    return {
        "inputs_embeds": _as_numpy(embeds),
        "attention_mask": _as_numpy(attention_mask),
        "position_ids": _as_numpy(position_ids),
        **{name: _as_numpy(value) for name, value in zip(past_names, past, strict=True)},
    }


def _assert_parity(actual, expected):
    np.testing.assert_allclose(actual[0], expected[0].detach().cpu().numpy(), rtol=0, atol=1e-4)
    np.testing.assert_array_equal(
        actual[0].argmax(axis=-1), expected[0].detach().cpu().numpy().argmax(axis=-1)
    )
    for actual_cache, expected_cache in zip(actual[1:], expected[1:], strict=True):
        np.testing.assert_allclose(
            actual_cache,
            expected_cache.detach().cpu().numpy(),
            rtol=0,
            atol=1e-4,
        )


def _assert_cache_shapes(cache, num_layers, kv_heads, head_dim, expected_length):
    expected_shape = (1, kv_heads, expected_length, head_dim)
    assert len(cache) == 2 * num_layers
    for index, tensor in enumerate(cache):
        assert tuple(tensor.shape) == expected_shape, (
            f"cache tensor {index} has shape {tensor.shape}"
        )


def _dim_symbol(value_info, axis):
    dimension = value_info.type.tensor_type.shape.dim[axis]
    assert dimension.dim_param, f"{value_info.name} axis {axis} is not dynamic"
    return dimension.dim_param


def _assert_graph_dimensions(model_proto, past_names, present_names):
    graph_inputs = {value.name: value for value in model_proto.graph.input}
    graph_outputs = {value.name: value for value in model_proto.graph.output}
    input_names = ["inputs_embeds", "attention_mask", "position_ids", *past_names]
    batch_dimensions = {
        (
            dimension.dim_param,
            dimension.dim_value,
        )
        for name in input_names
        for dimension in [graph_inputs[name].type.tensor_type.shape.dim[0]]
    }
    assert len(batch_dimensions) == 1
    assert _dim_symbol(graph_inputs["inputs_embeds"], 1) == _dim_symbol(
        graph_inputs["position_ids"], 1
    )
    _dim_symbol(graph_inputs["attention_mask"], 1)
    past_symbols = {_dim_symbol(graph_inputs[name], 2) for name in past_names}
    assert len(past_symbols) == 1
    output_cache_symbols = {_dim_symbol(graph_outputs[name], 2) for name in present_names}
    assert len(output_cache_symbols) == 1


def _run_case(session, wrapper, embeds, attention_mask, position_ids, past):
    with torch.inference_mode():
        expected = wrapper(embeds, attention_mask, position_ids, past)
    actual = session.run(
        None,
        _feed(embeds, attention_mask, position_ids, past),
    )
    _assert_parity(actual, expected)
    return actual


def test_tiny_qwen3_onnx_prefill_dynamic_cache_and_decode(tmp_path, tiny_qwen3, monkeypatch):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    source_dir = tmp_path / "source"
    source_lm_dir = source_dir / "lm"
    source_lm_dir.mkdir(parents=True)
    (source_lm_dir / "config.json").write_text(
        '{"model_type":"qwen3","dtype":"bfloat16"}', encoding="utf-8"
    )
    (source_lm_dir / "model.safetensors").write_bytes(b"tiny-model-placeholder")
    model = tiny_qwen3
    monkeypatch.setattr(
        "kitten2_onnx.export_lm.load_hf_lm",
        lambda repo_dir, dtype: model.to(dtype),
    )
    export_dir = tmp_path / "exported-lm"
    diagnostics_dir = tmp_path / "onnx-diagnostics"
    path = export_lm(
        source_dir,
        export_dir,
        profile_name="dev-fp32",
        diagnostics_dir=diagnostics_dir,
    )
    wrapper = KittenQwen3ExportWrapper(model).eval()
    hidden = int(model.config.hidden_size)
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    num_layers = int(model.config.num_hidden_layers)
    past_names, present_names = _io_names(num_layers)

    assert any(file.suffix == ".md" for file in diagnostics_dir.iterdir())
    assert (export_dir / "embed_tokens.npy").is_file()
    assert json.loads((export_dir / "lm.meta.json").read_text(encoding="utf-8"))["runtime_profile"] == "dev-fp32"
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False)
    locations = {
        entry.value
        for tensor in graph.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    assert locations
    assert all(not Path(location).is_absolute() for location in locations)
    assert all((export_dir / location).is_file() for location in locations)
    _assert_graph_dimensions(graph, past_names, present_names)

    relocated_dir = tmp_path / "relocated-lm"
    shutil.copytree(export_dir, relocated_dir)
    session = ort.InferenceSession(
        str(relocated_dir / "lm.onnx"), providers=["CPUExecutionProvider"]
    )

    for past_length in (0, 3, 32):
        for seq_length in (1, 4, 17):
            embeds = torch.randn(1, seq_length, hidden)
            attention_mask = torch.ones(1, past_length + seq_length, dtype=torch.long)
            position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long)[
                None, :
            ]
            past = [torch.randn(1, kv_heads, past_length, head_dim) for _ in range(2 * num_layers)]
            outputs = _run_case(session, wrapper, embeds, attention_mask, position_ids, past)
            _assert_cache_shapes(
                outputs[1:], num_layers, kv_heads, head_dim, past_length + seq_length
            )

    embeddings = model.get_input_embeddings()
    prompt_ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    prompt_embeds = embeddings(prompt_ids)
    prompt_length = prompt_ids.shape[1]
    result = _run_case(
        session,
        wrapper,
        prompt_embeds,
        torch.ones(1, prompt_length, dtype=torch.long),
        torch.arange(prompt_length, dtype=torch.long)[None, :],
        _empty_cache(model),
    )
    _assert_cache_shapes(result[1:], num_layers, kv_heads, head_dim, prompt_length)

    with torch.inference_mode():
        torch_outputs = wrapper(
            prompt_embeds,
            torch.ones(1, prompt_length, dtype=torch.long),
            torch.arange(prompt_length, dtype=torch.long)[None, :],
            _empty_cache(model),
        )
    torch_logits, torch_cache = torch_outputs[0], list(torch_outputs[1:])
    onnx_logits, onnx_cache = result[0], list(result[1:])

    for step in range(5):
        assert int(onnx_logits.argmax(axis=-1)[0]) == int(torch_logits.argmax(dim=-1)[0])
        token_id = int(torch_logits.argmax(dim=-1)[0])
        token_embed = embeddings(torch.tensor([[token_id]], dtype=torch.long))
        total_length = prompt_length + step + 1
        attention_mask = torch.ones(1, total_length, dtype=torch.long)
        position_ids = torch.tensor([[total_length - 1]], dtype=torch.long)
        onnx_result = session.run(
            None,
            _feed(token_embed, attention_mask, position_ids, onnx_cache),
        )
        with torch.inference_mode():
            torch_result = wrapper(token_embed, attention_mask, position_ids, torch_cache)
        _assert_parity(onnx_result, torch_result)
        _assert_cache_shapes(onnx_result[1:], num_layers, kv_heads, head_dim, total_length)
        _assert_cache_shapes(torch_result[1:], num_layers, kv_heads, head_dim, total_length)
        onnx_logits, onnx_cache = onnx_result[0], list(onnx_result[1:])
        torch_logits, torch_cache = torch_result[0], list(torch_result[1:])


def test_export_failure_preserves_diagnostic(tmp_path, monkeypatch):
    export_options = {}

    def fail_export(*args, **kwargs):
        export_options.update(kwargs)
        raise RuntimeError("synthetic export failure")

    monkeypatch.setattr(torch.onnx, "export", fail_export)
    diagnostics_dir = tmp_path / "debug"
    with pytest.raises(RuntimeError, match="synthetic export failure"):
        _export_with_diagnostics(
            torch.nn.Identity(),
            (),
            tmp_path / "failed.onnx",
            artifacts_dir=diagnostics_dir,
        )

    failure_report = (diagnostics_dir / "export-failure.txt").read_text(encoding="utf-8")
    assert "RuntimeError: synthetic export failure" in failure_report
    assert export_options["report"] is True
    assert export_options["verify"] is True
    assert export_options["artifacts_dir"] == str(diagnostics_dir)
