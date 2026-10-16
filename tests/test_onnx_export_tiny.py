from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from kitten2_onnx.export_lm import (
    KittenQwen3ExportWrapper,
    KittenQwen3InputIdsExportWrapper,
    _export_with_diagnostics,
    _io_names,
    _share_tied_vocabulary_weight,
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
    lm_metadata = json.loads((export_dir / "lm.meta.json").read_text(encoding="utf-8"))
    assert lm_metadata["runtime_profile"] == "dev-fp32"
    assert lm_metadata["embedding_storage_dtype"] == "bfloat16-u16"
    embedding = np.load(export_dir / lm_metadata["embedding_file"], allow_pickle=False)
    assert embedding.dtype == np.uint16
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


def test_export_lm_input_ids_mode_omits_external_embedding_table(tmp_path, tiny_qwen3, monkeypatch):
    onnx = pytest.importorskip("onnx")
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
    export_dir = tmp_path / "input-ids-export"
    path = export_lm(
        source_dir,
        export_dir,
        profile_name="dev-fp32",
        input_mode="input_ids",
        diagnostics_dir=tmp_path / "diagnostics",
    )

    metadata = json.loads((export_dir / "lm.meta.json").read_text(encoding="utf-8"))
    assert metadata["input_mode"] == "input_ids"
    assert "embedding_file" not in metadata
    assert metadata["input_names"][:4] == [
        "input_ids",
        "speaker_hidden",
        "attention_mask",
        "position_ids",
    ]
    assert not (export_dir / "embed_tokens.npy").exists()
    assert not (export_dir / "embed_tokens.bf16.npy").exists()
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False)
    vocabulary_weight_count = sum(
        int(np.prod(tensor.dims)) == int(model.config.vocab_size * model.config.hidden_size)
        for tensor in graph.graph.initializer
    )
    assert vocabulary_weight_count == 1


def test_input_ids_graph_injects_speaker_and_shares_tied_embedding(tmp_path, tiny_qwen3):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = tiny_qwen3.eval()
    wrapper = KittenQwen3InputIdsExportWrapper(model).eval()
    layers = int(model.config.num_hidden_layers)
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    hidden = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)
    past_names, present_names = _io_names(layers)
    input_names = ["input_ids", "speaker_hidden", "attention_mask", "position_ids", *past_names]
    output_names = ["logits", *present_names]
    batch_dim = torch.export.Dim("batch", min=1)
    seq_dim = torch.export.Dim("seq", min=1)
    total_dim = torch.export.Dim("total", min=1)
    past_dim = torch.export.Dim("past", min=0)
    dynamic_shapes = {
        "input_ids": {0: batch_dim, 1: seq_dim},
        "speaker_hidden": {0: batch_dim},
        "attention_mask": {0: batch_dim, 1: total_dim},
        "position_ids": {0: batch_dim, 1: seq_dim},
        "past": [{0: batch_dim, 2: past_dim} for _ in range(len(past_names))],
    }
    dummy = (
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        torch.randn(1, hidden),
        torch.ones(1, 7, dtype=torch.long),
        torch.arange(3, 7, dtype=torch.long)[None, :],
        [torch.zeros(1, kv_heads, 3, head_dim) for _ in range(2 * layers)],
    )
    path = tmp_path / "input_ids.onnx"
    _export_with_diagnostics(
        wrapper,
        dummy,
        path,
        verify=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        dynamo=True,
        external_data=True,
    )
    assert _share_tied_vocabulary_weight(path, vocab_size=vocab, hidden_size=hidden)

    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False)
    vocabulary_initializers = [
        tensor for tensor in graph.graph.initializer if int(np.prod(tensor.dims)) == vocab * hidden
    ]
    assert len(vocabulary_initializers) == 1
    vocabulary_weight = vocabulary_initializers[0]
    consumers = {node.op_type for node in graph.graph.node if vocabulary_weight.name in node.input}
    assert "Gather" in consumers
    assert consumers & {"Transpose", "MatMul"}
    assert {value.name for value in graph.graph.input} >= set(input_names)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    torch.manual_seed(13)

    def run(input_ids, speaker_hidden, past, position):
        attention_mask = torch.ones(1, position + input_ids.shape[1], dtype=torch.long)
        position_ids = torch.arange(
            position, position + input_ids.shape[1], dtype=torch.long
        )[None, :]
        with torch.inference_mode():
            expected = wrapper(input_ids, speaker_hidden, attention_mask, position_ids, past)
        feed = {
            "input_ids": input_ids.numpy(),
            "speaker_hidden": speaker_hidden.numpy(),
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
            **dict(zip(past_names, [value.numpy() for value in past], strict=True)),
        }
        actual = session.run(None, feed)
        _assert_parity(actual, expected)
        return actual

    empty_cache = [torch.zeros(1, kv_heads, 0, head_dim) for _ in range(2 * layers)]
    prefill_outputs = []
    for length in (2, 5):
        input_ids = torch.arange(1, length + 1, dtype=torch.long)[None, :] % vocab
        speaker_hidden = torch.randn(1, hidden)
        output = run(input_ids, speaker_hidden, empty_cache, 0)
        _assert_cache_shapes(output[1:], layers, kv_heads, head_dim, length)
        prefill_outputs.append((input_ids, output))

    input_ids, output = prefill_outputs[-1]
    decode_id = torch.tensor([[7 % vocab]], dtype=torch.long)
    decode_speaker = torch.randn(1, hidden)
    decode_past = [torch.from_numpy(value) for value in output[1:]]
    decode_output = run(decode_id, decode_speaker, decode_past, input_ids.shape[1])
    zero_speaker_output = run(
        decode_id,
        torch.zeros_like(decode_speaker),
        decode_past,
        input_ids.shape[1],
    )
    np.testing.assert_array_equal(decode_output[0], zero_speaker_output[0])
    _assert_cache_shapes(decode_output[1:], layers, kv_heads, head_dim, input_ids.shape[1] + 1)


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
