from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from kitten2_onnx.export_lm import (
    KittenQwen3ExportWrapper,
    KittenQwen3InputIdsExportWrapper,
    _dummy_input_ids,
    _dummy_inputs,
)
from kitten2_onnx.export_speaker import build_speaker_projection


def test_student_width_export_wrappers_accept_speaker_hidden_and_cache(student_width_qwen3):
    model = student_width_qwen3
    config = model.config
    assert config.hidden_size == 1024
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128

    input_ids = torch.tensor([[3, 4]], dtype=torch.long)
    speaker_hidden = torch.randn(1, config.hidden_size)
    attention_mask = torch.ones(1, 2, dtype=torch.long)
    position_ids = torch.arange(2, dtype=torch.long)[None, :]
    empty_cache = [torch.zeros(1, 8, 0, 128) for _ in range(2)]

    with torch.inference_mode():
        by_ids = KittenQwen3InputIdsExportWrapper(model)(
            input_ids, speaker_hidden, attention_mask, position_ids, empty_cache
        )
        token_embeddings = model.get_input_embeddings()(input_ids)
        injected_embeddings = torch.where(
            position_ids.eq(0).unsqueeze(-1), speaker_hidden.unsqueeze(1), token_embeddings
        )
        by_embeddings = KittenQwen3ExportWrapper(model)(
            injected_embeddings, attention_mask, position_ids, empty_cache
        )

    assert by_ids[0].shape == (1, config.vocab_size)
    assert len(by_ids[1:]) == 2
    assert all(tuple(cache.shape) == (1, 8, 2, 128) for cache in by_ids[1:])
    torch.testing.assert_close(by_ids[0], by_embeddings[0])


def test_student_export_dummy_inputs_follow_full_architecture_dimensions():
    config = SimpleNamespace(
        hidden_size=1024,
        intermediate_size=3072,
        vocab_size=158241,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )

    inputs_embeds, attention_mask, position_ids, past = _dummy_inputs(config, torch.float32)
    input_ids, speaker_hidden, ids_attention_mask, ids_position_ids, ids_past = _dummy_input_ids(
        config, torch.float32
    )

    assert inputs_embeds.shape == (1, 4, 1024)
    assert attention_mask.shape == (1, 7)
    assert position_ids.shape == (1, 4)
    assert len(past) == 56
    assert all(tuple(cache.shape) == (1, 8, 3, 128) for cache in past)
    assert input_ids.shape == (1, 4)
    assert int(input_ids.max()) < 158241
    assert speaker_hidden.shape == (1, 1024)
    assert ids_attention_mask.shape == (1, 7)
    assert ids_position_ids.shape == (1, 4)
    assert len(ids_past) == 56
    assert all(tuple(cache.shape) == (1, 8, 3, 128) for cache in ids_past)


def test_student_speaker_projection_uses_configured_hidden_width(tmp_path):
    lm_dir = tmp_path / "lm"
    lm_dir.mkdir()
    (lm_dir / "config.json").write_text(json.dumps({"hidden_size": 1024}), encoding="utf-8")
    save_file(
        {
            "spk_proj.0.weight": torch.randn(1024, 512),
            "spk_proj.0.bias": torch.randn(1024),
            "spk_proj.1.weight": torch.ones(1024),
            "spk_proj.1.bias": torch.zeros(1024),
        },
        str(lm_dir / "model.safetensors"),
    )

    projection = build_speaker_projection(tmp_path)
    output = projection(torch.randn(2, 512))

    assert projection[0].in_features == 512
    assert projection[0].out_features == 1024
    assert output.shape == (2, 1024)
