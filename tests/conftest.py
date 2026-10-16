from __future__ import annotations

import pytest
import torch


@pytest.fixture()
def tiny_qwen3():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("installed transformers does not expose Qwen3")

    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    torch.manual_seed(7)
    model = transformers.Qwen3ForCausalLM(config).float().eval()
    return model


@pytest.fixture()
def student_width_qwen3():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("installed transformers does not expose Qwen3")

    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=1024,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    torch.manual_seed(7)
    model = transformers.Qwen3ForCausalLM(config).float().eval()
    return model
