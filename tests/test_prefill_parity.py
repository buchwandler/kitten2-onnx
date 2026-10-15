from __future__ import annotations

import torch

from kitten2_onnx.export_lm import KittenQwen3ExportWrapper


def test_prefill_wrapper_matches_qwen3(tiny_qwen3):
    model = tiny_qwen3
    wrapper = KittenQwen3ExportWrapper(model)

    ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    embeds = model.get_input_embeddings()(ids)
    mask = torch.ones(1, ids.shape[1], dtype=torch.long)
    pos = torch.arange(ids.shape[1], dtype=torch.long)[None, :]
    empty = [
        torch.zeros(
            1,
            model.config.num_key_value_heads,
            0,
            model.config.head_dim,
            dtype=embeds.dtype,
        )
        for _ in range(2 * model.config.num_hidden_layers)
    ]

    with torch.no_grad():
        wrapped = wrapper(embeds, mask, pos, empty)[0]
        direct = model(input_ids=ids, use_cache=True).logits[:, -1, :].float()

    torch.testing.assert_close(wrapped, direct, rtol=1e-5, atol=1e-5)
