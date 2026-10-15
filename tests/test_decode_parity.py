from __future__ import annotations

import torch

from kitten2_onnx.export_lm import KittenQwen3ExportWrapper


def _flatten(cache):
    return [tensor for layer in cache.to_legacy_cache() for tensor in layer]


def test_cached_decode_wrapper_matches_qwen3(tiny_qwen3):
    model = tiny_qwen3
    wrapper = KittenQwen3ExportWrapper(model)

    prompt = torch.tensor([[3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        prefill = model(input_ids=prompt, use_cache=True)

    token = torch.tensor([[6]], dtype=torch.long)
    embeds = model.get_input_embeddings()(token)
    past = _flatten(prefill.past_key_values)
    mask = torch.ones(1, prompt.shape[1] + 1, dtype=torch.long)
    pos = torch.tensor([[prompt.shape[1]]], dtype=torch.long)

    with torch.no_grad():
        wrapped = wrapper(embeds, mask, pos, past)[0]
        direct = (
            model(
                input_ids=token,
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            .logits[:, -1, :]
            .float()
        )

    torch.testing.assert_close(wrapped, direct, rtol=1e-5, atol=1e-5)
