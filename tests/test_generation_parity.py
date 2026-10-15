from __future__ import annotations

import torch

from kitten2_onnx.export_lm import KittenQwen3ExportWrapper


def _empty_cache(model, dtype):
    return [
        torch.zeros(
            1,
            model.config.num_key_value_heads,
            0,
            model.config.head_dim,
            dtype=dtype,
        )
        for _ in range(2 * model.config.num_hidden_layers)
    ]


def _wrapper_greedy(model, prompt, steps):
    wrapper = KittenQwen3ExportWrapper(model)
    embeddings = model.get_input_embeddings()
    ids = torch.tensor([prompt], dtype=torch.long)
    embeds = embeddings(ids)
    mask = torch.ones(1, len(prompt), dtype=torch.long)
    pos = torch.arange(len(prompt), dtype=torch.long)[None, :]
    past = _empty_cache(model, embeds.dtype)

    generated = []
    with torch.no_grad():
        out = wrapper(embeds, mask, pos, past)
        logits, past = out[0], list(out[1:])
        for _ in range(steps):
            token = int(torch.argmax(logits[0]))
            generated.append(token)
            token_ids = torch.tensor([[token]], dtype=torch.long)
            embeds = embeddings(token_ids)
            total = len(prompt) + len(generated)
            mask = torch.ones(1, total, dtype=torch.long)
            pos = torch.tensor([[total - 1]], dtype=torch.long)
            out = wrapper(embeds, mask, pos, past)
            logits, past = out[0], list(out[1:])
    return generated


def _direct_greedy(model, prompt, steps):
    token_ids = torch.tensor([prompt], dtype=torch.long)
    generated = []
    cache = None
    with torch.no_grad():
        for _ in range(steps):
            out = model(
                input_ids=token_ids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            token = int(torch.argmax(out.logits[0, -1]))
            generated.append(token)
            token_ids = torch.tensor([[token]], dtype=torch.long)
    return generated


def test_greedy_generation_matches(tiny_qwen3):
    prompt = [3, 4, 5]
    assert _wrapper_greedy(tiny_qwen3, prompt, 5) == _direct_greedy(tiny_qwen3, prompt, 5)
