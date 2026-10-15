# kitten2-onnx

Experimental ONNX export, validation, and optimization tooling for **KittenTTS 2**.

The first MVP deliberately focuses on the part that is already well understood:
the KittenTTS 2 speech language model. KittenTTS 2 uses a `Qwen3ForCausalLM`
backbone, but speaker identity is injected by replacing the first token embedding
with the output of a learned `spk_proj` head.

This repository therefore exports:

1. `lm.onnx` — Qwen3 transformer + LM head with explicit KV cache and
   **`inputs_embeds`** rather than `input_ids`.
2. `embed_tokens.npy` — the token embedding table used outside the graph.
3. `speaker.onnx` — Kitten's 512-dimensional speaker projection head.
4. `lm.meta.json` / `speaker.meta.json` — graph contracts and model metadata.

The S3 codec/vocoder is intentionally not claimed as complete in this MVP.
`export_codec.py` records the proposed split and exits with a clear message until
the S3Gen flow and vocoder have parity-tested ONNX wrappers.

## Why `inputs_embeds`?

KittenTTS 2 performs speaker injection at prefill:

```text
prompt token ids
      |
      v
embed_tokens
      |
      +---- embedding[0] <- spk_proj(speaker[512])
      |
      v
Qwen3 transformer
      |
      v
S3 codec token logits
```

Keeping the embedding lookup outside `lm.onnx` makes that behavior explicit and
avoids patching a generated ONNX graph.

## Expected source layout

Download or clone `KittenML/kitten-tts-2` so the local directory contains at
least:

```text
kitten-tts-2/
├── config.json
└── lm/
    ├── config.json
    ├── tokenizer.json
    └── model.safetensors   # full/plain checkpoint, not only packed weights
```

The exporter is designed around the full Hugging Face-compatible LM checkpoint.

## Export

```bash
python -m kitten2_onnx.export_lm \
  /path/to/kitten-tts-2 \
  --output build/lm \
  --dtype bf16

python -m kitten2_onnx.export_speaker \
  /path/to/kitten-tts-2 \
  --output build/speaker
```

The LM exporter uses `torch.onnx.export(..., dynamo=True)` and opset 18 by
default. It exports a single graph that accepts an explicit KV cache. Prefill
uses zero-length cache tensors; decode uses the returned cache.

## Inspect

```bash
python scripts/inspect_model.py /path/to/kitten-tts-2
python scripts/inspect_model.py --onnx build/lm/lm.onnx
```

## Parity

Start with deterministic logits, not generated audio:

```bash
python scripts/compare_torch_onnx.py \
  /path/to/kitten-tts-2 \
  build/lm/lm.onnx \
  --steps 5
```

The goal is:

1. prefill last-token logits match;
2. cached decode logits match for several steps;
3. greedy codec token IDs match;
4. only then add Kitten sampling processors;
5. only then export the S3 decoder.

## Quantization

After FP/BF16 parity is established:

```bash
python scripts/quantize.py build/lm/lm.onnx build/lm/lm.int8.onnx
```

Dynamic INT8 is only a convenient first experiment. It is not intended to
preserve Kitten's packed ternary representation.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

No `src/` directory is used. Versioning is Git-tag driven through
`setuptools-scm`.
