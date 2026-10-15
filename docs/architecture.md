# Architecture

## Upstream KittenTTS 2 pipeline

KittenTTS 2 is not the older single-graph Kitten ONNX architecture. Its core
pipeline is:

```text
text + reference voice
        |
        v
generation prompt
        |
        +---------------- speaker embedding [512]
        |                         |
        |                         v
        |                      spk_proj
        |                         |
        v                         v
token IDs -> embeddings -> replace embedding position 0
                         |
                         v
                Qwen3ForCausalLM
                         |
                         v
                  S3 codec tokens
                         |
                         v
                     S3Gen
                         |
                         v
                  24 kHz waveform
```

The upstream implementation uses a standard Hugging Face
`AutoModelForCausalLM`/Qwen3 architecture. `spk_proj` is checkpoint-specific and
is attached separately because the base Qwen3 model does not define it.

## MVP graph split

The first version intentionally uses small, composable contracts:

```text
embed_tokens.npy
      |
token ids -> lookup
      |
speaker[512] -> speaker.onnx -> hidden vector
      |                    |
      +------ replace position 0
                           |
                           v
                       lm.onnx
                  inputs_embeds + KV
                           |
                           v
                         logits
```

`lm.onnx` returns last-position logits and full present KV tensors. A single graph
serves both prefill and decode.

## Why not a monolithic graph?

Keeping orchestration outside ONNX makes the first parity milestone much easier:

- Kitten has custom logits processors and sampling policy.
- speaker injection happens only at prefill.
- the generation loop naturally owns the KV cache.
- the S3 audio decoder is a separate subsystem.
- failures can be localized to LM, speaker projection, sampling, or codec.

A later runtime can fuse or specialize parts after parity is established.

## Codec boundary

Upstream S3Gen performs both reference conditioning and generated-token decoding.
The initial plan is to validate three boundaries independently:

1. reference conditioning;
2. S3 token + conditioning -> mel/flow output;
3. mel -> 24 kHz waveform.

Do not claim an end-to-end ONNX KittenTTS 2 runtime until the waveform path has
objective and listening parity tests.
