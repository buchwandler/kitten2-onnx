# Exporter notes

This repository starts with a direct PyTorch ONNX exporter because it gives the
project complete control over the graph contract required by KittenTTS 2.

## Candidate toolchains

### PyTorch `torch.onnx.export`

Current MVP choice.

Advantages:

- direct control over `inputs_embeds`;
- explicit flattening of KV cache tensors;
- one graph can support prefill and decode;
- no dependency on a higher-level model registry;
- straightforward parity against the original Hugging Face model.

The MVP uses the dynamo exporter and opset 18.

### ONNX Runtime GenAI Model Builder

A strong alternative because it has first-class Qwen3 model support and can
build graphs with embeddings excluded. It should be evaluated after the direct
export establishes a known-good graph contract.

Questions to test:

- exact `inputs_embeds` contract;
- zero-length prefill cache support;
- external data layout;
- BF16/FP16 behavior on target providers;
- whether its generated graph is easier to quantize than the direct export.

### Hugging Face / Optimum exporters

Useful references and possibly suitable for a future simplified pipeline. They
are less attractive for the first milestone because Kitten's speaker embedding
replacement is intentionally outside ordinary Qwen text-generation behavior.

### Olive

Useful as an optimization/orchestration layer once a correct Qwen3 graph exists.
Do not mix conversion and optimization questions during initial parity work.

## Relevant open-source patterns

When comparing implementations, focus on projects that export Qwen3 as an
autoregressive audio-token LM rather than only ordinary text chat models. Useful
patterns include:

- explicit past/present KV tensors;
- last-token-only logits;
- prefill with a zero-length cache;
- dynamic sequence/cache dimensions;
- separate embedding lookup;
- separate full-vocabulary and audio-vocabulary LM heads;
- PyTorch-vs-ONNX parity across several cached decode steps.

## Acceptance order

1. model inspection and shape contract;
2. speaker projection parity;
3. LM prefill logits parity;
4. LM cached decode logits parity;
5. deterministic greedy token parity;
6. Kitten logits-processor/sampling parity;
7. S3 decoder parity;
8. end-to-end waveform parity;
9. quantization/optimization.
