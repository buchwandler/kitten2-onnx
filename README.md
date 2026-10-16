# kitten2-onnx

Experimental ONNX export, validation, and optimization tooling for **KittenTTS 2**.

The first MVP deliberately focuses on the part that is already well understood:
the KittenTTS 2 speech language model. KittenTTS 2 uses a `Qwen3ForCausalLM`
backbone, but speaker identity is injected by replacing the first token embedding
with the output of a learned `spk_proj` head.

The project exports complete KittenTTS 2 ONNX bundles with speaker projection, Qwen3 language model, Flow, and mel-to-waveform graphs. `Kitten2OnnxTTS` runs the inference pipeline.

The LM supports two graph input contracts:

- `inputs_embeds` is the default and keeps token lookup outside the LM graph. For a BF16 source checkpoint, the external table is stored as `embed_tokens.bf16.npy` using packed `uint16` BF16 bits.
- `input_ids` is an opt-in mode that performs tied token lookup and position-zero speaker injection inside the graph. It omits the separate embedding file.

Both modes use the same tied vocabulary weight for token lookup and the LM head. The input_ids exporter normalizes the transposed head weight into one shared initializer. The CPU benchmark currently favors the default mode for latency, so input_ids is not the default.

Older LM metadata without `input_mode` continues to use `inputs_embeds`.
The `cuda-fp16` bundle keeps the speaker projection and S3 decoder graphs in FP32. `Kitten2OnnxLM` casts the FP32 speaker output to the LM I/O dtype; the bundle metadata records this boundary and the decoder precision choice. CUDA execution still requires a CUDA-enabled environment.

## LM input contracts

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

In the default mode, the runtime projects the speaker vector and replaces the first
embedding before calling `lm.onnx`. The opt-in `input_ids` mode performs the same
lookup and injection inside the graph.

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
  --profile dev-fp32

python -m kitten2_onnx.export_speaker \
  /path/to/kitten-tts-2 \
  --output build/speaker
```

The full bundle defaults to `inputs_embeds`. To build the opt-in `input_ids` variant:

```bash
python -m kitten2_onnx.export_all \
  /path/to/kitten-tts-2 \
  --output build/input-ids \
  --profile dev-fp32 \
  --input-mode input_ids \
  --validate
```

The LM exporter uses `torch.onnx.export(..., dynamo=True)` and opset 18 by
default. It exports a single graph that accepts an explicit KV cache. Prefill
uses zero-length cache tensors; decode uses the returned cache.

## Inspect

```bash
python scripts/inspect_model.py /path/to/kitten-tts-2
python scripts/inspect_model.py --onnx build/lm/lm.onnx
```



## Initialize a 0.6B student

Create an initial Kitten-compatible student checkpoint from the Hugging Face PyTorch Qwen3-0.6B base model and the pinned local teacher assets:

```bash
uv run python scripts/build_qwen3_06b_student.py \
  --base Qwen/Qwen3-0.6B-Base \
  --teacher .models/kitten-tts-2 \
  --output .models/kitten-tts-2-qwen3-0.6b-student-init
```

Use `--base-revision` to select a specific base checkpoint revision. The resolved base revision and pinned Kitten teacher revision are recorded in `model_revision.json`.

This command creates an initialization checkpoint, not a trained or usable TTS model. It preserves the shared text-token rows, initializes the Kitten-specific vocabulary rows, copies the Kitten tokenizer and non-LM assets, and adds a randomly initialized speaker projection. Distillation and speech-quality evaluation are required before treating the student as a speech model or using it as a release candidate.
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
