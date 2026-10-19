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


### Exact ternary INT8 storage, measured result

The signed-zero-preserving exact-storage path transforms all 196 eligible LM MatMuls. The measured candidate bundle is 3,994,123,323 bytes (about 3.72 GiB), compared with an 8,177,466,136-byte baseline. It replaces 5,637,144,576 FP32 operand bytes with 1,453,326,336 bytes of INT8 codes and FP32 scales. Code `-2` preserves negative zero exactly.

The candidate at `build/experiments/ternary-exact-lowram-all196-retry16-task0011` passed the trusted ONNX-only diagnostic, strict real-LM parity, bundle validation, the five-case E2E corpus, and fixed-seed synthesis. Strict parity and E2E passed under a user-approved 16 GiB process-tree RSS cap. The highest measured peak was 15,617,724,416 bytes during E2E. The worker requires at least 16 GiB available memory before launch, keeps an 8 GiB available-memory reserve, and limits additional swap use to 2 GiB. At the original 12 GiB cap, strict parity and E2E were stopped by the memory guard.

The LM metadata selects the `low-memory-exact-ternary` CPU session profile. This is an exact-storage format, not execution quantization. The full TTS path was measured within the approved 16 GiB worker cap.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

No `src/` directory is used. Versioning is Git-tag driven through
`setuptools-scm`.


## Text controls and voice selection

`Kitten2OnnxTTS.generate` and `generate_with_metadata` accept `normalize` and `use_emotion`. Normalization is enabled by default. Set `normalize=False` for multilingual text that should not pass through the English-oriented language normalizer. Basic formatting and sentence termination still apply, while Arabic, Chinese, and Devanagari terminal punctuation is preserved.

`use_emotion=None` enables the configured emotion control when recognized expression or vocal-event markup is present. Set it to `True` or `False` to force or suppress the control token. `generate_with_metadata` reports `expression_detected` and `emotion_control_used`.

```python
from kitten2_onnx.tts import Kitten2OnnxTTS

tts = Kitten2OnnxTTS("build/kitten2-onnx-47voices")
print(tts.available_voices)
audio = tts.generate(
    "早上好。你今天感觉怎么样？",
    voice="Bruno",
    preset="stable",
    seed=7,
    normalize=False,
)
```

The synthesis CLI exposes the same controls. `--no-normalize` disables language normalization, while `--use-emotion` and `--no-emotion` override automatic emotion selection.

```bash
python -m kitten2_onnx.synthesize build/kitten2-onnx-47voices \
  "Guten Morgen." --voice Sable --preset stable --seed 7 \
  --max-new-tokens 32 --no-normalize --output build/sable.wav
```

`available_voices` lists the names packaged in the selected bundle. The default voice remains Bruno.

## Building a local 47-voice bundle

The accepted voice set comes from the pinned `voices/voices.json` at model revision `fa8ce5c7547e322a065cad9b813d27a78fad9640` and upstream code revision `36ef3090ca1e59122a81fc23be4a659a502b9d55`. Download the pinned local reference model and all voice assets:

```bash
python scripts/download_reference_model.py KittenML/kitten-tts-2 \
  --revision fa8ce5c7547e322a065cad9b813d27a78fad9640 \
  --output .models/kitten-tts-2 --plain-lm --all-voices
```

Capture voice-specific S3Gen references sequentially. The command writes one `.npz` and metadata file per voice, a resumable `capture_report.json`, and peak RSS information. Re-running it skips valid captures and recaptures missing or stale ones.

```bash
python -m scripts.capture_all_voices \
  --model-dir .models/kitten-tts-2 \
  --output-dir build/voices-all \
  --token-count 32
```

To reuse already exported ONNX graphs, augment a validated source bundle locally:

```bash
python -m scripts.augment_bundle_voices \
  build/baseline-hosted-onnx build/voices-all \
  --model-dir .models/kitten-tts-2 \
  --output build/kitten2-onnx-47voices

python -m kitten2_onnx.validate_bundle build/kitten2-onnx-47voices \
  --output build/reports/validate-47-voice-bundle.json
```

The output directory must not already exist. Augmentation stages a sibling copy, verifies the source bundle, adds only missing voice references, regenerates `bundle.json` file hashes, validates the staged bundle, and compares every unchanged source file hash before atomic promotion. The source bundle is not modified. The workflow only reads local and pinned upstream assets. It does not publish to or modify the Hugging Face repository.

When exporting directly from the pinned model instead of reusing an existing bundle, `python -m kitten2_onnx.export_all ... --all-voices` packages the selected voice set while retaining a single shared decoder graph pair. Without `--all-voices`, export keeps the Bruno-only default.

## Release checks and resource evidence

Run the exact expression parity and multilingual parity cases against the final bundle. The multilingual fixture explicitly sets `normalize=false` for German, Chinese, Arabic, and Hindi.

```bash
python -m scripts.validate_e2e_corpus .models/kitten-tts-2 \
  build/kitten2-onnx-47voices \
  --corpus tests/fixtures/expression_cases.json \
  --output build/reports/e2e-expression.json

python -m scripts.validate_e2e_corpus .models/kitten-tts-2 \
  build/kitten2-onnx-47voices \
  --corpus tests/fixtures/multilingual_cases.json \
  --output build/reports/e2e-multilingual.json
```

The local CPU evidence in `build/reports/feature-parity-47voices.json` records 47 captured voices, 167/167 bundle files verified, preserved source file hashes, exact parity for three expression cases and four multilingual cases, finite waveforms, and a non-default Sable CLI smoke test. The underlying report and resource-guard logs are retained beside it.

| Check | Observed peak RSS | Elapsed | Result |
| --- | ---: | ---: | --- |
| Capture all 47 voices | 3.03 GB | 15m 36s | 47/47 captured and verified |
| Augment and validate bundle | 3.69 GB | 46s | 167/167 files verified, source hashes preserved |
| Expression PyTorch/ONNX parity | 15.88 GB | 17m 23s | Exact token parity, 3/3 cases, finite audio |
| Multilingual PyTorch/ONNX parity | 16.04 GB | 7m 47s | Exact token parity, 4/4 cases, finite audio |
| Sable CLI synthesis | 8.68 GB | 1m 44s | 24 kHz finite audio, 0.74s |

The parity runs used `scripts/run_bounded.py` with an 18 GiB RSS cap and a 6 GiB available-memory reserve. Two earlier expression attempts were stopped by stricter swap and reserve guards; the completed run used a 4.5 GiB additional-swap ceiling. Check available memory before running full CPU parity and retain the guard reports. These measurements depend on the host and are not universal sizing guarantees.

The augmented bundle and reports are local build artifacts. The release boundary is: pin source revisions, download and capture, stage and validate local augmentation, run parity and finite-audio checks, then review the hash and resource reports. Remote publication is a separate, explicit operation and is not part of this workflow.


Run repository tests and lint before accepting a release bundle:

```bash
pytest -q
ruff check .
```
