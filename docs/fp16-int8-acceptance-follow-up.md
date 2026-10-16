# FP16 and INT8 acceptance follow-up

## Current status

This checkout exposes only `AzureExecutionProvider` and `CPUExecutionProvider`. `nvidia-smi` is unavailable, so CUDA-FP16 candidate validation and the real-model INT8 sweep were not run. No FP16 or INT8 candidate is certified or promoted.

The current `build/experiments/cuda-fp16-input-embeds` directory does not contain `bundle.json`, a decoder, or tokenizer assets, so it is not currently a complete TTS bundle. The `cuda-fp16-input-embeds-full` candidate does contain a bundle manifest. Repackage or restore the first candidate as a complete bundle before running the commands below.

The source model is expected at `.models/kitten-tts-2`, and the original FP32 bundle is `build/export-all-relocated`. Override `MODEL_DIR` if the CUDA host stores the model elsewhere.

## CUDA-FP16 candidates

First confirm that the host can run CUDA ONNX sessions:

```bash
uv run python -c 'import onnxruntime as ort; print(ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'
nvidia-smi
```

For each complete candidate, run bundle validation, size audit, the profile-appropriate quality report, and a synthesis smoke test. The quality report uses a CUDA FP16 PyTorch reference and the original FP32 bundle for side-by-side audio artifacts. Token differences are reported, not treated as an automatic quality failure.

```bash
MODEL_DIR=.models/kitten-tts-2
BASELINE=build/export-all-relocated

for NAME in cuda-fp16-input-embeds cuda-fp16-input-embeds-full; do
  CANDIDATE="build/experiments/$NAME"
  REPORT_NAME="${NAME#cuda-}"
  REPORT="build/reports/$REPORT_NAME"

  if [ ! -f "$CANDIDATE/bundle.json" ]; then
    echo "Skipping incomplete bundle: $CANDIDATE" >&2
    continue
  fi

  mkdir -p "$REPORT"
  uv run python -m kitten2_onnx.validate_bundle "$CANDIDATE" \
    --provider CUDAExecutionProvider \
    --provider CPUExecutionProvider \
    --output "$REPORT/bundle-validation.json"
  uv run python -m scripts.audit_bundle_size "$CANDIDATE" \
    --output "$REPORT/size.json"
  uv run python -m scripts.validate_e2e_corpus "$MODEL_DIR" "$CANDIDATE" \
    --corpus tests/fixtures/e2e_corpus.json \
    --output "$REPORT/quality-e2e.json" \
    --provider CUDAExecutionProvider \
    --provider CPUExecutionProvider \
    --reference-device cuda \
    --reference-dtype float16 \
    --quality-report \
    --reference-bundle "$BASELINE"
  uv run python -m kitten2_onnx.synthesize "$CANDIDATE" "Hello from Kitten." \
    --seed 42 \
    --max-new-tokens 64 \
    --provider CUDAExecutionProvider \
    --provider CPUExecutionProvider \
    --output "$REPORT/synthesis-smoke.wav"
done
```

Expected report directories are `build/reports/fp16-input-embeds/` and `build/reports/fp16-input-embeds-full/`. The E2E command also writes candidate and original WAVs under each report directory's `quality-e2e-audio/` subdirectory. These reports require manual quality review. Do not claim equivalence based on numeric reports or token parity alone.

## Bounded mixed-precision INT8 sweep

The helper verifies the canonical graph inventory and proposes four candidates that keep the LM head and the final 4, 8, 14, or 20 transformer blocks in FP32. It uses the existing `int8-dynamic-chunked` optimizer, writes a machine-readable `results.json`, and refuses to reuse existing output roots.

Run this only on the CUDA-capable follow-up host as requested. The transformation and quality diagnostics use the CPU profile; the search is intentionally not run in this checkout.

```bash
MODEL_DIR=.models/kitten-tts-2
BASELINE=build/export-all-relocated

uv run python -m scripts.search_int8_mixed_precision "$BASELINE" \
  --model-dir "$MODEL_DIR" \
  --corpus tests/fixtures/e2e_corpus.json \
  --output-root build/experiments/int8-mixed-precision-bounded \
  --reports-root build/reports/int8-mixed-precision-bounded
```

The inventory and candidate summary will be written to:

- `build/experiments/int8-mixed-precision-bounded/inventory.json`
- `build/experiments/int8-mixed-precision-bounded/results.json`

Each candidate gets a separate bundle directory and report directory. The default run records the `very_short` quality diagnostic and reviewable audio. Add `--full-corpus` only when a full-corpus evaluation is explicitly desired. Result status remains diagnostic and requires manual review; it does not assert that an INT8 model is as good as the original.
