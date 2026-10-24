# Packed and emb4 variant bundles

The original `kitten2-onnx-bundle-v1` format remains unchanged for the full-weight export path and existing exact-int8 bundles. Variant exports use `kitten2-onnx-bundle-v2`, which records source weight semantics separately from ONNX storage.

## Build flow

Export a selected upstream variant with one shared vocabulary asset and separate transformer and output-head graphs:

```bash
python -m kitten2_onnx.export_all MODEL_DIR \
  --weights packed \
  --all-voices \
  --output build/packed-source
```

Use `--weights emb4` for the lossy upstream emb4 vocabulary. Optimize each v2 source bundle to q4 before runtime validation:

```bash
python scripts/optimize_bundle.py build/packed-source build/packed-q4 --lm ternary-nbits4 --validate
```

The q4 method is required for v2 packed and emb4 bundles. It records four-bit `MatMulNBits`, block size, source group size 128, integer zero point 8, graph hashes, source weight hash, and runtime provenance. It rejects partial conversion through MatMul exclusions. Keep source and output directories separate.

## v2 metadata contract

The LM entry identifies `weight_variant`, `core_model`, `head_model`, and the vocabulary asset. `storage.transformer` describes the ONNX execution representation. `storage.vocabulary` records either exact `bfloat16-u16` storage for packed or the upstream compact `kitten-emb4` payload with `lossy: true`. Source weight SHA256, model revision, emb4 reconstruction version when relevant, and the lowest ONNX Runtime version actually tested are retained in LM metadata.

The emb4 asset contains only the vocabulary tier map, codebooks, packed codes, and scales. At runtime, one `SharedVocabulary` instance reconstructs the BF16 table for token lookup and the output projection. The transformer core and output head contain no vocabulary initializer. Validation and the size audit reject multiple vocabulary assets or any vocabulary-sized initializer in those graphs, and report logical parameters, physical copies, and stored bytes.

The exporter records ONNX Runtime 1.31.0 as the tested minimum because the complete v2 bundle workflow was exercised there. Separate strict q4 LM cache checks under ONNX Runtime 1.22.1 and 1.23.2 reproduced the same failure. Other execution providers remain unverified. The kernel benchmarks do not establish full-model parity, end-to-end audio parity, or release suitability. Do not publish a variant until its reference, storage, and resource checks pass.

## Fail-closed local release qualification

Use the tracked release builder after the source model and the selected variant reference are pinned locally. Each invocation requires new output and report directories; existing candidates and reports are never replaced.

```bash
python -m scripts.build_variant_release .models/kitten-tts-2 \
  --weights packed \
  --variant-reference build/reference-variants/packed \
  --output build/releases/packed \
  --reports build/release-reports/packed

python -m scripts.build_variant_release .models/kitten-tts-2 \
  --weights emb4 \
  --variant-reference build/reference-variants/emb4 \
  --output build/releases/emb4 \
  --reports build/release-reports/emb4
```

The builder stages the selected weights without changing either source directory, exports all indexed voices, applies `ternary-nbits4`, and runs bundle validation, a size audit, strict 32-step LM parity, expression and multilingual reference checks, and finite 24 kHz CPU synthesis for every source-indexed voice. It also validates and synthesizes the existing exact-int8 bundle, then verifies that its tree hash is unchanged. Each subprocess runs through the resource guard. Reports, per-stage resource records, and logs remain under the selected reports directory.

Promotion is atomic and happens only after every gate passes. On failure, the temporary candidate is removed and `release_status.json` records the failed gate. The builder does not publish or upload artifacts. For a retry, select a fresh reports directory so the failed attempt remains available for diagnosis.

The currently measured packed and emb4 q4 candidates fail the unchanged KV-cache parity gate; see [variant benchmark results](variant_benchmarks.md). They are diagnostic bundles, not verified release candidates. Do not treat their existing size, token, or audio smoke results as release qualification.

The candidate destinations are `build/releases/packed` and `build/releases/emb4`; their reports belong under `build/release-reports/packed` and `build/release-reports/emb4`. Neither verified destination currently exists. The measured bundles under `build/variant-bundles/` remain diagnostic artifacts.
