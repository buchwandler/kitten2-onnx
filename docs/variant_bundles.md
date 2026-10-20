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

The exporter records ONNX Runtime 1.31.0 as the tested minimum because it is the only runtime version tested for this path. Earlier ONNX Runtime versions and other execution providers are not verified. The kernel benchmarks do not establish full-model parity, end-to-end audio parity, or release suitability. Do not publish a variant until its reference, storage, and resource checks pass.
