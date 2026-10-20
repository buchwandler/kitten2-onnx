# Packed and emb4 q4 benchmark status

## Scope and provenance

The local real-model checks used `KittenML/kitten-tts-2` revision `fa8ce5c7547e322a065cad9b813d27a78fad9640` with the matching upstream code revision `36ef3090ca1e59122a81fc23be4a659a502b9d55`. The packed and emb4 checkpoint files were downloaded selectively into ignored `build/reference-variants/` directories. They were not published.

Environment: Linux x86_64 under WSL2, Python 3.13.15, ONNX 1.23.2, ONNX Runtime 1.31.0, CPUExecutionProvider. Other ORT versions and providers remain untested.

## q4 build and size

Both 47-voice bundle-v2 candidates converted all 196 transformer MatMuls to q4 MatMulNBits, reported zero non-ternary initializer conversions, and passed full bundle validation. The q4 external body file was 754,683,904 bytes for each variant.

| Variant | Shared vocabulary asset | LM directory bytes | Full 47-voice bundle bytes |
| --- | ---: | ---: | ---: |
| packed | BF16-u16 NPY, 648,155,264 bytes | 1,406,879,325 | 2,078,395,122 |
| emb4 | compact SafeTensors, 170,430,275 bytes | 929,154,352 | 1,600,670,180 |

The packed bundle stores one vocabulary table. The emb4 bundle marks the vocabulary as lossy and records reconstruction version `kitten2-tl2-emb4-v1`. These figures are measured q4 bundle sizes, not upstream checkpoint sizes or q2 target claims.
The packed and emb4 staging helpers copy bundle files instead of hard-linking mutable manifests or ONNX external data. The base exact-int8 bundle manifest was restored to bundle-v1 and passed validation after the helper repair.

## Real LM reference comparison

`scripts/benchmark_variant_lm.py` compared each q4 bundle with its corresponding upstream packed or emb4 PyTorch loader using the same 294-token Bruno prompt and 32 greedy decode steps. Existing limits were retained: logits `rtol=1e-4, atol=2e-4`; cache `rtol=1e-4, atol=2e-3`.

| Variant | Prefill and decode logits | Greedy tokens | KV cache |
| --- | --- | --- | --- |
| packed | Allclose; maximum absolute error `4.58e-5` at prefill and `3.05e-5` during decode | 32/32 matched | **Failed** existing allclose limits. Prefill had 9 violating elements in 4 value-cache tensors, maximum absolute error `0.00579834`; relative L2 error was `1.85e-6`. Decode cache also exceeded tolerance. |
| emb4 | Allclose; maximum absolute error `5.72e-5` at prefill and `3.05e-5` during decode | 32/32 matched | **Failed** existing allclose limits. Prefill had 12 violating elements in 4 value-cache tensors, maximum absolute error `0.00579834`; relative L2 error was `1.91e-6`. Decode cache also exceeded tolerance. |

The parity benchmark therefore does **not** pass. The cache tolerances were not weakened. Both q4 exports also emitted a PyTorch ONNX exporter output-accuracy warning. Treat the small relative cache error as a diagnostic, not as acceptance.
An unquantized packed FP32 source-bundle control passed the same 32-step comparison, including all 56 prefill and decode cache tensors. The maximum cache absolute error was `0.00231934`, within the unchanged allclose rule. The packed q4 candidate rebuilt with `accuracy_level=0` and another using block size 16 both passed bundle validation but reproduced the q4 cache failures. This isolates the packed mismatch to the q4 MatMulNBits path for the tested CPU configuration. It does not establish behavior on other providers.

The reference benchmark measured q4 ORT decode at approximately 116.7 ms/token on this host. The PyTorch reference took approximately 756 ms/token. These measurements are specific to the benchmark setup and are not an end-to-end real-time factor claim.

## Emb4 reconstruction reference

The full emb4 checkpoint was decoded with the port and compared element-for-element to the pinned upstream `kittenml.kittentts2.tl2_emb4.dequant_embedding` implementation. The BF16 tables matched exactly. Against the full BF16 vocabulary, relative L2 error was `0.11391475`, maximum absolute error was `0.06787109`, and per-tier relative L2 errors were `0.01945937`, `0.11808104`, and `0.11583712`.

## Voice, feature, and resource smoke checks

The packed q4 v2 bundle generated finite, non-empty 24 kHz audio for all 47 packaged voices. The multilingual and expression fixture cases also generated finite audio, a Bruno sample from the emb4 q4 v2 bundle and the restored exact-int8 base both passed the synthesis CLI's audio validation. These are runtime smoke checks, not upstream variant-specific audio parity or a listening-quality review.

The guarded packed all-voice and feature smoke took 735 seconds and peaked at 5,094,522,880 bytes RSS. The real LM parity processes peaked at 14,274,568,192 bytes RSS for packed and 13,428,240,384 bytes for emb4; the FP32 control peaked at 16,133,144,576 bytes. The full emb4 embedding comparison peaked at 5,330,079,744 bytes RSS. The exact-int8 synthesis smoke took 195 seconds and peaked at 8,661,098,496 bytes RSS. Machine-readable outputs and guard records are under `build/reports/` and `build/variant-bundles/*-reports/`.

## Acceptance status

The q4 builds, bundle validation, measured sizes, full-checkpoint emb4 reconstruction, 32-step logits, and runtime audio smoke checks are recorded. Acceptance remains open because both variants fail the unchanged KV-cache allclose limit, and upstream variant-specific audio/reference checks have not passed. Do not publish either variant or describe these checks as full parity.
