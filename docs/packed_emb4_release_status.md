# Packed and emb4 release backend status

## Decision

Do not use `ternary-nbits4` / ONNX Runtime `MatMulNBits` as the packed or emb4 release backend. Keep q4 available for diagnostics only. The strict KV-cache tolerances remain unchanged. The next backend must store compact ternary values and execute reconstructed weights through standard ONNX `MatMul`.

## Evidence retained

The tracked [variant benchmark report](variant_benchmarks.md) records the real-model q4 bundle measurements and parity results. The detailed packed ablation matrix is retained locally at `build/variant-bundles/packed-q4-reports/ablation-matrix.json` (under ignored `build/`). Do not overwrite this report or its candidate directories.

The matrix records ten packed configurations. All-q4 candidates failed strict cache parity under ONNX Runtime 1.31.0, 1.23.2, and 1.22.1. Tested fallbacks included v-projection, selected Q/K/V and attention layers, all attention, and early/late attention subsets. The only passing mixed-storage diagnostic restored 115 of 196 MatMuls to FP32, yielding a 3,430,060,584-byte bundle. This is about 14% smaller than the 3,994,140,402-byte exact-int8 baseline and about 65% larger than the 2,078,395,122-byte all-q4 packed bundle. No practical small q4 fallback was found.

The measured q4 full-bundle sizes were 2,078,395,122 bytes for packed and 1,600,670,180 bytes for emb4. Both variants passed their bundle checks, logits comparison, and 32 greedy-token comparison, but failed the unchanged KV-cache allclose criteria. An unquantized packed FP32 source-bundle control passed the same strict 32-step comparison. The small cache relative error in q4 is diagnostic evidence, not release acceptance.

The existing q4 candidates, mixed-storage diagnostics, and reports are local build artifacts. At task start, they were present under `build/variant-bundles/`; no `build/releases/packed`, `build/releases/emb4`, or `build/release-reports/` release output existed. These ignored outputs are preserved and are not verified release candidates.

## One-MatMul prototype evidence

The synthetic fixture and the smallest eligible real packed transformer MatMul (`node_linear_1`, shape `[2048, 1024]`) pass checker and CPUExecutionProvider output comparison on ONNX Runtime 1.22.1, 1.23.2, and 1.31.0. Maximum output absolute error was 0 for each. Each optimized graph folded the tritpack decode nodes into a single standard `MatMul` with an initializer weight; no `MatMulNBits` remained.

| ONNX Runtime | Session creation, baseline/candidate (ms) | First run, baseline/candidate (ms) | Warm median, baseline/candidate (ms) | Process peak RSS after baseline/candidate (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1.22.1 | 25.1 / 54.5 | 1.181 / 0.358 | 0.169 / 0.107 | 140.8 / 185.2 |
| 1.23.2 | 14.0 / 52.9 | 3.643 / 0.302 | 0.058 / 0.047 | 141.9 / 177.8 |
| 1.31.0 | 17.1 / 36.5 | 0.468 / 0.464 | 0.041 / 0.051 | 144.2 / 177.2 |

Each warm median uses 50 repeated runs. RSS values are process high-water observations after each stage, not isolated memory deltas. These small-matmul timings do not establish full-model startup, resource, or decode performance; the full packed diagnostic still has to pass its strict parity and runtime gates. Machine-readable local reports are `build/reports/tritpack-one-matmul-ort-1.22.1.json`, `build/reports/tritpack-one-matmul-ort-1.23.2.json`, and `build/reports/tritpack-one-matmul-ort-1.31.0-v2.json`.

## Full packed diagnostic structural result

The source-only ternary inventory in `build/reports/packed-source-tritpack-inventory-v1.json` confirmed 196 eligible MatMuls across 253 total MatMuls. The inventory is tied to source graph SHA256 `7d560542e6ec0a16baf9332708f058ca5e6b9ab97a2b984d86e6927c4924c2d8` and inventory SHA256 `eaf3dc88d81e41c970b83fe0f116ea8471e7ec4c130d1290cc286a95d6ec96e6`. It checks exact ternary structure directly in the selected packed ONNX graph, since the packed SafeTensors checkpoint stores encoded tensors rather than matching dense checkpoint weights.

The diagnostic at `build/variant-bundles/packed-tritpack-diagnostic-v1/` converted all 196 MatMuls and 196 unique initializers. Storage reconstruction was verified against each source weight, the ONNX checker passed, no `MatMulNBits` remained, and the original FP32 target initializers were absent. The compact codes occupy 286,261,248 bytes and scales 44,040,192 bytes, versus 5,637,144,576 logical FP32 operand bytes. The candidate LM graph is 4,349,347 bytes and its external data is 330,797,056 bytes.

The conversion report is `build/reports/packed-tritpack-conversion-v1.json`; structural validation is `build/reports/packed-tritpack-structural-validation-v2.json`. Both resource-guard reports passed. Conversion took 47.0 seconds with 151,416,832 bytes peak RSS; the exact inventory took 25.1 seconds with 106,405,888 bytes peak RSS. The graph SHA256 is `dc1234c32bd1295629d6fc6af0ebe9d423009ea8451dae11774f6f11b2cad40b` and external-data SHA256 is `8ac57068e289bedb9fa81b92ed19ffb2325234403ecf7313c09cd55f2008fc98`.

This establishes only the full-model structural and storage gate. The diagnostic has not yet run strict 32-step packed LM parity or the full-model runtime comparison, and it is not a validated release candidate. Do not proceed to emb4 or release integration before the packed parity and usability gates pass.

## Strategy A full-model runtime attempt

The 32-step packed LM benchmark was run under the 16 GiB RSS guard. The guard terminated the process during ONNX session construction after 104.6 seconds at 17,184,563,200 bytes peak RSS; minimum available memory remained 11,064,119,296 bytes. The report `build/reports/packed-tritpack-lm-benchmark-v1.json` was not produced, so this is a measured Strategy A resource failure, not a strict-parity result. No logits, cache, tokens, or warm-decode measurements were collected.

The in-graph route is not usable under the configured RSS limit. Use the planned local materialization fallback only to attempt the same strict packed parity gate. Do not integrate the backend, qualify emb4, or create release candidates unless the materialized runtime passes that unchanged 32-step gate.

## Strategy B materialization result

The first in-graph full-model attempt did not produce a parity result. It exceeded the 16 GiB RSS guard during ONNX session construction. A separate materialization cache was therefore built for the unchanged packed tritpack graph. The original `packed-tritpack-diagnostic-v1` bundle and reports remain untouched. The updated diagnostic is `build/variant-bundles/packed-tritpack-materialized-diagnostic-v2/`; structural validation passed for all 196 MatMuls in `build/reports/packed-tritpack-structural-validation-v3.json`.

The cache is outside the diagnostic bundle at `build/cache/packed-tritpack-materialized-v4/`. Its cache key is `3d771a2708887a4148869d82bdf484868351c5f59cf0c633398f397918632d00`. It contains a standard-MatMul graph and 5,637,144,576 bytes of materialized FP32 weights, with graph SHA256 `62d5e03a30945ef24200d53c89d774f70623e6e135b0a66c27ce6686b18120bb` and external-data SHA256 `55a823f6c68f4edd49c8d133b610a2c30c69a830c206476357b96d156f83bdaf`. The prewarm report is `build/reports/packed-tritpack-materialized-cache-v2.json`; standalone materialization took 26.54 seconds with 113,238,016 bytes peak RSS. The cache is not listed in the bundle manifest.

The unchanged 32-step packed benchmark passed with ONNX Runtime 1.31.0 and CPUExecutionProvider. Report: `build/reports/packed-tritpack-materialized-lm-parity-v2.json`. Prefill logits, all 56 prefill cache tensors, decode logits, every decode cache tensor, and all 32 greedy tokens passed the existing tolerances. The maximum prefill cache absolute error was 0.0023193359375 and still passed the unchanged elementwise relative-plus-absolute rule. The report records no cache divergence.

The successful resource-guard report is `build/reports/packed-tritpack-materialized-lm-parity-v2-resource-rss16.json`: status `ok`, peak RSS 15,494,057,984 bytes under a 16 GiB cap, and minimum available memory 12,811,079,680 bytes. The run used a 3 GiB maximum additional-swap guard; observed additional swap was 1,745,166,336 bytes. Two earlier attempts stopped before parity under stricter 1 GiB and 2 GiB swap-delta caps. They remain recorded as resource-guard reports and are not parity results.

Warm ONNX decode measured 371.74 ms/token versus 255.74 ms/token for the passing unquantized packed FP32 control in `build/reports/packed-source-lm-parity.json`, a 1.45x ratio that passes the 1.5x usability gate. The materialized benchmark reports 69.69 seconds for runtime initialization, including 20.78 seconds to verify the existing cache. The separate first cache build took 26.54 seconds. Strategy B is selected for the diagnostic path. The packed strict-LM and runtime gates passed. Bundle-v2 backend integration is implemented with focused regression coverage, but full packed release qualification and all promotion gates remain outstanding.
## Bundle-v2 backend integration

The bundle optimizer accepts `ternary-tritpack-exact` for packed and emb4 bundle-v2 variants, and the release builder now selects it while rejecting q4 or incomplete storage metadata. It records source and payload hashes, tritpack mappings, standard-MatMul runtime semantics, and required materialization metadata. The validator checks all 196 transformed MatMuls, initializer mappings, payload integrity, storage byte counts, and duplicate FP32 source weights, then validates ONNX Runtime using an external materialization cache. Release subprocess gates use a 3 GiB additional-swap cap, matching the successful packed diagnostic guard, while retaining the 16 GiB RSS limit and 2 GiB available-memory reserve. The size audit reports packed codes and scales separately from the logical FP32 operand size and local cache requirement. `ternary-nbits4` remains available as a diagnostic optimizer method.

Focused regression coverage uses a 196-MatMul bundle fixture, verifies exact outputs after materialization, checks bundle size accounting, and exercises validator runtime materialization. This is integration evidence only. No packed candidate has passed the complete release qualification or been promoted, and no emb4 qualification has started.

The latest packed attempt is recorded in `build/release-reports/packed-tritpack-v5/`. Export, tritpack transformation, bundle validation, and size audit passed. The staged bundle measured 1,655,174,583 bytes. The run stopped at the strict 32-step LM parity gate before producing parity results: additional system swap reached 3,225,411,584 bytes, exceeding the 3 GiB guard. Peak RSS was 12,475,940,864 bytes and minimum available memory was 15,454,523,392 bytes. The staged directory was removed, `build/releases/packed` was not created, and `release_status.json` records `candidate_promoted: false`. E2E, voice, exact-int8 integrity, final manifest, and promotion gates did not run. Keep packed qualification open and do not start emb4.
## Qualification order and stop gates



1. Implement and test the pure five-trits-per-byte codec without ONNX integration.
2. Prove synthetic and real packed one-MatMul reconstruction through standard `MatMul`; retain parity and runtime evidence. Stop if numerical parity fails.
3. **Passed:** stream-converted all 196 eligible packed transformer MatMuls and structurally validated the diagnostic bundle under the resource guard. See the measured evidence above.
4. **Passed:** materialized packed runtime passes strict 32-step logits, KV-cache, and greedy-token parity with unchanged tolerances. See `build/reports/packed-tritpack-materialized-lm-parity-v2.json`.
5. **Passed:** Strategy A failed the 16 GiB session-construction resource gate. Strategy B materialization passes packed parity and the 1.5x warm-decode usability gate; it is selected. No in-graph parity result was produced.
6. Integrate the qualified backend, then qualify packed before emb4. Promote neither local release candidate unless every applicable validation, parity, voice, resource, immutability, manifest, size, and atomic-promotion gate passes.

Never publish or upload automatically. Keep materialization caches outside release bundles. Record measured sizes and resource use, not projections, in final release documentation.
