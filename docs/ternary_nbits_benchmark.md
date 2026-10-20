# Ternary MatMulNBits CPU feasibility

## Environment and method

The q2 feasibility run compared q2 and q4 on the same Kitten layer-27 activations for `self_attn.v_proj`, `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj`. It measured prefill (`M=294`) and one cached decode step (`M=1`), with block sizes 128, 64, 32, and 16, `accuracy_level=1`, one intra-op thread, and one timed iteration per case.

- Platform: Linux 6.18.33.2 under WSL2, x86_64
- Python: 3.13.15
- ONNX: 1.23.2
- ONNX Runtime: 1.31.0, CPUExecutionProvider
- Report: `build/reports/nbits-q2-q4-feasibility.json`

The q2/q4 feasibility comparison tested only ONNX Runtime 1.31.0. The declared package minimum, 1.22, remains unverified. This benchmark does not establish support on other platforms or ORT versions, and it is not full-model evidence.

## Results

Median operator latency across the four projections and four block sizes:

| Activation case | q2 median | q4 median | q2/q4 |
| --- | ---: | ---: | ---: |
| Prefill, M=294 | 100.10 ms | 60.43 ms | 1.66x |
| Decode, M=1 | 39.75 ms | 1.01 ms | 39.47x |


## q4 repeated timing

A q4-only follow-up used five timed iterations per case across layer 27, all four projections, and block sizes 128, 64, 32, and 16. On the same ORT 1.31.0 CPU environment, the median across those 16 projection/block combinations was 64.94 ms for prefill (M=294) and 1.04 ms for decode (M=1). The report is `build/reports/nbits-q4-repeat-feasibility.json`. These are kernel-only measurements, not full-model latency or parity evidence.
The q2 decode measurements ranged from 4.31 to 43.84 ms, while q4 ranged from 0.146 to 1.262 ms. Based on this CPU feasibility screen, the user selected q4 for the next implementation phase and q2 remains diagnostic-only. This short kernel benchmark does not establish full-model parity, end-to-end latency, or storage-size suitability. The comparison script's elementwise errors are diagnostic data, not a substitute for the accepted end-to-end parity gates.

## Decision gate

Use q4 as the portable backend for the next implementation phase, subject to full-model parity and size review. Do not build or publish a full q2 bundle based on these results. Keep the existing exact-int8 bundle unchanged. No tolerance change or external publication is implied by this report.
