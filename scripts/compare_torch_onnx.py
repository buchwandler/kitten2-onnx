#!/usr/bin/env python3
"""Compare exported LM logits against the source Qwen3 model.

This script deliberately tests deterministic prefill/decode tensor parity before
attempting Kitten's stochastic logits processors or waveform synthesis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from kitten2_onnx.export_lm import DTYPES, load_hf_lm
from kitten2_onnx.runtime import Kitten2OnnxLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", type=Path)
    parser.add_argument("onnx_dir", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--prompt", default="Hello from Kitten.")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model = load_hf_lm(args.repo_dir, DTYPES[args.dtype])
    tokenizer = AutoTokenizer.from_pretrained(str(args.repo_dir / "lm"))
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not ids:
        raise SystemExit("tokenizer produced no ids")

    ort_lm = Kitten2OnnxLM(args.onnx_dir)
    ort_logits, ort_past = ort_lm.step(np.asarray([ids], dtype=np.int64))

    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        ref = model(input_ids=input_ids, use_cache=True)
    ref_logits = ref.logits[:, -1, :].float().cpu().numpy()

    print(
        "prefill max_abs=",
        float(np.max(np.abs(ort_logits.astype(np.float32) - ref_logits))),
    )

    ref_cache = ref.past_key_values
    token = int(np.argmax(ref_logits[0]))
    for step in range(args.steps):
        ort_logits, ort_past = ort_lm.step(
            np.asarray([[token]], dtype=np.int64),
            past=ort_past,
        )
        with torch.no_grad():
            ref = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=ref_cache,
                use_cache=True,
            )
        ref_cache = ref.past_key_values
        ref_logits = ref.logits[:, -1, :].float().cpu().numpy()
        diff = float(np.max(np.abs(ort_logits.astype(np.float32) - ref_logits)))
        ort_token = int(np.argmax(ort_logits[0]))
        ref_token = int(np.argmax(ref_logits[0]))
        print(f"decode[{step}] max_abs={diff:.6g} argmax_ref={ref_token} argmax_onnx={ort_token}")
        token = ref_token


if __name__ == "__main__":
    main()
