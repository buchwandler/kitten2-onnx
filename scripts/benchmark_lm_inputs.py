"""Compare CPU latency for the legacy and input_ids LM graph contracts."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

from kitten2_onnx.reference import load_reference_artifacts
from kitten2_onnx.runtime import Kitten2OnnxLM


def _benchmark_profile(
    lm_dir: Path,
    speaker_path: Path,
    prompt: np.ndarray,
    speaker_embedding: np.ndarray,
    *,
    steps: int,
    repeats: int,
) -> dict[str, object]:
    started = time.perf_counter()
    runtime = Kitten2OnnxLM(lm_dir, speaker_model=speaker_path)
    startup_seconds = time.perf_counter() - started

    def run_chain(runtime):
        prefill_started = time.perf_counter()
        logits, past = runtime.step(prompt, speaker_embedding=speaker_embedding)
        prefill_seconds = time.perf_counter() - prefill_started
        generated: list[int] = []
        decode_seconds: list[float] = []
        for _ in range(steps):
            token = int(np.argmax(logits[0]))
            generated.append(token)
            decode_started = time.perf_counter()
            logits, past = runtime.step(np.asarray([[token]], dtype=np.int64), past=past)
            decode_seconds.append(time.perf_counter() - decode_started)
        return generated, prefill_seconds, decode_seconds

    run_chain(runtime)
    samples = [run_chain(runtime) for _ in range(repeats)]
    report = {
        "input_mode": runtime.input_mode,
        "lm_bytes": sum(path.stat().st_size for path in lm_dir.iterdir() if path.is_file()),
        "session_startup_seconds": startup_seconds,
        "prefill_seconds_median": statistics.median(sample[1] for sample in samples),
        "decode_steps_seconds_median": statistics.median(sum(sample[2]) for sample in samples),
        "decode_step_seconds_median": statistics.median(
            duration for sample in samples for duration in sample[2]
        ),
        "repeats": repeats,
        "greedy_raw_token_ids": samples[-1][0],
    }
    del runtime
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("speaker_model", type=Path)
    parser.add_argument("inputs_embeds_lm", type=Path)
    parser.add_argument("input_ids_lm", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", default="Hello from Kitten.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    artifacts = load_reference_artifacts(args.model_dir, "Bruno")
    prompt = np.asarray([artifacts.build_prompt(args.prompt)], dtype=np.int64)
    results = {
        "inputs_embeds_bfloat16_u16": _benchmark_profile(
            args.inputs_embeds_lm,
            args.speaker_model,
            prompt,
            artifacts.speaker_embedding,
            steps=args.steps,
            repeats=args.repeats,
        ),
        "input_ids_shared_tied_weight": _benchmark_profile(
            args.input_ids_lm,
            args.speaker_model,
            prompt,
            artifacts.speaker_embedding,
            steps=args.steps,
            repeats=args.repeats,
        ),
    }
    if (
        results["inputs_embeds_bfloat16_u16"]["greedy_raw_token_ids"]
        != results["input_ids_shared_tied_weight"]["greedy_raw_token_ids"]
    ):
        raise ValueError("Greedy token sequences differ between LM input contracts")

    report = {
        "format": "kitten2-onnx-lm-input-contract-benchmark-v1",
        "provider": "CPUExecutionProvider",
        "prompt": args.prompt,
        "prompt_tokens": int(prompt.shape[1]),
        "decode_steps": args.steps,
        "graph_dtype": "float32",
        "source_dtype": "bfloat16",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
