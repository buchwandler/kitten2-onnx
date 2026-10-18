from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from kitten2_onnx.lm_parity_reference import load_reference_pack

_GRAPH_OPTIMIZATION = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if os.uname().sysname == "Darwin" else peak * 1024)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _reference_prefill_feed(reference_path: Path) -> dict[str, np.ndarray]:
    reference = load_reference_pack(reference_path)
    metadata = reference.metadata
    lm_metadata = metadata["lm_metadata"]
    input_mode = lm_metadata.get("input_mode", "inputs_embeds")
    past_names = [name for name in lm_metadata["input_names"] if name.startswith("past_")]
    return {
        "attention_mask": np.ones((1, reference.prompt_length), dtype=np.int64),
        "position_ids": np.arange(reference.prompt_length, dtype=np.int64)[None],
        **dict(
            zip(
                past_names,
                [
                    np.zeros(
                        (
                            1,
                            int(metadata["num_key_value_heads"]),
                            0,
                            int(metadata["head_dim"]),
                        ),
                        dtype=np.float32,
                    )
                    for _ in past_names
                ],
                strict=True,
            )
        ),
        **reference.prefill_inputs(input_mode),
    }


def _zero_feed(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    dtypes = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }
    feed = {}
    for item in session.get_inputs():
        dtype = dtypes[item.type]
        shape = tuple(dimension if isinstance(dimension, int) and dimension > 0 else 1 for dimension in item.shape)
        feed[item.name] = np.zeros(shape, dtype=dtype)
    return feed


def check_session_memory(
    model_path: Path,
    *,
    reference_path: Path | None = None,
    run_zero_inference: bool = False,
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
    graph_optimization: str = "disable",
    disable_memory_pattern: bool = True,
    disable_cpu_arena: bool = True,
) -> dict[str, Any]:
    if graph_optimization not in _GRAPH_OPTIMIZATION:
        raise ValueError(f"unsupported graph optimization level: {graph_optimization}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = not disable_memory_pattern
    options.enable_cpu_mem_arena = not disable_cpu_arena
    options.graph_optimization_level = _GRAPH_OPTIMIZATION[graph_optimization]
    runtime_options = {
        "intra_op_threads": intra_op_threads,
        "inter_op_threads": inter_op_threads,
        "execution_mode": "sequential",
        "enable_mem_pattern": not disable_memory_pattern,
        "enable_cpu_mem_arena": not disable_cpu_arena,
        "graph_optimization": graph_optimization,
    }
    result: dict[str, Any] = {
        "format": "kitten2-onnx-session-memory-v1",
        "status": "failed",
        "model": str(model_path.resolve()),
        "runtime_options": runtime_options,
        "session_create_seconds": None,
        "inference_seconds": None,
        "inference_performed": False,
        "output_count": None,
        "peak_rss_bytes": 0,
        "error": None,
    }
    started = time.perf_counter()
    try:
        session_started = time.perf_counter()
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        result["session_create_seconds"] = time.perf_counter() - session_started
        if reference_path is not None:
            feed = _reference_prefill_feed(reference_path)
        elif run_zero_inference:
            feed = _zero_feed(session)
        else:
            feed = None
        if feed is not None:
            inference_started = time.perf_counter()
            outputs = session.run(None, feed)
            result["inference_seconds"] = time.perf_counter() - inference_started
            result["inference_performed"] = True
            result["output_count"] = len(outputs)
        result["status"] = "ok"
    except (RuntimeError, ValueError, TypeError, KeyError, IndexError, OSError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.perf_counter() - started
    result["peak_rss_bytes"] = _peak_rss_bytes()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ONNX Runtime session memory in an isolated worker.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--run-zero-inference", action="store_true")
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument(
        "--graph-optimization", choices=tuple(_GRAPH_OPTIMIZATION), default="disable"
    )
    parser.add_argument("--enable-memory-pattern", action="store_true")
    parser.add_argument("--enable-cpu-arena", action="store_true")
    args = parser.parse_args()
    if args.reference is not None and args.run_zero_inference:
        parser.error("--reference and --run-zero-inference are mutually exclusive")
    result = check_session_memory(
        args.model,
        reference_path=args.reference,
        run_zero_inference=args.run_zero_inference,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        graph_optimization=args.graph_optimization,
        disable_memory_pattern=not args.enable_memory_pattern,
        disable_cpu_arena=not args.enable_cpu_arena,
    )
    _write_json_atomic(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
