"""Compare FP32 and NBits MatMul kernels on identical real model activations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.quantize_ternary_nbits import (
    TernaryNBitsConfig,
    _array_sha256,
    rewrite_matmul_to_nbits4,
)

_TARGET_PROJECTIONS = (
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_BLOCK_SIZES = (128, 64, 32, 16)
_ATOL = 2e-3
_RTOL = 1e-4


def _source_matmul_model(weight: np.ndarray) -> onnx.ModelProto:
    if weight.dtype != np.float32 or weight.ndim != 2:
        raise ValueError("weight must be a rank-2 float32 [K, N] matrix")
    k, n = weight.shape
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["activation", "weight"], ["output"], name="diagnostic_matmul")],
        "nbits-kernel-diagnostic",
        [helper.make_tensor_value_info("activation", TensorProto.FLOAT, ["rows", k])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["rows", n])],
        [numpy_helper.from_array(np.ascontiguousarray(weight), name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model


def _session(model: onnx.ModelProto, intra_op_threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _run_timed(
    session: ort.InferenceSession,
    activation: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, float]:
    inputs = {"activation": activation}
    result = session.run(["output"], inputs)[0]
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = session.run(["output"], inputs)[0]
        timings.append((time.perf_counter() - started) * 1000.0)
    return result, statistics.median(timings)


def _compare_sessions(
    fp32_session: ort.InferenceSession,
    nbits_session: ort.InferenceSession,
    activation: np.ndarray,
    *,
    config: TernaryNBitsConfig,
    iterations: int,
) -> dict[str, Any]:
    fp32_output, fp32_runtime_ms = _run_timed(fp32_session, activation, iterations)
    nbits_output, nbits_runtime_ms = _run_timed(nbits_session, activation, iterations)
    absolute_difference = np.abs(fp32_output - nbits_output)
    max_index = tuple(int(index) for index in np.unravel_index(np.argmax(absolute_difference), absolute_difference.shape))
    max_absolute_difference = float(absolute_difference[max_index])
    relative_difference = absolute_difference / np.maximum(np.abs(fp32_output), 1e-12)
    allowed_difference = _ATOL + _RTOL * np.abs(fp32_output)
    return {
        "block_size": config.block_size,
        "accuracy_level": config.accuracy_level,
        "input_shape": list(activation.shape),
        "activation_sha256": hashlib.sha256(
            memoryview(np.ascontiguousarray(activation)).cast("B")
        ).hexdigest(),
        "output_shape": list(fp32_output.shape),
        "atol": _ATOL,
        "rtol": _RTOL,
        "max_absolute_difference": max_absolute_difference,
        "max_relative_difference": float(np.max(relative_difference)),
        "outside_tolerance_count": int(np.count_nonzero(absolute_difference > allowed_difference)),
        "output_element_count": int(fp32_output.size),
        "max_error_index": list(max_index),
        "fp32_value_at_max_error": float(fp32_output[max_index]),
        "nbits_value_at_max_error": float(nbits_output[max_index]),
        "fp32_runtime_median_ms": fp32_runtime_ms,
        "nbits_runtime_median_ms": nbits_runtime_ms,
        "runtime_speedup": fp32_runtime_ms / nbits_runtime_ms if nbits_runtime_ms else None,
    }


def compare_matmul(
    activation: np.ndarray,
    weight: np.ndarray,
    config: TernaryNBitsConfig,
    *,
    iterations: int = 3,
    intra_op_threads: int = 4,
) -> dict[str, Any]:
    """Compare one exact NBits MatMul with FP32 using the same activation."""
    activation = np.asarray(activation)
    weight = np.asarray(weight)
    if activation.dtype != np.float32 or activation.ndim != 2:
        raise ValueError("activation must be a rank-2 float32 [M, K] matrix")
    if weight.dtype != np.float32 or weight.ndim != 2 or activation.shape[1] != weight.shape[0]:
        raise ValueError("weight must be float32 [K, N] and match the activation K dimension")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    source_model = _source_matmul_model(weight)
    nbits_model = rewrite_matmul_to_nbits4(source_model, config=config)
    onnx.checker.check_model(source_model)
    onnx.checker.check_model(nbits_model)
    fp32_session = _session(source_model, intra_op_threads)
    nbits_session = _session(nbits_model, intra_op_threads)
    return _compare_sessions(
        fp32_session,
        nbits_session,
        activation,
        config=config,
        iterations=iterations,
    )


def _capture_real_activations(
    model_dir: Path,
    *,
    layers: list[int],
    decode_steps: list[int],
    intra_op_threads: int,
) -> tuple[torch.nn.Module, dict[str, dict[str, np.ndarray]]]:
    from kitten2_onnx.export_lm import load_hf_lm
    from kitten2_onnx.reference import _load_speaker_projection, load_reference_artifacts

    fixture_path = Path(__file__).resolve().parents[1] / "tests/fixtures/reference_bruno.npz"
    artifacts = load_reference_artifacts(model_dir, "Bruno")
    with np.load(fixture_path, allow_pickle=False) as fixture:
        prompt_ids = fixture["prompt_ids"]
        expected_generated_ids = fixture["generated_raw_ids"]
    np.testing.assert_array_equal(prompt_ids, artifacts.build_prompt("Hello from Kitten."))
    if not decode_steps or min(decode_steps) < 0 or max(decode_steps) >= len(expected_generated_ids):
        raise ValueError("decode steps must select available non-negative fixture token indices")

    torch.set_num_threads(intra_op_threads)
    model = load_hf_lm(model_dir, torch.float32)
    model.eval()
    named_modules = dict(model.named_modules())
    target_names = [
        f"model.layers.{layer}.{projection}"
        for layer in layers
        for projection in _TARGET_PROJECTIONS
    ]
    missing = [name for name in target_names if name not in named_modules]
    if missing:
        raise ValueError(f"Model is missing diagnostic projection modules: {missing}")

    captured: dict[str, dict[str, np.ndarray]] = {}
    current_case: list[str | None] = [None]
    hooks = []
    for name in target_names:
        def capture_input(_module, inputs, *, target_name=name):
            case = current_case[0]
            if case is None:
                return
            activation = inputs[0].detach().to(dtype=torch.float32).cpu().numpy()
            captured.setdefault(case, {})[target_name] = np.ascontiguousarray(
                activation.reshape(-1, activation.shape[-1])
            )

        hooks.append(named_modules[name].register_forward_pre_hook(capture_input))

    prompt_tensor = torch.from_numpy(prompt_ids[None]).to(dtype=torch.long)
    with torch.inference_mode():
        token_embeddings = model.get_input_embeddings()(prompt_tensor)
        speaker_projection = _load_speaker_projection(
            model_dir, artifacts.config, torch.float32, torch.device("cpu")
        )
        speaker_embedding = torch.from_numpy(artifacts.speaker_embedding[None])
        speaker_hidden = speaker_projection(speaker_embedding)
        inputs_embeds = token_embeddings.clone()
        inputs_embeds[:, 0, :] = speaker_hidden
        attention_mask = torch.ones((1, len(prompt_ids)), dtype=torch.long)
        position_ids = torch.arange(len(prompt_ids), dtype=torch.long)[None, :]
        current_case[0] = "prefill"
        output = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        current_case[0] = None

        for step in range(max(decode_steps) + 1):
            predicted = int(output.logits[:, -1, :].argmax(dim=-1)[0])
            expected = int(expected_generated_ids[step])
            if predicted != expected:
                raise ValueError(
                    f"FP32 reference generated token {predicted} at step {step}, expected {expected}"
                )
            token_ids = torch.tensor([[expected]], dtype=torch.long)
            token_embeds = model.get_input_embeddings()(token_ids)
            past_length = len(prompt_ids) + step
            next_mask = torch.ones((1, past_length + 1), dtype=torch.long)
            next_positions = torch.tensor([[past_length]], dtype=torch.long)
            current_case[0] = f"decode_step_{step}" if step in decode_steps else None
            output = model(
                inputs_embeds=token_embeds,
                attention_mask=next_mask,
                position_ids=next_positions,
                past_key_values=output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            current_case[0] = None

    for hook in hooks:
        hook.remove()
    return model, captured


def _diagnose_real_activations(
    model_dir: Path,
    baseline_dir: Path,
    *,
    inventory_path: Path,
    layers: list[int],
    decode_steps: list[int],
    accuracy_level_0_block_size: int | None,
    iterations: int,
    intra_op_threads: int,
) -> dict[str, Any]:
    model, captured = _capture_real_activations(
        model_dir,
        layers=layers,
        decode_steps=decode_steps,
        intra_op_threads=intra_op_threads,
    )
    named_modules = dict(model.named_modules())
    configs = [TernaryNBitsConfig(block_size=size, accuracy_level=1) for size in _BLOCK_SIZES]
    if accuracy_level_0_block_size is not None:
        configs.append(TernaryNBitsConfig(block_size=accuracy_level_0_block_size, accuracy_level=0))

    rows = []
    for layer in layers:
        for projection in _TARGET_PROJECTIONS:
            module_name = f"model.layers.{layer}.{projection}"
            module = named_modules[module_name]
            source_weight = np.ascontiguousarray(
                module.weight.detach().to(dtype=torch.float32).cpu().numpy().T
            )
            source_weight_sha256 = _array_sha256(source_weight)
            if captured.get("prefill", {}).get(module_name) is None:
                raise ValueError(f"No prefill activation was captured for {module_name}")
            source_model = _source_matmul_model(source_weight)
            fp32_session = _session(source_model, intra_op_threads)
            nbits_sessions = []
            for config in configs:
                nbits_model = rewrite_matmul_to_nbits4(source_model, config=config)
                onnx.checker.check_model(nbits_model)
                nbits_sessions.append((config, _session(nbits_model, intra_op_threads)))
            for case, activations in captured.items():
                activation = activations.get(module_name)
                if activation is None:
                    continue
                for config, nbits_session in nbits_sessions:
                    result = _compare_sessions(
                        fp32_session,
                        nbits_session,
                        activation,
                        config=config,
                        iterations=iterations,
                    )
                    rows.append(
                        {
                            "case": case,
                            "layer": layer,
                            "module": projection,
                            "checkpoint_key": f"model.layers.{layer}.{projection}.weight",
                            "source_weight_sha256": source_weight_sha256,
                            **result,
                        }
                    )
            del fp32_session, nbits_sessions, source_model, source_weight
            gc.collect()

    bundle = json.loads((baseline_dir / "bundle.json").read_text(encoding="utf-8"))
    lm_metadata = json.loads((baseline_dir / "lm/lm.meta.json").read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["source_graph_sha256"] != bundle["files"]["lm/lm.onnx"]:
        raise ValueError("MatMul inventory does not describe the supplied baseline LM graph")
    return {
        "format": "kitten2-onnx-nbits-kernel-diagnostic-v1",
        "diagnostic_only": True,
        "acceptance_gate": "tests/test_real_lm_parity.py",
        "baseline_bundle": str(baseline_dir.resolve()),
        "baseline_lm_graph_sha256": bundle["files"]["lm/lm.onnx"],
        "input_mode": lm_metadata["input_mode"],
        "model_revision": lm_metadata["source_model_revision"],
        "inventory_sha256": inventory["inventory_sha256"],
        "activation_source": "Bruno prompt fixture with speaker injection and greedy reference decode tokens",
        "captured_cases": sorted(captured),
        "layers": layers,
        "projections": list(_TARGET_PROJECTIONS),
        "decode_steps": decode_steps,
        "configurations": [
            {"block_size": config.block_size, "accuracy_level": config.accuracy_level}
            for config in configs
        ],
        "iterations_per_case": iterations,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_architecture": platform.machine(),
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "intra_op_threads": intra_op_threads,
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_bundle", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[23, 24, 25, 26, 27])
    parser.add_argument("--decode-steps", type=int, nargs="+", default=[0, 1, 4])
    parser.add_argument("--accuracy-level-0-block-size", type=int, choices=_BLOCK_SIZES)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("build/reports/nbits-kernel-diagnostic.json")
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.intra_op_threads <= 0:
        parser.error("--iterations and --intra-op-threads must be positive")
    report = _diagnose_real_activations(
        args.model_dir,
        args.baseline_bundle,
        inventory_path=args.inventory,
        layers=args.layers,
        decode_steps=args.decode_steps,
        accuracy_level_0_block_size=args.accuracy_level_0_block_size,
        iterations=args.iterations,
        intra_op_threads=args.intra_op_threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(report['results'])} same-activation kernel comparisons")


if __name__ == "__main__":
    main()
