from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from kitten2_onnx.export_lm import RUNTIME_PROFILES, load_hf_lm
from kitten2_onnx.lm_parity_reference import (
    CACHE_ATOL,
    CACHE_RTOL,
    LOGITS_ATOL,
    LOGITS_RTOL,
    REFERENCE_FORMAT,
    pack_sha256,
    sha256_file,
    source_identity,
    verify_baseline_against_reference,
)
from kitten2_onnx.reference import _load_speaker_projection, load_reference_artifacts

PROMPT = "Hello from Kitten."
GENERATED_TOKEN_COUNT = 32
SAMPLING_SETTINGS = {
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "token_run_penalty": 0.0,
    "seed": 7,
}


def _git_revision(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, ...], ...]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple(tuple(layer) for layer in cache)


def _flatten_cache(cache: Any) -> list[torch.Tensor]:
    return [tensor for layer in _legacy_cache(cache) for tensor in layer]


def _numpy_cache(cache: Any) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy().copy() for tensor in _flatten_cache(cache)]


def _verify_append_only(previous: list[np.ndarray], current: list[np.ndarray], step: int) -> None:
    if len(previous) != len(current):
        raise ValueError(f"Cache tensor count changed at decode step {step}")
    for index, (old, new) in enumerate(zip(previous, current, strict=True)):
        if new.shape[-2] != old.shape[-2] + 1 or not np.array_equal(new[..., :-1, :], old):
            raise ValueError(
                f"PyTorch reference cache is not append-only at decode step {step}, tensor {index}"
            )


def _write_array(root: Path, name: str, array: np.ndarray) -> dict[str, Any]:
    filename = f"{name}.npy"
    path = root / filename
    np.save(path, np.asarray(array), allow_pickle=False)
    saved = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "file": filename,
        "shape": list(saved.shape),
        "dtype": str(saved.dtype),
        "sha256": sha256_file(path),
    }


def _capture_arrays(model_dir: Path, baseline_bundle: Path, fixture_path: Path):
    import onnxruntime as ort

    artifacts = load_reference_artifacts(model_dir, "Bruno")
    with np.load(fixture_path, allow_pickle=False) as fixture:
        expected_prompt_ids = np.asarray(fixture["prompt_ids"], dtype=np.int64)
        generated_ids = np.asarray(fixture["generated_raw_ids"], dtype=np.int64)
    if generated_ids.shape != (GENERATED_TOKEN_COUNT,):
        raise ValueError(f"Expected exactly {GENERATED_TOKEN_COUNT} fixture token IDs")
    prompt_ids = np.asarray(artifacts.build_prompt(PROMPT), dtype=np.int64)
    np.testing.assert_array_equal(prompt_ids, expected_prompt_ids)

    lm_metadata = json.loads(
        (baseline_bundle / "lm/lm.meta.json").read_text(encoding="utf-8")
    )
    if lm_metadata["source_model_revision"] != artifacts.model_revision:
        raise ValueError("Baseline ONNX graph and source model revisions do not match")
    if lm_metadata["runtime_profile"] != "dev-fp32":
        raise ValueError("Parity reference capture requires the dev-fp32 baseline")
    profile = RUNTIME_PROFILES[lm_metadata["runtime_profile"]]
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = load_hf_lm(model_dir, profile.graph_dtype)
    model.eval()
    device = torch.device("cpu")
    speaker_projection = _load_speaker_projection(
        model_dir, artifacts.config, profile.graph_dtype, device
    )
    speaker_embedding = torch.from_numpy(artifacts.speaker_embedding[None])
    with torch.inference_mode():
        speaker_hidden_torch = speaker_projection(speaker_embedding).cpu().numpy()

    speaker_session = ort.InferenceSession(
        str(baseline_bundle / "speaker/speaker.onnx"), providers=["CPUExecutionProvider"]
    )
    speaker_hidden = speaker_session.run(
        ["speaker_hidden"], {"speaker_embedding": artifacts.speaker_embedding[None]}
    )[0].astype(np.float32, copy=False)
    np.testing.assert_allclose(speaker_hidden, speaker_hidden_torch, rtol=1e-5, atol=1e-5)

    prompt_tensor = torch.from_numpy(prompt_ids[None])
    with torch.inference_mode():
        token_embeddings = model.get_input_embeddings()(prompt_tensor)
    prefill_inputs_embeds = token_embeddings.detach().cpu().numpy()[0].copy()
    prefill_inputs_embeds[0] = speaker_hidden[0]
    input_mode = lm_metadata.get("input_mode", "inputs_embeds")
    if input_mode not in {"inputs_embeds", "input_ids"}:
        raise ValueError(f"Unsupported baseline LM input mode: {input_mode}")

    layers = int(model.config.num_hidden_layers)
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    attention_mask = torch.ones((1, len(prompt_ids)), dtype=torch.long)
    position_ids = torch.arange(len(prompt_ids), dtype=torch.long)[None, :]
    with torch.inference_mode():
        output = model(
            inputs_embeds=torch.from_numpy(prefill_inputs_embeds[None]),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    logits = [output.logits[:, -1, :].float().cpu().numpy()[0].copy()]
    reference_cache = output.past_key_values
    previous_cache = _numpy_cache(reference_cache)
    decode_token_embeds: list[np.ndarray] = []

    for step, expected_id in enumerate(generated_ids):
        predicted_id = int(np.argmax(logits[-1]))
        if predicted_id != int(expected_id):
            raise ValueError(
                f"PyTorch reference token differs from fixture at step {step}: "
                f"{predicted_id} != {int(expected_id)}"
            )
        token_ids = torch.tensor([[int(expected_id)]], dtype=torch.long)
        with torch.inference_mode():
            token_embeds = model.get_input_embeddings()(token_ids)
        decode_token_embeds.append(token_embeds.detach().cpu().numpy()[0, 0].copy())
        past_length = len(prompt_ids) + step
        next_mask = torch.ones((1, past_length + 1), dtype=torch.long)
        next_positions = torch.tensor([[past_length]], dtype=torch.long)
        with torch.inference_mode():
            output = model(
                inputs_embeds=token_embeds,
                attention_mask=next_mask,
                position_ids=next_positions,
                past_key_values=reference_cache,
                use_cache=True,
                return_dict=True,
            )
        reference_cache = output.past_key_values
        current_cache = _numpy_cache(reference_cache)
        _verify_append_only(previous_cache, current_cache, step)
        previous_cache = current_cache
        logits.append(output.logits[:, -1, :].float().cpu().numpy()[0].copy())

    cache_keys = np.stack(previous_cache[::2], axis=0)
    cache_values = np.stack(previous_cache[1::2], axis=0)
    arrays = {
        "prompt_ids": prompt_ids,
        "prefill_inputs_embeds": prefill_inputs_embeds.astype(np.float32, copy=False),
        "speaker_hidden": speaker_hidden[0],
        "generated_ids": generated_ids,
        "token_embeds": np.stack(decode_token_embeds).astype(np.float32, copy=False),
        "logits": np.stack(logits).astype(np.float32, copy=False)[:, None, :],
        "cache_keys": cache_keys.astype(np.float32, copy=False),
        "cache_values": cache_values.astype(np.float32, copy=False),
    }
    reference_details = {
        "lm_metadata": lm_metadata,
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": layers,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "prompt_length": len(prompt_ids),
        "generated_token_count": len(generated_ids),
        "input_mode": input_mode,
    }
    del output, reference_cache, model, speaker_projection, speaker_session
    gc.collect()
    return arrays, reference_details, artifacts.model_revision


def capture_reference_pack(
    model_dir: Path,
    baseline_bundle: Path,
    output: Path,
    *,
    fixture_path: Path,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    baseline_bundle = baseline_bundle.resolve()
    output = output.resolve()
    fixture_path = fixture_path.resolve()
    if output.exists():
        raise FileExistsError(f"Reference pack output already exists: {output}")
    identity = source_identity(model_dir, baseline_bundle, fixture_path)
    if identity["model_revision"] is None:
        raise ValueError("Source model revision is required for a parity reference pack")
    arrays, details, model_revision = _capture_arrays(model_dir, baseline_bundle, fixture_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        array_metadata = {
            name: _write_array(temporary, name, array) for name, array in arrays.items()
        }
        try:
            commit = _git_revision(Path(__file__).resolve().parents[1])
        except OSError:
            commit = None
        metadata = {
            "format": REFERENCE_FORMAT,
            "source_identity": identity,
            "model_revision": model_revision,
            "prompt": PROMPT,
            "voice": "Bruno",
            "sampling_settings": SAMPLING_SETTINGS,
            "expected_token_ids": arrays["generated_ids"].tolist(),
            "generated_token_count": details["generated_token_count"],
            "logits_rtol": LOGITS_RTOL,
            "logits_atol": LOGITS_ATOL,
            "cache_rtol": CACHE_RTOL,
            "cache_atol": CACHE_ATOL,
            "repository_commit": commit,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            **details,
            "arrays": array_metadata,
            "baseline_verification": None,
            "trusted": False,
        }
        metadata["pack_sha256"] = pack_sha256(metadata)
        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        from kitten2_onnx.lm_parity_reference import load_reference_pack

        reference = load_reference_pack(temporary)
        verification = verify_baseline_against_reference(
            baseline_bundle / "lm/lm.onnx", reference
        )
        metadata["baseline_verification"] = verification
        metadata["trusted"] = verification["diagnostic_score"] == [0, 0, 0.0, 0.0] and verification[
            "generated_ids_match"
        ]
        metadata["pack_sha256"] = pack_sha256(metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        load_reference_pack(temporary)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    result = {
        "output": str(output),
        "pack_sha256": metadata["pack_sha256"],
        "trusted": metadata["trusted"],
        "baseline_verification": metadata["baseline_verification"],
    }
    if not metadata["trusted"]:
        raise ValueError("FP32 baseline did not reproduce the golden reference pack")
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Capture a reusable FP32 LM parity reference pack.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("baseline_bundle", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("build/reports/nbits-reference")
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=project_root / "tests/fixtures/reference_bruno.npz",
    )
    args = parser.parse_args()
    result = capture_reference_pack(
        args.model_dir, args.baseline_bundle, args.output, fixture_path=args.fixture
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
