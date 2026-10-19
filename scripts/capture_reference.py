#!/usr/bin/env python3
"""Capture deterministic KittenTTS 2 PyTorch reference tensors."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kitten2_onnx.reference import (
    UPSTREAM_CODE_REVISION,
    ReferenceArtifacts,
    load_reference_artifacts,
    load_reference_model,
)
from kitten2_onnx.sampling import SamplingSettings, sample


def _legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def _cache_length(cache: Any) -> int:
    return int(_legacy_cache(cache)[0][0].shape[-2])


def capture_reference(
    artifacts: ReferenceArtifacts,
    *,
    text: str,
    max_new_tokens: int = 32,
    device: str = "cpu",
    reference_model: Any | None = None,
    logits_callback: Callable[[int, np.ndarray], None] | None = None,
    sampling_settings: SamplingSettings | None = None,
    seed: int | None = None,

    normalize: bool = True,
    use_emotion: bool | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference = reference_model or load_reference_model(artifacts, device=device)
    model = reference.model
    prepared = artifacts.prepare_text(text, normalize=normalize, use_emotion=use_emotion)
    prompt_ids = artifacts.build_prompt_from_prepared(prepared)
    device_obj = reference.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device_obj)
    embedding_table = model.get_input_embeddings().weight

    with torch.inference_mode():
        speaker_hidden = reference.project_speaker()
        token_embeds = model.get_input_embeddings()(input_ids)
        prefill_inputs_embeds = torch.cat(
            [speaker_hidden.unsqueeze(1), token_embeds[:, 1:, :]], dim=1
        )
        prompt_length = len(prompt_ids)
        history = list(prompt_ids)
        rng = np.random.default_rng(seed)
        prefill_position_ids = torch.arange(prompt_length, device=device_obj)[None, :]
        prefill_attention_mask = torch.ones((1, prompt_length), dtype=torch.long, device=device_obj)
        prefill = model(
            inputs_embeds=prefill_inputs_embeds,
            position_ids=prefill_position_ids,
            attention_mask=prefill_attention_mask,
            use_cache=True,
            return_dict=True,
        )
        logits = prefill.logits[:, -1, :]
        cache = prefill.past_key_values
        prefill_cache = _legacy_cache(cache)
        prefill_kv_shapes = np.asarray(
            [[list(key.shape), list(value.shape)] for key, value in prefill_cache],
            dtype=np.int64,
        )
        kv_lengths = [_cache_length(cache)]
        generated_ids: list[int] = []
        decode_logits: list[np.ndarray] = []
        eos_ids = {artifacts.token_map.speech_end_id, artifacts.token_map.stop_id}

        for step_index in range(max_new_tokens):
            step_logits = logits[0].float()
            if logits_callback is not None:
                logits_callback(step_index, step_logits.cpu().numpy().copy())
            if len(decode_logits) < 5:
                decode_logits.append(step_logits.cpu().numpy().copy())
            if sampling_settings is None:
                next_id = int(torch.argmax(step_logits).item())
            else:
                next_id = sample(
                    step_logits.cpu().numpy(),
                    history,
                    artifacts.token_map,
                    sampling_settings,
                    float(rng.random()),
                    prompt_length=prompt_length,
                )
            generated_ids.append(next_id)
            if next_id in eos_ids:
                break

            history.append(next_id)
            past_length = _cache_length(cache)
            next_input = torch.tensor([[next_id]], dtype=torch.long, device=device_obj)
            attention_mask = torch.ones((1, past_length + 1), dtype=torch.long, device=device_obj)
            position_ids = torch.tensor([[past_length]], dtype=torch.long, device=device_obj)
            output = model(
                input_ids=next_input,
                past_key_values=cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
            kv_lengths.append(_cache_length(cache))
            logits = output.logits[:, -1, :]

    arrays = {
        "prompt_ids": np.asarray(prompt_ids, dtype=np.int64),
        "speaker_embedding_512": artifacts.speaker_embedding.astype(np.float32, copy=False),
        "speaker_hidden": speaker_hidden[0].float().cpu().numpy(),
        "prefill_inputs_embeds": prefill_inputs_embeds[0].float().cpu().numpy(),
        "prefill_position_ids": prefill_position_ids[0].cpu().numpy(),
        "prefill_attention_mask": prefill_attention_mask[0].cpu().numpy(),
        "prefill_last_logits": prefill.logits[0, -1].float().cpu().numpy(),
        "prefill_kv_shapes": prefill_kv_shapes,
        "generated_raw_ids": np.asarray(generated_ids, dtype=np.int64),
        "decode_logits_first5": np.asarray(decode_logits, dtype=np.float32),
        "kv_lengths": np.asarray(kv_lengths, dtype=np.int64),
        "input_embedding_shape": np.asarray(embedding_table.shape, dtype=np.int64),
    }
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "unknown"
    metadata = {
        "source_model_path": str(artifacts.repo_dir),
        "source_model_revision": artifacts.model_revision,
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "dtype": str(embedding_table.dtype).removeprefix("torch."),
        "device": str(device_obj),
        "voice": artifacts.voice,
        "reference_transcript": artifacts.transcript,
        "target_text": text,

        "processed_text": prepared.processed_text,
        "normalize": prepared.normalize,
        "expression_detected": prepared.expression_detected,
        "emotion_control_used": prepared.emotion_control_used,
        "prompt_length": len(prompt_ids),
        "generated_token_count": len(generated_ids),
    }
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", type=Path)
    parser.add_argument("--voice")
    parser.add_argument("--text", default="Hello from Kitten.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-normalize", action="store_true")
    emotion_group = parser.add_mutually_exclusive_group()
    emotion_group.add_argument("--use-emotion", dest="use_emotion", action="store_true")
    emotion_group.add_argument("--no-emotion", dest="use_emotion", action="store_false")
    parser.set_defaults(use_emotion=None)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output basename for .npz and .json files"
    )
    args = parser.parse_args()

    artifacts = load_reference_artifacts(args.repo_dir, args.voice)
    arrays, metadata = capture_reference(
        artifacts,
        text=args.text,
        max_new_tokens=args.max_new_tokens,
        device=args.device,

        normalize=not args.no_normalize,
        use_emotion=args.use_emotion,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.output.with_suffix(".npz")
    json_path = args.output.with_suffix(".json")
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"npz": str(npz_path), "metadata": str(json_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
