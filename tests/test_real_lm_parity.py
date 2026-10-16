from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from kitten2_onnx.export_lm import RUNTIME_PROFILES, _io_names, load_hf_lm
from kitten2_onnx.reference import _load_speaker_projection, load_reference_artifacts
from kitten2_onnx.runtime import Kitten2OnnxLM
from kitten2_onnx.sampling import SamplingSettings


def _legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def _flatten_cache(cache):
    return [tensor for layer in _legacy_cache(cache) for tensor in layer]


def _assert_outputs(actual, expected_logits, expected_cache):
    np.testing.assert_allclose(
        actual[0], expected_logits.detach().cpu().numpy(), rtol=1e-4, atol=2e-4
    )
    actual_cache = actual[1:]
    expected_cache = _flatten_cache(expected_cache)
    assert len(actual_cache) == len(expected_cache)
    for actual_tensor, expected_tensor in zip(actual_cache, expected_cache, strict=True):
        np.testing.assert_allclose(
            actual_tensor,
            expected_tensor.detach().cpu().numpy(),
            rtol=1e-4,
            atol=2e-3,
        )


def test_real_bruno_prompt_injection_and_32_step_lm_parity(monkeypatch):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model_dir_value = os.environ.get("KITTEN2_MODEL_DIR")
    onnx_dir_value = os.environ.get("KITTEN2_ONNX_DIR")
    if not model_dir_value or not onnx_dir_value:
        pytest.skip("set KITTEN2_MODEL_DIR and KITTEN2_ONNX_DIR to enable real LM parity")
    model_dir = Path(model_dir_value)
    onnx_dir = Path(onnx_dir_value)
    speaker_path_value = os.environ.get("KITTEN2_SPEAKER_ONNX")
    speaker_path = (
        Path(speaker_path_value)
        if speaker_path_value
        else onnx_dir.parent / "speaker" / "speaker.onnx"
    )
    if not speaker_path.is_file():
        pytest.skip(f"speaker graph is not present: {speaker_path}")

    metadata = json.loads((onnx_dir / "lm.meta.json").read_text(encoding="utf-8"))
    profile_name = metadata["runtime_profile"]
    if profile_name != "dev-fp32":
        pytest.skip("real CPU parity currently requires the dev-fp32 profile")
    profile = RUNTIME_PROFILES[profile_name]
    assert metadata["graph_dtype"] == profile.graph_dtype_name == "float32"
    assert metadata["io_dtype"] == profile.io_dtype == "float32"
    assert metadata["source_dtype"] == "bfloat16"
    assert metadata["runtime_provider"] == "CPUExecutionProvider"
    assert metadata["external_data"] is True
    graph = onnx.load(str(onnx_dir / "lm.onnx"), load_external_data=False)
    locations = {
        entry.value
        for tensor in graph.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    assert locations
    assert all(not Path(location).is_absolute() for location in locations)
    assert all((onnx_dir / location).is_file() for location in locations)
    onnx.checker.check_model(str(onnx_dir / "lm.onnx"))

    artifacts = load_reference_artifacts(model_dir, "Bruno")
    assert metadata["source_model_revision"] == artifacts.model_revision
    with np.load(Path(__file__).parent / "fixtures" / "reference_bruno.npz", allow_pickle=False) as capture:
        expected_prompt_ids = capture["prompt_ids"]
        expected_generated_ids = capture["generated_raw_ids"]
    prompt_ids = np.asarray(artifacts.build_prompt("Hello from Kitten."), dtype=np.int64)
    np.testing.assert_array_equal(prompt_ids, expected_prompt_ids)
    assert expected_generated_ids.shape == (32,)

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = load_hf_lm(model_dir, profile.graph_dtype)
    model.eval()
    device = torch.device("cpu")
    speaker_projection = _load_speaker_projection(
        model_dir, artifacts.config, profile.graph_dtype, device
    )
    speaker_embedding = torch.from_numpy(artifacts.speaker_embedding[None])
    with torch.inference_mode():
        speaker_hidden_torch = speaker_projection(speaker_embedding)
    speaker_session = ort.InferenceSession(
        str(speaker_path), providers=["CPUExecutionProvider"]
    )
    speaker_hidden_onnx = speaker_session.run(
        ["speaker_hidden"], {"speaker_embedding": artifacts.speaker_embedding[None]}
    )[0]
    np.testing.assert_allclose(
        speaker_hidden_onnx,
        speaker_hidden_torch.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    embedding_table = np.load(onnx_dir / metadata["embedding_file"], mmap_mode="r")
    prompt_tensor = torch.from_numpy(prompt_ids[None])
    with torch.inference_mode():
        token_embeddings = model.get_input_embeddings()(prompt_tensor)
    exported_embeddings = torch.from_numpy(
        np.asarray(embedding_table[prompt_ids], dtype=np.float32).copy()[None]
    )
    torch.testing.assert_close(exported_embeddings, token_embeddings, rtol=0, atol=0)
    inputs_embeds = token_embeddings.clone()
    inputs_embeds[:, 0, :] = torch.from_numpy(speaker_hidden_onnx)
    torch.testing.assert_close(
        inputs_embeds[:, 0, :], torch.from_numpy(speaker_hidden_onnx), rtol=0, atol=0
    )
    torch.testing.assert_close(inputs_embeds[:, 1:, :], token_embeddings[:, 1:, :], rtol=0, atol=0)

    layers = int(model.config.num_hidden_layers)
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    past_names, _ = _io_names(layers)
    attention_mask = torch.ones((1, len(prompt_ids)), dtype=torch.long)
    position_ids = torch.arange(len(prompt_ids), dtype=torch.long)[None, :]
    with torch.inference_mode():
        reference = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(onnx_dir / "lm.onnx"),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    empty_cache = [
        np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32)
        for _ in range(2 * layers)
    ]
    onnx_outputs = session.run(
        None,
        {
            "inputs_embeds": inputs_embeds.numpy(),
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
            **dict(zip(past_names, empty_cache, strict=True)),
        },
    )
    logits = reference.logits[:, -1, :].float()
    _assert_outputs(onnx_outputs, logits, reference.past_key_values)
    onnx_logits = onnx_outputs[0]
    onnx_cache = onnx_outputs[1:]
    reference_cache = reference.past_key_values
    generated_ids = []

    for step, expected_id in enumerate(expected_generated_ids):
        torch_id = int(logits.argmax(dim=-1)[0])
        onnx_id = int(onnx_logits.argmax(axis=-1)[0])
        assert onnx_id == torch_id, f"ONNX/PyTorch token mismatch at generated token {step}"
        assert torch_id == int(expected_id), f"BF16 reference token mismatch at generated token {step}"
        generated_ids.append(torch_id)

        token_ids = torch.tensor([[torch_id]], dtype=torch.long)
        with torch.inference_mode():
            token_embeds = model.get_input_embeddings()(token_ids)
        past_length = len(prompt_ids) + step
        next_mask = torch.ones((1, past_length + 1), dtype=torch.long)
        next_positions = torch.tensor([[past_length]], dtype=torch.long)
        onnx_outputs = session.run(
            None,
            {
                "inputs_embeds": token_embeds.numpy(),
                "attention_mask": next_mask.numpy(),
                "position_ids": next_positions.numpy(),
                **dict(
                    zip(
                        past_names,
                        onnx_cache,
                        strict=True,
                    )
                ),
            },
        )
        with torch.inference_mode():
            reference = model(
                inputs_embeds=token_embeds,
                attention_mask=next_mask,
                position_ids=next_positions,
                past_key_values=reference_cache,
                use_cache=True,
                return_dict=True,
            )
        logits = reference.logits[:, -1, :].float()
        _assert_outputs(onnx_outputs, logits, reference.past_key_values)
        onnx_logits = onnx_outputs[0]
        onnx_cache = onnx_outputs[1:]
        reference_cache = reference.past_key_values

    np.testing.assert_array_equal(generated_ids, expected_generated_ids)


    original_inference_session = ort.InferenceSession

    def reuse_session(path, *args, **kwargs):
        if Path(path) == onnx_dir / "lm.onnx":
            return session
        if Path(path) == speaker_path:
            return speaker_session
        return original_inference_session(path, *args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", reuse_session)
    runtime = Kitten2OnnxLM(onnx_dir, speaker_model=speaker_path)
    result = runtime.generate(
        prompt_ids.tolist(),
        token_map=artifacts.token_map,
        max_new_tokens=32,
        speaker_embedding=artifacts.speaker_embedding,
        settings=SamplingSettings(
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            token_run_penalty=0.0,
        ),
        seed=7,
    )
    expected_raw_ids = expected_generated_ids.tolist()
    expected_s3_ids = [
        token - artifacts.token_map.audio_id_base
        for token in expected_raw_ids
        if artifacts.token_map.audio_id_base <= token < artifacts.token_map.audio_id_end
    ]
    assert result.raw_ids == expected_raw_ids
    assert result.s3_ids == expected_s3_ids
