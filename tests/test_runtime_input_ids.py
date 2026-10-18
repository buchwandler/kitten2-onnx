from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kitten2_onnx.runtime import Kitten2OnnxLM


@pytest.mark.parametrize(
    (
        "hidden_size",
        "kv_heads",
        "head_dim",
        "io_dtype",
        "expected_type",
        "expected_numpy_dtype",
        "provider",
        "runtime_memory_profile",
    ),
    [
        (4, 1, 1, "float32", "tensor(float)", np.float32, "CPUExecutionProvider", "default"),
        (4, 1, 1, "float16", "tensor(float16)", np.float16, "CUDAExecutionProvider", "default"),
        (1024, 8, 128, "float32", "tensor(float)", np.float32, "CPUExecutionProvider", "default"),
        (4, 1, 1, "float32", "tensor(float)", np.float32, "CPUExecutionProvider", "low-memory-exact-ternary"),
    ],
)
def test_input_ids_runtime_projects_speaker_only_during_prefill(
    tmp_path,
    monkeypatch,
    hidden_size,
    kv_heads,
    head_dim,
    io_dtype,
    expected_type,
    expected_numpy_dtype,
    provider,
    runtime_memory_profile,
):

    class FakeSession:
        def __init__(self, path, providers, sess_options=None):
            self.path = Path(path)
            self.providers = providers
            self.sess_options = sess_options
            self.feeds = []

        def get_inputs(self):
            if self.path.name == "speaker.onnx":
                return [SimpleNamespace(name="speaker_embedding", type="tensor(float)")]
            return [
                SimpleNamespace(name="input_ids", type="tensor(int64)"),
                SimpleNamespace(name="speaker_hidden", type=expected_type),
                SimpleNamespace(name="attention_mask", type="tensor(int64)"),
                SimpleNamespace(name="position_ids", type="tensor(int64)"),
                SimpleNamespace(name="past_key_0", type="tensor(float)"),
                SimpleNamespace(name="past_value_0", type="tensor(float)"),
            ]

        def run(self, output_names, feeds):
            self.feeds.append(feeds)
            if self.path.name == "speaker.onnx":
                return [np.arange(hidden_size, dtype=np.float32)[None, :]]
            past_length = feeds["past_key_0"].shape[2]
            seq_length = feeds["input_ids"].shape[1]
            cache = np.zeros((1, kv_heads, past_length + seq_length, head_dim), dtype=np.float32)
            logits = np.zeros((1, 16), dtype=np.float32)
            return [logits, cache, cache]

    sessions = []

    def create_session(path, providers, sess_options=None):
        session = FakeSession(path, providers, sess_options)
        sessions.append(session)
        return session

    class FakeSessionOptions:
        pass

    fake_runtime = SimpleNamespace(
        InferenceSession=create_session,
        SessionOptions=FakeSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_DISABLE_ALL="disable"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    metadata = {
        "input_mode": "input_ids",
        "input_names": [
            "input_ids",
            "speaker_hidden",
            "attention_mask",
            "position_ids",
            "past_key_0",
            "past_value_0",
        ],
        "runtime_profile": "cuda-fp16" if io_dtype == "float16" else "dev-fp32",
        "runtime_memory_profile": runtime_memory_profile,
        "runtime_provider": provider,
        "io_dtype": io_dtype,
        "hidden_size": hidden_size,
        "num_hidden_layers": 1,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
    }
    (tmp_path / "lm.meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "lm.onnx").write_bytes(b"fake graph")
    speaker_path = tmp_path / "speaker.onnx"
    speaker_path.write_bytes(b"fake graph")

    runtime = Kitten2OnnxLM(tmp_path, speaker_model=speaker_path, providers=[provider])
    assert runtime._empty_cache()["past_key_0"].shape == (1, kv_heads, 0, head_dim)
    logits, past = runtime.step(
        np.asarray([[3, 5]], dtype=np.int64),
        speaker_embedding=np.asarray([0.25, -0.5], dtype=np.float32),
    )

    assert past["past_key_0"].shape == (1, kv_heads, 2, head_dim)
    assert runtime.embeddings is None
    assert logits.shape == (1, 16)
    lm_session = sessions[0]
    if runtime_memory_profile == "low-memory-exact-ternary":
        options = lm_session.sess_options
        assert options.intra_op_num_threads == 1
        assert options.inter_op_num_threads == 1
        assert options.execution_mode == "sequential"
        assert options.enable_mem_pattern is False
        assert options.enable_cpu_mem_arena is False
        assert options.graph_optimization_level == "disable"
    np.testing.assert_array_equal(lm_session.feeds[0]["input_ids"], [[3, 5]])
    np.testing.assert_array_equal(
        lm_session.feeds[0]["speaker_hidden"], np.arange(hidden_size)[None, :]
    )
    assert lm_session.feeds[0]["speaker_hidden"].dtype == expected_numpy_dtype
    np.testing.assert_array_equal(lm_session.feeds[0]["position_ids"], [[0, 1]])

    runtime.step(np.asarray([[7]], dtype=np.int64), past=past)
    np.testing.assert_array_equal(lm_session.feeds[1]["input_ids"], [[7]])
    np.testing.assert_array_equal(lm_session.feeds[1]["speaker_hidden"], np.zeros((1, hidden_size)))
    assert lm_session.feeds[1]["speaker_hidden"].dtype == expected_numpy_dtype
    np.testing.assert_array_equal(lm_session.feeds[1]["position_ids"], [[2]])
