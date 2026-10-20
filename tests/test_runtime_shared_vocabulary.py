from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_shared_vocabulary_reconstructs_emb4_for_lookup_and_projection(tmp_path, monkeypatch):
    import kitten2_onnx.tl2_emb4 as emb4
    from kitten2_onnx.runtime import SharedVocabulary

    table = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.bfloat16)
    monkeypatch.setattr(emb4, "read_emb4_embedding_asset", lambda path: table)

    class FakeOrtValue:
        @staticmethod
        def ortvalue_from_numpy(array):
            return array

    ort = SimpleNamespace(OrtValue=FakeOrtValue)
    vocabulary = SharedVocabulary(
        tmp_path,
        {
            "embedding_storage_dtype": "kitten-emb4",
            "embedding_file": "vocabulary.emb4.safetensors",
            "vocab_size": 2,
            "hidden_size": 2,
        },
        ort,
    )

    np.testing.assert_array_equal(vocabulary.lookup(np.asarray([1]), np.dtype("float32")), [[0.5, 3.0]])
    assert vocabulary.ortvalue is vocabulary.weights

def test_cpu_initializer_override_probe_is_isolated():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    script = textwrap.dedent(
        """
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper

        hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
        placeholder = numpy_helper.from_array(np.zeros((3, 2), dtype=np.float32), name="weight")
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["hidden", "weight"], ["output"])],
            "initializer_override_probe",
            [hidden],
            [output],
            initializer=[placeholder],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = min(model.ir_version, 10)
        weight = np.arange(6, dtype=np.float32).reshape(3, 2)
        options = ort.SessionOptions()
        options.add_initializer("weight", ort.OrtValue.ortvalue_from_numpy(weight))
        try:
            session = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            actual = session.run(None, {"hidden": np.ones((1, 3), dtype=np.float32)})[0]
        except Exception as error:
            print(f"unsupported:{type(error).__name__}")
        else:
            print("supported" if np.array_equal(actual, weight.sum(axis=0, keepdims=True)) else "incorrect")
        """
    )

    def disable_core_dump():
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=disable_core_dump if os.name == "posix" else None,
    )
    if result.returncode == -signal.SIGSEGV:
        return
    assert result.returncode == 0, result.stderr
    outcome = result.stdout.strip()
    assert outcome == "supported" or outcome.startswith("unsupported:")
