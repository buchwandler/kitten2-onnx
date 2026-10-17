from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.lm_parity_reference import REFERENCE_FORMAT, pack_sha256, sha256_file
from kitten2_onnx.quantize_ternary_nbits import _array_sha256
from scripts.measure_nbits_sensitivity import measure_sensitivity


def _write_reference(root: Path) -> None:
    root.mkdir()
    arrays = {
        "prompt_ids": np.asarray([10, 11, 12], dtype=np.int64),
        "prefill_inputs_embeds": np.ones((3, 128), dtype=np.float32),
        "speaker_hidden": np.zeros(128, dtype=np.float32),
        "generated_ids": np.asarray([2], dtype=np.int64),
        "token_embeds": np.zeros((1, 128), dtype=np.float32),
        "logits": np.zeros((3, 1, 5), dtype=np.float32),
        "cache_keys": np.zeros((2, 1, 1, 4, 2), dtype=np.float32),
        "cache_values": np.zeros((2, 1, 1, 4, 2), dtype=np.float32),
    }
    records = {}
    for name, array in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        records[name] = {
            "file": path.name,
            "shape": list(loaded.shape),
            "dtype": str(loaded.dtype),
            "sha256": sha256_file(path),
        }
    input_names = [
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "past_key_0",
        "past_value_0",
        "past_key_1",
        "past_value_1",
    ]
    metadata = {
        "format": REFERENCE_FORMAT,
        "source_identity": {},
        "lm_metadata": {"input_mode": "inputs_embeds", "input_names": input_names},
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "prompt_length": 3,
        "generated_token_count": 1,
        "arrays": records,
    }
    metadata["pack_sha256"] = pack_sha256(metadata)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _inputs():
    return [
        helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, 3, 128]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 3]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, 3]),
        helper.make_tensor_value_info("past_key_0", TensorProto.FLOAT, [1, 1, "past", 2]),
        helper.make_tensor_value_info("past_value_0", TensorProto.FLOAT, [1, 1, "past", 2]),
        helper.make_tensor_value_info("past_key_1", TensorProto.FLOAT, [1, 1, "past", 2]),
        helper.make_tensor_value_info("past_value_1", TensorProto.FLOAT, [1, 1, "past", 2]),
    ]


def _write_models(root: Path) -> tuple[Path, Path, Path]:
    baseline_path = root / "baseline.onnx"
    parent_path = root / "parent.onnx"
    inventory_path = root / "inventory.json"
    weight = np.zeros((128, 2), dtype=np.float32)
    baseline_graph = helper.make_graph(
        [helper.make_node("MatMul", ["inputs_embeds", "W"], ["Y"], name="linear")],
        "baseline",
        _inputs(),
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 2])],
        [numpy_helper.from_array(weight, "W")],
    )
    baseline = helper.make_model(baseline_graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(baseline, baseline_path)

    parent_nodes = [
        helper.make_node(
            "MatMulNBits",
            ["inputs_embeds", "W_q4", "W_scales", "W_zero"],
            ["Y"],
            name="linear",
            domain="com.microsoft",
            K=128,
            N=2,
            bits=4,
            block_size=128,
            accuracy_level=1,
        )
    ]
    initializers = [
        numpy_helper.from_array(np.zeros((128, 2), dtype=np.uint8), "W_q4"),
        numpy_helper.from_array(np.ones((1, 2), dtype=np.float32), "W_scales"),
        numpy_helper.from_array(np.zeros((1, 2), dtype=np.uint8), "W_zero"),
    ]
    parent_graph = helper.make_graph(
        parent_nodes,
        "parent",
        _inputs(),
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 2])],
        initializers,
    )
    parent = helper.make_model(
        parent_graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    onnx.save(parent, parent_path)

    inventory = {
        "source_graph_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "eligible_exact_ternary_matmul_count": 1,
        "nodes": [
            {
                "eligible_exact_ternary": True,
                "node": "linear",
                "activation": "inputs_embeds",
                "output": "Y",
                "initializer": "W",
                "source_sha256": _array_sha256(weight),
                "layer": 0,
                "module": "mlp.up_proj",
            }
        ],
    }
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    return baseline_path, parent_path, inventory_path


def test_measure_sensitivity_uses_reference_activation_and_cleans_instrumented_files(
    tmp_path, monkeypatch
):
    baseline_path, parent_path, inventory_path = _write_models(tmp_path)
    reference_path = tmp_path / "reference"
    _write_reference(reference_path)
    instrumented_path = tmp_path / "scratch" / "instrumented.onnx"
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()

    class SessionOptions:
        intra_op_num_threads = 0

    class Session:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, output_names, feed):
            activation = feed["inputs_embeds"]
            output = np.matmul(activation, np.zeros((128, 2), dtype=np.float32)) + 0.01
            return [activation if name == "inputs_embeds" else output for name in output_names]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=Session),
    )

    result = measure_sensitivity(
        parent_path,
        baseline_path,
        inventory_path,
        reference_path,
        instrumented_path,
    )

    row = result["ranked_nodes"][0]
    assert result["status"] == "sensitivity_complete"
    assert result["measured_node_count"] == 1
    assert row["node"] == "linear"
    assert row["output_error_count"] == 6
    assert row["local_max_abs_error"] == pytest.approx(0.01)
    assert row["fallback_byte_cost"] == 128 * 2 * 4
    assert row["nbits_bytes_if_quantized"] == 128 * 2 + 2 * 4 + 2
    assert result["ranked_nodes"] == sorted(
        result["ranked_nodes"], key=lambda item: (-item["priority"], -item["local_normalized_error"], item["fallback_byte_cost"], item["node"])
    )
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline_hash
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == parent_hash
    assert not instrumented_path.exists()
