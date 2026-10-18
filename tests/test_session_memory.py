from __future__ import annotations

import json
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.resource_guard import run_bounded_process


def test_session_memory_checker_runs_tiny_model_under_bounded_runner(tmp_path):
    graph_path = tmp_path / "tiny.onnx"
    result_path = tmp_path / "session.json"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"])],
        "tiny",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(np.asarray([[1.0], [2.0]], dtype=np.float32), name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, graph_path)

    outer_result = run_bounded_process(
        [
            sys.executable,
            "scripts/check_onnx_session_memory.py",
            "--model",
            str(graph_path),
            "--result",
            str(result_path),
            "--run-zero-inference",
        ],
        max_rss_gib=2,
        min_available_gib=0,
        reserve_available_gib=0,
        timeout_seconds=30,
        output_json=tmp_path / "bounded.json",
        log_path=tmp_path / "bounded.log",
        max_additional_swap_gib=None,
        poll_interval_seconds=0.05,
    )

    child_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert outer_result["status"] == "ok"
    assert child_result["status"] == "ok"
    assert child_result["inference_performed"] is True
    assert child_result["output_count"] == 1
