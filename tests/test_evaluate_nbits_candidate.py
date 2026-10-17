from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kitten2_onnx.lm_parity_reference import (
    REFERENCE_FORMAT,
    pack_sha256,
    sha256_file,
)


def _write_pack(root: Path) -> Path:
    root.mkdir(parents=True)
    arrays = {
        "prompt_ids": np.asarray([10, 11, 12], dtype=np.int64),
        "prefill_inputs_embeds": np.zeros((3, 4), dtype=np.float32),
        "speaker_hidden": np.zeros(4, dtype=np.float32),
        "generated_ids": np.asarray([2, 1], dtype=np.int64),
        "token_embeds": np.zeros((2, 4), dtype=np.float32),
        "logits": np.zeros((3, 1, 5), dtype=np.float32),
        "cache_keys": np.zeros((2, 1, 1, 5, 2), dtype=np.float32),
        "cache_values": np.zeros((2, 1, 1, 5, 2), dtype=np.float32),
    }
    arrays["logits"][0, 0, 2] = 5.0
    arrays["logits"][1, 0, 1] = 5.0
    arrays["logits"][2, 0, 4] = 5.0
    records = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        path = root / filename
        np.save(path, array, allow_pickle=False)
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        records[name] = {
            "file": filename,
            "shape": list(loaded.shape),
            "dtype": str(loaded.dtype),
            "sha256": sha256_file(path),
        }
    metadata = {
        "format": REFERENCE_FORMAT,
        "trusted": True,
        "source_identity": {"model_revision": "tiny-model"},
        "baseline_verification": {"diagnostic_score": [0, 0, 0.0, 0.0]},
        "lm_metadata": {
            "input_mode": "inputs_embeds",
            "input_names": [
                "inputs_embeds",
                "attention_mask",
                "position_ids",
                "past_key_0",
                "past_value_0",
                "past_key_1",
                "past_value_1",
            ],
        },
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "prompt_length": 3,
        "generated_token_count": 2,
        "arrays": records,
    }
    metadata["pack_sha256"] = pack_sha256(metadata)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _write_tiny_model(path: Path, reference_path: Path) -> None:
    logits = np.load(reference_path / "logits.npy", allow_pickle=False)[:, 0, :]
    nodes = [
        helper.make_node("ReduceMax", ["position_ids"], ["last_position"], axes=[1], keepdims=0),
        helper.make_node("Sub", ["last_position", "position_offset"], ["logit_index"]),
        helper.make_node("Gather", ["logit_table", "logit_index"], ["selected_logits"], axis=0),
        helper.make_node("ReduceSum", ["inputs_embeds"], ["embed_sum"], keepdims=0),
        helper.make_node("Mul", ["embed_sum", "zero_scalar"], ["zero_embed_sum"]),
        helper.make_node("Add", ["selected_logits", "zero_embed_sum"], ["logits"]),
        helper.make_node("Shape", ["attention_mask"], ["mask_shape"]),
        helper.make_node("Gather", ["mask_shape", "mask_sequence_axis"], ["total_length"], axis=0),
        helper.make_node("Shape", ["past_key_0"], ["past_shape"]),
        helper.make_node("Gather", ["past_shape", "past_sequence_axis"], ["past_length"], axis=0),
        helper.make_node("Sub", ["total_length", "past_length"], ["new_length"]),
        helper.make_node("Unsqueeze", ["new_length", "unsqueeze_axis"], ["new_length_vector"]),
        helper.make_node(
            "Concat", ["cache_shape_prefix", "new_length_vector", "cache_shape_suffix"],
            ["new_cache_shape"], axis=0
        ),
    ]
    initializers = [
        numpy_helper.from_array(logits.astype(np.float32), "logit_table"),
        numpy_helper.from_array(np.asarray(2, dtype=np.int64), "position_offset"),
        numpy_helper.from_array(np.asarray(0, dtype=np.float32), "zero_scalar"),
        numpy_helper.from_array(np.asarray(1, dtype=np.int64), "mask_sequence_axis"),
        numpy_helper.from_array(np.asarray(2, dtype=np.int64), "past_sequence_axis"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), "unsqueeze_axis"),
        numpy_helper.from_array(np.asarray([1, 1], dtype=np.int64), "cache_shape_prefix"),
        numpy_helper.from_array(np.asarray([2], dtype=np.int64), "cache_shape_suffix"),
    ]
    inputs = [
        helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, "sequence", 4]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "total_sequence"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, "sequence"]),
    ]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 5])]
    for layer in range(2):
        for kind in ("key", "value"):
            past_name = f"past_{kind}_{layer}"
            present_name = f"present_{kind}_{layer}"
            new_name = f"new_{kind}_{layer}"
            inputs.append(
                helper.make_tensor_value_info(
                    past_name, TensorProto.FLOAT, [1, 1, "past_sequence", 2]
                )
            )
            nodes.append(helper.make_node("ConstantOfShape", ["new_cache_shape"], [new_name]))
            nodes.append(
                helper.make_node("Concat", [past_name, new_name], [present_name], axis=2)
            )
            outputs.append(
                helper.make_tensor_value_info(
                    present_name, TensorProto.FLOAT, [1, 1, "present_sequence", 2]
                )
            )
    graph = helper.make_graph(
        nodes,
        "tiny-nbits-candidate",
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_candidate_worker_runs_onnx_only_and_matches_tiny_reference(tmp_path):
    reference = _write_pack(tmp_path / "reference")
    model_path = tmp_path / "candidate.onnx"
    output_path = tmp_path / "diagnostic.json"
    _write_tiny_model(model_path, reference)
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_nbits_candidate.py",
            "--model",
            str(model_path),
            "--reference",
            str(reference),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["diagnostic_score"] == [0, 0, 0.0, 0.0]
    assert result["candidate_token_ids"] == [2, 1]
    assert result["onnx_runtime_calls"] == 3
    assert result["heavy_ml_imports_loaded"] == []


def test_candidate_worker_refuses_wrong_reference_source_identity(tmp_path):
    reference = _write_pack(tmp_path / "reference")
    model_path = tmp_path / "not-run.onnx"
    model_path.write_bytes(b"not used")
    output_path = tmp_path / "diagnostic.json"
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_nbits_candidate.py",
            "--model",
            str(model_path),
            "--reference",
            str(reference),
            "--output",
            str(output_path),
            "--expected-source-identity-sha256",
            "incorrect",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "source identity does not match" in completed.stderr
    assert not output_path.exists()


def test_candidate_worker_supervisor_records_resource_metrics(tmp_path):
    from scripts.nbits_worker_process import run_candidate_worker

    search_root = tmp_path / "search"
    search_root.mkdir()
    output_path = tmp_path / "reports/result.json"
    code = (
        "import json,pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
        "'status':'diagnostic_complete','peak_rss_bytes':4096}))"
    )

    result = run_candidate_worker(
        [sys.executable, "-c", code, str(output_path)],
        output_path=output_path,
        search_root=search_root,
        max_worker_rss_gib=1,
        max_search_disk_gib=1,
        timeout_seconds=10,
    )

    assert result["status"] == "diagnostic_complete"
    assert result["worker_exit_code"] == 0
    assert result["peak_rss_bytes"] >= 4096
    assert result["available_memory_before_bytes"] is not None
    assert result["available_memory_after_bytes"] is not None


def test_candidate_worker_supervisor_stops_before_launch_when_disk_budget_is_exceeded(tmp_path):
    from scripts.nbits_worker_process import run_candidate_worker

    search_root = tmp_path / "search"
    search_root.mkdir()
    (search_root / "large.bin").write_bytes(b"more than one byte")
    output_path = tmp_path / "reports/result.json"
    result = run_candidate_worker(
        [sys.executable, "-c", "raise SystemExit(9)"],
        output_path=output_path,
        search_root=search_root,
        max_worker_rss_gib=1,
        max_search_disk_gib=1e-9,
    )

    assert result["status"] == "disk_budget_exceeded"
    assert result["worker_exit_code"] is None


def test_candidate_worker_supervisor_terminates_rss_guarded_process_group(tmp_path, monkeypatch):
    import scripts.nbits_worker_process as supervisor

    search_root = tmp_path / "search"
    search_root.mkdir()
    output_path = tmp_path / "reports/result.json"
    monkeypatch.setattr(supervisor, "_process_tree_rss_bytes", lambda _pid: 2 * 1024**3)

    result = supervisor.run_candidate_worker(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        output_path=output_path,
        search_root=search_root,
        max_worker_rss_gib=1,
        max_search_disk_gib=1,
        timeout_seconds=10,
    )

    assert result["status"] == "worker_oom_guard"
    assert result["oom_guard_triggered"] is True
    assert result["worker_exit_code"] != 0



def test_candidate_worker_terminates_when_search_disk_budget_is_exceeded(tmp_path):
    from scripts.nbits_worker_process import run_candidate_worker

    search_root = tmp_path / "search"
    search_root.mkdir()
    output_path = tmp_path / "reports/result.json"
    worker_code = (
        "import pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_bytes(b'x'*4096); time.sleep(30)"
    )

    result = run_candidate_worker(
        [sys.executable, "-c", worker_code, str(search_root / "growing.bin")],
        output_path=output_path,
        search_root=search_root,
        max_worker_rss_gib=1,
        max_search_disk_gib=1e-7,
        timeout_seconds=10,
    )

    assert result["status"] == "disk_budget_exceeded"
    assert result["disk_budget_triggered"] is True
    assert "exceeding" in result["disk_budget_error"]
