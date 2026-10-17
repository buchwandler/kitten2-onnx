from __future__ import annotations

import json

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from safetensors.torch import save_file

from scripts.audit_bundle_size import _graph_size_report, audit_bundle
from scripts.audit_checkpoint_weights import audit_checkpoint


def _write_tiny_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    lm_dir = bundle / "lm"
    lm_dir.mkdir(parents=True)
    embedding = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.save(lm_dir / "embed_tokens.npy", embedding, allow_pickle=False)
    (lm_dir / "lm.meta.json").write_text(
        json.dumps({"vocab_size": 3, "hidden_size": 2, "embedding_file": "embed_tokens.npy"}),
        encoding="utf-8",
    )
    (bundle / "bundle.json").write_text(
        json.dumps({"export": {"runtime_profile": "test-fp32"}}), encoding="utf-8"
    )

    initializers = [
        onnx.numpy_helper.from_array(
            np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(2, 3), "vocab_weight"
        ),
        onnx.numpy_helper.from_array(
            np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(2, 3), "vocab_weight_copy"
        ),
        onnx.numpy_helper.from_array(
            np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(3, 2), "gather_table"
        ),
        onnx.numpy_helper.from_array(
            np.asarray([1, 2, 3, 4], dtype=np.float32).reshape(1, 1, 2, 2), "conv_weight"
        ),
        onnx.numpy_helper.from_array(np.arange(2048, dtype=np.float32), "large_weight"),
    ]
    nodes = [
        helper.make_node("MatMul", ["hidden", "vocab_weight"], ["matmul_out"], name="lm_head"),
        helper.make_node(
            "Gemm", ["hidden", "vocab_weight_copy"], ["gemm_out"], name="duplicate_head"
        ),
        helper.make_node("Gather", ["gather_table", "indices"], ["gather_out"], axis=0),
        helper.make_node("Conv", ["image", "conv_weight"], ["conv_out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_lm",
        [
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 3, 3]),
        ],
        [
            helper.make_tensor_value_info("matmul_out", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("gemm_out", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("gather_out", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, [1, 1, 2, 2]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save_model(
        model,
        lm_dir / "lm.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="lm.onnx.data",
        size_threshold=0,
        convert_attribute=False,
    )
    return bundle


def test_bundle_audit_reports_exact_files_initializers_operators_and_vocabulary(tmp_path):
    bundle = _write_tiny_bundle(tmp_path)

    report = audit_bundle(bundle)
    graph = report["onnx"]["lm"]
    duplicates = graph["duplicate_initializer_groups"]

    assert report["bundle_bytes"] == sum(report["files"].values())
    assert report["directories"]["lm"] == sum(
        size for name, size in report["files"].items() if name.startswith("lm/")
    )
    assert graph["initializer_count"] == 5
    assert graph["initializer_bytes_by_dtype"] == {"FLOAT": 8280}
    assert graph["external_data_bytes"] == (bundle / "lm/lm.onnx.data").stat().st_size
    assert graph["node_type_counts"] == {"Conv": 1, "Gather": 1, "Gemm": 1, "MatMul": 1}
    assert [row["initializer"] for row in graph["initializer_consumers"]["MatMul"]] == [
        "vocab_weight"
    ]
    assert len(duplicates) == 1
    assert duplicates[0]["names"] == ["vocab_weight", "vocab_weight_copy"]
    assert graph["vocabulary"]["tied_weight_storage"] == "multiple"
    assert graph["vocabulary"]["initializer_count"] == 3
    assert (
        graph["vocabulary"]["embedding_file"]["file_bytes"]
        == (bundle / "lm/embed_tokens.npy").stat().st_size
    )
    assert graph["vocabulary"]["total_bytes_attributable_to_vocabulary"] > 0


def test_checkpoint_audit_reports_byte_weighted_ternary_fraction(tmp_path):
    ternary = torch.zeros((2, 128), dtype=torch.bfloat16)
    ternary[0, 0] = 1.0
    ternary[0, 1] = -1.0
    ternary[1, 0] = 2.0
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"ternary.weight": ternary, "dense.weight": dense}, str(checkpoint))

    report = audit_checkpoint(checkpoint)

    assert report["tensor_count"] == 2
    assert report["ternary_tensor_count"] == 1
    assert report["total_parameter_bytes_source_dtype"] == 528
    assert report["ternary_parameter_bytes_bf16"] == 512
    assert report["nonternary_parameter_bytes_bf16"] == 8
    assert report["ternary_fraction_by_bytes"] == 512 / 520
    assert report["ternary_tensors"][0]["name"] == "ternary.weight"



def test_size_audit_reports_matmul_nbits4_storage_by_operand_role(tmp_path):
    graph_path = tmp_path / "lm.onnx"
    packed = onnx.numpy_helper.from_array(
        np.zeros((2, 2, 64), dtype=np.uint8), "packed_codes"
    )
    scales = onnx.numpy_helper.from_array(np.ones((4,), dtype=np.float32), "scales")
    zero_points = onnx.numpy_helper.from_array(np.full((2,), 0x88, dtype=np.uint8), "zero_points")
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMulNBits",
                ["A", "packed_codes", "scales", "zero_points"],
                ["Y"],
                name="ternary_projection",
                domain="com.microsoft",
                K=256,
                N=2,
                bits=4,
                block_size=128,
                accuracy_level=1,
            )
        ],
        "nbits4-audit",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 256])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        [packed, scales, zero_points],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, str(graph_path))

    report = _graph_size_report(graph_path)["matmul_nbits4"]

    assert report["node_count"] == 1
    assert report["initializer_count"] == 3
    assert report["storage_bytes_by_role"] == {
        "packed_codes": 256,
        "scales": 16,
        "zero_points": 2,
    }
    assert report["total_storage_bytes"] == 274
    assert report["nodes"][0]["node"] == "ternary_projection"
