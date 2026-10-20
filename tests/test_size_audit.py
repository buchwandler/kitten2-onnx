from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
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
    storage = report["storage_breakdown"]
    assert storage["lm_bytes"] == report["directories"]["lm"]
    assert storage["non_lm_bytes"] == report["bundle_bytes"] - storage["lm_bytes"]
    assert storage["vocabulary_initializer_storage_bytes"] == graph["vocabulary"][
        "initializer_storage_bytes"
    ]
    assert storage["embedding_file_bytes"] == graph["vocabulary"]["embedding_file"][
        "file_bytes"
    ]


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

    graph_report = _graph_size_report(graph_path)
    report = graph_report["matmul_nbits"]
    assert graph_report["matmul_nbits4"] == report

    assert report["node_count"] == 1
    assert report["initializer_count"] == 3
    assert report["storage_bytes_by_role"] == {
        "packed_codes": 256,
        "scales": 16,
        "zero_points": 2,
    }
    assert report["total_storage_bytes"] == 274
    assert report["nodes"][0]["node"] == "ternary_projection"
    configurations = report["configurations"]
    assert len(configurations) == 1
    assert {
        key: configurations[0][key]
        for key in (
            "bits",
            "block_size",
            "accuracy_level",
            "node_count",
            "initializer_count",
            "storage_bytes_by_role",
            "total_storage_bytes",
        )
    } == {
        "bits": 4,
        "block_size": 128,
        "accuracy_level": 1,
        "node_count": 1,
        "initializer_count": 3,
        "storage_bytes_by_role": {
            "packed_codes": 256,
            "scales": 16,
            "zero_points": 2,
        },
        "total_storage_bytes": 274,
    }
    assert report["nodes"][0]["accuracy_level"] == 1
    assert report["nodes"][0]["packed_code_bytes"] == 256
    assert report["nodes"][0]["scale_bytes"] == 16
    assert report["nodes"][0]["zero_point_bytes"] == 2



def test_size_audit_groups_nbits_storage_by_configuration(tmp_path):
    graph_path = tmp_path / "lm.onnx"
    initializers = [
        onnx.numpy_helper.from_array(np.zeros((2, 2, 32), dtype=np.uint8), "codes_64"),
        onnx.numpy_helper.from_array(np.ones((4,), dtype=np.float32), "scales_64"),
        onnx.numpy_helper.from_array(np.full((2,), 0x88, dtype=np.uint8), "zeros_64"),
        onnx.numpy_helper.from_array(np.zeros((2, 4, 16), dtype=np.uint8), "codes_32"),
        onnx.numpy_helper.from_array(np.ones((8,), dtype=np.float32), "scales_32"),
        onnx.numpy_helper.from_array(np.full((4,), 0x88, dtype=np.uint8), "zeros_32"),
    ]
    nodes = [
        helper.make_node(
            "MatMulNBits",
            ["A", "codes_64", "scales_64", "zeros_64"],
            ["Y64"],
            name="projection_64",
            domain="com.microsoft",
            K=128,
            N=2,
            bits=4,
            block_size=64,
            accuracy_level=1,
        ),
        helper.make_node(
            "MatMulNBits",
            ["A", "codes_32", "scales_32", "zeros_32"],
            ["Y32"],
            name="projection_32",
            domain="com.microsoft",
            K=128,
            N=2,
            bits=4,
            block_size=32,
            accuracy_level=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "nbits-config-audit",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 128])],
        [
            helper.make_tensor_value_info("Y64", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("Y32", TensorProto.FLOAT, [1, 2]),
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, str(graph_path))

    report = _graph_size_report(graph_path)["matmul_nbits"]

    assert report["node_count"] == 2
    assert report["storage_bytes_by_role"] == {
        "packed_codes": 256,
        "scales": 48,
        "zero_points": 6,
    }
    assert [
        (row["bits"], row["block_size"], row["accuracy_level"], row["total_storage_bytes"])
        for row in report["configurations"]
    ] == [(4, 32, 0, 164), (4, 64, 1, 146)]
    assert report["nodes"][0]["packed_code_bytes"] == 128
    assert report["nodes"][0]["scale_bytes"] == 16
    assert report["nodes"][0]["zero_point_bytes"] == 2


def test_size_audit_exposes_fallback_storage_and_provenance(tmp_path):
    bundle = _write_tiny_bundle(tmp_path)
    metadata_path = bundle / "lm/lm.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fallback_storage = {
        "fallback_initializer_count": 1,
        "fallback_fp32_bytes": 1024,
        "nbits_bytes_removed": 164,
        "net_bundle_byte_cost": 860,
    }
    provenance = {"onnx_version": "test", "cpu_architecture": "test-cpu"}
    metadata["quantization"] = {
        "fallback_storage": fallback_storage,
        "provenance": provenance,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    graph = _graph_size_report(bundle / "lm/lm.onnx")
    report = audit_bundle(bundle)

    assert graph["fallback_storage"] == fallback_storage
    assert graph["quantization_provenance"] == provenance
    assert report["storage_breakdown"]["fallback_storage"] == fallback_storage


@pytest.mark.parametrize(
    ("signed_zero_encoding", "encoding_version", "reconstruction"),
    [(False, 1, None), (True, 2, "where")],
)
def test_size_audit_recognizes_legacy_and_signed_zero_exact_storage(
    tmp_path, signed_zero_encoding, encoding_version, reconstruction
):
    graph_path = tmp_path / "lm.onnx"
    initializers = [
        onnx.numpy_helper.from_array(np.zeros((1, 128, 2), dtype=np.int8), "codes"),
        onnx.numpy_helper.from_array(np.ones((1, 1, 2), dtype=np.float32), "scales"),
        onnx.numpy_helper.from_array(np.asarray([128, 2], dtype=np.int64), "shape"),
    ]
    nodes = []
    product_name = "scaled_codes"
    if signed_zero_encoding:
        initializers.extend(
            [
                onnx.numpy_helper.from_array(np.asarray(-2, dtype=np.int8), "marker"),
                onnx.numpy_helper.from_array(np.asarray(-0.0, dtype=np.float32), "negative_zero"),
            ]
        )
        nodes.append(helper.make_node("Equal", ["codes", "marker"], ["negative_zero_mask"]))
    nodes.extend(
        [
            helper.make_node("Cast", ["codes"], ["float_codes"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["float_codes", "scales"], [product_name]),
        ]
    )
    if signed_zero_encoding:
        nodes.append(
            helper.make_node(
                "Where",
                ["negative_zero_mask", "negative_zero", product_name],
                ["signed_zero_codes"],
            )
        )
        product_name = "signed_zero_codes"
    nodes.extend(
        [
            helper.make_node("Reshape", [product_name, "shape"], ["weight"]),
            helper.make_node("MatMul", ["A", "weight"], ["Y"], name="exact_matmul"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "exact-storage-audit",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 128])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, str(graph_path))

    report = _graph_size_report(graph_path)["ternary_exact_int8"]

    assert report["node_count"] == 1
    assert report["encoding_versions"] == [encoding_version]
    assert report["negative_zero_code"] == (-2 if signed_zero_encoding else None)
    assert report["negative_zero_preserved"] is signed_zero_encoding
    assert report["negative_zero_reconstruction"] == reconstruction
    assert report["initializer_count"] == 2
    assert report["storage_bytes_by_role"] == {"int8_codes": 256, "float32_scales": 8}
    assert report["total_storage_bytes"] == 264
    assert report["source_fp32_bytes_replaced"] == 1024
    assert report["storage_reduction_bytes"] == 760
    assert report["exact_storage_ratio"] == 264 / 1024


@pytest.mark.parametrize("variant", ["packed", "emb4"])
def test_size_audit_v2_reports_one_vocabulary_copy(tmp_path, variant):
    bundle = tmp_path / "bundle-v2"
    lm_dir = bundle / "lm"
    lm_dir.mkdir(parents=True)
    embedding_name = "vocabulary.bf16.npy" if variant == "packed" else "vocabulary.emb4.safetensors"
    embedding_path = lm_dir / embedding_name
    if variant == "packed":
        np.save(embedding_path, np.zeros((3, 8), dtype=np.uint16), allow_pickle=False)
        method = "bfloat16-u16"
    else:
        from safetensors.torch import save_file

        save_file(
            {
                "emb4/tier_map": torch.zeros(3, dtype=torch.uint8),
                "emb4/t0_codes": torch.zeros(3, dtype=torch.uint8),
                "emb4/t0_centers": torch.zeros((2, 8), dtype=torch.float32),
                "emb4/t0_s": torch.ones(3, dtype=torch.float32),
                "emb4/t0_t": torch.ones(8, dtype=torch.float32),
            },
            str(embedding_path),
            metadata={
                "schema": "kitten_ternary_tl2_rl_v1",
                "embedding_metadata": json.dumps({"V": 3, "D": 8, "Ks": "[2]"}),
            },
        )
        method = "kitten-emb4"

    (lm_dir / "lm.meta.json").write_text(
        json.dumps(
            {
                "graph": "lm.onnx",
                "vocab_size": 3,
                "hidden_size": 8,
                "embedding_file": embedding_name,
                "embedding_storage_dtype": method,
            }
        ),
        encoding="utf-8",
    )
    core_inputs = [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 8])]
    core_outputs = [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, 8])]
    core = helper.make_graph([], "core", core_inputs, core_outputs)
    head_inputs = [
        helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, 8]),
        helper.make_tensor_value_info("vocabulary_weight", TensorProto.FLOAT, [3, 8]),
    ]
    head_outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])]
    head = helper.make_graph([], "head", head_inputs, head_outputs)
    for path, graph in ((lm_dir / "lm.onnx", core), (lm_dir / "lm_output.onnx", head)):
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = min(model.ir_version, 10)
        onnx.save_model(model, str(path))
    manifest = {
        "format": "kitten2-onnx-bundle-v2",
        "lm": {
            "model": "lm/lm.onnx",
            "core_model": "lm/lm.onnx",
            "head_model": "lm/lm_output.onnx",
            "metadata": "lm/lm.meta.json",
            "embeddings": f"lm/{embedding_name}",
            "storage": {
                "transformer": {
                    "method": "matmul-nbits",
                    "bits": 4,
                    "block_size": 128,
                    "source_group_size": 128,
                    "zero_point": 8,
                },
                "vocabulary": {
                    "method": method,
                    "file": f"lm/{embedding_name}",
                    "lossy": variant == "emb4",
                },
            },
        },
    }
    (bundle / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_bundle(bundle)

    assert report["vocabulary_storage"]["logical_parameters"] == 24
    assert report["vocabulary_storage"]["physical_copies"] == 1
    assert report["vocabulary_storage"]["stored_bytes"] == embedding_path.stat().st_size
    assert report["vocabulary_storage"]["lossy"] is (variant == "emb4")


def test_size_audit_v2_rejects_duplicate_vocabulary_assets(tmp_path):
    bundle = tmp_path / "bundle-v2"
    lm_dir = bundle / "lm"
    lm_dir.mkdir(parents=True)
    embedding_path = lm_dir / "vocabulary.bf16.npy"
    vocabulary = np.zeros((3, 8), dtype=np.uint16)
    np.save(embedding_path, vocabulary, allow_pickle=False)
    np.save(lm_dir / "vocabulary-copy.npy", vocabulary, allow_pickle=False)
    metadata_path = lm_dir / "lm.meta.json"
    metadata_path.write_text(
        json.dumps({"vocab_size": 3, "hidden_size": 8, "embedding_file": embedding_path.name}),
        encoding="utf-8",
    )
    for name in ("lm.onnx", "lm_output.onnx"):
        graph = helper.make_graph(
            [],
            name,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = min(model.ir_version, 10)
        onnx.save_model(model, str(lm_dir / name))
    manifest = {
        "format": "kitten2-onnx-bundle-v2",
        "lm": {
            "model": "lm/lm.onnx",
            "core_model": "lm/lm.onnx",
            "head_model": "lm/lm_output.onnx",
            "metadata": "lm/lm.meta.json",
            "embeddings": "lm/vocabulary.bf16.npy",
            "storage": {
                "vocabulary": {
                    "method": "bfloat16-u16",
                    "file": "lm/vocabulary.bf16.npy",
                    "lossy": False,
                }
            },
        },
    }
    (bundle / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="physical vocabulary copies"):
        audit_bundle(bundle)
