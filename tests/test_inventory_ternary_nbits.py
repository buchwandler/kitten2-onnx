from __future__ import annotations

import hashlib
import json

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from safetensors.torch import save_file

from scripts.inventory_ternary_nbits import build_inventory


def _write_inventory_inputs(tmp_path, *, node_name="node_linear", duplicate_layer=False):
    weight = torch.zeros((2, 128), dtype=torch.bfloat16)
    weight[0] = torch.where(torch.arange(128) % 3 == 0, 1, 0).to(torch.bfloat16)
    weight[1] = torch.where(torch.arange(128) % 4 == 0, -2, 0).to(torch.bfloat16)
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint_weights = {"model.layers.0.self_attn.q_proj.weight": weight}
    if duplicate_layer:
        checkpoint_weights["model.layers.1.self_attn.q_proj.weight"] = weight.clone()
    save_file(checkpoint_weights, str(checkpoint))

    source_weight = np.ascontiguousarray(weight.float().numpy().T)
    initializer = numpy_helper.from_array(source_weight, name="ternary_weight")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "ternary_weight"], ["hidden"], name=node_name),
            helper.make_node("MatMul", ["hidden", "dynamic_weight"], ["output"], name="dynamic_matmul"),
        ],
        "inventory-test",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128]),
            helper.make_tensor_value_info("dynamic_weight", TensorProto.FLOAT, [2, 2]),
        ],
        [
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2]),
        ],
        [initializer],
    )
    source_graph = tmp_path / "source.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), source_graph)
    return source_graph, checkpoint


def test_inventory_maps_exact_ternary_weight_and_hashes_rows(tmp_path):
    source_graph, checkpoint = _write_inventory_inputs(tmp_path)

    report = build_inventory(source_graph, checkpoint, expected_eligible_matmul_count=1)

    assert report["eligible_exact_ternary_matmul_count"] == 1
    assert report["mapped_eligible_matmul_count"] == 1
    assert report["ambiguous_mapping_count"] == 0
    assert report["repeated_seven_matrix_structure"] is False
    row = report["nodes"][0]
    assert row["node"] == "node_linear"
    assert row["shape"] == [128, 2]
    assert row["eligible_exact_ternary"] is True
    assert row["layer"] == 0
    assert row["module"] == "self_attn.q_proj"
    assert row["checkpoint_key"] == "model.layers.0.self_attn.q_proj.weight"
    assert row["mapping_method"] == "sha256+shape"
    assert set(report["nodes"][1]) == set(row)
    assert report["nodes"][1]["eligible_exact_ternary"] is False
    inventory_identity = {
        "format": report["format"],
        "source_group_size": report["source_group_size"],
        "source_graph_sha256": report["source_graph_sha256"],
        "source_external_data_sha256": report["source_external_data_sha256"],
        "checkpoint_sha256s": [row["sha256"] for row in report["checkpoint_files"]],
        "expected_eligible_matmul_count": 1,
        "nodes": report["nodes"],
    }
    expected_hash = hashlib.sha256(
        json.dumps(inventory_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert report["inventory_sha256"] == expected_hash


def test_inventory_uses_layer_neighborhood_to_resolve_hash_collision(tmp_path):
    source_graph, checkpoint = _write_inventory_inputs(tmp_path, duplicate_layer=True)

    report = build_inventory(source_graph, checkpoint, expected_eligible_matmul_count=1)

    assert report["nodes"][0]["checkpoint_key"] == "model.layers.0.self_attn.q_proj.weight"
    assert report["nodes"][0]["mapping_method"] == "sha256+shape+layer-neighborhood"


def test_inventory_fails_when_hash_collision_remains_ambiguous(tmp_path):
    source_graph, checkpoint = _write_inventory_inputs(
        tmp_path, node_name="unnamed_matmul", duplicate_layer=True
    )

    with pytest.raises(ValueError, match="ambiguous"):
        build_inventory(source_graph, checkpoint, expected_eligible_matmul_count=1)
