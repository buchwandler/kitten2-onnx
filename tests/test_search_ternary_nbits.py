from __future__ import annotations

from scripts.search_ternary_nbits import (
    _diagnostic_score,
    _semantic_fallback_groups,
    candidate_key,
    exact_configurations,
)


def test_exact_search_configurations_follow_declared_order():
    assert exact_configurations() == [
        {"bits": 4, "block_size": 64, "accuracy_level": 1},
        {"bits": 4, "block_size": 32, "accuracy_level": 1},
        {"bits": 4, "block_size": 16, "accuracy_level": 1},
    ]


def test_candidate_key_is_stable_and_binds_source_config_and_exclusions():
    source_files = {"lm/lm.onnx": "graph-hash", "lm/lm.onnx.data": "data-hash"}
    config = {"bits": 4, "block_size": 32, "accuracy_level": 1}
    first = candidate_key(source_files, "inventory-hash", config, ["node_b", "node_a"])
    reordered = candidate_key(source_files, "inventory-hash", config, ["node_a", "node_b", "node_a"])

    assert first == reordered
    assert first != candidate_key(source_files, "other-inventory", config, ["node_a", "node_b"])
    assert first != candidate_key(source_files, "inventory-hash", {**config, "block_size": 16})
    assert first != candidate_key(source_files, "inventory-hash", config, ["node_a"])
    assert first != candidate_key({"lm/lm.onnx": "other-graph"}, "inventory-hash", config, ["node_a", "node_b"])



def test_semantic_fallback_groups_cover_late_layer_projection_sets():
    modules = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    inventory = {
        "nodes": [
            {
                "eligible_exact_ternary": True,
                "layer": layer,
                "module": module,
                "node": f"layer-{layer}-{module}",
            }
            for layer in range(28)
            for module in modules
        ]
    }

    groups = {group["name"]: group["nodes"] for group in _semantic_fallback_groups(inventory)}

    assert len(groups) == 76
    assert len(groups["layer-23"]) == 7
    assert len(groups["v-proj-layers-23-27"]) == 5
    assert len(groups["attention-layer-23"]) == 4
    assert len(groups["mlp-layer-23"]) == 3
    assert len(groups["mlp-gate_proj-layer-23"]) == 1
    assert len(groups["attention-layers-23-27"]) == 20
    assert len(groups["mlp-projections-layers-23-27"]) == 15
    assert len(groups["layer-0"]) == 7
    assert len(groups["self_attn-q_proj-layer-23"]) == 1


def test_diagnostic_score_uses_declared_lexicographic_fields():
    assert _diagnostic_score({"diagnostic_score": [2, 8, 1.5, 0.25]}) == (2, 8, 1.5, 0.25)
