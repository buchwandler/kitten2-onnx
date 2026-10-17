from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import scripts.search_ternary_nbits as search
from scripts.search_ternary_nbits import (
    BoundedDeltaEvaluator,
    _diagnostic_score,
    _evaluate_delta_worker,
    _legacy_seed_sets,
    _promote_finalist,
    _run_beam_search,
    _semantic_fallback_groups,
    beam_expansions,
    bounded_frontier,
    candidate_key,
    delta_candidate_key,
    exact_configurations,
    finalist_eligible,
    import_legacy_cache,
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



def _row(key: str, score: list[float | int], byte_cost: int) -> dict:
    return {
        "candidate_key": key,
        "excluded_nodes": [key],
        "diagnostic_score": score,
        "bundle_bytes_estimate": byte_cost,
        "status": "diagnostic_complete",
    }


def test_legacy_cache_is_imported_only_as_untrusted_seed_data(tmp_path):
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "baseline_bundle_files": {"lm/lm.onnx": "hash"},
                "inventory_sha256": "inventory",
                "cache": {
                    "old-key": {
                        "candidate_key": "old-key",
                        "excluded_nodes": ["node-b", "node-a"],
                        "diagnostic_score": [1, 2, 3.0, 4.0],
                        "bundle": "/protected/old-candidate",
                        "status": "diagnostic_complete",
                    }
                },
                "group_evaluations": [{"name": "old-group"}],
            }
        ),
        encoding="utf-8",
    )

    imported = import_legacy_cache(
        legacy_path,
        baseline_bundle_files={"lm/lm.onnx": "hash"},
        inventory_sha256="inventory",
    )

    assert imported["reusable_count"] == 0
    assert imported["rows"][0]["reusable"] is False
    assert "bundle" not in imported["rows"][0]
    assert _legacy_seed_sets(imported, width=2) == [["node-a", "node-b"]]
    assert imported["group_evaluations"] == [{"name": "old-group"}]


def test_evaluator_retries_worker_failure_and_reuses_identity_bound_results():
    identity = {"reference": "trusted-pack"}
    calls = []
    updates = []
    responses = iter(
        [
            {"status": "worker_failed"},
            {"status": "diagnostic_complete", "diagnostic_score": [1, 2, 3.0, 4.0]},
        ]
    )

    def evaluate(nodes, _label):
        calls.append(nodes)
        return next(responses)

    cache = {}
    evaluator = BoundedDeltaEvaluator(
        identity,
        cache,
        max_evaluations=2,
        max_retries=1,
        evaluate_one=evaluate,
        on_update=lambda current, count: updates.append((len(current), count)),
    )

    row = evaluator.evaluate(["node-b", "node-a", "node-a"], "seed")
    assert row["status"] == "diagnostic_complete"
    assert row["attempts"] == 2
    assert calls == [["node-a", "node-b"], ["node-a", "node-b"]]
    assert evaluator.evaluation_count == 2
    assert updates[-1] == (1, 2)

    cached = evaluator.evaluate(["node-a", "node-b"], "another-label")
    assert cached is row
    assert len(calls) == 2
    assert delta_candidate_key(identity, ["node-a", "node-b"]) != delta_candidate_key(
        {"reference": "different-pack"}, ["node-a", "node-b"]
    )


def test_evaluator_enforces_absolute_budget_and_stops_on_resource_guards():
    evaluator = BoundedDeltaEvaluator(
        {},
        {},
        max_evaluations=1,
        max_retries=2,
        evaluate_one=lambda _nodes, _label: {"status": "worker_failed"},
        on_update=lambda _cache, _count: None,
    )

    assert evaluator.evaluate(["one"], "retry") is None
    assert evaluator.evaluation_count == 1
    assert evaluator.stop_condition["reason"] == "evaluation_budget_exhausted"

    for status in ("worker_oom_guard", "disk_budget_exceeded", "worker_timeout"):
        calls = []
        guarded = BoundedDeltaEvaluator(
            {},
            {},
            max_evaluations=3,
            max_retries=2,
            evaluate_one=lambda _nodes, _label, status=status, calls=calls: calls.append(status)
            or {"status": status},
            on_update=lambda _cache, _count: None,
        )
        row = guarded.evaluate([status], "resource-stop")
        assert row["status"] == status
        assert guarded.evaluation_count == 1
        assert guarded.stop_condition["reason"] == status
        assert calls == [status]


def test_bounded_frontier_orders_candidates_and_caps_width():
    rows = [
        _row("best-score", [0, 0, 0.0, 0.0], 500),
        _row("dominated", [1, 1, 1.0, 1.0], 1000),
        _row("same-score-cheaper", [1, 1, 1.0, 1.0], 300),
        _row("worse-score-cheapest", [2, 2, 2.0, 2.0], 100),
    ]

    frontier = bounded_frontier(rows, width=3)

    assert [row["candidate_key"] for row in frontier] == [
        "best-score",
        "same-score-cheaper",
        "dominated",
    ]
    assert [state for state in beam_expansions([_row("b", [1, 1, 1, 1], 1)], ["c", "b", "d"])] == [
        ["b", "c"],
        ["b", "d"],
    ]


def test_beam_checks_strict_parity_only_for_zero_diagnostic_candidates():
    scores = {
        ("node-a",): [0, 0, 0.0, 0.0],
        (): [1, 1, 1.0, 1.0],
        ("node-b",): [1, 1, 1.0, 1.0],
    }
    strict_calls = []
    evaluator = BoundedDeltaEvaluator(
        {"identity": "fixed"},
        {},
        max_evaluations=5,
        max_retries=0,
        evaluate_one=lambda nodes, _label: {
            "status": "diagnostic_complete",
            "diagnostic_score": scores[tuple(nodes)],
            "bundle_bytes_estimate": 100,
        },
        on_update=lambda _cache, _count: None,
    )

    finalist = _run_beam_search(
        evaluator,
        [["node-a"]],
        ["node-b"],
        beam_width=2,
        strict_check=lambda row: strict_calls.append(row["excluded_nodes"]) or {"status": "passed"},
    )

    assert finalist["status"] == "strict_parity_passed"
    assert finalist["excluded_nodes"] == ["node-a"]
    assert strict_calls == [["node-a"]]
    assert evaluator.evaluation_count == 2
    assert finalist_eligible(finalist, 100)
    assert not finalist_eligible(finalist, 99)
    assert not finalist_eligible({**finalist, "diagnostic_score": [1, 0, 0, 0]}, 100)


def test_beam_never_returns_a_zero_diagnostic_strict_failure():
    evaluator = BoundedDeltaEvaluator(
        {},
        {},
        max_evaluations=2,
        max_retries=0,
        evaluate_one=lambda _nodes, _label: {
            "status": "diagnostic_complete",
            "diagnostic_score": [0, 0, 0.0, 0.0],
        },
        on_update=lambda _cache, _count: None,
    )

    finalist = _run_beam_search(
        evaluator,
        [["node-a"]],
        [],
        beam_width=1,
        strict_check=lambda _row: {"status": "failed"},
    )

    assert finalist is None
    assert next(iter(evaluator.cache.values()))["status"] == "strict_parity_failed"



def test_beam_retries_cached_strict_worker_failure_on_resume():
    identity = {"reference": "fixed"}
    nodes = ["node-a"]
    row = {
        "candidate_key": delta_candidate_key(identity, nodes),
        "excluded_nodes": nodes,
        "diagnostic_score": [0, 0, 0.0, 0.0],
        "bundle_bytes_estimate": 100,
        "status": "strict_parity_worker_failed",
        "strict_delta_parity": {"status": "worker_failed"},
    }
    evaluator = BoundedDeltaEvaluator(
        identity,
        {row["candidate_key"]: row},
        max_evaluations=2,
        max_retries=0,
        evaluate_one=lambda current_nodes, _label: {
            "status": "diagnostic_complete",
            "diagnostic_score": [1, 1, 1.0, 1.0],
        }
        if not current_nodes
        else {},
        on_update=lambda _cache, _count: None,
    )
    strict_calls = []

    finalist = _run_beam_search(
        evaluator,
        [nodes],
        [],
        beam_width=1,
        strict_check=lambda _row: strict_calls.append(True) or {"status": "passed"},
    )

    assert finalist["status"] == "strict_parity_passed"
    assert strict_calls == [True]
    assert evaluator.evaluation_count == 1



def test_beam_uses_evaluation_budget_instead_of_stopping_on_stagnant_levels():
    evaluator = BoundedDeltaEvaluator(
        {},
        {},
        max_evaluations=30,
        max_retries=0,
        evaluate_one=lambda nodes, _label: {
            "status": "diagnostic_complete",
            "diagnostic_score": (
                [0, 0, 0.0, 0.0]
                if len(nodes) == 4
                else [4 - len(nodes), 4 - len(nodes), 1.0, 1.0]
            ),
        },
        on_update=lambda _cache, _count: None,
    )

    finalist = _run_beam_search(
        evaluator,
        [["a"]],
        ["a", "b", "c", "d"],
        beam_width=2,
        strict_check=lambda _row: {"status": "passed"},
        max_depth=4,
    )

    assert finalist is not None
    assert len(finalist["excluded_nodes"]) == 4
    assert finalist["status"] == "strict_parity_passed"



def test_finalist_promotion_rejects_nonpassing_rows_before_materialization(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "run_candidate_worker", lambda *_args, **_kwargs: pytest.fail("unexpected worker"))
    row = {
        "candidate_key": "candidate-key",
        "excluded_nodes": ["node-a"],
        "diagnostic_score": [0, 0, 0.0, 0.0],
        "bundle_bytes_estimate": 10,
        "strict_delta_parity": {"status": "failed"},
    }

    with pytest.raises(ValueError, match="Finalist must have zero diagnostic score"):
        _promote_finalist(
            row,
            tmp_path / "parent",
            tmp_path / "output",
            tmp_path / "reports",
            tmp_path / "model",
            size_budget_bytes=20,
            max_worker_rss_gib=2,
            max_search_disk_gib=5,
            worker_timeout_seconds=10,
        )


def test_delta_worker_cleans_scratch_and_records_delta_cost(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    reports = tmp_path / "reports"

    def build(_parent, _baseline, _inventory, nodes, model_path, fallback_path):
        assert nodes == {"node-a"}
        model_path.write_bytes(b"model")
        fallback_path.write_bytes(b"weights")
        return SimpleNamespace(fallback_storage={"net_bundle_byte_cost": 32})

    def run(_command, **kwargs):
        assert kwargs["model_path"].read_bytes() == b"model"
        assert kwargs["fallback_data_path"].read_bytes() == b"weights"
        return {
            "status": "diagnostic_complete",
            "diagnostic_score": [1, 2, 3.0, 4.0],
            "fallback_data_bytes": 7,
        }

    monkeypatch.setattr(search, "build_fallback_delta", build)
    monkeypatch.setattr(search, "run_candidate_worker", run)
    evaluate = _evaluate_delta_worker(
        tmp_path / "parent.onnx",
        tmp_path / "baseline.onnx",
        tmp_path / "inventory.json",
        {"nodes": []},
        tmp_path / "reference",
        "reference-identity",
        scratch,
        reports,
        100,
        max_worker_rss_gib=2,
        max_search_disk_gib=5,
        worker_timeout_seconds=10,
    )

    result = evaluate(["node-a"], "test")

    assert result["status"] == "diagnostic_complete"
    assert result["bundle_bytes_estimate"] == 132
    assert result["fallback_data_bytes"] == 7
    assert list(scratch.iterdir()) == []


def test_delta_worker_build_failure_is_recorded_and_scratch_is_removed(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"

    def fail_build(*_args):
        raise OSError("temporary build failure")

    monkeypatch.setattr(search, "build_fallback_delta", fail_build)
    evaluate = _evaluate_delta_worker(
        tmp_path / "parent.onnx",
        tmp_path / "baseline.onnx",
        tmp_path / "inventory.json",
        {"nodes": []},
        tmp_path / "reference",
        "reference-identity",
        scratch,
        tmp_path / "reports",
        100,
        max_worker_rss_gib=2,
        max_search_disk_gib=5,
        worker_timeout_seconds=10,
    )
    evaluator = BoundedDeltaEvaluator(
        {},
        {},
        max_evaluations=1,
        max_retries=0,
        evaluate_one=evaluate,
        on_update=lambda _cache, _count: None,
    )

    row = evaluator.evaluate(["node-a"], "test")

    assert row["status"] == "delta_build_failed"
    assert "temporary build failure" in row["error"]
    assert list(scratch.iterdir()) == []


def test_controller_import_does_not_load_framework_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scripts.search_ternary_nbits; "
                "assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules"
            ),
        ],
        check=True,
    )
