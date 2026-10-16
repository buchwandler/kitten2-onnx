from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import scripts.search_int8_mixed_precision as search


def _make_bundle(path: Path, *, layer_count: int = 28, vocab_size: int = 8) -> Path:
    lm_dir = path / "lm"
    lm_dir.mkdir(parents=True)
    target_count = layer_count * 7 + 1
    nodes = []
    initializers = []
    value_info = []
    for index in range(target_count):
        node_name = "node_linear" if index == 0 else f"node_linear_{index}"
        output_name = f"linear_{index}"
        output_width = vocab_size if index == target_count - 1 else 2
        weight = np.ones((2, output_width), dtype=np.float32)
        initializers.append(numpy_helper.from_array(weight, name=f"weight_{index}"))
        nodes.append(
            helper.make_node(
                "MatMul",
                ["hidden", f"weight_{index}"],
                [output_name],
                name=node_name,
            )
        )
        value_info.append(
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, output_width])
        )
    nodes.append(helper.make_node("Identity", [f"linear_{target_count - 1}"], ["logits"]))
    graph = helper.make_graph(
        nodes,
        "tiny-kitten-lm",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, vocab_size])],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save_model(
        model,
        lm_dir / "lm.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    (lm_dir / "lm.meta.json").write_text(
        json.dumps({"num_hidden_layers": layer_count, "vocab_size": vocab_size}),
        encoding="utf-8",
    )
    (path / "bundle.json").write_text(
        json.dumps({"lm": {"model": "lm/lm.onnx", "metadata": "lm/lm.meta.json"}}),
        encoding="utf-8",
    )
    return path


def test_inventory_verifies_transformer_blocks_and_lm_head(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")

    inventory = search.inventory_matmuls(bundle)

    assert inventory["matmul_count"] == 197
    assert inventory["supported_matmul_count"] == 197
    assert len(inventory["blocks"]) == 28
    assert all(len(block["targets"]) == 7 for block in inventory["blocks"])
    assert inventory["lm_head"]["node"] == "node_linear_196"
    assert inventory["lm_head"]["initializer"] == "weight_196"
    nodes, initializers = search.exclusions_for_candidate(
        inventory,
        preserve_last_blocks=4,
    )
    assert len(nodes) == 29
    assert len(initializers) == 29
    assert nodes[-1] == "node_linear_196"


def test_inventory_rejects_unverified_layer_count(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle", layer_count=27)

    with pytest.raises(ValueError, match="Expected the Kitten2 28-layer LM"):
        search.inventory_matmuls(bundle)


def test_execute_candidate_reuses_optimizer_and_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "bundle.json"
    source_file.write_text('{"original": true}\n', encoding="utf-8")
    original_bytes = source_file.read_bytes()
    seen = {}

    def fake_optimize(input_dir, output_dir, **kwargs):
        seen["input"] = input_dir
        seen["output"] = output_dir
        seen["kwargs"] = kwargs
        return {"status": "ok"}

    monkeypatch.setattr(search, "optimize_bundle", fake_optimize)
    output = tmp_path / "candidate"
    reports = tmp_path / "candidate-reports"
    result = search.execute_candidate(
        source,
        output,
        reports,
        exclude_matmul_nodes=["node_linear_196"],
        exclude_matmul_initializers=["weight_196"],
    )

    assert result == {"status": "ok"}
    assert seen["kwargs"]["lm_method"] == "int8-dynamic-chunked"
    assert seen["kwargs"]["exclude_matmul_nodes"] == ["node_linear_196"]
    assert seen["kwargs"]["exclude_matmul_initializers"] == ["weight_196"]
    assert source_file.read_bytes() == original_bytes

    output.mkdir()
    with pytest.raises(FileExistsError):
        search.execute_candidate(
            source,
            output,
            reports,
            exclude_matmul_nodes=[],
            exclude_matmul_initializers=[],
        )



def test_candidate_matrix_uses_four_bounded_preservation_values(tmp_path):
    inventory = search.inventory_matmuls(_make_bundle(tmp_path / "bundle"))

    candidates = search.candidate_matrix(inventory)

    assert [candidate["preserve_last_blocks"] for candidate in candidates] == [4, 8, 14, 20]
    assert all(inventory["lm_head"]["node"] in candidate["exclude_matmul_nodes"] for candidate in candidates)


def test_run_search_records_diagnostic_rows_without_overwriting_source(tmp_path, monkeypatch):
    import scripts.validate_e2e_corpus as validator

    source = _make_bundle(tmp_path / "source")
    source_manifest = (source / "bundle.json").read_bytes()
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "voice": "voice",
                "cases": [{"name": "very_short", "text": "Hello", "normalized_text": "Hello"}],
            }
        ),
        encoding="utf-8",
    )
    seen = []

    def fake_execute(input_bundle, output_bundle, report_dir, **kwargs):
        seen.append((Path(input_bundle), Path(output_bundle), kwargs))
        return {"output_bundle_bytes": 1234, "quantization": {"transformed_matmul_count": 100}}

    def fake_validate(*args, **kwargs):
        return {
            "quality_review": {"status": "requires_manual_review"},
            "first_divergence": {"case": "very_short", "step": 1},
            "cases": [
                {
                    "name": "very_short",
                    "token_divergence_step": 1,
                    "logits_max_abs_error": 0.5,
                    "logits_rmse": 0.1,
                    "logits_compared_steps": 2,
                    "waveform_finite": True,
                    "quality_audio": {"waveform_comparison": {"rmse": 0.02}},
                }
            ],
        }

    monkeypatch.setattr(search, "execute_candidate", fake_execute)
    monkeypatch.setattr(validator, "validate_corpus", fake_validate)
    output_root = tmp_path / "experiments"
    reports_root = tmp_path / "reports"

    result = search.run_search(
        source,
        output_root,
        reports_root,
        model_dir=tmp_path / "model",
        corpus_path=corpus_path,
    )

    assert [candidate["preserve_last_blocks"] for candidate in result["candidates"]] == [
        4,
        8,
        14,
        20,
    ]
    assert all(candidate["status"] == "diagnostic_complete" for candidate in result["candidates"])
    assert all(candidate["bundle_bytes"] == 1234 for candidate in result["candidates"])
    assert all(candidate["transformed_matmul_count"] == 100 for candidate in result["candidates"])
    assert all(candidate["quality"]["review_status"] == "requires_manual_review" for candidate in result["candidates"])
    assert len(seen) == 4
    assert (output_root / "results.json").is_file()
    assert (source / "bundle.json").read_bytes() == source_manifest

    with pytest.raises(FileExistsError):
        search.run_search(
            source,
            output_root,
            reports_root,
            model_dir=tmp_path / "model",
            corpus_path=corpus_path,
        )
