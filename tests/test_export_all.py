import hashlib
import json

import pytest

import kitten2_onnx.export_all as exporter


def test_export_all_writes_bundle_paths_hashes_and_revisions(tmp_path, monkeypatch):
    model_dir = tmp_path / "source"
    (model_dir / "lm").mkdir(parents=True)
    (model_dir / "lm" / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model_revision.json").write_text(
        json.dumps(
            {
                "repo_id": "KittenML/kitten-tts-2",
                "source_model_revision": "model-rev",
                "upstream_code_revision": exporter.UPSTREAM_CODE_REVISION,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "sample_rate": 24000,
        "default_voice": "Bruno",
        "default_preset": "stable",
        "decode_presets": {"stable": {"temperature": 0.8}},
        "generation": {"repetition_window": 50},
        "token_map": {"audio_id_base": 10},
    }
    monkeypatch.setattr(
        exporter,
        "inspect_checkpoint",
        lambda _: {"valid": True, "kitten_config": config, "errors": []},
    )

    def touch(path, content=b"onnx"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def fake_export_speaker(_, output_dir, *, opset):
        touch(output_dir / "speaker.onnx")
        (output_dir / "speaker.meta.json").write_text(
            json.dumps({"hidden_size": 8}), encoding="utf-8"
        )

    def fake_export_lm(_, output_dir, *, profile_name, opset):
        touch(output_dir / "lm.onnx")
        touch(output_dir / "lm.onnx.data", b"external")
        touch(output_dir / "embed_tokens.npy", b"embeddings")
        (output_dir / "lm.meta.json").write_text(
            json.dumps(
                {
                    "embedding_file": "embed_tokens.npy",
                    "source_dtype": "bfloat16",
                    "graph_dtype": "float32",
                }
            ),
            encoding="utf-8",
        )

    def fake_capture_reference(_, voice, *, token_count):
        return {}, {"voice": voice, "token_count": token_count}

    def fake_write_capture(path, arrays, metadata):
        touch(path, b"voice")
        path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path, path.with_suffix(".meta.json")

    def fake_export_decoder(path, output_dir, *, opset):
        graph_name = "flow.onnx" if output_dir.name == "flow" else "mel2wav.onnx"
        touch(output_dir / graph_name)
        metadata_path = output_dir / graph_name.replace(".onnx", ".meta.json")
        metadata_path.write_text(
            json.dumps({"graph": graph_name, "parity_passed": True}), encoding="utf-8"
        )
        return output_dir / graph_name, metadata_path

    monkeypatch.setattr(exporter, "export_speaker", fake_export_speaker)
    monkeypatch.setattr(exporter, "export_lm", fake_export_lm)
    monkeypatch.setattr(exporter, "capture_reference", fake_capture_reference)
    monkeypatch.setattr(exporter, "write_capture", fake_write_capture)
    monkeypatch.setattr(exporter, "export_flow", fake_export_decoder)
    monkeypatch.setattr(exporter, "export_mel2wav", fake_export_decoder)
    checked = []

    def fake_check(graphs):
        checked.extend(graphs)
        assert len(graphs) == 4
        return [path.name for path in graphs]

    monkeypatch.setattr(exporter, "_check_onnx_graphs", fake_check)
    validation_roots = []
    monkeypatch.setattr(
        exporter,
        "validate_bundle",
        lambda path: validation_roots.append(path) or {"valid": True, "errors": []},
    )

    bundle_path = exporter.export_all(
        model_dir,
        tmp_path / "bundle",
        validate=True,
    )
    bundle_dir = bundle_path.parent
    assert validation_roots == [bundle_dir]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert bundle["format"] == "kitten2-onnx-bundle-v1"
    assert bundle["sample_rate"] == 24000
    assert bundle["lm"]["embeddings"] == "lm/embed_tokens.npy"
    assert bundle["voices"] == {"Bruno": "voices/Bruno.npz"}
    assert bundle["tokenizer_assets"] == ["tokenizer.json"]
    assert bundle["decode_presets"] == config["decode_presets"]
    assert bundle["revisions"]["source_model_revision"] == "model-rev"
    assert bundle["revisions"]["upstream_code_revision"] == exporter.UPSTREAM_CODE_REVISION
    assert bundle["versions"]["packages"]["torch"]
    assert len(checked) == 4
    assert "bundle.json" not in bundle["files"]
    assert "lm/lm.onnx.data" in bundle["files"]

    for relative_path, expected_hash in bundle["files"].items():
        actual_hash = hashlib.sha256((bundle_dir / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def test_export_all_rejects_invalid_source_checkpoint(tmp_path, monkeypatch):
    model_dir = tmp_path / "source"
    model_dir.mkdir()
    monkeypatch.setattr(
        exporter,
        "inspect_checkpoint",
        lambda _: {"valid": False, "kitten_config": {}, "errors": ["invalid checkpoint"]},
    )

    with pytest.raises(ValueError, match="invalid checkpoint"):
        exporter.export_all(model_dir, tmp_path / "bundle")
