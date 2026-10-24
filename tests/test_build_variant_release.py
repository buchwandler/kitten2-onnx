import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import build_variant_release, smoke_variant_bundle

REVISION = "a" * 40
WEIGHT_PAYLOADS = {"packed": b"packed-weight-payload", "emb4": b"emb4-weight-payload"}
WEIGHT_SHA256 = {name: hashlib.sha256(payload).hexdigest() for name, payload in WEIGHT_PAYLOADS.items()}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_model_root(path: Path, *, variant: str | None = None) -> None:
    _write_json(
        path / "config.json",
        {
            "lm_dir": "lm",
            "lm_packed": "lm/model-ternary.safetensors",
            "lm_emb4": "lm/model-tl2-emb4.safetensors",
            "voices": "voices/voices.json",
        },
    )
    _write_json(
        path / "model_revision.json",
        {"source_model_revision": REVISION, "upstream_code_revision": "b" * 40, "weight_variant": variant},
    )
    _write_json(path / "bundle.json", {"source_marker": True})
    if variant == "packed":
        checkpoint = path / "lm/model-ternary.safetensors"
    elif variant == "emb4":
        checkpoint = path / "lm/model-tl2-emb4.safetensors"
    else:
        checkpoint = path / "lm/model.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(WEIGHT_PAYLOADS[variant] if variant else b"full-weight-payload")
    _write_json(path / "voices/voices.json", {"Amy": {}, "Bruno": {}})
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _write_exact_int8_bundle(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(path / "bundle.json", {"format": "test"})
    (path / "lm.onnx").write_bytes(b"unchanged exact int8")


def _write_candidate_bundle(path: Path, variant: str) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "bundle.json",
        {
            "format": "kitten2-onnx-bundle-v1",
            "revisions": {"source_model_revision": REVISION, "upstream_code_revision": "b" * 40},
            "voices": {"Amy": "voices/Amy.npz", "Bruno": "voices/Bruno.npz"},
            "lm": {
                "metadata": "lm/lm.meta.json",
                "weight_variant": variant,
                "source_weight_sha256": WEIGHT_SHA256[variant],
                "storage": {
                    "transformer": {"format": "ternary-nbits4"},
                    "vocabulary": {"format": "float32"},
                },
            },
        },
    )
    _write_json(
        path / "lm/lm.meta.json",
        {
            "source_model_revision": REVISION,
            "source_weight_sha256": WEIGHT_SHA256[variant],
            "quantization": {"format": "ternary-nbits4", "bits": 4, "block_size": 128, "accuracy_level": 1},
        },
    )
    (path / "lm/lm.onnx").write_bytes(b"candidate payload")

def _make_runner(*, variant: str = "packed", parity_passed: bool = True):
    def runner(command, **kwargs):
        args = [str(item) for item in command]
        output_json = kwargs["output_json"]
        log_path = kwargs["log_path"]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        log_path.write_text("completed", encoding="utf-8")

        if any("export_all" in arg for arg in args):
            assert args[args.index("--weights") + 1] == variant
            export_dir = Path(args[args.index("--output") + 1])
            export_dir.mkdir(parents=True)
            (export_dir / "export.marker").write_text("exported", encoding="utf-8")
        elif any("optimize_bundle.py" in arg for arg in args):
            candidate_dir = Path(args[3])
            _write_candidate_bundle(candidate_dir, variant)
            report_dir = Path(args[args.index("--report-dir") + 1])
            _write_json(report_dir / "optimization.json", {"validation": {"status": "passed"}})
        elif any("validate_bundle" in arg for arg in args):
            validation_path = Path(args[args.index("--output") + 1])
            _write_json(validation_path, {"valid": True, "bundle_dir": args[3]})
        elif any("audit_bundle_size.py" in arg for arg in args):
            size_path = Path(args[args.index("--output") + 1])
            _write_json(size_path, {"bundle_bytes": 1234, "bundle_dir": args[2]})
        elif any("benchmark_variant_lm.py" in arg for arg in args):
            assert args[args.index("--variant") + 1] == variant
            _write_json(
                Path(args[args.index("--output") + 1]),
                {
                    "parity_passed": parity_passed,
                    "variant": variant,
                    "source_model_revision": REVISION,
                    "source_weight_sha256": WEIGHT_SHA256[variant],
                },
            )
        elif "scripts.validate_e2e_corpus" in args:
            assert args[args.index("--weights") + 1] == variant
            assert args[args.index("--provider") + 1] == "CPUExecutionProvider"
            corpus_path = Path(args[args.index("--corpus") + 1])
            candidate_dir = Path(args[4])
            cases = json.loads(corpus_path.read_text(encoding="utf-8"))["cases"]
            _write_json(
                Path(args[args.index("--output") + 1]),
                {
                    "weight_variant": variant,
                    "validation_mode": "exact_parity",
                    "source_model_revision": REVISION,
                    "source_weight_sha256": WEIGHT_SHA256[variant],
                    "upstream_code_revision": "b" * 40,
                    "onnx_bundle_revisions": {"source_model_revision": REVISION},
                    "onnx_bundle_provenance": {
                        "path": str(candidate_dir),
                        "status": "available",
                        "tree_sha256": build_variant_release._tree_hash(candidate_dir),
                    },
                    "requested_providers": ["CPUExecutionProvider"],
                    "actual_providers": ["CPUExecutionProvider"],
                    "sample_rate": 24000,
                    "first_divergence": None,
                    "cases": [
                        {
                            "name": case["name"],
                            "token_ids_match_reference": True,
                            "waveform_finite": True,
                            "waveform_samples": 120,
                        }
                        for case in cases
                    ],
                },
            )
        elif "scripts.smoke_variant_bundle" in args:
            if "--voice" in args:
                voices = [args[args.index("--voice") + 1]]
            else:
                voices = ["Amy", "Bruno"]
            bundle_dir = Path(args[3])
            _write_json(
                Path(args[args.index("--output") + 1]),
                {
                    "status": "passed",
                    "bundle": str(bundle_dir),
                    "bundle_tree_sha256": smoke_variant_bundle._bundle_tree_sha256(bundle_dir),
                    "expected_voices": voices,
                    "cases": [
                        {
                            "voice": voice,
                            "status": "passed",
                            "sample_rate": 24000,
                            "waveform_finite": True,
                            "waveform_samples": 120,
                        }
                        for voice in voices
                    ],
                },
            )
        else:
            raise AssertionError(f"Unexpected command: {args}")
        return {"status": "ok", "returncode": 0}

    return runner


def _build_fixture(tmp_path: Path, *, variant: str = "packed", parity_passed: bool = True):
    model_dir = tmp_path / "source-model"
    variant_dir = tmp_path / f"{variant}-reference"
    _write_model_root(model_dir)
    _write_model_root(variant_dir, variant=variant)
    exact_int8_dir = tmp_path / "exact-int8"
    _write_exact_int8_bundle(exact_int8_dir)
    sibling_variant = "emb4" if variant == "packed" else "packed"
    sibling = tmp_path / f"releases/{sibling_variant}"
    sibling.mkdir(parents=True)
    (sibling / "sentinel.txt").write_text("keep", encoding="utf-8")
    _write_json(sibling / "bundle.json", {"format": "sibling-candidate"})
    output_dir = tmp_path / f"releases/{variant}"
    reports_dir = tmp_path / f"release-reports/{variant}"
    hashes = {
        "source": build_variant_release._tree_hash(model_dir),
        "variant": build_variant_release._tree_hash(variant_dir),
        "exact": build_variant_release._tree_hash(exact_int8_dir),
        "sibling": build_variant_release._tree_hash(sibling),
    }
    return {
        "model_dir": model_dir,
        "variant_dir": variant_dir,
        "exact_int8_dir": exact_int8_dir,
        "output_dir": output_dir,
        "reports_dir": reports_dir,
        "sibling": sibling,
        "hashes": hashes,
        "runner": _make_runner(variant=variant, parity_passed=parity_passed),
    }


def test_prepare_variant_source_copies_selected_weights_without_touching_source(tmp_path):
    model_dir = tmp_path / "source"
    variant_dir = tmp_path / "variant"
    staged_dir = tmp_path / "staged"
    _write_model_root(model_dir)
    _write_model_root(variant_dir, variant="packed")
    source_hash = build_variant_release._tree_hash(model_dir)
    variant_hash = build_variant_release._tree_hash(variant_dir)

    result = build_variant_release._prepare_variant_source(
        model_dir, variant_dir, "packed", staged_dir
    )

    assert result == staged_dir
    assert (staged_dir / "lm/model-ternary.safetensors").read_bytes() == b"packed-weight-payload"
    assert not (staged_dir / "lm/model.safetensors").exists()
    assert build_variant_release._tree_hash(model_dir) == source_hash
    assert build_variant_release._tree_hash(variant_dir) == variant_hash




def test_prepare_variant_source_isolates_when_source_and_reference_are_the_same(tmp_path):
    model_dir = tmp_path / "source"
    staged_dir = tmp_path / "staged"
    _write_model_root(model_dir, variant="emb4")
    source_hash = build_variant_release._tree_hash(model_dir)

    result = build_variant_release._prepare_variant_source(model_dir, model_dir, "emb4", staged_dir)

    assert result == staged_dir
    assert (staged_dir / "lm/model-tl2-emb4.safetensors").read_bytes() == WEIGHT_PAYLOADS["emb4"]
    assert build_variant_release._tree_hash(model_dir) == source_hash

@pytest.mark.parametrize("variant", ["packed", "emb4"])
def test_build_release_promotes_only_after_all_gates_and_preserves_inputs(tmp_path, variant):
    fixture = _build_fixture(tmp_path, variant=variant)

    result = build_variant_release.build_release(
        fixture["model_dir"],
        weights=variant,
        output_dir=fixture["output_dir"],
        reports_dir=fixture["reports_dir"],
        variant_reference_dir=fixture["variant_dir"],
        exact_int8_bundle_dir=fixture["exact_int8_dir"],
        command_runner=fixture["runner"],
    )

    assert result["status"] == "verified_local_candidate"
    assert result["candidate_promoted"] is True
    assert (fixture["output_dir"] / "release_manifest.json").is_file()
    manifest = json.loads((fixture["output_dir"] / "release_manifest.json").read_text())
    assert manifest["variant"] == variant
    assert manifest["build_config"]["nbits"] == 4
    assert manifest["build_config"]["block_size"] == 128
    assert manifest["build_config"]["accuracy_level"] == 1
    for relative, record in manifest["files"].items():
        payload = (fixture["output_dir"] / relative).read_bytes()
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    assert "lm/lm.onnx" in manifest["files"]
    assert build_variant_release._tree_hash(fixture["model_dir"]) == fixture["hashes"]["source"]
    assert build_variant_release._tree_hash(fixture["variant_dir"]) == fixture["hashes"]["variant"]
    assert build_variant_release._tree_hash(fixture["exact_int8_dir"]) == fixture["hashes"]["exact"]
    assert build_variant_release._tree_hash(fixture["sibling"]) == fixture["hashes"]["sibling"]



def test_emb4_build_preserves_previously_promoted_packed_candidate(tmp_path):
    model_dir = tmp_path / "source-model"
    packed_reference = tmp_path / "packed-reference"
    emb4_reference = tmp_path / "emb4-reference"
    _write_model_root(model_dir)
    _write_model_root(packed_reference, variant="packed")
    _write_model_root(emb4_reference, variant="emb4")
    exact_int8_dir = tmp_path / "exact-int8"
    _write_exact_int8_bundle(exact_int8_dir)
    source_hash = build_variant_release._tree_hash(model_dir)
    exact_hash = build_variant_release._tree_hash(exact_int8_dir)

    packed_output = tmp_path / "releases/packed"
    packed_result = build_variant_release.build_release(
        model_dir,
        weights="packed",
        output_dir=packed_output,
        reports_dir=tmp_path / "release-reports/packed",
        variant_reference_dir=packed_reference,
        exact_int8_bundle_dir=exact_int8_dir,
        command_runner=_make_runner(variant="packed"),
    )
    packed_hash = build_variant_release._tree_hash(packed_output)

    emb4_output = tmp_path / "releases/emb4"
    emb4_result = build_variant_release.build_release(
        model_dir,
        weights="emb4",
        output_dir=emb4_output,
        reports_dir=tmp_path / "release-reports/emb4",
        variant_reference_dir=emb4_reference,
        exact_int8_bundle_dir=exact_int8_dir,
        command_runner=_make_runner(variant="emb4"),
    )

    assert packed_result["candidate_promoted"] is True
    assert emb4_result["candidate_promoted"] is True
    assert build_variant_release._tree_hash(packed_output) == packed_hash
    assert build_variant_release._tree_hash(model_dir) == source_hash
    assert build_variant_release._tree_hash(exact_int8_dir) == exact_hash


@pytest.mark.parametrize("variant", ["packed", "emb4"])
def test_failed_lm_parity_does_not_promote_or_mutate_any_input(tmp_path, variant):
    fixture = _build_fixture(tmp_path, variant=variant, parity_passed=False)

    with pytest.raises(build_variant_release.ReleaseGateError, match="Strict 32-step"):
        build_variant_release.build_release(
            fixture["model_dir"],
            weights=variant,
            output_dir=fixture["output_dir"],
            reports_dir=fixture["reports_dir"],
            variant_reference_dir=fixture["variant_dir"],
            exact_int8_bundle_dir=fixture["exact_int8_dir"],
            command_runner=fixture["runner"],
        )

    assert not fixture["output_dir"].exists()
    failure = json.loads((fixture["reports_dir"] / "release_status.json").read_text())
    assert failure["candidate_promoted"] is False
    assert failure["failed_gate"] == "lm-parity"
    assert build_variant_release._tree_hash(fixture["model_dir"]) == fixture["hashes"]["source"]
    assert build_variant_release._tree_hash(fixture["variant_dir"]) == fixture["hashes"]["variant"]
    assert build_variant_release._tree_hash(fixture["exact_int8_dir"]) == fixture["hashes"]["exact"]
    assert build_variant_release._tree_hash(fixture["sibling"]) == fixture["hashes"]["sibling"]


def test_build_release_rejects_revision_mismatch_before_running_commands(tmp_path):
    fixture = _build_fixture(tmp_path)
    _write_json(
        fixture["variant_dir"] / "model_revision.json",
        {"source_model_revision": "c" * 40, "upstream_code_revision": "b" * 40, "weight_variant": "packed"},
    )

    def unexpected_runner(*args, **kwargs):
        raise AssertionError("No build command should run for a revision mismatch")

    with pytest.raises(ValueError, match="revisions do not match"):
        build_variant_release.build_release(
            fixture["model_dir"],
            weights="packed",
            output_dir=fixture["output_dir"],
            reports_dir=fixture["reports_dir"],
            variant_reference_dir=fixture["variant_dir"],
            exact_int8_bundle_dir=fixture["exact_int8_dir"],
            command_runner=unexpected_runner,
        )

    assert not fixture["output_dir"].exists()
    failure = json.loads((fixture["reports_dir"] / "release_status.json").read_text())
    assert failure["candidate_promoted"] is False
    assert failure["failed_gate"] == "source-preflight"
    assert build_variant_release._tree_hash(fixture["model_dir"]) == fixture["hashes"]["source"]
    assert build_variant_release._tree_hash(fixture["exact_int8_dir"]) == fixture["hashes"]["exact"]
    assert build_variant_release._tree_hash(fixture["sibling"]) == fixture["hashes"]["sibling"]



def test_release_reports_cannot_be_written_inside_a_sibling_candidate(tmp_path):
    fixture = _build_fixture(tmp_path)
    sibling_hash = build_variant_release._tree_hash(fixture["sibling"])

    with pytest.raises(ValueError, match="inside an existing bundle"):
        build_variant_release.build_release(
            fixture["model_dir"],
            weights="packed",
            output_dir=fixture["output_dir"],
            reports_dir=fixture["sibling"] / "reports",
            variant_reference_dir=fixture["variant_dir"],
            exact_int8_bundle_dir=fixture["exact_int8_dir"],
            command_runner=fixture["runner"],
        )

    assert not fixture["output_dir"].exists()
    assert not (fixture["sibling"] / "reports").exists()
    assert build_variant_release._tree_hash(fixture["sibling"]) == sibling_hash


def test_smoke_bundle_reports_every_voice_and_resumes_only_matching_bundle(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_json(bundle_dir / "bundle.json", {"default_preset": "stable", "lm": {"weight_variant": "packed"}})
    (bundle_dir / "lm.onnx").write_bytes(b"test model")
    output = tmp_path / "smoke.json"
    calls = []

    class FakeTTS:
        sample_rate = 24000
        available_voices = ("Amy", "Bruno")

        def __init__(self, *args, **kwargs):
            pass

        def generate_with_metadata(self, text, *, voice, **kwargs):
            calls.append(voice)
            return SimpleNamespace(waveform=np.array([0.1, -0.2], dtype=np.float32), s3_ids=[1, 2])

    monkeypatch.setattr(smoke_variant_bundle, "Kitten2OnnxTTS", FakeTTS)
    first = smoke_variant_bundle.smoke_bundle(bundle_dir, output)
    assert first["status"] == "passed"
    assert first["passed_voices"] == 2
    assert calls == ["Amy", "Bruno"]

    second = smoke_variant_bundle.smoke_bundle(bundle_dir, output)
    assert second["status"] == "passed"
    assert calls == ["Amy", "Bruno"]


def test_smoke_bundle_records_voice_failures_and_continues(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_json(bundle_dir / "bundle.json", {"default_preset": "stable"})
    output = tmp_path / "smoke.json"

    class FakeTTS:
        sample_rate = 24000
        available_voices = ("Amy", "Bruno")

        def __init__(self, *args, **kwargs):
            pass

        def generate_with_metadata(self, text, *, voice, **kwargs):
            if voice == "Amy":
                raise RuntimeError("synthetic voice failure")
            return SimpleNamespace(waveform=np.array([0.1], dtype=np.float32), s3_ids=[1])

    monkeypatch.setattr(smoke_variant_bundle, "Kitten2OnnxTTS", FakeTTS)
    report = smoke_variant_bundle.smoke_bundle(bundle_dir, output)

    assert report["status"] == "failed"
    assert report["failed_voices"] == ["Amy"]
    assert [case["voice"] for case in report["cases"]] == ["Amy", "Bruno"]
