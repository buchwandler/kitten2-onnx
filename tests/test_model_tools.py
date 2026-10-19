from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
_DOWNLOAD_TOOLS = runpy.run_path(str(_SCRIPT_DIR / "download_reference_model.py"))
_INSPECT_TOOLS = runpy.run_path(str(_SCRIPT_DIR / "inspect_model.py"))
BASE_FILES = _DOWNLOAD_TOOLS["BASE_FILES"]
UPSTREAM_CODE_REVISION = _DOWNLOAD_TOOLS["UPSTREAM_CODE_REVISION"]
download_model = _DOWNLOAD_TOOLS["download_model"]
voice_files = _DOWNLOAD_TOOLS["voice_files"]
all_voice_files = _DOWNLOAD_TOOLS["all_voice_files"]
PINNED_VOICE_FIXTURE = Path(__file__).parent / "fixtures" / "voices_fa8ce5c.json"
PINNED_VOICE_DATA = json.loads(PINNED_VOICE_FIXTURE.read_text(encoding="utf-8"))
PINNED_VOICE_NAMES = PINNED_VOICE_DATA["voices"]
TOKEN_FIELDS = _INSPECT_TOOLS["TOKEN_FIELDS"]
inspect_checkpoint = _INSPECT_TOOLS["inspect_checkpoint"]


def _write_checkpoint(repo_dir: Path, *, final_seg_id: int = 105) -> None:
    lm_dir = repo_dir / "lm"
    lm_dir.mkdir(parents=True)
    token_map = {name: index for index, name in enumerate(TOKEN_FIELDS)}
    token_map.update(audio_id_base=100, num_audio_tokens=5, final_seg_id=final_seg_id)
    (repo_dir / "config.json").write_text(
        json.dumps({"token_map": token_map, "speaker_embedding_dim": 4}), encoding="utf-8"
    )
    (lm_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 10,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    state = {
        "model.embed_tokens.weight": torch.zeros(10, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(4, 8),
        "model.layers.1.self_attn.k_proj.weight": torch.zeros(4, 8),
        "spk_proj.0.weight": torch.zeros(8, 4),
        "spk_proj.0.bias": torch.zeros(8),
        "spk_proj.1.weight": torch.ones(8),
        "spk_proj.1.bias": torch.zeros(8),
    }
    save_file(state, str(lm_dir / "model.safetensors"))
    (repo_dir / "model_revision.json").write_text(
        json.dumps({"source_model_revision": "pinned-commit"}), encoding="utf-8"
    )


def test_inspector_reports_checkpoint_contract_and_revision(tmp_path):
    repo_dir = tmp_path / "checkpoint"
    repo_dir.mkdir()
    _write_checkpoint(repo_dir)

    report = inspect_checkpoint(repo_dir)

    assert report["valid"]
    assert report["model_revision"] == "pinned-commit"
    assert report["token_map"]["audio_id_base"] == 100
    assert report["architecture"]["num_key_value_heads"] == 1
    assert report["speaker_projection_tensors"]["spk_proj.0.weight"]["shape"] == [8, 4]
    assert report["embedding_tensor"] == {
        "name": "model.embed_tokens.weight",
        "shape": [10, 8],
        "dtype": "F32",
        "file": "lm/model.safetensors",
    }
    assert all(report["invariants"].values())
    assert report["source_safetensor_files"][0]["size_bytes"] > 0
    assert report["warnings"]


def test_inspector_rejects_inconsistent_token_map(tmp_path):
    repo_dir = tmp_path / "checkpoint"
    repo_dir.mkdir()
    _write_checkpoint(repo_dir, final_seg_id=106)

    report = inspect_checkpoint(repo_dir)

    assert not report["valid"]
    assert report["invariants"]["audio_range_ends_at_final_seg"] is False
    assert "Checkpoint invariant failed: audio_range_ends_at_final_seg" in report["errors"]

def test_pinned_voice_fixture_has_47_sorted_names():
    assert PINNED_VOICE_DATA["source_model_revision"] == "fa8ce5c7547e322a065cad9b813d27a78fad9640"
    assert len(PINNED_VOICE_NAMES) == 47
    assert PINNED_VOICE_NAMES == sorted(set(PINNED_VOICE_NAMES))


def test_all_voice_files_resolve_every_pinned_voice_asset_once():
    config = {"voices": "voices/voices.json", "default_decoder": "default"}
    voice_index = {
        name: {"reference": f"{name.lower()}.wav", "artifacts": f"{name.lower()}.npz"}
        for name in PINNED_VOICE_NAMES
    }
    voice_assets = {
        f"voices/{name.lower()}.{extension}"
        for name in PINNED_VOICE_NAMES
        for extension in ("wav", "npz")
    }
    available = {"voices/voices.json", "cpp/default/decoder.pt"} | voice_assets

    files = all_voice_files(config, voice_index, available)

    assert files == available
    assert sum(path == "cpp/default/decoder.pt" for path in files) == 1



def test_voice_files_selects_one_voice_and_default_decoder():
    config = {"voices": "voices/voices.json", "default_decoder": "default"}
    voices = {"Bruno": {"reference": "bruno.wav", "artifacts": "bruno.npz"}}
    available = {
        "voices/voices.json",
        "voices/bruno.wav",
        "voices/bruno.npz",
        "voices/other_voice.wav",
        "cpp/default/decoder.pt",
    }

    assert voice_files(config, voices, "Bruno", available) == {
        "voices/voices.json",
        "voices/bruno.wav",
        "voices/bruno.npz",
        "cpp/default/decoder.pt",
    }


def test_voice_files_rejects_paths_outside_repository():
    config = {"voices": "voices/voices.json", "default_decoder": "default"}
    voices = {"Bruno": {"reference": "../../outside.wav", "artifacts": "bruno.npz"}}
    available = {"voices/voices.json", "voices/bruno.npz", "cpp/default/decoder.pt"}

    with pytest.raises(ValueError, match="Invalid repository-relative path"):
        voice_files(config, voices, "Bruno", available)


def test_download_uses_resolved_revision_and_requested_files(tmp_path, monkeypatch):
    import huggingface_hub

    requested: dict[str, object] = {}
    available = set(BASE_FILES) | {
        "lm/model.safetensors",
        "lm/model-ternary.safetensors",
        "voices/voices.json",
        "voices/bruno.wav",
        "voices/bruno.npz",
        "voices/other_voice.wav",
        "cpp/default/decoder.pt",
    }

    class FakeApi:
        def model_info(self, repo_id, *, revision):
            assert repo_id == "KittenML/kitten-tts-2"
            assert revision == "requested-tag"
            return SimpleNamespace(
                sha="resolved-commit",
                siblings=[SimpleNamespace(rfilename=path) for path in sorted(available)],
            )

    files_dir = tmp_path / "remote"
    files_dir.mkdir()
    remote_config = files_dir / "config.json"
    remote_config.write_text(
        json.dumps({"voices": "voices/voices.json", "default_decoder": "default"}),
        encoding="utf-8",
    )
    remote_index = files_dir / "voices.json"
    remote_index.write_text(
        json.dumps({"Bruno": {"reference": "bruno.wav", "artifacts": "bruno.npz"}}),
        encoding="utf-8",
    )

    def fake_hf_hub_download(repo_id, filename, *, revision):
        assert revision == "resolved-commit"
        return remote_config if filename == "config.json" else remote_index

    def fake_snapshot_download(repo_id, *, revision, local_dir, allow_patterns):
        requested.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        Path(local_dir).mkdir(parents=True)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    output = tmp_path / "download"
    manifest = download_model(
        "KittenML/kitten-tts-2",
        "requested-tag",
        output,
        plain_lm=True,
        voice="Bruno",
    )

    assert requested["revision"] == "resolved-commit"
    assert set(requested["allow_patterns"]) == set(BASE_FILES) | {
        "lm/model.safetensors",
        "voices/voices.json",
        "voices/bruno.wav",
        "voices/bruno.npz",
        "cpp/default/decoder.pt",
    }
    assert "voices/other_voice.wav" not in requested["allow_patterns"]
    assert manifest["source_model_revision"] == "resolved-commit"
    assert manifest["upstream_code_revision"] == UPSTREAM_CODE_REVISION
    assert json.loads((output / "model_revision.json").read_text(encoding="utf-8")) == manifest



def test_download_all_voices_selects_every_voice_and_one_shared_decoder(tmp_path, monkeypatch):
    import huggingface_hub

    requested: dict[str, object] = {}
    voice_index = {
        name: {"reference": f"{name.lower()}.wav", "artifacts": f"{name.lower()}.npz"}
        for name in PINNED_VOICE_NAMES
    }
    voice_assets = {
        f"voices/{name.lower()}.{extension}"
        for name in PINNED_VOICE_NAMES
        for extension in ("wav", "npz")
    }
    available = set(BASE_FILES) | {"voices/voices.json", "cpp/default/decoder.pt"} | voice_assets

    class FakeApi:
        def model_info(self, repo_id, *, revision):
            return SimpleNamespace(
                sha="resolved-commit",
                siblings=[SimpleNamespace(rfilename=path) for path in sorted(available)],
            )

    files_dir = tmp_path / "remote"
    files_dir.mkdir()
    remote_config = files_dir / "config.json"
    remote_config.write_text(
        json.dumps({"voices": "voices/voices.json", "default_decoder": "default"}),
        encoding="utf-8",
    )
    remote_index = files_dir / "voices.json"
    remote_index.write_text(json.dumps(voice_index), encoding="utf-8")

    def fake_hf_hub_download(repo_id, filename, *, revision):
        assert revision == "resolved-commit"
        return remote_config if filename == "config.json" else remote_index

    def fake_snapshot_download(repo_id, *, revision, local_dir, allow_patterns):
        requested.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        Path(local_dir).mkdir(parents=True)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    output = tmp_path / "download-all"
    manifest = download_model(
        "KittenML/kitten-tts-2",
        "requested-tag",
        output,
        plain_lm=False,
        voice=None,
        all_voices=True,
    )

    expected_files = set(BASE_FILES) | {"voices/voices.json", "cpp/default/decoder.pt"} | voice_assets
    assert set(requested["allow_patterns"]) == expected_files
    assert manifest["voice"] is None
    assert manifest["voices"] == PINNED_VOICE_NAMES
    assert manifest["downloaded_files"] == sorted(expected_files)
