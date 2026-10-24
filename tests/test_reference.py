from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from kitten2_onnx.reference import (
    UPSTREAM_CODE_REVISION,
    _load_speaker_projection,
    load_reference_artifacts,
)


def _token_map():
    return {
        "audio_id_base": 100,
        "num_audio_tokens": 8,
        "speech_start_id": 20,
        "speech_end_id": 21,
        "text_start_id": 22,
        "text_end_id": 23,
        "start_id": 24,
        "stop_id": 25,
        "final_seg_id": 108,
        "reference_speech_start_id": 109,
        "reference_speech_end_id": 110,
        "reference_text_start_id": 111,
        "reference_text_end_id": 112,
    }


def test_load_reference_artifacts_and_prompt(tmp_path, monkeypatch):
    import transformers

    (tmp_path / "lm").mkdir()
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "token_map": _token_map(),
                "speaker_embedding_dim": 4,
                "default_voice": "Bruno",
                "generation": {"emotion_control": "{emo: 1}"},
                "voices": "voices/voices.json",
            }
        ),
        encoding="utf-8",
    )
    (voice_dir / "voices.json").write_text(
        json.dumps(
            {
                "Bruno": {
                    "artifacts": "bruno.npz",
                    "transcript": "Reference words.",
                }
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        voice_dir / "bruno.npz",
        embedding=np.asarray([1, 2, 3, 4], dtype=np.float32),
        reference_tokens=np.asarray([0, 7], dtype=np.int32),
    )
    (tmp_path / "model_revision.json").write_text(
        json.dumps({"source_model_revision": "pinned-model"}), encoding="utf-8"
    )

    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            if text == "Reference words.":
                return [31, 32]
            if text == "{emo: 1}":
                return [51]
            if text == "[joyful] Hello.":
                return [43, 44]
            assert text == "Target words."
            return [41, 42]
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    artifacts = load_reference_artifacts(tmp_path)

    assert artifacts.model_revision == "pinned-model"
    assert artifacts.voice == "Bruno"
    assert artifacts.speaker_embedding.shape == (4,)
    assert artifacts.reference_tokens.tolist() == [0, 7]
    assert artifacts.reference_prefix == [24, 111, 31, 32, 112, 109, 100, 107, 110]
    assert artifacts.build_prompt("Target words.") == [
        24,
        111,
        31,
        32,
        112,
        109,
        100,
        107,
        110,
        22,
        41,
        42,
        108,
        20,
    ]

    assert artifacts.build_prompt("[joyful] hello") == [
        24,
        111,
        31,
        32,
        112,
        109,
        100,
        107,
        110,
        51,
        22,
        43,
        44,
        108,
        20,
    ]
    assert artifacts.build_prompt("[joyful] hello", use_emotion=False) == [
        24,
        111,
        31,
        32,
        112,
        109,
        100,
        107,
        110,
        22,
        43,
        44,
        108,
        20,
    ]


def test_load_reference_artifacts_uses_variant_weights_and_matching_speaker_projection(
    tmp_path, monkeypatch
):
    import transformers

    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return [31, 32]

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    revision = "same-upstream-revision"

    def write_model(root, variant, checkpoint_name, speaker_projection):
        lm_dir = root / "lm"
        voices_dir = root / "voices"
        lm_dir.mkdir(parents=True)
        voices_dir.mkdir()
        config = {
            "token_map": _token_map(),
            "speaker_embedding_dim": 4,
            "default_voice": "Bruno",
            "generation": {},
            "voices": "voices/voices.json",
            "lm_packed": "lm/model-ternary.safetensors",
            "lm_emb4": "lm/model-tl2-emb4.safetensors",
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (voices_dir / "voices.json").write_text(
            json.dumps({"Bruno": {"artifacts": "bruno.npz", "transcript": "Reference."}}),
            encoding="utf-8",
        )
        np.savez(
            voices_dir / "bruno.npz",
            embedding=np.ones(4, dtype=np.float32),
            reference_tokens=np.asarray([0, 7], dtype=np.int64),
        )
        (root / "model_revision.json").write_text(
            json.dumps({"source_model_revision": revision, "weight_variant": variant}),
            encoding="utf-8",
        )
        tensors = {"spk_proj.0.weight": torch.ones((8, 4))} if speaker_projection else {
            "model.weight": torch.ones((2, 2))
        }
        save_file(tensors, str(lm_dir / checkpoint_name))
        return lm_dir / checkpoint_name

    packed_root = tmp_path / "packed"
    packed_checkpoint = write_model(
        packed_root, "packed", "model-ternary.safetensors", speaker_projection=True
    )
    packed = load_reference_artifacts(packed_root, weight_variant="packed")
    assert packed.weight_variant == "packed"
    assert packed.source_weight_sha256 == hashlib.sha256(packed_checkpoint.read_bytes()).hexdigest()
    assert packed.speaker_projection_checkpoint == packed_checkpoint
    assert packed.speaker_projection_weight_variant == "packed"

    emb4_root = tmp_path / "emb4"
    write_model(emb4_root, "emb4", "model-tl2-emb4.safetensors", speaker_projection=False)
    full_root = tmp_path / "full"
    full_checkpoint = write_model(
        full_root, "full", "model.safetensors", speaker_projection=True
    )
    emb4 = load_reference_artifacts(
        full_root,
        weight_variant="emb4",
        weight_model_dir=emb4_root,
        speaker_projection_model_dir=full_root,
    )
    assert emb4.weight_variant == "emb4"
    assert emb4.repo_dir == emb4_root
    assert emb4.voice_artifacts_dir == full_root
    assert emb4.source_weight_sha256 == hashlib.sha256(
        (emb4_root / "lm/model-tl2-emb4.safetensors").read_bytes()
    ).hexdigest()
    assert emb4.speaker_projection_checkpoint == full_checkpoint
    assert emb4.speaker_projection_weight_variant == "full"
    assert emb4.speaker_projection_source_model_revision == revision

    speaker_root = tmp_path / "speaker"
    write_model(speaker_root, "full", "model.safetensors", speaker_projection=True)
    manifest_path = speaker_root / "model_revision.json"
    manifest_path.write_text(
        json.dumps({"source_model_revision": "different-revision", "weight_variant": "full"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="same pinned source revision"):
        load_reference_artifacts(
            full_root,
            weight_variant="emb4",
            weight_model_dir=emb4_root,
            speaker_projection_model_dir=speaker_root,
        )


def test_load_reference_model_loads_the_requested_upstream_weight_variant(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kitten2_onnx import export_lm, reference

    calls = {}

    class FakeModel:
        dtype = torch.float32

        def to(self, *, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    monkeypatch.setattr(
        export_lm,
        "load_hf_lm",
        lambda repo_dir, dtype, *, weight_variant: calls.update(
            repo_dir=repo_dir, dtype=dtype, weight_variant=weight_variant
        ) or FakeModel(),
    )
    monkeypatch.setattr(
        reference,
        "_load_speaker_projection",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    artifacts = SimpleNamespace(
        weight_variant="emb4",
        config={"dtype": "float32"},
        repo_dir=tmp_path,
        speaker_projection_checkpoint=tmp_path / "speaker.safetensors",
    )

    loaded = reference.load_reference_model(artifacts, dtype=torch.float32)

    assert calls["repo_dir"] == tmp_path
    assert calls["weight_variant"] == "emb4"
    assert calls["dtype"] is torch.float32
    assert calls["eval"] is True
    assert loaded.device == torch.device("cpu")


def test_load_speaker_projection_from_checkpoint_head(tmp_path):
    repo_dir = tmp_path / "checkpoint"
    lm_dir = repo_dir / "lm"
    lm_dir.mkdir(parents=True)
    (lm_dir / "config.json").write_text(json.dumps({"hidden_size": 8}), encoding="utf-8")
    state = {
        "spk_proj.0.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "spk_proj.0.bias": torch.zeros(8),
        "spk_proj.1.weight": torch.ones(8),
        "spk_proj.1.bias": torch.zeros(8),
        "model.embed_tokens.weight": torch.zeros(10, 8),
    }
    save_file(state, str(lm_dir / "model.safetensors"))

    projection = _load_speaker_projection(
        repo_dir,
        {"speaker_embedding_dim": 4},
        torch.float32,
        torch.device("cpu"),
    )

    assert projection[0].weight.shape == (8, 4)
    torch.testing.assert_close(projection[0].weight, state["spk_proj.0.weight"])


def test_checked_in_reference_capture_has_required_boundaries():
    fixture_dir = Path(__file__).parent / "fixtures"
    metadata = json.loads((fixture_dir / "reference_bruno.json").read_text(encoding="utf-8"))
    with np.load(fixture_dir / "reference_bruno.npz", allow_pickle=False) as capture:
        required_arrays = {
            "prompt_ids",
            "speaker_embedding_512",
            "speaker_hidden",
            "prefill_inputs_embeds",
            "prefill_position_ids",
            "prefill_attention_mask",
            "prefill_last_logits",
            "prefill_kv_shapes",
            "generated_raw_ids",
            "decode_logits_first5",
            "kv_lengths",
            "input_embedding_shape",
        }
        assert required_arrays <= set(capture.files)
        assert capture["prompt_ids"].shape == (metadata["prompt_length"],)
        assert capture["generated_raw_ids"].shape == (32,)
        assert capture["decode_logits_first5"].shape == (5, 158241)
        assert capture["prefill_kv_shapes"].shape == (28, 2, 4)
        assert capture["kv_lengths"].shape == (33,)

    assert metadata["source_model_revision"] == "fa8ce5c7547e322a065cad9b813d27a78fad9640"
    assert metadata["upstream_code_revision"] == UPSTREAM_CODE_REVISION
    assert metadata["torch_version"]
    assert metadata["transformers_version"] == "4.57.1"
    assert metadata["dtype"] == "bfloat16"
    assert metadata["voice"] == "Bruno"
