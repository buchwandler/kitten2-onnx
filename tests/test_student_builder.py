from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

import scripts.build_qwen3_06b_student as builder
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from scripts.build_qwen3_06b_student import (
    STUDENT_VOCAB_SIZE,
    TEACHER_MODEL_REVISION,
    _copy_teacher_assets,
    _validate_teacher,
)
from scripts.inspect_model import TOKEN_FIELDS


def _write_teacher(path: Path) -> None:
    lm_dir = path / "lm"
    lm_dir.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"speaker_embedding_dim": 512, "token_map": {"audio_id_base": 151675}}),
        encoding="utf-8",
    )
    teacher_config = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": STUDENT_VOCAB_SIZE,
        "tie_word_embeddings": True,
        "rope_theta": 1000000,
    }
    (lm_dir / "config.json").write_text(json.dumps(teacher_config), encoding="utf-8")
    (path / "model_revision.json").write_text(
        json.dumps(
            {
                "source_model_revision": TEACHER_MODEL_REVISION,
                "upstream_code_revision": UPSTREAM_CODE_REVISION,
            }
        ),
        encoding="utf-8",
    )
    for filename in (
        "added_tokens.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (lm_dir / filename).write_text(filename, encoding="utf-8")
    (lm_dir / "model.safetensors").write_bytes(b"teacher weights")
    (path / "speaker").mkdir()
    (path / "speaker" / "model.safetensors").write_bytes(b"speaker weights")
    (path / "cpp" / "default").mkdir(parents=True)
    (path / "cpp" / "default" / "decoder.pt").write_bytes(b"decoder")
    (path / "voices").mkdir()
    (path / "voices" / "voices.json").write_text("{}", encoding="utf-8")
    (path / ".cache").mkdir()
    (path / ".cache" / "private-cache-file").write_text("ignored", encoding="utf-8")


def test_builder_accepts_only_pinned_teacher_metadata(tmp_path):
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    _write_teacher(teacher)

    manifest, config = _validate_teacher(teacher)

    assert manifest["source_model_revision"] == TEACHER_MODEL_REVISION
    assert config["hidden_size"] == 2048
    assert config["vocab_size"] == STUDENT_VOCAB_SIZE


def test_builder_copies_kitten_assets_without_teacher_lm_weights(tmp_path):
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    _write_teacher(teacher)
    output = tmp_path / "student"
    output.mkdir()

    _copy_teacher_assets(teacher, output)

    assert (output / "config.json").is_file()
    assert (output / "speaker" / "model.safetensors").read_bytes() == b"speaker weights"
    assert (output / "cpp" / "default" / "decoder.pt").read_bytes() == b"decoder"
    assert (output / "voices" / "voices.json").is_file()
    assert (output / "lm" / "tokenizer.json").read_text(encoding="utf-8") == "tokenizer.json"
    assert not (output / "lm" / "model.safetensors").exists()
    assert not (output / ".cache").exists()



def test_student_architecture_matches_handoff():
    assert builder.STUDENT_ARCHITECTURE == {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 158241,
        "tie_word_embeddings": True,
    }


def test_student_vocabulary_preserves_text_rows_and_ties_after_reload(
    tmp_path, monkeypatch, tiny_qwen3
):
    monkeypatch.setattr(builder, "SOURCE_VOCAB_SIZE", 97)
    monkeypatch.setattr(builder, "KITTEN_TOKEN_START", 80)
    monkeypatch.setattr(builder, "STUDENT_VOCAB_SIZE", 101)
    monkeypatch.setattr(builder, "STUDENT_HIDDEN_SIZE", 32)
    model = tiny_qwen3
    original = model.get_input_embeddings().weight.detach().clone()

    builder._initialize_kitten_vocabulary(model)
    initialized = model.get_input_embeddings().weight.detach().clone()

    torch.testing.assert_close(initialized[:80], original[:80])
    assert not torch.equal(initialized[80:97], original[80:97])
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()

    model.save_pretrained(tmp_path, safe_serialization=True)
    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.config.vocab_size == 101
    assert reloaded.get_input_embeddings().weight.data_ptr() == reloaded.get_output_embeddings().weight.data_ptr()
    torch.testing.assert_close(reloaded.get_input_embeddings().weight, initialized)


def test_build_student_writes_provenance_and_inspectable_checkpoint(tmp_path, monkeypatch):
    import transformers

    dimensions = {
        "model_type": "qwen3",
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 8,
        "tie_word_embeddings": True,
    }
    monkeypatch.setattr(builder, "SOURCE_VOCAB_SIZE", 8)
    monkeypatch.setattr(builder, "KITTEN_TOKEN_START", 5)
    monkeypatch.setattr(builder, "STUDENT_VOCAB_SIZE", 12)
    monkeypatch.setattr(builder, "STUDENT_HIDDEN_SIZE", 4)
    monkeypatch.setattr(builder, "STUDENT_INTERMEDIATE_SIZE", 6)
    monkeypatch.setattr(builder, "STUDENT_NUM_LAYERS", 2)
    monkeypatch.setattr(builder, "STUDENT_NUM_ATTENTION_HEADS", 2)
    monkeypatch.setattr(builder, "STUDENT_NUM_KEY_VALUE_HEADS", 1)
    monkeypatch.setattr(builder, "STUDENT_HEAD_DIM", 2)
    monkeypatch.setattr(builder, "SPEAKER_EMBEDDING_DIM", 4)
    monkeypatch.setattr(
        builder,
        "STUDENT_ARCHITECTURE",
        {**dimensions, "vocab_size": 12},
    )

    teacher = tmp_path / "teacher"
    teacher.mkdir()
    _write_teacher(teacher)
    teacher_lm_config_path = teacher / "lm" / "config.json"
    teacher_lm_config = json.loads(teacher_lm_config_path.read_text(encoding="utf-8"))
    teacher_lm_config["vocab_size"] = 12
    teacher_lm_config_path.write_text(json.dumps(teacher_lm_config), encoding="utf-8")
    top_config_path = teacher / "config.json"
    top_config = json.loads(top_config_path.read_text(encoding="utf-8"))
    top_config["speaker_embedding_dim"] = 4
    token_map = {name: 0 for name in TOKEN_FIELDS}
    token_map.update(audio_id_base=10, num_audio_tokens=3, final_seg_id=13)
    top_config["token_map"] = token_map
    top_config_path.write_text(json.dumps(top_config), encoding="utf-8")

    class TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = nn.Linear(4, 2, bias=False)

    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = TinyAttention()

    class TinyStudent(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(8, 4)
            self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
            self.lm_head = nn.Linear(4, 8, bias=False)
            self.tie_weights()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def tie_weights(self):
            self.lm_head.weight = self.model.embed_tokens.weight

        def resize_token_embeddings(self, vocab_size, *, mean_resizing):
            old = self.model.embed_tokens
            embedding = nn.Embedding(vocab_size, 4)
            with torch.no_grad():
                embedding.weight[: old.num_embeddings].copy_(old.weight)
            self.model.embed_tokens = embedding
            self.lm_head = nn.Linear(4, vocab_size, bias=False)
            self.config.vocab_size = vocab_size
            self.tie_weights()
            return embedding

        def save_pretrained(self, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text(
                json.dumps(vars(self.config)), encoding="utf-8"
            )
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in self.state_dict().items()
                if name != "lm_head.weight"
            }
            save_file(state, str(output_dir / "model.safetensors"))

    def fake_config_from_pretrained(*args, **kwargs):
        return SimpleNamespace(**dimensions, initializer_range=0.02)

    def fake_model_from_pretrained(*args, config, **kwargs):
        return TinyStudent(config)

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fake_model_from_pretrained),
    )

    output = tmp_path / "student"
    report = builder.build_student("synthetic-base", teacher, output, experiment_id="test")

    assert report["valid"]
    assert report["architecture"]["hidden_size"] == 4
    assert report["speaker_projection_tensors"]["spk_proj.0.weight"]["shape"] == [4, 4]
    provenance = json.loads((output / "model_revision.json").read_text(encoding="utf-8"))
    assert provenance["source_model_revision"] == "student-qwen3-0.6b-test"
    assert provenance["teacher_model_revision"] == TEACHER_MODEL_REVISION
    assert provenance["upstream_code_revision"] == UPSTREAM_CODE_REVISION
    assert provenance["initialization_only"] is True
    assert (output / "lm" / "tokenizer.json").is_file()
    assert (output / "speaker" / "model.safetensors").is_file()
