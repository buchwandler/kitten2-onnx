from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kitten2_onnx import synthesize as synthesize_module
from kitten2_onnx import tts as tts_module
from kitten2_onnx.reference import ReferenceArtifacts
from kitten2_onnx.synthesize import main, validate_audio
from kitten2_onnx.tts import Kitten2OnnxTTS


def test_validate_audio_rejects_invalid_waveforms():
    with pytest.raises(ValueError, match="non-empty mono"):
        validate_audio(np.zeros((1, 0), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        validate_audio(np.asarray([np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="peak"):
        validate_audio(np.asarray([1.1], dtype=np.float32))


def test_synthesis_cli_generates_valid_builtin_voice_wav(tmp_path):
    bundle_value = os.environ.get("KITTEN2_TTS_BUNDLE")
    if not bundle_value:
        pytest.skip("set KITTEN2_TTS_BUNDLE to run built-in-voice ONNX synthesis")

    sf = pytest.importorskip("soundfile")
    output = tmp_path / "bruno.wav"
    main(
        [
            bundle_value,
            "Hello from Kitten.",
            "--voice",
            "Bruno",
            "--preset",
            "stable",
            "--seed",
            "7",
            "--max-new-tokens",
            "32",
            "--output",
            str(output),
        ]
    )

    audio, sample_rate = sf.read(output, dtype="float32")
    assert sample_rate == 24000
    assert audio.ndim == 1
    assert audio.size > 0
    assert np.isfinite(audio).all()
    assert 1e-5 <= np.max(np.abs(audio)) <= 1.0
    rms = np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
    assert 1e-6 <= rms <= 0.5



def test_emotion_conditioning_is_independent_from_sampling_preset(monkeypatch):
    from types import SimpleNamespace

    calls = {"prompts": [], "presets": []}

    class Tokenizer:
        def encode(self, value, *, add_special_tokens):
            assert not add_special_tokens
            if value == "{emo: 1}":
                return [71, 72]
            return [11, 12]

    tts = object.__new__(Kitten2OnnxTTS)
    tts.bundle = {
        "default_voice": "Bruno",
        "voices": {"Bruno": "voices/Bruno.npz", "Kiki": "voices/Kiki.npz"},
        "decode_presets": {"stable": {}, "expressive": {}},
        "generation": {"emotion_control": "{emo: 1}"},
    }
    tts.tokenizer = Tokenizer()
    tts.token_map = SimpleNamespace(start_id=1, final_seg_id=5, text_start_id=2, speech_start_id=3)
    tts._voice = lambda voice: (np.zeros(1, dtype=np.float32), [7, 8], {})

    def generate(prompt_ids, **kwargs):
        calls["prompts"].append(prompt_ids)
        return SimpleNamespace(raw_ids=[101], s3_ids=[1])

    tts.lm = SimpleNamespace(generate=generate)
    tts.decoder = SimpleNamespace(
        decode=lambda *args, **kwargs: np.asarray([0.1, 0.2], dtype=np.float32)
    )

    def resolve_settings(bundle, preset):
        calls["presets"].append(preset)
        return f"settings:{preset}"

    monkeypatch.setattr(tts_module, "sampling_settings_from_config", resolve_settings)

    automatic = tts.generate_with_metadata("[joyful] hello", preset="expressive")
    unknown = tts.generate_with_metadata("[reverent] hello", preset="stable")
    forced = tts.generate_with_metadata("plain hello", use_emotion=True)
    suppressed = tts.generate_with_metadata("[joyful] hello", use_emotion=False)

    reference = ReferenceArtifacts(
        repo_dir=Path("."),
        config={"generation": {"emotion_control": "{emo: 1}"}},
        model_revision=None,
        token_map=tts.token_map,
        tokenizer=tts.tokenizer,
        voice="Bruno",
        transcript="",
        speaker_embedding=np.zeros(1, dtype=np.float32),
        reference_tokens=np.asarray([], dtype=np.int64),
        reference_prefix=[7, 8],
    )
    reference_prepared = reference.prepare_text("[joyful] hello")
    assert reference_prepared.processed_text == automatic.normalized_text
    assert reference.build_prompt("[joyful] hello") == automatic.prompt_ids

    assert automatic.prompt_ids == [7, 8, 71, 72, 2, 11, 12, 5, 3]
    assert automatic.expression_detected and automatic.emotion_control_used
    assert unknown.prompt_ids == [7, 8, 2, 11, 12, 5, 3]
    assert not unknown.expression_detected and not unknown.emotion_control_used
    assert forced.prompt_ids == [7, 8, 71, 72, 2, 11, 12, 5, 3]
    assert suppressed.prompt_ids == [7, 8, 2, 11, 12, 5, 3]
    assert calls["presets"] == ["expressive", "stable", None, None]
    assert tts.available_voices == ["Bruno", "Kiki"]
    assert tts.available_presets == ["stable", "expressive"]


def test_synthesis_cli_passes_normalization_and_emotion_options(tmp_path, monkeypatch):
    seen = {}

    class FakeTTS:
        sample_rate = 24000

        def __init__(self, bundle, *, providers=None):
            pass

        def generate(self, text, **kwargs):
            seen.update(kwargs)
            return np.asarray([0.1, 0.2], dtype=np.float32)

    monkeypatch.setattr(synthesize_module, "Kitten2OnnxTTS", FakeTTS)
    output = tmp_path / "cli.wav"
    main(
        [
            "bundle",
            "hello",
            "--no-normalize",
            "--use-emotion",
            "--output",
            str(output),
        ]
    )

    assert output.is_file()
    assert seen["normalize"] is False
    assert seen["use_emotion"] is True


def test_synthesis_cli_rejects_conflicting_emotion_options(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "bundle",
                "hello",
                "--use-emotion",
                "--no-emotion",
                "--output",
                str(tmp_path / "cli.wav"),
            ]
        )
