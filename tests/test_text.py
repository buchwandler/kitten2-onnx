from __future__ import annotations

import importlib

import pytest

from kitten2_onnx import text


@pytest.mark.parametrize("tag", sorted(text.EXPRESSION_TAGS))
def test_detects_each_supported_emotion_tag(tag: str) -> None:
    assert text.has_expression_tag(f"[{tag}] hello")


@pytest.mark.parametrize("tag", sorted(text.VOCAL_EVENT_TAGS))
def test_detects_each_supported_vocal_event(tag: str) -> None:
    assert text.has_expression_tag(f"hello <{tag}>")


def test_detects_emphasis_up_to_eighty_characters() -> None:
    assert text.has_expression_tag("(((short emphasis)))")
    assert text.has_expression_tag(f"((({'x' * 80})))")
    assert not text.has_expression_tag(f"((({'x' * 81})))")
    assert not text.has_expression_tag("(((line\nbreak)))")


@pytest.mark.parametrize(
    "source,marker",
    [
        ("[joyful] Hello.", "[joyful]"),
        ("Hello <laugh>.", "<laugh>"),
        ("I said (((hardly))).", "(((hardly)))"),
        ("[JOYFUL] Hello.", "[JOYFUL]"),
    ],
)
def test_normalization_preserves_recognized_markup_verbatim(source: str, marker: str) -> None:
    assert marker in text.normalize_text(source)


def test_normalization_handles_mixed_expression_markup() -> None:
    source = "[joyful] We won the grant <laugh> I can (((hardly))) believe it!"
    normalized = text.normalize_text(source)

    assert "[joyful]" in normalized
    assert "<laugh>" in normalized
    assert "(((hardly)))" in normalized
    assert text.has_expression_tag(source)


@pytest.mark.parametrize("source", ["[reverent] hello", "section [3]", "x < 5"])
def test_unknown_bracketed_text_is_not_expression_markup(source: str) -> None:
    assert not text.has_expression_tag(source)
    assert text.normalize_text(source)


def test_unknown_bracketed_text_is_normalized_as_ordinary_text() -> None:
    normalized = text.normalize_text("section [3]")

    assert "three" in normalized.lower()
    assert "[3]" not in normalized
    assert not text.has_expression_tag("section [3]")


def test_normalize_false_skips_english_written_form_expansion() -> None:
    source = "in 2024, it cost $19.99"

    normalized = text.normalize_text(source, normalize=False)
    expanded = text.normalize_text(source, normalize=True)

    assert "2024" in normalized
    assert "$19.99" in normalized
    assert "2024" not in expanded
    assert "nineteen dollars" in expanded.lower()




@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("早上好？", "早上好？"),
        ("كيف حالك؟", "كيف حالك؟"),
        ("सुप्रभात।", "सुप्रभात।"),
        ("你好。", "你好。"),
    ],
)
def test_normalize_false_preserves_unicode_terminal_punctuation(source: str, expected: str) -> None:
    assert text.normalize_text(source, normalize=False) == expected

def test_normalizer_requires_sentinel_preservation(monkeypatch: pytest.MonkeyPatch) -> None:
    original = text._backend.PRESERVES_SENTINELS
    monkeypatch.setattr(text._backend, "PRESERVES_SENTINELS", False)
    with pytest.raises(RuntimeError, match="no longer preserves private-use sentinels"):
        importlib.reload(text)

    monkeypatch.setattr(text._backend, "PRESERVES_SENTINELS", original)
    importlib.reload(text)


@pytest.mark.parametrize(
    "source,use_emotion,expected",
    [
        ("[joyful] hello", None, True),
        ("[reverent] hello", None, False),
        ("plain hello", True, True),
        ("[joyful] hello", False, False),
    ],
)
def test_prepare_generation_text_resolves_emotion_controls(
    source: str, use_emotion: bool | None, expected: bool
) -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.encoded: list[str] = []

        def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            self.encoded.append(value)
            return [91, 92]

    tokenizer = Tokenizer()
    prepared = text.prepare_generation_text(
        source,
        tokenizer=tokenizer,
        generation_config={"emotion_control": "{emo: 1}"},
        normalize=False,
        use_emotion=use_emotion,
    )

    assert prepared.expression_detected is text.has_expression_tag(source)
    assert prepared.emotion_control_used is expected
    assert prepared.emotion_ids == ([91, 92] if expected else None)
    assert ("{emo: 1}" in tokenizer.encoded) is expected
