from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import kitten_text_processing as _backend

EXPRESSION_TAGS = frozenset(
    {
        "mundane",
        "nervous",
        "tender",
        "angry",
        "excited",
        "stern",
        "sad",
        "contemplative",
        "surprised",
        "joyful",
    }
)
VOCAL_EVENT_TAGS = frozenset(
    {"pause", "sigh", "gasp", "laugh", "giggle", "sob", "scoff", "growl", "um", "gulp"}
)

_TAG_SPAN = re.compile(
    r"\[(?:" + "|".join(re.escape(tag) for tag in EXPRESSION_TAGS) + r")\]"
    r"|<(?:"
    + "|".join(re.escape(tag) for tag in VOCAL_EVENT_TAGS)
    + r")>",
    re.IGNORECASE,
)
_EMPHASIS_SPAN = re.compile(r"\(\(\(([^()\n]{1,80})\)\)\)")
_PLACEHOLDER = "\ue000{}\ue001"

_UI_GLYPHS = re.compile(r"[\u2022\u2023\u2043\u25e6\u25b8\u21b3\u2192\u2190\u2191\u2193]+")
_MATH_GLYPH_WORDS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "ϕ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Γ": "gamma",
    "Δ": "delta",
    "Θ": "theta",
    "Λ": "lambda",
    "Ξ": "xi",
    "Π": "pi",
    "Σ": "sigma",
    "Φ": "phi",
    "Ψ": "psi",
    "Ω": "omega",
    "=": "equals",
    "≠": "is not equal to",
    "≈": "approximately equals",
    "≤": "is less than or equal to",
    "≥": "is greater than or equal to",
    "±": "plus or minus",
    "×": "times",
    "÷": "divided by",
    "·": "dot",
    "−": "minus",
    "∗": "star",
    "∘": "composed with",
    "⊤": "transpose",
    "∼": "distributed as",
    "∝": "proportional to",
    "≡": "is equivalent to",
    "∈": "in",
    "∉": "not in",
    "⊂": "subset of",
    "⊆": "subset of",
    "∪": "union",
    "∩": "intersect",
    "⊕": "plus",
    "⊗": "times",
    "^": "to the",
    "∑": "sum of",
    "∏": "product of",
    "∫": "integral of",
    "∂": "partial",
    "∇": "gradient of",
    "√": "square root of",
    "∞": "infinity",
    "∆": "delta",
    "∀": "for all",
    "∃": "there exists",
    "⇒": "implies",
    "⇔": "if and only if",
    "∥": "norm",
    "‖": "norm",
    "⌊": "floor of",
    "⌋": "",
    "⌈": "ceiling of",
    "⌉": "",
    "ℝ": "R",
    "ℕ": "N",
    "ℤ": "Z",
    "ℚ": "Q",
    "ℂ": "C",
    "𝔼": "the expectation of",
}
_MATH_GLYPH_WORDS.update(
    {
        char: str(digit)
        for digit, pair in enumerate(
            zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "₀₁₂₃₄₅₆₇₈₉", strict=True)
        )
        for char in pair
    }
)
_MATH_GLYPH_RE = re.compile(
    "|".join(re.escape(glyph) for glyph in sorted(_MATH_GLYPH_WORDS, key=len, reverse=True))
)
_ELLIPSIS = re.compile(r"(?:\.\s+){2,}\.|…|\.\.\.")
_STRAIGHT_QUOTE_PAIR = re.compile(r'"([^"\n]*)"')
_NONTAG_BRACKET_SPAN = re.compile(r"\[([^\[\]\n]{1,400})\]|<([^<>\n]{1,60})>|\(([^()\n]{1,400})\)")
_COMMA_COLLISION = re.compile(r",\s*([,.!?;:])")
_TRAILING_NONTERMINAL = re.compile(r"[,;:\-–—]+\s*$")
_ALREADY_TERMINATED = set('.!?؟。！？।॥"’”\')]')
_DASH = re.compile(r"\s*(?:[—–]|--)\s*")
_DASH_FIX = (
    (re.compile(r",\s*,+"), ", "),
    (re.compile(r"\s+([,.!?;:])"), r"\1"),
    (re.compile(r",\s*([.!?])"), r"\1"),
    (re.compile(r"\s{2,}"), " "),
    (re.compile(r"^\s*,\s*"), ""),
)

if not getattr(_backend, "PRESERVES_SENTINELS", False):
    raise RuntimeError(
        "kitten_text_processing no longer preserves private-use sentinels; "
        "expression tags would be corrupted. Pin an earlier version."
    )


def _protect_tags(text: str) -> tuple[str, list[str]]:
    tags: list[str] = []

    def replace(match: re.Match[str]) -> str:
        tags.append(match.group(0))
        return _PLACEHOLDER.format(len(tags) - 1)

    return _EMPHASIS_SPAN.sub(replace, _TAG_SPAN.sub(replace, text)), tags


def _restore_tags(text: str, tags: list[str]) -> str:
    for index, tag in enumerate(tags):
        text = text.replace(_PLACEHOLDER.format(index), tag)
    return text


def has_expression_tag(text: str) -> bool:
    value = str(text or "")
    return bool(_TAG_SPAN.search(value) or _EMPHASIS_SPAN.search(value))


def _speak_math_glyphs(text: str) -> str:
    text = _ELLIPSIS.sub(", ", text)
    text = _MATH_GLYPH_RE.sub(lambda match: f" {_MATH_GLYPH_WORDS[match.group(0)]} ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _reformat_nontag_brackets(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        inner = next(group for group in match.groups() if group is not None).strip()
        return f", {inner},"

    text = _NONTAG_BRACKET_SPAN.sub(replace, text)
    while collapsed := _COMMA_COLLISION.sub(r"\1", text):
        if collapsed == text:
            return collapsed
        text = collapsed
    return text


def _smart_quote_pairs(text: str) -> str:
    return _STRAIGHT_QUOTE_PAIR.sub(lambda match: f"“{match.group(1)}”", text)


def _dashes_to_commas(text: str) -> str:
    if not text:
        return text
    text = _DASH.sub(", ", text)
    for pattern, replacement in _DASH_FIX:
        text = pattern.sub(replacement, text)
    return text.strip()


def _capitalize_and_terminate(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\s+([?!])", r"\1", text)
    for index, char in enumerate(text):
        if char.isalpha():
            text = text[:index] + char.upper() + text[index + 1 :]
            break
    stripped = text.rstrip()
    if not stripped or stripped[-1] in _ALREADY_TERMINATED:
        return stripped
    stripped = _TRAILING_NONTERMINAL.sub("", stripped).rstrip()
    return f"{stripped}." if stripped else stripped


def normalize_text(text: str, normalize: bool = True) -> str:
    protected, tags = _protect_tags(str(text or ""))
    protected = re.sub(r"\s*\n+\s*", " ", protected)
    protected = re.sub(r"(?<=[a-z])([.!?;:])(?=[A-Z])", r"\1 ", protected)
    protected = _speak_math_glyphs(protected)
    protected = re.sub(
        r"\b([vV])(\d+(?:\.\d+)+)",
        lambda match: match.group(1) + " " + " point ".join(match.group(2).split(".")),
        protected,
    )
    protected = re.sub(r"(?<=\d)\s*([?!])(?=\s|$)", r" \1", protected)
    raw = _dashes_to_commas(_UI_GLYPHS.sub(" ", _reformat_nontag_brackets(protected)))
    if not raw.strip():
        return _restore_tags(raw, tags)
    if not normalize:
        return _restore_tags(_capitalize_and_terminate(raw), tags)
    try:
        spoken = (_backend.normalize_text(_smart_quote_pairs(raw)) or "").strip()
    except Exception:  # noqa: BLE001
        spoken = ""
    if not spoken:
        return _restore_tags(_capitalize_and_terminate(raw), tags)
    return _restore_tags(_capitalize_and_terminate(_dashes_to_commas(spoken)), tags)


@dataclass(frozen=True)
class PreparedGenerationText:
    original_text: str
    processed_text: str
    normalize: bool
    expression_detected: bool
    emotion_control_used: bool
    emotion_ids: list[int] | None


def prepare_generation_text(
    text: str,
    *,
    tokenizer: Any,
    generation_config: Mapping[str, Any],
    normalize: bool = True,
    use_emotion: bool | None = None,
) -> PreparedGenerationText:
    processed_text = normalize_text(text, normalize=normalize).strip()
    expression_detected = has_expression_tag(text)
    emotion_enabled = expression_detected if use_emotion is None else use_emotion
    emotion_control = generation_config.get("emotion_control")
    emotion_ids = None
    if emotion_control and emotion_enabled:
        emotion_ids = tokenizer.encode(emotion_control, add_special_tokens=False)
    return PreparedGenerationText(
        original_text=text,
        processed_text=processed_text,
        normalize=normalize,
        expression_detected=expression_detected,
        emotion_control_used=bool(emotion_ids),
        emotion_ids=emotion_ids,
    )
