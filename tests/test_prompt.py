from __future__ import annotations

import json
from pathlib import Path

import pytest

from kitten2_onnx.prompt import build_generation_prompt, build_reference_prefix
from kitten2_onnx.tokens import TokenMap

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "upstream_prompts.json"
_FIXTURES = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


@pytest.mark.parametrize("fixture", _FIXTURES, ids=[item["name"] for item in _FIXTURES])
def test_prompt_matches_pinned_upstream_fixture(fixture):
    token_map = TokenMap.from_config(fixture["config"])
    reference_prefix = fixture["expected_reference_prefix"]

    if reference_prefix is not None:
        reference_prefix = build_reference_prefix(
            token_map,
            fixture["reference_text_ids"],
            fixture["reference_audio_tokens"],
        )
        assert reference_prefix == fixture["expected_reference_prefix"]

    prompt = build_generation_prompt(
        token_map,
        fixture["target_text_ids"],
        reference_prefix=reference_prefix,
        emotion_ids=fixture["emotion_ids"],
    )

    assert prompt == fixture["expected_prompt_ids"]


def test_token_map_rejects_unknown_config_fields():
    with pytest.raises(ValueError, match="unknown token_map keys"):
        TokenMap.from_config({"token_map": {"unexpected_id": 1}})


def test_token_map_exposes_exclusive_audio_end():
    token_map = TokenMap.from_config(_FIXTURES[0]["config"])

    assert token_map.audio_id_end == 158236
    assert token_map.uses_reference_role_tokens
