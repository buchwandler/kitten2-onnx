from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark_variant_lm import (
    MAX_FAILURE_INDEX_SAMPLE,
    _compare_cache,
    _first_cache_failure,
)


def test_compare_cache_reports_layer_kind_and_worst_failure_values():
    reference = {"past_value_3": np.asarray([[[[0.0, 1.0, -2.0]]]], dtype=np.float32)}
    candidate = {"past_value_3": np.asarray([[[[0.003, 1.0, -2.003]]]], dtype=np.float32)}

    report = _compare_cache(reference, candidate)
    failure = report["failures"]["past_value_3"]

    assert report["allclose"] is False
    assert failure["layer"] == 3
    assert failure["cache_kind"] == "value"
    assert failure["failing_elements"] == 2
    assert failure["failing_indices"] == [[0, 0, 0, 0], [0, 0, 0, 2]]
    assert failure["failing_indices_truncated"] is False
    assert failure["max_absolute_error"] == pytest.approx(0.003, abs=1e-6)
    assert failure["max_relative_error"] == pytest.approx(0.0015, abs=1e-6)
    assert failure["worst_index"] == [0, 0, 0, 2]
    assert failure["reference_value_at_worst_index"] == pytest.approx(-2.0)
    assert failure["candidate_value_at_worst_index"] == pytest.approx(-2.003, abs=1e-6)
    assert report["first_failure"]["tensor"] == "past_value_3"


def test_compare_cache_bounds_failure_index_samples():
    reference = {"past_key_1": np.zeros((1, 1, 1, MAX_FAILURE_INDEX_SAMPLE + 3), dtype=np.float32)}
    reference["past_key_1"][..., -1] = 1.0
    candidate = {"past_key_1": reference["past_key_1"] + 0.003}

    failure = _compare_cache(reference, candidate)["failures"]["past_key_1"]

    assert failure["failing_elements"] == MAX_FAILURE_INDEX_SAMPLE + 3
    assert len(failure["failing_indices"]) == MAX_FAILURE_INDEX_SAMPLE
    assert failure["failing_indices_truncated"] is True
    assert failure["max_relative_error"] == pytest.approx(0.003, abs=1e-6)


def test_first_cache_failure_reports_phase_and_zero_based_decode_step():
    prefill = {"allclose": True, "first_failure": None}
    decode = [
        {"allclose": True, "first_failure": None},
        {
            "allclose": False,
            "first_failure": {"tensor": "past_value_7", "layer": 7},
        },
    ]

    assert _first_cache_failure(prefill, decode) == {
        "phase": "decode",
        "step": 1,
        "tensor": "past_value_7",
        "layer": 7,
    }


def test_first_cache_failure_prefers_prefill_failure():
    prefill = {
        "allclose": False,
        "first_failure": {"tensor": "past_value_2", "layer": 2},
    }
    decode = [{"allclose": False, "first_failure": {"tensor": "past_value_5", "layer": 5}}]

    assert _first_cache_failure(prefill, decode) == {
        "phase": "prefill",
        "step": 0,
        "tensor": "past_value_2",
        "layer": 2,
    }
