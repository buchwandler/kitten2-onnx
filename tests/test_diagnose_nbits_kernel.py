from __future__ import annotations

import hashlib

import numpy as np
import pytest

from kitten2_onnx.quantize_ternary_nbits import TernaryNBitsConfig
from scripts.diagnose_nbits_kernel import compare_matmul


def _exact_weight() -> np.ndarray:
    rng = np.random.default_rng(723)
    codes = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(128, 3))
    scales = np.array([0.25, 0.5, 1.25], dtype=np.float32)
    return codes.astype(np.float32) * scales[None, :]


def test_compare_matmul_uses_same_activation_and_reports_kernel_diagnostics():
    rng = np.random.default_rng(991)
    activation = rng.normal(size=(5, 128)).astype(np.float32)
    weight = _exact_weight()

    report = compare_matmul(
        activation,
        weight,
        TernaryNBitsConfig(block_size=64, accuracy_level=1),
        iterations=1,
        intra_op_threads=1,
    )

    assert report["block_size"] == 64
    assert report["accuracy_level"] == 1
    assert report["input_shape"] == [5, 128]
    assert report["output_shape"] == [5, 3]
    assert report["activation_sha256"] == hashlib.sha256(activation.tobytes()).hexdigest()
    assert report["max_absolute_difference"] >= 0
    assert report["max_relative_difference"] >= 0
    assert 0 <= report["outside_tolerance_count"] <= report["output_element_count"]
    assert len(report["max_error_index"]) == 2
    assert report["fp32_runtime_median_ms"] > 0
    assert report["nbits_runtime_median_ms"] > 0
    assert report["runtime_speedup"] > 0


def test_compare_matmul_rejects_non_float32_activations():
    with pytest.raises(ValueError, match="activation must be a rank-2 float32"):
        compare_matmul(
            np.zeros((1, 128), dtype=np.float16),
            _exact_weight(),
            TernaryNBitsConfig(block_size=128),
            iterations=1,
        )
