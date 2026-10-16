from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from kitten2_onnx.reference import _load_speaker_projection, load_reference_artifacts


def _speaker_paths() -> tuple[Path, Path]:
    model_dir = os.environ.get("KITTEN2_MODEL_DIR")
    speaker_path = os.environ.get("KITTEN2_SPEAKER_ONNX")
    if not model_dir:
        pytest.skip("set KITTEN2_MODEL_DIR to enable real speaker parity")
    if speaker_path:
        path = Path(speaker_path)
    else:
        onnx_dir = os.environ.get("KITTEN2_ONNX_DIR")
        if not onnx_dir:
            pytest.skip("set KITTEN2_SPEAKER_ONNX or KITTEN2_ONNX_DIR")
        path = Path(onnx_dir).parent / "speaker" / "speaker.onnx"
    if not path.is_file():
        pytest.skip(f"speaker graph is not present: {path}")
    return Path(model_dir), path


def test_real_bruno_speaker_projection_matches_onnx():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model_dir, speaker_path = _speaker_paths()
    artifacts = load_reference_artifacts(model_dir, "Bruno")
    projection = _load_speaker_projection(
        model_dir,
        artifacts.config,
        torch.float32,
        torch.device("cpu"),
    )
    speaker_embedding = torch.from_numpy(artifacts.speaker_embedding[None])
    with torch.inference_mode():
        expected = projection(speaker_embedding).numpy()

    onnx.checker.check_model(str(speaker_path))
    session = ort.InferenceSession(str(speaker_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["speaker_hidden"], {"speaker_embedding": artifacts.speaker_embedding[None]}
    )[0]

    metadata = json.loads(
        (speaker_path.parent / "speaker.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["speaker_dim"] == 512
    assert metadata["hidden_size"] == 2048
    assert actual.shape == expected.shape == (1, 2048)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
