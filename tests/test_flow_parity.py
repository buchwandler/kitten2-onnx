from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from kitten2_onnx.decoder_reference import load_decoder_reference
from kitten2_onnx.export_flow import _metrics
from kitten2_onnx.runtime import append_s3_silence_tail


def test_captured_flow_onnx_matches_pytorch_reference():
    reference_path = os.environ.get("KITTEN2_DECODER_REFERENCE")
    graph_dir = os.environ.get("KITTEN2_FLOW_ONNX")
    if not reference_path or not graph_dir:
        pytest.skip("set KITTEN2_DECODER_REFERENCE and KITTEN2_FLOW_ONNX")
    ort = pytest.importorskip("onnxruntime")

    reference = load_decoder_reference(Path(reference_path))
    arrays = reference.arrays
    graph_dir = Path(graph_dir)
    metadata = json.loads((graph_dir / "flow.meta.json").read_text(encoding="utf-8"))
    assert metadata["silence_tail"]["graph_expects_padded_tokens"] is True
    assert metadata["silence_tail"]["append_applied_by"] == "runtime"

    speech_tokens = append_s3_silence_tail(arrays["speech_tokens"])
    session = ort.InferenceSession(str(graph_dir / metadata["graph"]), providers=["CPUExecutionProvider"])
    mel = session.run(
        ["mel"],
        {
            "speech_tokens": speech_tokens,
            "speech_token_lengths": np.asarray([speech_tokens.shape[1]], dtype=np.int64),
            "prompt_token": arrays["conditioning_prompt_token"],
            "prompt_token_len": arrays["conditioning_prompt_token_len"],
            "prompt_feat": arrays["conditioning_prompt_feat"],
            "embedding": arrays["conditioning_embedding"],
            "flow_noise": arrays["flow_noise"],
        },
    )[0]

    expected = arrays["flow_mel"]
    assert mel.shape == expected.shape
    assert np.isfinite(mel).all()
    metrics = _metrics(expected, mel)
    assert metrics["correlation"] > 0.999
    assert metrics["max_abs_error"] < 1e-4
