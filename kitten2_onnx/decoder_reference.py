"""Loading and validating deterministic decoder reference bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DECODER_REFERENCE_FORMAT = "kitten2-onnx-decoder-reference-v1"
REQUIRED_ARRAYS = {
    "conditioning_prompt_token",
    "speaker_embedding",
    "reference_tokens",
    "conditioning_prompt_token_len",
    "conditioning_prompt_feat",
    "conditioning_embedding",
    "speech_tokens",
    "speech_token_lengths",
    "flow_noise",
    "flow_mel",
    "mel2wav_input",
    "vocoder_phase_noise",
    "vocoder_sine_noise",
    "mel2wav_waveform",
    "waveform",
}


@dataclass(frozen=True)
class DecoderReference:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def load_decoder_reference(path: str | Path) -> DecoderReference:
    path = Path(path)
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != DECODER_REFERENCE_FORMAT:
        raise ValueError(f"Unsupported decoder reference format in {metadata_path}")

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    if missing := REQUIRED_ARRAYS - arrays.keys():
        raise ValueError(f"Decoder reference is missing arrays: {sorted(missing)}")
    metadata_arrays = metadata.get("arrays")
    if not isinstance(metadata_arrays, dict):
        raise TypeError("Decoder reference metadata has no array contract")
    if metadata_arrays.keys() != arrays.keys():
        raise ValueError("Decoder reference array names do not match the metadata")
    for name, array in arrays.items():
        expected = metadata_arrays[name]
        if expected.get("shape") != list(array.shape):
            raise ValueError(f"Decoder reference shape mismatch for {name!r}")
        if expected.get("dtype") != str(array.dtype):
            raise ValueError(f"Decoder reference dtype mismatch for {name!r}")

    if not np.array_equal(arrays["flow_mel"], arrays["mel2wav_input"]):
        raise ValueError("The captured mel2wav input differs from the flow output mel")
    return DecoderReference(arrays=arrays, metadata=metadata)



def write_decoder_reference(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Decoder reference output must use the .npz extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

    metadata = {
        **metadata,
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
    }
    metadata_path = path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, metadata_path
