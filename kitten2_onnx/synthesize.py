"""Synthesize a WAV file from a packaged Kitten2 ONNX bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from kitten2_onnx.tts import Kitten2OnnxTTS


def validate_audio(audio: np.ndarray) -> None:
    """Reject empty, invalid, silent, or clipping-level synthesis output."""
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("generated audio must be a non-empty mono waveform")
    if not np.isfinite(audio).all():
        raise ValueError("generated audio contains non-finite samples")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if not 1e-5 <= peak <= 1.0:
        raise ValueError(f"generated audio peak {peak:.6g} is outside the sensible range")
    if not 1e-6 <= rms <= 0.5:
        raise ValueError(f"generated audio RMS {rms:.6g} is outside the sensible range")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Kitten2 ONNX bundle directory")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--voice", help="Packaged built-in voice")
    parser.add_argument("--preset", help="Decode preset from bundle.json")
    parser.add_argument("--seed", type=int, help="Seed for sampling and decoder noise")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--output", type=Path, required=True, help="Output WAV path")
    args = parser.parse_args(argv)

    import soundfile as sf

    tts = Kitten2OnnxTTS(args.bundle, providers=args.providers)
    audio = tts.generate(
        args.text,
        voice=args.voice,
        preset=args.preset,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    validate_audio(audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, tts.sample_rate, subtype="PCM_16")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_rate": tts.sample_rate,
                "samples": int(audio.size),
                "duration_seconds": audio.size / tts.sample_rate,
                "peak": float(np.max(np.abs(audio))),
                "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
