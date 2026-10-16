"""Plan the KittenTTS 2 S3 codec/vocoder ONNX export.

The LM and speaker projection have clear ONNX contracts. The audio side is more
complex: upstream S3Gen combines reference conditioning, speech-token processing,
a mean-flow mel generator, and a mel-to-waveform vocoder.

This MVP refuses to emit a misleading "codec.onnx" until each stage has parity
tests. Run this module to write a machine-readable export plan.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

EXPORT_PLAN = {
    "format": "kitten2-onnx-codec-plan-v1",
    "status": "not-yet-exported",
    "sample_rate": 24000,
    "token_rate_hz": 25,
    "stages": [
        {
            "name": "reference_conditioning",
            "purpose": "reference waveform -> prompt_token/prompt_feat/embedding",
            "mvp": "keep in PyTorch first",
        },
        {
            "name": "flow",
            "purpose": "generated S3 tokens + reference conditioning -> mel",
            "mvp": "export and validate independently",
        },
        {
            "name": "mel2wav",
            "purpose": "mel -> 24 kHz waveform",
            "mvp": "export after flow parity",
        },
    ],
    "acceptance": [
        "deterministic tensor parity for fixed reference conditioning",
        "waveform length parity",
        "waveform correlation / max-abs comparison",
        "end-to-end audio smoke test",
    ],
}


def write_plan(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "codec.export-plan.json"
    path.write_text(json.dumps(EXPORT_PLAN, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-onnx",
        action="store_true",
        help="Exit non-zero after writing the plan because codec ONNX is not implemented.",
    )
    args = parser.parse_args(argv)
    path = write_plan(args.output)
    print(path)
    if args.require_onnx:
        raise SystemExit(
            f"S3 codec ONNX export is intentionally not implemented in the MVP; see {path}"
        )


if __name__ == "__main__":
    main()
