#!/usr/bin/env python3
"""Apply simple dynamic INT8 quantization to an ONNX LM.

This is an experiment after floating-point parity, not a replacement for
KittenTTS 2's packed ternary checkpoint format.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(args.input),
        str(args.output),
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )
    print(args.output)


if __name__ == "__main__":
    main()
