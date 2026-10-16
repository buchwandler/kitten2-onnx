"""Export the pinned non-streaming S3Gen mel-to-waveform stage to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from kitten2_onnx.decoder_reference import load_decoder_reference
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from kitten2_onnx.s3gen_reference import (
    S3GEN_FILENAME,
    S3GEN_REPOSITORY,
    S3GEN_REVISION,
    load_s3gen_model,
)

MEL2WAV_FORMAT = "kitten2-onnx-mel2wav-v1"
INPUT_NAMES = ("mel", "phase_noise", "sine_noise")
OUTPUT_NAMES = ("waveform",)


class Mel2WavExportWrapper(nn.Module):
    """Run the HiFT vocoder with its phase and harmonic noise supplied as inputs."""

    def __init__(self, vocoder: nn.Module):
        super().__init__()
        self.vocoder = vocoder
        self.sine_generator = vocoder.m_source.l_sin_gen

    def forward(
        self,
        mel: torch.Tensor,
        phase_noise: torch.Tensor,
        sine_noise: torch.Tensor,
    ) -> torch.Tensor:
        f0 = self.vocoder.f0_predictor(mel)
        source_f0 = self.vocoder.f0_upsamp(f0[:, None])
        harmonics = torch.arange(
            1,
            self.sine_generator.harmonic_num + 2,
            dtype=mel.dtype,
            device=mel.device,
        ).view(1, -1, 1)
        frequencies = source_f0 * harmonics / self.sine_generator.sampling_rate
        phase = 2 * np.pi * torch.remainder(torch.cumsum(frequencies, dim=-1), 1.0)
        voiced = (source_f0 > self.sine_generator.voiced_threshold).to(mel)
        sine = self.sine_generator.sine_amp * torch.sin(phase + phase_noise)
        noise_amplitude = (
            voiced * self.sine_generator.noise_std
            + (1 - voiced) * self.sine_generator.sine_amp / 3
        )
        sine = sine * voiced + noise_amplitude * sine_noise
        source = self.vocoder.m_source.l_linear(sine.transpose(1, 2))
        source = self.vocoder.m_source.l_tanh(source).transpose(1, 2)
        return self.vocoder.decode(x=mel, s=source)


def _torch_inputs(arrays: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    names = ("mel2wav_input", "vocoder_phase_noise", "vocoder_sine_noise")
    return tuple(torch.from_numpy(arrays[name]) for name in names)


def _validate_noise_contract(inputs: tuple[torch.Tensor, ...], vocoder: nn.Module) -> None:
    mel, phase_noise, sine_noise = inputs
    harmonics = vocoder.m_source.l_sin_gen.harmonic_num + 1
    samples_per_frame = int(vocoder.f0_upsamp.scale_factor)
    expected_samples = mel.shape[2] * samples_per_frame
    if mel.shape[0] != 1:
        raise ValueError("The mel-to-waveform exporter supports batch size one")
    if phase_noise.shape != (1, harmonics, 1):
        raise ValueError(f"phase_noise must have shape (1, {harmonics}, 1)")
    if sine_noise.shape != (1, harmonics, expected_samples):
        raise ValueError(
            "sine_noise must have shape "
            f"(1, {harmonics}, {expected_samples}), found {tuple(sine_noise.shape)}"
        )


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise ValueError(f"Waveform length mismatch: expected {reference.shape}, found {actual.shape}")
    if not np.isfinite(reference).all():
        raise ValueError("PyTorch reference waveform contains non-finite values")
    if not np.isfinite(actual).all():
        raise ValueError("ONNX mel-to-waveform output contains non-finite values")
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    difference = actual64 - reference64
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(reference64))))
    indices = sorted(
        {
            0,
            1,
            2,
            reference.shape[-1] // 4,
            reference.shape[-1] // 2,
            reference.shape[-1] * 3 // 4,
            reference.shape[-1] - 2,
            reference.shape[-1] - 1,
        }
    )
    return {
        "correlation": float(np.corrcoef(reference64.reshape(-1), actual64.reshape(-1))[0, 1]),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "relative_rmse": rmse / reference_rms,
        "reference_rms": reference_rms,
        "actual_rms": float(np.sqrt(np.mean(np.square(actual64)))),
        "sample_metrics": [
            {
                "index": index,
                "reference": float(reference64.reshape(-1)[index]),
                "actual": float(actual64.reshape(-1)[index]),
                "abs_error": float(abs(difference.reshape(-1)[index])),
            }
            for index in indices
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ort_inputs(inputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(INPUT_NAMES, inputs, strict=True)
    }


def export_mel2wav(
    reference_path: str | Path,
    output_dir: str | Path,
    *,
    opset: int = 18,
) -> tuple[Path, Path]:
    import onnx
    import onnxruntime as ort

    reference_path = Path(reference_path)
    reference = load_decoder_reference(reference_path)
    arrays = reference.arrays
    model, weights_path = load_s3gen_model()
    vocoder = model.mel2wav
    wrapper = Mel2WavExportWrapper(vocoder).eval()
    del model

    example_inputs = _torch_inputs(arrays)
    _validate_noise_contract(example_inputs, vocoder)
    with torch.inference_mode():
        torch_waveform = wrapper(*example_inputs).cpu().numpy()
    reference_waveform = arrays["mel2wav_waveform"]
    torch_metrics = _metrics(reference_waveform, torch_waveform)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "mel2wav.onnx"
    dynamic_axes = {
        "mel": {2: "mel_frames"},
        "sine_noise": {2: "vocoder_samples"},
        "waveform": {1: "waveform_samples"},
    }
    mel_frames = torch.export.Dim("mel_frames", min=2)
    dynamic_shapes = ({2: mel_frames}, {}, {2: mel_frames * int(vocoder.f0_upsamp.scale_factor)})
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(onnx_path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_shapes=dynamic_shapes,
        opset_version=int(opset),
        dynamo=True,
        external_data=True,
    )

    model_proto = onnx.load(str(onnx_path), load_external_data=False)
    graph_nodes = []
    scatter_index_casts = 0
    for node_index, node in enumerate(model_proto.graph.node):
        if node.op_type == "ScatterND":
            cast_output = f"{node.input[1]}_scatter_index_{node_index}"
            graph_nodes.append(
                onnx.helper.make_node(
                    "Cast", [node.input[1]], [cast_output], to=onnx.TensorProto.INT64
                )
            )
            node.input[1] = cast_output
            scatter_index_casts += 1
        graph_nodes.append(node)
    if scatter_index_casts:
        del model_proto.graph.node[:]
        model_proto.graph.node.extend(graph_nodes)
        onnx.save_model(model_proto, str(onnx_path))
    onnx.checker.check_model(str(onnx_path))
    model_proto = onnx.load(str(onnx_path), load_external_data=False)
    random_ops = sorted(
        {node.op_type for node in model_proto.graph.node}
        & {"RandomNormal", "RandomNormalLike", "Multinomial"}
    )
    if random_ops:
        raise ValueError(f"Exported mel2wav graph contains random operators: {random_ops}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_waveform = session.run(list(OUTPUT_NAMES), _ort_inputs(example_inputs))[0]
    metrics = _metrics(reference_waveform, ort_waveform)

    dynamic_mel = example_inputs[0][:, :, :31].contiguous()
    dynamic_phase = example_inputs[1]
    dynamic_samples = int(vocoder.f0_upsamp.scale_factor) * dynamic_mel.shape[2]
    dynamic_noise = example_inputs[2][:, :, :dynamic_samples].contiguous()
    dynamic_inputs = (dynamic_mel, dynamic_phase, dynamic_noise)
    _validate_noise_contract(dynamic_inputs, vocoder)
    with torch.inference_mode():
        dynamic_torch_waveform = wrapper(*dynamic_inputs).cpu().numpy()
    dynamic_ort_waveform = session.run(list(OUTPUT_NAMES), _ort_inputs(dynamic_inputs))[0]
    dynamic_metrics = _metrics(dynamic_torch_waveform, dynamic_ort_waveform)

    correlation_threshold = 0.999
    relative_rmse_threshold = 1e-4
    parity_passed = all(
        result["correlation"] > correlation_threshold
        and result["relative_rmse"] < relative_rmse_threshold
        for result in (metrics, dynamic_metrics)
    )
    external_data = sorted(
        path.name for path in output_dir.glob("mel2wav.onnx*") if path != onnx_path
    )
    metadata: dict[str, Any] = {
        "format": MEL2WAV_FORMAT,
        "graph": onnx_path.name,
        "external_data": external_data,
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "voice": reference.metadata["voice"],
        "voice_reference": reference_path.name,
        "voice_reference_sha256": _sha256(reference_path),
        "source_model_revision": reference.metadata.get("source_model_revision"),
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "s3gen_repository": S3GEN_REPOSITORY,
        "s3gen_revision": S3GEN_REVISION,
        "s3gen_weights": S3GEN_FILENAME,
        "s3gen_weights_sha256": _sha256(weights_path),
        "precision": "fp32",
        "provider": "CPUExecutionProvider",
        "opset": int(opset),
        "sample_rate": 24000,
        "trim_fade_samples": 24000 // 50,
        "samples_per_mel_frame": int(vocoder.f0_upsamp.scale_factor),
        "fade_policy": "the upstream 20 ms trim fade is applied outside this graph",
        "noise_contract": {
            "phase_noise": "uniform phase angles with the fundamental phase set to zero",
            "sine_noise": "explicit Gaussian harmonic noise",
        },
        "input_shapes": {
            name: list(value.shape) for name, value in zip(INPUT_NAMES, example_inputs, strict=True)
        },
        "dynamic_axes": dynamic_axes,
        "output_shape": list(ort_waveform.shape),
        "reference_torch_metrics": torch_metrics,
        "onnx_parity": metrics,
        "dynamic_length_check": {
            "mel_frames": int(dynamic_mel.shape[2]),
            "waveform_shape": list(dynamic_ort_waveform.shape),
            "metrics": dynamic_metrics,
        },
        "correlation_threshold": correlation_threshold,
        "relative_rmse_threshold": relative_rmse_threshold,
        "parity_passed": parity_passed,
        "random_operators": random_ops,
        "scatter_index_casts_to_int64": scatter_index_casts,
        "graph_sha256": _sha256(onnx_path),
        "external_data_sha256": {name: _sha256(output_dir / name) for name in external_data},
    }
    metadata_path = output_dir / "mel2wav.meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return onnx_path, metadata_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Captured decoder reference NPZ")
    parser.add_argument("--output", type=Path, required=True, help="Mel-to-waveform output directory")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)

    graph_path, metadata_path = export_mel2wav(args.reference, args.output, opset=args.opset)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(json.dumps({"graph": str(graph_path), "metadata": str(metadata_path), **metadata}, indent=2))
    if not metadata["parity_passed"]:
        raise SystemExit("Mel-to-waveform ONNX parity did not meet the correlation threshold")


if __name__ == "__main__":
    main()
