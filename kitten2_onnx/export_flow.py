"""Export the pinned two-step S3 mean-flow mel stage to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from kitten2_onnx.decoder_reference import load_decoder_reference
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from kitten2_onnx.runtime import (
    S3GEN_SILENCE_TAIL_TOKENS,
    S3GEN_SILENCE_TOKEN,
    S3GEN_SPEECH_VOCAB_SIZE,
    append_s3_silence_tail,
)
from kitten2_onnx.s3gen_reference import (
    S3GEN_FILENAME,
    S3GEN_REPOSITORY,
    S3GEN_REVISION,
    load_s3gen_model,
)

FLOW_FORMAT = "kitten2-onnx-flow-v1"
MEANFLOW_STEPS = 2
INPUT_NAMES = (
    "speech_tokens",
    "speech_token_lengths",
    "prompt_token",
    "prompt_token_len",
    "prompt_feat",
    "embedding",
    "flow_noise",
)
OUTPUT_NAMES = ("mel",)


class MeanFlowExportWrapper(nn.Module):
    """Expose S3Gen flow inference with full initial noise as an explicit input."""

    def __init__(self, flow: nn.Module):
        super().__init__()
        self.flow = flow
        self.estimator = flow.decoder.estimator

    def forward(
        self,
        speech_tokens: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        flow_noise: torch.Tensor,
    ) -> torch.Tensor:
        embedding = F.normalize(embedding, dim=1)
        speaker = self.flow.spk_embed_affine_layer(embedding)

        token = torch.cat((prompt_token, speech_tokens), dim=1).long()
        token_lengths = prompt_token_len + speech_token_lengths
        token_positions = torch.arange(token.shape[1], device=token.device)
        token_mask = (token_positions.unsqueeze(0) < token_lengths.unsqueeze(1))
        token_mask = token_mask.unsqueeze(-1).to(speaker)
        token_embeddings = self.flow.input_embedding(token) * token_mask

        hidden, hidden_mask = self.flow.encoder(token_embeddings, token_lengths)
        hidden_lengths = hidden_mask.sum(dim=-1).squeeze(-1)
        hidden = self.flow.encoder_proj(hidden)
        mean = hidden.transpose(1, 2).contiguous()

        prompt_frames = prompt_feat.shape[1]
        prompt_condition = prompt_feat.transpose(1, 2).contiguous()
        generated_condition = torch.zeros_like(mean)[:, :, prompt_frames:]
        condition = torch.cat((prompt_condition, generated_condition), dim=2)

        mel_positions = torch.arange(mean.shape[2], device=mean.device)
        mel_mask = mel_positions.unsqueeze(0) < hidden_lengths.unsqueeze(1)
        mel_mask = mel_mask.unsqueeze(1).to(mean)

        times = mean.new_tensor((0.0, 0.5, 1.0))
        sample = flow_noise
        for index in range(MEANFLOW_STEPS):
            start = times[index].unsqueeze(0)
            end = times[index + 1].unsqueeze(0)
            derivative = self.estimator(
                sample,
                mask=mel_mask,
                mu=mean,
                t=start,
                spks=speaker,
                cond=condition,
                r=end,
            )
            sample = sample + (end - start) * derivative

        return sample[:, :, prompt_frames:]


def _torch_inputs(arrays: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    speech_tokens = append_s3_silence_tail(arrays["speech_tokens"])
    speech_lengths = np.asarray([speech_tokens.shape[1]], dtype=np.int64)
    names = (
        "conditioning_prompt_token",
        "conditioning_prompt_token_len",
        "conditioning_prompt_feat",
        "conditioning_embedding",
        "flow_noise",
    )
    return (
        torch.from_numpy(speech_tokens),
        torch.from_numpy(speech_lengths),
        *(torch.from_numpy(arrays[name]) for name in names),
    )

def _validate_noise_contract(inputs: tuple[torch.Tensor, ...], flow: nn.Module) -> None:
    speech_tokens, speech_lengths, prompt_tokens, prompt_lengths, _, _, noise = inputs
    if speech_tokens.shape[0] != 1 or prompt_tokens.shape[0] != 1:
        raise ValueError("The S3 flow exporter supports batch size one")
    if speech_lengths.tolist() != [speech_tokens.shape[1]]:
        raise ValueError("speech_token_lengths must match the speech_tokens sequence")
    expected_frames = int(flow.token_mel_ratio) * int(
        prompt_lengths[0].item() + speech_lengths[0].item()
    )
    if noise.shape != (1, int(flow.output_size), expected_frames):
        raise ValueError(
            "flow_noise must contain the full initial latent noise with shape "
            f"(1, {int(flow.output_size)}, {expected_frames}), found {tuple(noise.shape)}"
        )


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.shape != actual.shape:
        raise ValueError(f"Mel shape mismatch: expected {reference.shape}, found {actual.shape}")
    if not np.isfinite(actual).all():
        raise ValueError("ONNX flow output contains non-finite values")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    correlation = float(np.corrcoef(reference.reshape(-1), actual.reshape(-1))[0, 1])
    return {
        "correlation": correlation,
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
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


def _make_dynamic_inputs(
    inputs: tuple[torch.Tensor, ...], flow: nn.Module, token_count: int
) -> tuple[torch.Tensor, ...]:
    speech_tokens, _, prompt_tokens, prompt_lengths, prompt_feat, embedding, _ = inputs
    speech_tokens = speech_tokens[:, : -S3GEN_SILENCE_TAIL_TOKENS]
    if token_count < 1 or token_count > speech_tokens.shape[1]:
        raise ValueError("dynamic token_count must fit the captured speech token sequence")
    tail = torch.full(
        (1, S3GEN_SILENCE_TAIL_TOKENS), S3GEN_SILENCE_TOKEN, dtype=torch.long
    )
    speech_tokens = torch.cat((speech_tokens[:, :token_count], tail), dim=1).contiguous()
    speech_lengths = torch.tensor([speech_tokens.shape[1]], dtype=torch.long)
    noise_frames = int(flow.token_mel_ratio) * int(
        prompt_lengths[0].item() + speech_tokens.shape[1]
    )
    torch.manual_seed(8675309)
    noise = torch.randn((1, int(flow.output_size), noise_frames), dtype=torch.float32)
    return (
        speech_tokens,
        speech_lengths,
        prompt_tokens,
        prompt_lengths,
        prompt_feat,
        embedding,
        noise,
    )


def export_flow(
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
    flow = model.flow
    wrapper = MeanFlowExportWrapper(flow).eval()
    del model

    example_inputs = _torch_inputs(arrays)
    _validate_noise_contract(example_inputs, flow)
    with torch.inference_mode():
        torch_mel = wrapper(*example_inputs).cpu().numpy()
    reference_mel = arrays["flow_mel"]
    torch_metrics = _metrics(reference_mel, torch_mel)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "flow.onnx"
    dynamic_axes = {
        "speech_tokens": {1: "speech_tokens"},
        "prompt_token": {1: "prompt_tokens"},
        "prompt_feat": {1: "prompt_mel_frames"},
        "flow_noise": {2: "initial_noise_frames"},
        "mel": {2: "generated_mel_frames"},
    }
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(onnx_path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_axes=dynamic_axes,
        opset_version=int(opset),
        dynamo=False,
        external_data=True,
    )

    model_proto = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model_proto)
    random_ops = sorted(
        {node.op_type for node in model_proto.graph.node}
        & {"RandomNormal", "RandomNormalLike", "Multinomial"}
    )
    if random_ops:
        raise ValueError(f"Exported flow graph contains random operators: {random_ops}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_mel = session.run(list(OUTPUT_NAMES), _ort_inputs(example_inputs))[0]
    metrics = _metrics(reference_mel, ort_mel)

    raw_token_count = example_inputs[0].shape[1] - S3GEN_SILENCE_TAIL_TOKENS
    dynamic_count = min(16, int(raw_token_count))
    dynamic_inputs = _make_dynamic_inputs(example_inputs, flow, dynamic_count)
    _validate_noise_contract(dynamic_inputs, flow)
    with torch.inference_mode():
        dynamic_torch_mel = wrapper(*dynamic_inputs).cpu().numpy()
    dynamic_ort_mel = session.run(list(OUTPUT_NAMES), _ort_inputs(dynamic_inputs))[0]
    dynamic_metrics = _metrics(dynamic_torch_mel, dynamic_ort_mel)

    threshold = 0.999
    parity_passed = metrics["correlation"] > threshold and dynamic_metrics["correlation"] > threshold
    external_data = sorted(
        path.name for path in output_dir.glob("flow.onnx*") if path != onnx_path
    )
    metadata: dict[str, Any] = {
        "format": FLOW_FORMAT,
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
        "meanflow_steps": MEANFLOW_STEPS,
        "token_mel_ratio": int(flow.token_mel_ratio),
        "speech_vocab_size": S3GEN_SPEECH_VOCAB_SIZE,
        "silence_tail": {
            "token_id": S3GEN_SILENCE_TOKEN,
            "count": S3GEN_SILENCE_TAIL_TOKENS,
            "mel_frames_per_token": int(flow.token_mel_ratio),
            "append_applied_by": "runtime",
            "graph_expects_padded_tokens": True,
        },
        "noise_contract": (
            "full initial latent noise, shape [1, flow_channels, "
            "token_mel_ratio * (prompt_token_len + speech_token_len including silence tail)]"
        ),
        "input_shapes": {
            name: list(value.shape) for name, value in zip(INPUT_NAMES, example_inputs, strict=True)
        },
        "dynamic_axes": dynamic_axes,
        "output_shape": list(ort_mel.shape),
        "reference_torch_metrics": torch_metrics,
        "onnx_parity": metrics,
        "dynamic_length_check": {
            "speech_token_count": dynamic_count,
            "mel_shape": list(dynamic_ort_mel.shape),
            "metrics": dynamic_metrics,
        },
        "correlation_threshold": threshold,
        "parity_passed": parity_passed,
        "random_operators": random_ops,
        "graph_sha256": _sha256(onnx_path),
        "external_data_sha256": {name: _sha256(output_dir / name) for name in external_data},
    }
    metadata_path = output_dir / "flow.meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return onnx_path, metadata_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Captured decoder reference NPZ")
    parser.add_argument("--output", type=Path, required=True, help="Flow ONNX output directory")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)

    graph_path, metadata_path = export_flow(args.reference, args.output, opset=args.opset)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(json.dumps({"graph": str(graph_path), "metadata": str(metadata_path), **metadata}, indent=2))
    if not metadata["parity_passed"]:
        raise SystemExit("Flow ONNX parity did not meet the correlation threshold")


if __name__ == "__main__":
    main()
