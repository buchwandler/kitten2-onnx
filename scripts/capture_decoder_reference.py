from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from kitten2_onnx.decoder_reference import DECODER_REFERENCE_FORMAT, write_decoder_reference
from kitten2_onnx.reference import UPSTREAM_CODE_REVISION
from kitten2_onnx.runtime import (
    S3GEN_MEL_FRAMES_PER_TOKEN,
    S3GEN_SILENCE_TAIL_TOKENS,
    S3GEN_SILENCE_TOKEN,
)
from kitten2_onnx.s3gen_reference import (
    S3GEN_REPOSITORY,
    S3GEN_REVISION,
    load_s3gen_model,
)

MEANFLOW_STEPS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _voice_paths(
    model_dir: Path, config: dict[str, Any], voice: str
) -> tuple[Path, Path, dict[str, Any]]:
    voices_path = model_dir / config.get("voices", "voices/voices.json")
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    if voice not in voices:
        raise ValueError(f"Unknown voice {voice!r}; available voices: {', '.join(sorted(voices))}")
    entry = voices[voice]
    voice_dir = voices_path.parent
    return voice_dir / entry["reference"], voice_dir / entry["artifacts"], entry

def capture_reference(
    model_dir: str | Path,
    voice: str,
    *,
    token_count: int = 32,
    flow_seed: int = 314159,
    vocoder_seed: int = 271828,
    s3gen_context: tuple[Any, Path] | None = None,
    s3gen_weights_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import soundfile as sf
    import torch

    model_dir = Path(model_dir).resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_manifest_path = model_dir / "model_revision.json"
    if not model_manifest_path.is_file():
        raise FileNotFoundError(f"Pinned model revision metadata is missing: {model_manifest_path}")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    source_model_revision = model_manifest.get("source_model_revision")
    if not source_model_revision:
        raise ValueError(f"No pinned source model revision is recorded in {model_manifest_path}")
    upstream_revision = model_manifest.get("upstream_code_revision")
    if upstream_revision != UPSTREAM_CODE_REVISION:
        raise ValueError(
            f"Model assets use upstream revision {upstream_revision}, "
            f"expected {UPSTREAM_CODE_REVISION}"
        )

    reference_path, voice_artifact_path, voice_entry = _voice_paths(model_dir, config, voice)
    with np.load(voice_artifact_path, allow_pickle=False) as voice_archive:
        speaker_embedding = np.asarray(voice_archive["embedding"], dtype=np.float32)
        reference_tokens = np.asarray(voice_archive["reference_tokens"], dtype=np.int64)
    reference_waveform, reference_sample_rate = sf.read(reference_path, dtype="float32")
    if reference_waveform.ndim == 2:
        reference_waveform = reference_waveform.mean(axis=1)
    reference_waveform = torch.from_numpy(np.asarray(reference_waveform, dtype=np.float32))[None]

    model, weights_path = s3gen_context if s3gen_context is not None else load_s3gen_model()
    weights_sha256 = (
        s3gen_weights_sha256 if s3gen_weights_sha256 is not None else _sha256(weights_path)
    )

    with torch.inference_mode():
        conditioning = model.embed_ref(reference_waveform, reference_sample_rate)
    prompt_length = int(conditioning["prompt_token_len"][0].item())
    token_count = int(token_count)
    if token_count < 1 or token_count > prompt_length:
        raise ValueError(f"token_count must be between 1 and {prompt_length}")


    flow_conditioning = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in conditioning.items()
    }
    speech_tokens = conditioning["prompt_token"][:, :token_count].contiguous()
    speech_token_lengths = torch.tensor([token_count], dtype=torch.long)
    tail = torch.full(
        (1, S3GEN_SILENCE_TAIL_TOKENS), S3GEN_SILENCE_TOKEN, dtype=torch.long
    )
    flow_speech_tokens = torch.cat((speech_tokens, tail), dim=1)
    flow_token_lengths = torch.tensor([flow_speech_tokens.shape[1]], dtype=torch.long)
    noise_shape = (
        1,
        int(model.flow.output_size),
        int(model.flow.token_mel_ratio) * (prompt_length + flow_speech_tokens.shape[1]),
    )
    torch.manual_seed(flow_seed)
    flow_noise = torch.randn(noise_shape, dtype=model.dtype)
    with torch.inference_mode():
        flow_mel = model.forward(
            flow_speech_tokens,
            ref_wav=None,
            ref_sr=None,
            ref_dict=flow_conditioning,
            n_cfm_timesteps=MEANFLOW_STEPS,
            finalize=True,
            speech_token_lens=flow_token_lengths,
            skip_vocoder=True,
            noised_mels=flow_noise,
        )

        mel2wav_input = flow_mel.to(dtype=model.dtype)
        f0 = model.mel2wav.f0_predictor(mel2wav_input)
        upsampled_f0 = model.mel2wav.f0_upsamp(f0[:, None])
        sine_generator = model.mel2wav.m_source.l_sin_gen
        phase_shape = (1, sine_generator.harmonic_num + 1, 1)
        sine_noise_shape = (1, sine_generator.harmonic_num + 1, upsampled_f0.shape[-1])

        torch.manual_seed(vocoder_seed)
        phase_noise = torch.rand(phase_shape, dtype=model.dtype) * (2 * np.pi) - np.pi
        phase_noise[:, 0, :] = 0
        sine_noise = torch.randn(sine_noise_shape, dtype=model.dtype)

        torch.manual_seed(vocoder_seed)
        mel2wav_waveform, _ = model.mel2wav.inference(
            speech_feat=mel2wav_input,
            cache_source=torch.zeros((1, 1, 0), dtype=model.dtype),
        )
        waveform = mel2wav_waveform.clone()
        waveform[:, : len(model.trim_fade)] *= model.trim_fade
    tensors = {
        "conditioning_prompt_token": conditioning["prompt_token"],
        "conditioning_prompt_token_len": conditioning["prompt_token_len"],
        "conditioning_prompt_feat": conditioning["prompt_feat"],
        "conditioning_embedding": conditioning["embedding"],
        "speech_tokens": speech_tokens,
        "speech_token_lengths": speech_token_lengths,
        "flow_noise": flow_noise,
        "flow_mel": flow_mel,
        "mel2wav_input": mel2wav_input,
        "vocoder_phase_noise": phase_noise,
        "vocoder_sine_noise": sine_noise,
        "mel2wav_waveform": mel2wav_waveform,
        "waveform": waveform,
    }
    arrays = {
        name: np.asarray(tensor.detach().cpu().numpy()) for name, tensor in tensors.items()
    }
    arrays["speaker_embedding"] = speaker_embedding
    arrays["reference_tokens"] = reference_tokens
    if any(not np.isfinite(array).all() for array in arrays.values() if array.dtype.kind == "f"):
        raise ValueError("Decoder reference contains non-finite values")

    metadata = {
        "format": DECODER_REFERENCE_FORMAT,
        "voice": voice,
        "voice_id": voice_entry.get("voice_id"),
        "reference_transcript": voice_entry["transcript"],
        "reference_audio": reference_path.relative_to(model_dir).as_posix(),
        "reference_audio_sha256": _sha256(reference_path),
        "reference_sample_rate": int(reference_sample_rate),
        "source_model_revision": source_model_revision,
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "s3gen_repository": S3GEN_REPOSITORY,
        "s3gen_revision": S3GEN_REVISION,
        "s3gen_weights_sha256": weights_sha256,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cpu",
        "dtype": str(model.dtype),
        "sample_rate": 24000,
        "meanflow_steps": MEANFLOW_STEPS,
        "token_count": int(token_count),
        "flow_seed": int(flow_seed),
        "vocoder_seed": int(vocoder_seed),
        "fixed_tokens_source": "conditioning_prompt_token prefix",
        "flow_noise_contract": "full initial latent noise replaces all CFM random initialization",
        "flow_finalize": True,
        "s3_silence_tail": {
            "token_id": S3GEN_SILENCE_TOKEN,
            "count": S3GEN_SILENCE_TAIL_TOKENS,
            "mel_frames_per_token": S3GEN_MEL_FRAMES_PER_TOKEN,
        },
        "vocoder_noise_contract": "captured phase angles and Gaussian harmonic noise",
        "conditioning": {
            "prompt_feat_len": None,
            "shapes": {
                name: list(arrays[array_name].shape)
                for name, array_name in (
                    ("prompt_token", "conditioning_prompt_token"),
                    ("prompt_token_len", "conditioning_prompt_token_len"),
                    ("prompt_feat", "conditioning_prompt_feat"),
                    ("embedding", "conditioning_embedding"),
                )
            },
        },
    }
    return arrays, metadata


def write_capture(
    output_path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".meta.json")
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp.npz"
    )
    temporary_metadata_path = temporary_path.with_suffix(".meta.json")
    try:
        write_decoder_reference(temporary_path, arrays, metadata)
        os.replace(temporary_path, output_path)
        os.replace(temporary_metadata_path, metadata_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)
    return output_path, metadata_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture fixed-input S3Gen flow and waveform reference tensors."
    )
    parser.add_argument("--model-dir", type=Path, default=Path(".models/kitten-tts-2"))
    parser.add_argument("--voice", default="Bruno")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--token-count", type=int, default=32)
    parser.add_argument("--flow-seed", type=int, default=314159)
    parser.add_argument("--vocoder-seed", type=int, default=271828)
    args = parser.parse_args()

    arrays, metadata = capture_reference(
        args.model_dir,
        args.voice,
        token_count=args.token_count,
        flow_seed=args.flow_seed,
        vocoder_seed=args.vocoder_seed,
    )
    output_path = args.output or Path("build/voices") / f"{args.voice}.npz"
    output_path, metadata_path = write_capture(output_path, arrays, metadata)
    print(
        json.dumps(
            {
                "npz": str(output_path),
                "metadata": str(metadata_path),
                "voice": args.voice,
                "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
