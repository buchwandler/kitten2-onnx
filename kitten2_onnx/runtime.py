"""Minimal ONNX Runtime driver for the exported KittenTTS 2 speech LM."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kitten2_onnx.sampling import SamplingSettings, is_eos_token, sample
from kitten2_onnx.tokens import TokenMap

S3GEN_SILENCE_TOKEN = 4299
S3GEN_SILENCE_TAIL_TOKENS = 3
S3GEN_MEL_FRAMES_PER_TOKEN = 2
S3GEN_SAMPLE_RATE = 24000
S3GEN_SAMPLES_PER_MEL_FRAME = 480
S3GEN_SPEECH_VOCAB_SIZE = 6561


def _cpu_session_options(profile: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if profile == "default":
        return options
    if profile != "low-memory-exact-ternary":
        raise ValueError(f"Unsupported CPU runtime memory profile: {profile!r}")
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


def clean_s3_tokens(
    speech_tokens: list[int] | np.ndarray, vocab_size: int = S3GEN_SPEECH_VOCAB_SIZE
) -> np.ndarray:
    """Keep only valid zero-based codec IDs accepted by the flow graph."""
    tokens = np.asarray(speech_tokens, dtype=np.int64)
    if tokens.ndim == 2 and tokens.shape[0] == 1:
        tokens = tokens[0]
    elif tokens.ndim != 1:
        raise ValueError("speech_tokens must have shape [T] or [1, T]")
    return tokens[(tokens >= 0) & (tokens < vocab_size)]


def append_s3_silence_tail(
    speech_tokens: list[int] | np.ndarray,
    token_id: int = S3GEN_SILENCE_TOKEN,
    count: int = S3GEN_SILENCE_TAIL_TOKENS,
) -> np.ndarray:
    """Append the silence tokens required by upstream S3 lookahead."""
    tokens = np.asarray(speech_tokens, dtype=np.int64)
    if tokens.ndim == 1:
        tail_shape = (count,)
    elif tokens.ndim == 2 and tokens.shape[0] == 1:
        tail_shape = (1, count)
    else:
        raise ValueError("speech_tokens must have shape [T] or [1, T]")
    tail = np.full(tail_shape, token_id, dtype=np.int64)
    return np.concatenate((tokens, tail), axis=-1)


@dataclass(frozen=True)
class GenerationResult:
    raw_ids: list[int]
    s3_ids: list[int]


class SharedVocabulary:
    """One reconstructed vocabulary table shared by token lookup and output projection."""

    def __init__(self, model_dir: Path, metadata: Mapping[str, object], ort):
        self.storage_dtype = str(metadata["embedding_storage_dtype"])
        embedding_path = model_dir / str(metadata["embedding_file"])
        expected_shape = (int(metadata["vocab_size"]), int(metadata["hidden_size"]))
        if self.storage_dtype == "kitten-emb4":
            import torch

            from kitten2_onnx.tl2_emb4 import read_emb4_embedding_asset

            table = read_emb4_embedding_asset(embedding_path)
            if tuple(table.shape) != expected_shape:
                raise ValueError(
                    f"Vocabulary shape {tuple(table.shape)} differs from metadata {expected_shape}"
                )
            weights = table.to(torch.float32).numpy()
            self.storage = None
        else:
            self.storage = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            if self.storage.shape != expected_shape:
                raise ValueError(
                    f"Vocabulary shape {self.storage.shape} differs from metadata {expected_shape}"
                )
            if self.storage_dtype == "bfloat16-u16":
                if self.storage.dtype != np.uint16:
                    raise ValueError("bfloat16-u16 vocabulary storage must use uint16")
                bits = np.asarray(self.storage, dtype=np.uint32) << 16
                weights = bits.view(np.float32)
            elif self.storage_dtype in {"float16", "float32"}:
                if str(self.storage.dtype) != self.storage_dtype:
                    raise ValueError("Vocabulary array dtype differs from metadata")
                weights = self.storage.astype(np.float32, copy=False)
            else:
                raise ValueError(
                    f"Unsupported shared vocabulary storage dtype: {self.storage_dtype!r}"
                )
        self.weights = np.ascontiguousarray(weights)
        self.ortvalue = ort.OrtValue.ortvalue_from_numpy(self.weights)
    def lookup(self, token_ids: np.ndarray, dtype: np.dtype) -> np.ndarray:
        return np.asarray(self.weights[token_ids], dtype=dtype)


class Kitten2OnnxLM:
    """Drive the exported Qwen3 LM with explicit embeddings and KV cache.

    This class handles speaker projection and LM token generation. It does not yet
    reproduce the S3 waveform decoder.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        speaker_model: str | Path | None = None,
        providers: list[str] | None = None,
        materialization_cache_dir: str | Path | None = None,
    ):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "lm.meta.json").read_text(encoding="utf-8"))
        bundle_manifest_path = self.model_dir.parent / "bundle.json"
        self.bundle_manifest = (
            json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
            if bundle_manifest_path.is_file()
            else {}
        )
        bundle_lm = self.bundle_manifest.get("lm", {})
        bundle_storage = bundle_lm.get("storage", {}) if isinstance(bundle_lm, dict) else {}
        self.tritpack_storage = (
            bundle_storage.get("transformer") if isinstance(bundle_storage, dict) else None
        )
        if self.tritpack_storage is not None and not isinstance(self.tritpack_storage, dict):
            raise ValueError("LM bundle transformer storage metadata must be an object")
        runtime_profile = self.meta.get("runtime_profile")
        io_dtype = self.meta.get("io_dtype")
        if runtime_profile is None or io_dtype is None:
            raise ValueError("LM metadata must declare runtime_profile and io_dtype")
        if io_dtype == "bfloat16":
            raise NotImplementedError(
                "bfloat16 LM I/O requires ONNX Runtime I/O binding; NumPy session.run is unsupported"
            )

        runtime_provider = self.meta.get("runtime_provider")
        if runtime_provider not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
            raise ValueError(f"Unsupported runtime provider in LM metadata: {runtime_provider!r}")
        providers = providers or [runtime_provider]
        if runtime_provider == "CUDAExecutionProvider" and runtime_provider not in providers:
            raise ValueError(f"Profile {runtime_profile!r} requires CUDAExecutionProvider")

        self.input_mode = self.meta.get("input_mode", "inputs_embeds")
        self.vocabulary_mode = self.meta.get("vocabulary_mode", "embedded")
        if self.vocabulary_mode not in {"embedded", "shared"}:
            raise ValueError(f"Unsupported vocabulary mode: {self.vocabulary_mode!r}")
        self.vocabulary = None
        if self.vocabulary_mode == "shared":
            if self.input_mode != "inputs_embeds":
                raise ValueError("Shared vocabulary requires input_mode='inputs_embeds'")
            self.vocabulary = SharedVocabulary(self.model_dir, self.meta, ort)
            self.embeddings = None
            self.embedding_storage_dtype = self.vocabulary.storage_dtype
        elif self.input_mode == "inputs_embeds":
            self.embeddings = np.load(
                self.model_dir / self.meta["embedding_file"], mmap_mode="r", allow_pickle=False
            )
            self.embedding_storage_dtype = self.meta.get(
                "embedding_storage_dtype", str(self.embeddings.dtype)
            )
            if self.embedding_storage_dtype == "bfloat16-u16":
                if self.embeddings.dtype != np.uint16:
                    raise ValueError("bfloat16-u16 embedding storage must use uint16")
            elif str(self.embeddings.dtype) != self.embedding_storage_dtype:
                raise ValueError("Embedding array dtype differs from LM metadata")
        elif self.input_mode == "input_ids":
            if "embedding_file" in self.meta:
                raise ValueError(
                    "input_ids LM metadata must not reference a separate embedding table"
                )
            self.embeddings = None
            self.embedding_storage_dtype = None
        else:
            raise ValueError(f"Unsupported LM input mode: {self.input_mode!r}")
        storage_runtime_profile = (
            self.tritpack_storage.get("runtime_memory_profile")
            if isinstance(self.tritpack_storage, dict)
            else None
        )
        runtime_memory_profile = self.meta.get(
            "runtime_memory_profile", storage_runtime_profile or "default"
        )
        if runtime_memory_profile not in {"default", "low-memory-exact-ternary"}:
            raise ValueError(f"Unsupported runtime memory profile: {runtime_memory_profile!r}")
        self.model_path = self.model_dir / "lm.onnx"
        self.materialization_provenance = None
        self.materialization_seconds = 0.0
        storage = self.tritpack_storage
        if isinstance(storage, dict) and storage.get("method") == "ternary-tritpack-exact":
            if storage.get("runtime_materialization_required") is not True:
                raise ValueError("exact tritpack storage requires local runtime materialization")
            if (
                storage.get("transformed_matmul_count") != 196
                or type(storage.get("transformed_initializer_count")) is not int
                or storage.get("transformed_initializer_count") <= 0
            ):
                raise ValueError("runtime materialization requires all 196 packed MatMuls")
            if runtime_memory_profile != "low-memory-exact-ternary":
                raise ValueError("tritpack materialization requires the low-memory runtime profile")
            if runtime_provider != "CPUExecutionProvider" or providers != ["CPUExecutionProvider"]:
                raise ValueError("tritpack materialization requires CPUExecutionProvider only")
            from kitten2_onnx.materialized_weights import (
                default_materialization_cache_root,
                materialize_tritpack_model,
            )

            if materialization_cache_dir is None:
                cache_root = default_materialization_cache_root()
            else:
                cache_root = Path(materialization_cache_dir)
            materialization_started = time.perf_counter()
            materialized = materialize_tritpack_model(
                self.model_path,
                cache_root,
                storage_metadata=storage,
            )
            self.materialization_seconds = time.perf_counter() - materialization_started
            self.materialization_provenance = materialized.provenance
            self.model_path = materialized.model_path
        elif isinstance(storage, dict) and storage.get("runtime_materialization_required") is True:
            raise ValueError("unsupported LM runtime materialization storage method")
        if runtime_memory_profile == "low-memory-exact-ternary":
            if runtime_provider != "CPUExecutionProvider":
                raise ValueError("Low-memory exact-ternary profile requires CPUExecutionProvider")
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=_cpu_session_options(runtime_memory_profile),
                providers=providers,
            )
        else:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )
        self.output_session = None
        if self.vocabulary is not None:
            output_head_file = self.meta.get("output_head_file")
            if not isinstance(output_head_file, str):
                raise ValueError("Shared vocabulary metadata must declare output_head_file")
            self.output_session = ort.InferenceSession(
                str(self.model_dir / output_head_file), providers=providers
            )
            head_inputs = self.output_session.get_inputs()
            if [item.name for item in head_inputs] != self.meta.get("output_head_input_names"):
                raise ValueError("Output-head inputs differ from LM metadata")
            if {item.name: item.type for item in head_inputs} != {
                "hidden_states": "tensor(float)",
                "vocabulary_weight": "tensor(float)",
            }:
                raise ValueError(
                    "Shared output head must accept float32 hidden states and vocabulary"
                )
            if [item.name for item in self.output_session.get_outputs()] != ["logits"]:
                raise ValueError("Shared output head must return logits")
        self.speaker_session = (
            ort.InferenceSession(str(speaker_model), providers=providers)
            if speaker_model is not None
            else None
        )
        self.layers = int(self.meta["num_hidden_layers"])
        self.kv_heads = int(self.meta["num_key_value_heads"])
        self.head_dim = int(self.meta["head_dim"])

        graph_inputs = {item.name: item.type for item in self.session.get_inputs()}
        expected_type = {"float32": "tensor(float)", "float16": "tensor(float16)"}[io_dtype]
        if self.input_mode == "inputs_embeds":
            graph_input_type = graph_inputs.get("inputs_embeds")
            if graph_input_type != expected_type:
                raise ValueError(
                    f"LM metadata io_dtype {io_dtype!r} does not match inputs_embeds type "
                    f"{graph_input_type!r}"
                )
        else:
            if graph_inputs.get("input_ids") != "tensor(int64)":
                raise ValueError("input_ids LM graph must accept int64 token IDs")
            if graph_inputs.get("speaker_hidden") != expected_type:
                raise ValueError("LM metadata io_dtype does not match speaker_hidden graph input")
        self.model_numpy_dtype = {"float32": np.float32, "float16": np.float16}[io_dtype]

    def _empty_cache(self) -> dict[str, np.ndarray]:
        shape = (1, self.kv_heads, 0, self.head_dim)
        return {
            name: np.zeros(shape, dtype=self.model_numpy_dtype)
            for i in range(self.layers)
            for name in (f"past_key_{i}", f"past_value_{i}")
        }

    def _project_speaker(self, speaker_embedding: np.ndarray) -> np.ndarray:
        if self.speaker_session is None:
            raise ValueError("speaker_model was not supplied")
        speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32)
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding[None, :]
        return self.speaker_session.run(
            ["speaker_hidden"],
            {"speaker_embedding": speaker_embedding},
        )[0]

    def _embed(
        self,
        token_ids: np.ndarray,
        *,
        speaker_embedding: np.ndarray | None = None,
        prefill: bool,
    ) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if self.vocabulary is not None:
            values = self.vocabulary.lookup(token_ids, self.model_numpy_dtype)
        elif self.embedding_storage_dtype == "bfloat16-u16":
            bits = np.asarray(self.embeddings[token_ids], dtype=np.uint32) << 16
            values = bits.view(np.float32).astype(self.model_numpy_dtype)
        else:
            values = np.asarray(self.embeddings[token_ids], dtype=self.model_numpy_dtype)
        if prefill and speaker_embedding is not None:
            speaker_hidden = self._project_speaker(speaker_embedding)
            values = values.copy()
            values[0, 0, :] = speaker_hidden[0].astype(values.dtype, copy=False)
        return values

    def _project_logits(self, hidden_states: np.ndarray) -> np.ndarray:
        binding = self.output_session.io_binding()
        hidden_states = np.ascontiguousarray(hidden_states, dtype=np.float32)
        binding.bind_cpu_input("hidden_states", hidden_states)
        binding.bind_ortvalue_input("vocabulary_weight", self.vocabulary.ortvalue)
        binding.bind_output("logits", "cpu")
        self.output_session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]

    def step(
        self,
        token_ids: np.ndarray,
        *,
        past: dict[str, np.ndarray] | None = None,
        speaker_embedding: np.ndarray | None = None,
        position: int | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run prefill or one cached decode chunk.

        ``token_ids`` must have shape ``[1, S]``. If ``past`` is omitted this is
        treated as prefill and an empty cache is supplied.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("token_ids must have shape [1, S]")

        is_prefill = past is None
        past = self._empty_cache() if past is None else past
        past_len = next(iter(past.values())).shape[2]
        start = past_len if position is None else int(position)

        attention_mask = np.ones((1, past_len + token_ids.shape[1]), dtype=np.int64)
        position_ids = np.arange(
            start,
            start + token_ids.shape[1],
            dtype=np.int64,
        )[None, :]
        inputs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past,
        }
        if self.input_mode == "inputs_embeds":
            inputs["inputs_embeds"] = self._embed(
                token_ids,
                speaker_embedding=speaker_embedding,
                prefill=is_prefill,
            )
        else:
            speaker_hidden = np.zeros(
                (1, int(self.meta["hidden_size"])), dtype=self.model_numpy_dtype
            )
            if is_prefill and speaker_embedding is not None:
                speaker_hidden = self._project_speaker(speaker_embedding).astype(
                    self.model_numpy_dtype, copy=False
                )
            inputs["input_ids"] = token_ids
            inputs["speaker_hidden"] = speaker_hidden
        outputs = self.session.run(None, inputs)
        logits = self._project_logits(outputs[0]) if self.vocabulary is not None else outputs[0]
        new_past: dict[str, np.ndarray] = {}
        for i in range(self.layers):
            new_past[f"past_key_{i}"] = outputs[1 + 2 * i]
            new_past[f"past_value_{i}"] = outputs[2 + 2 * i]
        return logits, new_past

    def generate(
        self,
        prompt_ids: list[int],
        *,
        token_map: TokenMap,
        max_new_tokens: int,
        speaker_embedding: np.ndarray | None = None,
        settings: SamplingSettings | None = None,
        seed: int | None = None,
        stop_ids: set[int] | None = None,
        logits_callback: Callable[[int, np.ndarray], None] | None = None,
    ) -> GenerationResult:
        """Generate raw LM IDs and their in-range, zero-based S3 counterparts."""
        if not prompt_ids:
            raise ValueError("prompt_ids may not be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return GenerationResult(raw_ids=[], s3_ids=[])

        settings = settings or SamplingSettings()
        rng = np.random.default_rng(seed)
        history = [int(token) for token in prompt_ids]
        prompt_length = len(history)
        logits, past = self.step(
            np.asarray([history], dtype=np.int64),
            speaker_embedding=speaker_embedding,
        )
        raw_ids: list[int] = []

        for step_index in range(max_new_tokens):
            if logits_callback is not None:
                logits_callback(step_index, np.asarray(logits[0]).copy())
            token = sample(
                logits[0],
                history,
                token_map,
                settings,
                float(rng.random()),
                prompt_length=prompt_length,
            )
            raw_ids.append(token)
            if stop_ids is None:
                should_stop = is_eos_token(token, token_map)
            else:
                should_stop = token in stop_ids
            if should_stop:
                break
            history.append(token)
            logits, past = self.step(
                np.asarray([[token]], dtype=np.int64),
                past=past,
            )

        s3_ids = [
            token - token_map.audio_id_base
            for token in raw_ids
            if token_map.audio_id_base <= token < token_map.audio_id_end
        ]
        return GenerationResult(raw_ids=raw_ids, s3_ids=s3_ids)


class Kitten2OnnxDecoder:
    """Run the deterministic S3 flow and HiFT ONNX stages for zero-based S3 IDs."""

    def __init__(
        self,
        flow_dir: str | Path,
        mel2wav_dir: str | Path,
        *,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        self.flow_dir = Path(flow_dir)
        self.mel2wav_dir = Path(mel2wav_dir)
        self.flow_meta = json.loads((self.flow_dir / "flow.meta.json").read_text(encoding="utf-8"))
        self.mel2wav_meta = json.loads(
            (self.mel2wav_dir / "mel2wav.meta.json").read_text(encoding="utf-8")
        )
        if self.flow_meta.get("format") != "kitten2-onnx-flow-v1":
            raise ValueError("Unsupported S3 flow graph metadata")
        if self.mel2wav_meta.get("format") != "kitten2-onnx-mel2wav-v1":
            raise ValueError("Unsupported mel2wav graph metadata")
        tail = self.flow_meta["silence_tail"]
        self.token_mel_ratio = int(self.flow_meta["token_mel_ratio"])
        self.speech_vocab_size = int(self.flow_meta["speech_vocab_size"])
        self.silence_token_id = int(tail["token_id"])
        self.silence_tail_count = int(tail["count"])
        self.samples_per_mel_frame = int(self.mel2wav_meta["samples_per_mel_frame"])
        self.sample_rate = int(self.mel2wav_meta["sample_rate"])
        self.trim_fade_samples = int(self.mel2wav_meta["trim_fade_samples"])

        providers = providers or [self.flow_meta["provider"]]
        self.flow_session = ort.InferenceSession(
            str(self.flow_dir / self.flow_meta["graph"]), providers=providers
        )
        self.mel2wav_session = ort.InferenceSession(
            str(self.mel2wav_dir / self.mel2wav_meta["graph"]), providers=providers
        )
        self.flow_channels = int(self.flow_meta["input_shapes"]["flow_noise"][1])
        self.vocoder_harmonics = int(self.mel2wav_meta["input_shapes"]["phase_noise"][1])

    def decode(
        self,
        speech_tokens: list[int] | np.ndarray,
        conditioning: Mapping[str, np.ndarray],
        *,
        seed: int | None = None,
        flow_noise: np.ndarray | None = None,
        phase_noise: np.ndarray | None = None,
        sine_noise: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Decode S3 IDs and apply the upstream silence lookahead and trim fade."""
        tokens = clean_s3_tokens(speech_tokens, self.speech_vocab_size)
        if tokens.size == 0:
            return None
        tokens = append_s3_silence_tail(
            tokens, self.silence_token_id, self.silence_tail_count
        )[None, :]

        prompt_token = np.asarray(conditioning["conditioning_prompt_token"], dtype=np.int64)
        prompt_token_len = np.asarray(conditioning["conditioning_prompt_token_len"], dtype=np.int64)
        prompt_feat = np.asarray(conditioning["conditioning_prompt_feat"], dtype=np.float32)
        embedding = np.asarray(conditioning["conditioning_embedding"], dtype=np.float32)
        if prompt_token.shape[0] != 1 or prompt_token_len.shape != (1,):
            raise ValueError("conditioning prompt tokens must have batch size one")
        if int(prompt_token_len[0]) != prompt_token.shape[1]:
            raise ValueError("conditioning_prompt_token_len does not match its token array")

        rng = np.random.default_rng(seed)
        flow_shape = (
            1,
            self.flow_channels,
            self.token_mel_ratio * (int(prompt_token_len[0]) + tokens.shape[1]),
        )
        if flow_noise is None:
            flow_noise = rng.standard_normal(flow_shape, dtype=np.float32)
        else:
            flow_noise = np.asarray(flow_noise, dtype=np.float32)
        if flow_noise.shape != flow_shape:
            raise ValueError(f"flow_noise must have shape {flow_shape}, found {flow_noise.shape}")

        flow_inputs = {
            "speech_tokens": tokens,
            "speech_token_lengths": np.asarray([tokens.shape[1]], dtype=np.int64),
            "prompt_token": prompt_token,
            "prompt_token_len": prompt_token_len,
            "prompt_feat": prompt_feat,
            "embedding": embedding,
            "flow_noise": flow_noise,
        }
        mel = self.flow_session.run(["mel"], flow_inputs)[0]

        phase_shape = (1, self.vocoder_harmonics, 1)
        if phase_noise is None:
            phase_noise = rng.uniform(-np.pi, np.pi, size=phase_shape).astype(np.float32)
            phase_noise[:, 0, :] = 0
        else:
            phase_noise = np.asarray(phase_noise, dtype=np.float32)
        sine_shape = (
            1,
            self.vocoder_harmonics,
            mel.shape[2] * self.samples_per_mel_frame,
        )
        if sine_noise is None:
            sine_noise = rng.standard_normal(sine_shape, dtype=np.float32)
        else:
            sine_noise = np.asarray(sine_noise, dtype=np.float32)
        if phase_noise.shape != phase_shape:
            raise ValueError(f"phase_noise must have shape {phase_shape}, found {phase_noise.shape}")
        if sine_noise.shape != sine_shape:
            raise ValueError(f"sine_noise must have shape {sine_shape}, found {sine_noise.shape}")

        waveform = self.mel2wav_session.run(
            ["waveform"],
            {"mel": mel, "phase_noise": phase_noise, "sine_noise": sine_noise},
        )[0]
        n_trim = self.trim_fade_samples
        fade = np.zeros(2 * n_trim, dtype=np.float32)
        fade[n_trim:] = (np.cos(np.linspace(np.pi, 0, n_trim, dtype=np.float32)) + 1) / 2
        waveform[:, : fade.shape[0]] *= fade
        return waveform
