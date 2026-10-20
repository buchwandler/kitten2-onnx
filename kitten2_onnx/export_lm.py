"""Export the KittenTTS 2 Qwen3 speech LM to ONNX.

The default graph starts at ``inputs_embeds`` and keeps Kitten speaker injection
in the runtime. An optional ``input_ids`` graph performs tied token lookup and
position-zero speaker injection inside ONNX. ``--shared-vocabulary`` exports a
vocabulary-free transformer core with a separate, I/O-bound output projection.

The transformer graph serves prefill and decode:

    inputs_embeds [B, S, H]
    attention_mask [B, P + S]
    position_ids [B, S]
    past_key_i / past_value_i [B, KV, P, D]
        ->
    logits [B, V] or last_hidden_state [B, H]
    present_key_i / present_value_i     [B, KV, P + S, D]

Prefill uses P=0; cached decode normally uses S=1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ExportProfile:
    graph_dtype: torch.dtype
    graph_dtype_name: str
    io_dtype: str
    provider: str
    verify: bool


RUNTIME_PROFILES = {
    "dev-fp32": ExportProfile(torch.float32, "float32", "float32", "CPUExecutionProvider", True),
    "cuda-fp16": ExportProfile(torch.float16, "float16", "float16", "CUDAExecutionProvider", False),
    "cuda-bf16": ExportProfile(torch.bfloat16, "bfloat16", "bfloat16", "CUDAExecutionProvider", False),
}



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class KittenQwen3ExportWrapper(nn.Module):
    """Flatten Hugging Face's cache object into stable ONNX tensor I/O."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.transformer = model.model
        # Kitten/Qwen3 uses tied embeddings. Keeping a direct reference avoids
        # depending on the high-level CausalLM output object during export.
        self.lm_head_weight = model.get_output_embeddings().weight
        self.num_layers = int(model.config.num_hidden_layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        from transformers.cache_utils import DynamicCache

        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(self.num_layers))
        cache = DynamicCache.from_legacy_cache(legacy)

        out = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        # Full prefill logits are enormous and generation only consumes the last
        # row. Project only the last hidden state.
        last_hidden = out.last_hidden_state[:, -1:, :]
        logits = torch.nn.functional.linear(last_hidden, self.lm_head_weight)
        logits = logits[:, 0, :].float()

        present = out.past_key_values.to_legacy_cache()
        flat = [tensor for layer in present for tensor in layer]
        return (logits, *flat)



class KittenQwen3CoreExportWrapper(nn.Module):
    """Export the transformer body without a tied-vocabulary output head."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.transformer = model.model
        if (
            model.get_input_embeddings().weight.data_ptr()
            != model.get_output_embeddings().weight.data_ptr()
        ):
            raise ValueError("Shared vocabulary export requires tied input and output embeddings")
        self.num_layers = int(model.config.num_hidden_layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        from transformers.cache_utils import DynamicCache

        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(self.num_layers))
        cache = DynamicCache.from_legacy_cache(legacy)
        out = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        last_hidden = out.last_hidden_state[:, -1, :]
        present = out.past_key_values.to_legacy_cache()
        flat = [tensor for layer in present for tensor in layer]
        return (last_hidden, *flat)


class KittenQwen3InputIdsExportWrapper(nn.Module):
    """Run token lookup and Kitten speaker injection inside the ONNX graph."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.transformer = model.model
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            raise ValueError("Qwen3 input and output embeddings are not tied")
        self.embedding_weight = input_weight
        self.num_layers = int(model.config.num_hidden_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        speaker_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        from transformers.cache_utils import DynamicCache

        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(self.num_layers))
        cache = DynamicCache.from_legacy_cache(legacy)
        token_embeddings = torch.nn.functional.embedding(input_ids, self.embedding_weight)
        speaker_mask = position_ids.eq(0).unsqueeze(-1)
        inputs_embeds = torch.where(
            speaker_mask,
            speaker_hidden.to(token_embeddings.dtype).unsqueeze(1),
            token_embeddings,
        )
        out = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        last_hidden = out.last_hidden_state[:, -1:, :]
        logits = torch.matmul(last_hidden, self.embedding_weight.transpose(0, 1))
        logits = logits[:, 0, :].float()
        present = out.past_key_values.to_legacy_cache()
        flat = [tensor for layer in present for tensor in layer]
        return (logits, *flat)

def _io_names(num_layers: int) -> tuple[list[str], list[str]]:
    past = [name for i in range(num_layers) for name in (f"past_key_{i}", f"past_value_{i}")]
    present = [
        name for i in range(num_layers) for name in (f"present_key_{i}", f"present_value_{i}")
    ]
    return past, present



def _export_shared_vocabulary_head(
    output_path: Path, *, vocab_size: int, hidden_size: int, opset: int
) -> Path:
    import onnx
    from onnx import TensorProto, helper

    hidden = helper.make_tensor_value_info(
        "hidden_states", TensorProto.FLOAT, ["batch", hidden_size]
    )
    vocabulary = helper.make_tensor_value_info(
        "vocabulary_weight", TensorProto.FLOAT, [vocab_size, hidden_size]
    )
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", vocab_size])
    graph = helper.make_graph(
        [
            helper.make_node(
                "Transpose", ["vocabulary_weight"], ["vocabulary_transposed"], perm=[1, 0]
            ),
            helper.make_node("MatMul", ["hidden_states", "vocabulary_transposed"], ["logits"]),
        ],
        "kitten_shared_vocabulary_output_head",
        [hidden, vocabulary],
        [logits],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(output_path))
    return output_path


def _dummy_inputs(config, dtype: torch.dtype, seq: int = 4, past_len: int = 3):
    hidden = int(config.hidden_size)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    layers = int(config.num_hidden_layers)

    embeds = torch.randn(1, seq, hidden, dtype=dtype)
    mask = torch.ones(1, past_len + seq, dtype=torch.int64)
    pos = torch.arange(past_len, past_len + seq, dtype=torch.int64)[None, :]
    past = [torch.zeros(1, kv_heads, past_len, head_dim, dtype=dtype) for _ in range(2 * layers)]
    return embeds, mask, pos, past



def _dummy_input_ids(config, dtype: torch.dtype, seq: int = 4, past_len: int = 3):
    hidden = int(config.hidden_size)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    layers = int(config.num_hidden_layers)

    input_ids = torch.randint(int(config.vocab_size), (1, seq), dtype=torch.int64)
    speaker_hidden = torch.randn(1, hidden, dtype=dtype)
    mask = torch.ones(1, past_len + seq, dtype=torch.int64)
    pos = torch.arange(past_len, past_len + seq, dtype=torch.int64)[None, :]
    past = [torch.zeros(1, kv_heads, past_len, head_dim, dtype=dtype) for _ in range(2 * layers)]
    return input_ids, speaker_hidden, mask, pos, past


def load_hf_lm(
    repo_dir: Path, dtype: torch.dtype, *, weight_variant: str = "full"
):
    """Load a full or upstream ternary Kitten Qwen3 checkpoint for ONNX export."""
    from transformers import AutoConfig, AutoModelForCausalLM

    repo_dir = Path(repo_dir)
    lm_dir = repo_dir / "lm"
    config_path = lm_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Kitten LM config not found: {config_path}")
    config_values = json.loads(config_path.read_text(encoding="utf-8"))
    if config_values.get("model_type") != "qwen3":
        raise ValueError(f"Expected a Qwen3 checkpoint in {lm_dir}")
    if weight_variant == "plain":
        weight_variant = "full"
    if weight_variant not in {"full", "packed", "emb4"}:
        raise ValueError("weight_variant must be 'full', 'packed', or 'emb4'")

    if weight_variant == "full":
        checkpoint_path = lm_dir / "model.safetensors"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Plain Kitten LM checkpoint not found: {checkpoint_path}")
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            str(lm_dir),
            dtype=dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
            output_loading_info=True,
        )
        missing = set(loading_info["missing_keys"])
        unexpected = {
            name for name in loading_info["unexpected_keys"] if not name.startswith("spk_proj.")
        }
    else:
        from accelerate import init_empty_weights

        from kitten2_onnx.weights import load_lm_state_dict, resolve_weight_path

        checkpoint_path = resolve_weight_path(repo_dir, weight_variant)
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"No {weight_variant!r} LM checkpoint found in {repo_dir}")
        config = AutoConfig.from_pretrained(str(lm_dir), local_files_only=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        state = {
            name: tensor.to(dtype=dtype)
            for name, tensor in load_lm_state_dict(repo_dir, variant=weight_variant).items()
            if not name.startswith("spk_proj.")
        }
        loading_info = model.load_state_dict(state, strict=False, assign=True)
        missing = set(loading_info.missing_keys)
        unexpected = set(loading_info.unexpected_keys)
        del state

    if model.config.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing:
        raise ValueError(f"Missing transformer checkpoint weights: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected checkpoint weights: {sorted(unexpected)}")

    model.tie_weights()
    if model.config.tie_word_embeddings:
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            raise ValueError("Qwen3 input and output embeddings are not tied")
    model.eval()
    return model

def save_embedding_table(
    model: nn.Module, output_dir: Path, *, source_dtype: str
) -> tuple[Path, str]:
    """Store the selected token table compactly without expanding it at runtime."""
    table = model.get_input_embeddings().weight.detach().cpu()
    source_dtype = source_dtype.removeprefix("torch.").lower()
    if table.dtype == torch.bfloat16 or (
        table.dtype == torch.float32 and source_dtype == "bfloat16"
    ):
        packed = table.to(torch.bfloat16).contiguous()
        array = packed.view(torch.uint16).numpy()
        storage_dtype = "bfloat16-u16"
        path = output_dir / "embed_tokens.bf16.npy"
    else:
        array = table.contiguous().numpy()
        storage_dtype = str(array.dtype)
        path = output_dir / "embed_tokens.npy"
    np.save(path, array, allow_pickle=False)
    return path, storage_dtype

def _share_tied_vocabulary_weight(graph_path: Path, *, vocab_size: int, hidden_size: int) -> bool:
    """Replace an exported transposed tied-head copy with a graph transpose."""
    import onnx

    graph_path = Path(graph_path)
    model = onnx.load(str(graph_path), load_external_data=False)

    def consumers(value_name: str) -> set[str]:
        return {node.op_type for node in model.graph.node if value_name in node.input}

    source_candidates = [
        tensor
        for tensor in model.graph.initializer
        if tuple(tensor.dims) == (vocab_size, hidden_size) and "Gather" in consumers(tensor.name)
    ]
    if len(source_candidates) != 1:
        raise ValueError("Could not identify one tied vocabulary embedding initializer")
    source = source_candidates[0]

    transposed_candidates = [
        tensor
        for tensor in model.graph.initializer
        if tuple(tensor.dims) == (hidden_size, vocab_size) and "MatMul" in consumers(tensor.name)
    ]
    if not transposed_candidates:
        if any(
            node.op_type == "Transpose"
            and node.input[0] == source.name
            and next((list(attr.ints) for attr in node.attribute if attr.name == "perm"), [])
            == [1, 0]
            for node in model.graph.node
        ):
            return False
        raise ValueError("Could not identify the tied vocabulary output initializer")
    if len(transposed_candidates) != 1:
        raise ValueError("Found multiple tied vocabulary output initializers")
    transposed = transposed_candidates[0]
    removed_external = {entry.key: entry.value for entry in transposed.external_data}
    removed_location = removed_external.get("location")
    removed_offset = int(removed_external["offset"]) if "offset" in removed_external else None

    from onnx import helper

    transpose_node = helper.make_node(
        "Transpose",
        [source.name],
        [transposed.name],
        perm=[1, 0],
        name="kitten_tied_embedding_transpose",
    )
    nodes = list(model.graph.node)
    nodes.insert(0, transpose_node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.remove(transposed)

    external_records: dict[str, list[tuple[object, int, int | None]]] = {}
    for tensor in model.graph.initializer:
        entries = {entry.key: entry.value for entry in tensor.external_data}
        location = entries.get("location")
        if location is not None:
            if Path(location).is_absolute():
                raise ValueError("ONNX external-data locations must be relative")
            offset = int(entries.get("offset", "0"))
            length = int(entries["length"]) if "length" in entries else None
            external_records.setdefault(location, []).append((tensor, offset, length))

    replacements: list[tuple[Path, Path, Path]] = []
    truncations: list[tuple[Path, int]] = []
    for location, records in external_records.items():
        source_path = graph_path.parent / location
        source_size = source_path.stat().st_size
        records.sort(key=lambda item: item[1])
        lengths = []
        for index, (_, offset, length) in enumerate(records):
            if length is not None:
                lengths.append(length)
                continue
            end = source_size
            if index + 1 < len(records):
                end = min(end, records[index + 1][1])
            if (
                location == removed_location
                and removed_offset is not None
                and removed_offset > offset
            ):
                end = min(end, removed_offset)
            lengths.append(end - offset)
        if (
            location == removed_location
            and removed_offset is not None
            and all(
                offset + length <= removed_offset
                for (_, offset, _), length in zip(records, lengths, strict=True)
            )
        ):
            truncations.append((source_path, removed_offset))
            continue
        target_path = source_path.with_name(source_path.name + ".shared")
        target_location = Path(os.path.relpath(target_path, graph_path.parent)).as_posix()
        temporary_path = target_path.with_name(target_path.name + ".tmp")
        with source_path.open("rb") as source_file, temporary_path.open("wb") as target_file:
            for (tensor, old_offset, _), length in zip(records, lengths, strict=True):
                aligned_offset = (target_file.tell() + 63) // 64 * 64
                target_file.write(bytes(aligned_offset - target_file.tell()))
                new_offset = target_file.tell()
                source_file.seek(old_offset)
                remaining = length
                while remaining:
                    chunk = source_file.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("ONNX external data ended before the initializer length")
                    target_file.write(chunk)
                    remaining -= len(chunk)
                for entry in tensor.external_data:
                    if entry.key == "location":
                        entry.value = target_location
                    elif entry.key == "offset":
                        entry.value = str(new_offset)
                    elif entry.key == "length":
                        entry.value = str(length)
                if not any(entry.key == "offset" for entry in tensor.external_data):
                    tensor.external_data.add(key="offset", value=str(new_offset))
                if not any(entry.key == "length" for entry in tensor.external_data):
                    tensor.external_data.add(key="length", value=str(length))
        replacements.append((source_path, temporary_path, target_path))

    temporary_graph = graph_path.with_name(graph_path.name + ".tmp")
    onnx.save_model(model, str(temporary_graph))
    for _, temporary_path, target_path in replacements:
        os.replace(temporary_path, target_path)
    onnx.checker.check_model(str(temporary_graph))
    for source_path, offset in truncations:
        with source_path.open("r+b") as data_file:
            data_file.truncate(offset)
    os.replace(temporary_graph, graph_path)
    for source_path, _, target_path in replacements:
        if source_path != target_path:
            source_path.unlink(missing_ok=True)
    onnx.checker.check_model(str(graph_path))
    return True


def _export_with_diagnostics(
    model: nn.Module,
    args: tuple,
    output_path: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    verify: bool = True,
    **export_kwargs,
) -> None:
    artifacts_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else (Path(__file__).resolve().parents[1] / "build" / "debug" / "lm-export")
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    try:
        torch.onnx.export(
            model,
            args,
            str(output_path),
            report=True,
            verify=verify,
            artifacts_dir=str(artifacts_dir),
            **export_kwargs,
        )
    except Exception as error:
        failure_path = artifacts_dir / "export-failure.txt"
        failure_path.write_text(
            f"Output: {output_path}\n{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        raise


def export_lm(
    repo_dir: str | Path,
    output_dir: str | Path,
    *,
    profile_name: str = "dev-fp32",
    input_mode: str = "inputs_embeds",
    shared_vocabulary: bool = False,
    weight_variant: str = "full",
    opset: int = 18,
    diagnostics_dir: str | Path | None = None,
) -> Path:
    try:
        profile = RUNTIME_PROFILES[profile_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown export profile {profile_name!r}. Choose from {sorted(RUNTIME_PROFILES)}"
        ) from error

    if input_mode not in {"inputs_embeds", "input_ids"}:
        raise ValueError("input_mode must be 'inputs_embeds' or 'input_ids'")
    if shared_vocabulary and input_mode != "inputs_embeds":
        raise ValueError("shared_vocabulary requires input_mode='inputs_embeds'")

    if weight_variant == "plain":
        weight_variant = "full"
    if weight_variant not in {"full", "packed", "emb4"}:
        raise ValueError("weight_variant must be 'full', 'packed', or 'emb4'")
    if weight_variant != "full" and not shared_vocabulary:
        raise ValueError("packed and emb4 variants require shared_vocabulary=True")
    if weight_variant != "full" and profile_name != "dev-fp32":
        raise ValueError("packed and emb4 variants require the dev-fp32 CPU export profile")
    repo_dir = Path(repo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = repo_dir / "lm"
    if weight_variant == "full":
        checkpoint_path = lm_dir / "model.safetensors"
    else:
        from kitten2_onnx.weights import resolve_weight_path

        checkpoint_path = resolve_weight_path(repo_dir, weight_variant)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No {weight_variant!r} LM checkpoint is declared in {repo_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LM checkpoint not found: {checkpoint_path}")
    source_config = json.loads((lm_dir / "config.json").read_text(encoding="utf-8"))
    source_dtype = str(source_config.get("dtype", source_config.get("torch_dtype", "unknown")))
    revision_path = repo_dir / "model_revision.json"
    revision = json.loads(revision_path.read_text(encoding="utf-8")) if revision_path.is_file() else {}

    model = load_hf_lm(repo_dir, profile.graph_dtype, weight_variant=weight_variant)
    config = model.config
    past_names, present_names = _io_names(int(config.num_hidden_layers))
    if input_mode == "inputs_embeds":
        wrapper = (
            KittenQwen3CoreExportWrapper(model)
            if shared_vocabulary
            else KittenQwen3ExportWrapper(model)
        ).eval()
        input_names = ["inputs_embeds", "attention_mask", "position_ids", *past_names]
        dummy = _dummy_inputs(config, dtype=profile.graph_dtype, seq=4, past_len=3)
    else:
        wrapper = KittenQwen3InputIdsExportWrapper(model).eval()
        input_names = ["input_ids", "speaker_hidden", "attention_mask", "position_ids", *past_names]
        dummy = _dummy_input_ids(config, dtype=profile.graph_dtype, seq=4, past_len=3)
    output_names = ["last_hidden_state" if shared_vocabulary else "logits", *present_names]

    batch_dim = torch.export.Dim("batch", min=1)
    seq_dim = torch.export.Dim("seq", min=1)
    total_dim = torch.export.Dim("total", min=1)
    past_dim = torch.export.Dim("past", min=0)
    if input_mode == "inputs_embeds":
        dynamic_shapes = {
            "inputs_embeds": {0: batch_dim, 1: seq_dim},
            "attention_mask": {0: batch_dim, 1: total_dim},
            "position_ids": {0: batch_dim, 1: seq_dim},
            "past": [{0: batch_dim, 2: past_dim} for _ in range(len(past_names))],
        }
    else:
        dynamic_shapes = {
            "input_ids": {0: batch_dim, 1: seq_dim},
            "speaker_hidden": {0: batch_dim},
            "attention_mask": {0: batch_dim, 1: total_dim},
            "position_ids": {0: batch_dim, 1: seq_dim},
            "past": [{0: batch_dim, 2: past_dim} for _ in range(len(past_names))],
        }
    output_path = output_dir / "lm.onnx"
    _export_with_diagnostics(
        wrapper,
        dummy,
        output_path,
        artifacts_dir=diagnostics_dir,
        verify=profile.verify,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        dynamo=True,
        external_data=True,
    )
    shared_metadata = {}
    if shared_vocabulary:
        output_head_path = _export_shared_vocabulary_head(
            output_dir / "lm_output.onnx",
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            opset=opset,
        )
        shared_metadata = {
            "vocabulary_mode": "shared",
            "output_head_file": output_head_path.name,
            "output_head_input_names": ["hidden_states", "vocabulary_weight"],
            "output_head_output_names": ["logits"],
            "output_head_input_shapes": {
                "hidden_states": [None, int(config.hidden_size)],
                "vocabulary_weight": [int(config.vocab_size), int(config.hidden_size)],
            },
            "output_head_output_shape": [None, int(config.vocab_size)],
        }
    if input_mode == "input_ids":
        _share_tied_vocabulary_weight(
            output_path,
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
        )

    embedding_metadata = {}
    if input_mode == "inputs_embeds":
        if weight_variant == "emb4":
            from kitten2_onnx.tl2_emb4 import (
                EMB4_RECONSTRUCTION_VERSION,
                write_emb4_embedding_asset,
            )

            embedding_path = write_emb4_embedding_asset(
                checkpoint_path, output_dir / "vocabulary.emb4.safetensors"
            )
            embedding_storage_dtype = "kitten-emb4"
        else:
            embedding_path, embedding_storage_dtype = save_embedding_table(
                model, output_dir, source_dtype=source_dtype
            )
        embedding_metadata = {
            "embedding_file": embedding_path.name,
            "embedding_storage_dtype": embedding_storage_dtype,
            "embedding_runtime_dtype": profile.io_dtype,
        }
    variant_metadata = {}
    if weight_variant != "full":
        import onnxruntime as ort

        variant_metadata = {
            "source_weight_variant": weight_variant,
            "source_weight_file": checkpoint_path.resolve().relative_to(repo_dir.resolve()).as_posix(),
            "source_weight_sha256": _sha256(checkpoint_path),
            "embedding_reconstruction_version": (
                EMB4_RECONSTRUCTION_VERSION if weight_variant == "emb4" else None
            ),
            "onnxruntime_min_tested_version": ort.__version__,
        }
    meta = {
        "format": "kitten2-onnx-lm-v1",
        "architecture": (getattr(config, "architectures", None) or [config.model_type])[0],
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "vocab_size": int(config.vocab_size),
        "source_dtype": source_dtype,
        "graph_dtype": profile.graph_dtype_name,
        "io_dtype": profile.io_dtype,
        "logits_dtype": "float32",
        "runtime_profile": profile_name,
        "runtime_provider": profile.provider,
        "verify_on_export": profile.verify,
        "source_model_path": str(repo_dir.resolve()),
        "source_model_revision": revision.get("source_model_revision"),
        "loading_path": str(checkpoint_path.resolve()),
        "attention_implementation": "eager",
        "external_data": True,
        "opset": int(opset),
        "input_mode": input_mode,
        **embedding_metadata,
        **variant_metadata,
        **shared_metadata,
        "input_names": input_names,
        "output_names": output_names,
    }
    (output_dir / "lm.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_dir", type=Path, help="Local KittenML/kitten-tts-2 directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(RUNTIME_PROFILES), default="dev-fp32")
    parser.add_argument(
        "--input-mode", choices=("inputs_embeds", "input_ids"), default="inputs_embeds"
    )
    parser.add_argument(
        "--shared-vocabulary",
        action="store_true",
        help="Export a vocabulary-free transformer core and separate output-head graph",
    )
    parser.add_argument("--weights", choices=("full", "packed", "emb4"), default="full")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)

    path = export_lm(
        args.repo_dir,
        args.output,
        profile_name=args.profile,
        input_mode=args.input_mode,
        shared_vocabulary=args.shared_vocabulary,
        weight_variant=args.weights,
        opset=args.opset,
    )
    print(path)


if __name__ == "__main__":
    main()
