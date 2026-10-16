"""Load the pinned upstream S3Gen model for export and parity capture."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path

from kitten2_onnx.reference import UPSTREAM_CODE_REVISION

S3GEN_REPOSITORY = "ResembleAI/chatterbox-turbo"
S3GEN_REVISION = "749d1c1a46eb10492095d68fbcf55691ccf137cd"
S3GEN_FILENAME = "s3gen_meanflow.safetensors"


def load_s3gen_class():
    try:
        distribution = importlib.metadata.distribution("kittenml")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            'Install the pinned decoder reference dependencies with '
            '`uv pip install -e ".[decoder-reference]".'
        ) from error

    direct_url = distribution.read_text("direct_url.json")
    installed_revision = None
    if direct_url:
        installed_revision = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    if installed_revision != UPSTREAM_CODE_REVISION:
        raise RuntimeError(
            f"Expected KittenTTS source revision {UPSTREAM_CODE_REVISION}, "
            f"found {installed_revision!r}"
        )

    source_path = Path(distribution.locate_file("kittenml/kittentts2/s3gen.py"))
    spec = importlib.util.spec_from_file_location("kitten2_onnx_upstream_s3gen", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned S3Gen source from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.S3Gen


def load_s3gen_model():
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = Path(
        hf_hub_download(
            repo_id=S3GEN_REPOSITORY,
            filename=S3GEN_FILENAME,
            revision=S3GEN_REVISION,
        )
    )
    model = load_s3gen_class()(meanflow=True)
    model.load_state_dict(load_file(weights_path), strict=True)
    model.to("cpu").eval()
    if model.dtype != torch.float32:
        raise ValueError(f"Expected FP32 S3Gen reference weights, found {model.dtype}")
    return model, weights_path
