from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

_REQUIRED_MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "onnx": "onnx",
    "onnxscript": "onnxscript",
    "onnxruntime": "onnxruntime",
    "huggingface_hub": "huggingface-hub",
}


def _module_report() -> tuple[dict[str, str], dict[str, Any], list[str]]:
    versions: dict[str, str] = {}
    modules: dict[str, Any] = {}
    missing: list[str] = []

    for module_name, distribution_name in _REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError, RuntimeError) as exc:
            versions[module_name] = "unavailable"
            missing.append(f"{module_name} import failed: {exc}")
            continue

        modules[module_name] = module
        try:
            versions[module_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[module_name] = str(getattr(module, "__version__", "unknown"))

    return versions, modules, missing


def _model_report(model_dir: Path | None) -> tuple[dict[str, Any], list[str]]:
    if model_dir is None:
        return {
            "directory": None,
            "directory_present": None,
            "config_present": None,
            "lm_config_present": None,
            "plain_lm_present": None,
            "packed_lm_present": None,
        }, []

    paths = {
        "directory_present": model_dir.is_dir(),
        "config_present": (model_dir / "config.json").is_file(),
        "lm_config_present": (model_dir / "lm" / "config.json").is_file(),
        "plain_lm_present": (model_dir / "lm" / "model.safetensors").is_file(),
        "packed_lm_present": (model_dir / "lm" / "model-ternary.safetensors").is_file(),
    }
    missing: list[str] = []
    if not paths["directory_present"]:
        missing.append(f"model directory not found: {model_dir}")
    if not paths["config_present"]:
        missing.append(f"model config not found: {model_dir / 'config.json'}")
    if not paths["lm_config_present"]:
        missing.append(f"LM config not found: {model_dir / 'lm' / 'config.json'}")
    if not paths["plain_lm_present"]:
        missing.append(f"plain LM checkpoint not found: {model_dir / 'lm' / 'model.safetensors'}")

    return {"directory": str(model_dir), **paths}, missing


def collect_report(model_dir: Path | None) -> dict[str, Any]:
    versions, modules, missing = _module_report()
    model, model_missing = _model_report(model_dir)
    missing.extend(model_missing)

    torch = modules.get("torch")
    cuda_available: bool | None = None
    cuda_device: str | None = None
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device = str(torch.cuda.get_device_name(0))
        except (OSError, RuntimeError) as exc:
            missing.append(f"CUDA diagnostic failed: {exc}")

    runtime = modules.get("onnxruntime")
    providers = list(runtime.get_available_providers()) if runtime is not None else []

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": versions.get("torch", "unavailable"),
        "transformers": versions.get("transformers", "unavailable"),
        "onnx": versions.get("onnx", "unavailable"),
        "onnxscript": versions.get("onnxscript", "unavailable"),
        "onnxruntime": versions.get("onnxruntime", "unavailable"),
        "huggingface_hub": versions.get("huggingface_hub", "unavailable"),
        "providers": providers,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "model": model,
        "missing": missing,
        "ready": not missing,
    }


def _print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"python: {report['python']}")
    print(f"platform: {report['platform']}")
    for package in ("torch", "transformers", "onnx", "onnxscript", "onnxruntime", "huggingface_hub"):
        print(f"{package} version: {report[package]}")
    print(f"onnxruntime available providers: {', '.join(report['providers']) or 'none'}")
    cuda = report["cuda_available"]
    print(f"CUDA availability: {'unknown' if cuda is None else cuda}")
    print(f"CUDA device: {report['cuda_device'] or 'none'}")

    model = report["model"]
    if model["directory"] is None:
        print("model directory present?: not checked (no --model supplied)")
    else:
        print(f"model directory present?: {model['directory_present']}")
        print(f"config.json present?: {model['config_present']}")
        print(f"lm/config.json present?: {model['lm_config_present']}")
        print(f"plain lm/model.safetensors present?: {model['plain_lm_present']}")
        print(f"packed lm/model-ternary.safetensors present?: {model['packed_lm_present']}")

    if report["missing"]:
        print("Missing export prerequisites:")
        for item in report["missing"]:
            print(f"- {item}")
    else:
        print("Export prerequisites: ready")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report KittenTTS 2 ONNX export prerequisites.")
    parser.add_argument("--model", type=Path, help="Local KittenTTS 2 model directory.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args()

    report = collect_report(args.model)
    _print_report(report, args.json)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
