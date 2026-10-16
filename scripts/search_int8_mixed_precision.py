"""Inventory and orchestrate bounded mixed-precision INT8 Kitten2 experiments."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from scripts.optimize_bundle import optimize_bundle


def _bundle_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes its root: {relative}")
    return path


def inventory_matmuls(bundle_dir: str | Path) -> dict[str, Any]:
    import onnx

    from kitten2_onnx.quantize_int8_streaming import _matmul_targets

    bundle_root = Path(bundle_dir).resolve()
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    lm_model_path = _bundle_file(bundle_root, bundle["lm"]["model"])
    lm_metadata_path = _bundle_file(bundle_root, bundle["lm"]["metadata"])
    metadata = json.loads(lm_metadata_path.read_text(encoding="utf-8"))
    layer_count = int(metadata["num_hidden_layers"])
    vocab_size = int(metadata["vocab_size"])
    if layer_count != 28:
        raise ValueError(f"Expected the Kitten2 28-layer LM, found {layer_count} layers")

    model = onnx.load(str(lm_model_path), load_external_data=False)
    model = onnx.shape_inference.infer_shapes(model)
    targets, excluded, matmul_count = _matmul_targets(
        model.graph,
        lm_model_path,
        exclude_nodes=(),
        exclude_initializers=(),
    )
    expected_target_count = layer_count * 7 + 1
    if len(targets) != expected_target_count:
        raise ValueError(
            f"Expected {layer_count} x 7 transformer MatMuls plus one LM head, "
            f"found {len(targets)} supported targets"
        )

    expected_node_names = ["node_linear", *(f"node_linear_{index}" for index in range(1, expected_target_count))]
    actual_node_names = [target.name for target in targets]
    if actual_node_names != expected_node_names:
        raise ValueError("Supported MatMul graph order or node names differ from the verified Kitten2 layout")

    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    logits = next((value for value in model.graph.output if value.name == "logits"), None)
    if logits is None:
        raise ValueError("LM graph has no logits output")
    output_dims = logits.type.tensor_type.shape.dim
    if not output_dims or output_dims[-1].dim_value != vocab_size:
        raise ValueError("LM graph logits dimension differs from LM metadata vocab_size")

    head_targets = [
        target
        for target in targets
        if len(initializers[target.weight_name].dims) == 2
        and initializers[target.weight_name].dims[-1] == vocab_size
    ]
    if len(head_targets) != 1 or head_targets[0] is not targets[-1]:
        raise ValueError("Could not verify a unique final LM-head MatMul from graph dimensions and order")
    head_target = head_targets[0]

    target_records: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    body_targets = targets[:-1]
    for block_index in range(layer_count):
        block_targets = body_targets[block_index * 7 : (block_index + 1) * 7]
        blocks.append(
            {
                "index": block_index,
                "targets": [
                    {
                        "node": target.name,
                        "initializer": target.weight_name,
                        "graph_index": target.index,
                        "weight_shape": list(initializers[target.weight_name].dims),
                    }
                    for target in block_targets
                ],
            }
        )

    for block in blocks:
        for projection_index, target in enumerate(block["targets"]):
            target_records.append(
                {
                    **target,
                    "group": "transformer_block",
                    "block_index": block["index"],
                    "projection_index": projection_index,
                }
            )
    target_records.append(
        {
            "node": head_target.name,
            "initializer": head_target.weight_name,
            "graph_index": head_target.index,
            "weight_shape": list(initializers[head_target.weight_name].dims),
            "group": "lm_head",
            "block_index": None,
            "projection_index": None,
        }
    )
    return {
        "format": "kitten2-onnx-int8-matmul-inventory-v1",
        "source_bundle": str(bundle_root),
        "layer_count": layer_count,
        "projections_per_block": 7,
        "vocab_size": vocab_size,
        "matmul_count": matmul_count,
        "supported_matmul_count": len(targets),
        "unsupported_matmuls": excluded,
        "blocks": blocks,
        "lm_head": target_records[-1],
        "targets": target_records,
    }


def exclusions_for_candidate(
    inventory: dict[str, Any],
    *,
    preserve_last_blocks: int,
) -> tuple[list[str], list[str]]:
    layer_count = int(inventory["layer_count"])
    if not 0 <= preserve_last_blocks <= layer_count:
        raise ValueError(f"preserve_last_blocks must be between 0 and {layer_count}")
    first_preserved_block = layer_count - preserve_last_blocks
    excluded = [
        target
        for target in inventory["targets"]
        if target["group"] == "lm_head"
        or (
            target["group"] == "transformer_block"
            and target["block_index"] >= first_preserved_block
        )
    ]
    nodes = list(dict.fromkeys(target["node"] for target in excluded))
    initializers = list(dict.fromkeys(target["initializer"] for target in excluded))
    return nodes, initializers


def execute_candidate(
    input_bundle: str | Path,
    output_bundle: str | Path,
    report_dir: str | Path,
    *,
    exclude_matmul_nodes: list[str],
    exclude_matmul_initializers: list[str],
    chunk_rows: int = 64,
) -> dict[str, Any]:
    input_root = Path(input_bundle).resolve()
    output_root = Path(output_bundle).resolve()
    report_root = Path(report_dir).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    if report_root.exists():
        raise FileExistsError(report_root)
    if (
        output_root == input_root
        or output_root.is_relative_to(input_root)
        or input_root.is_relative_to(output_root)
        or output_root == report_root
        or output_root.is_relative_to(report_root)
        or report_root.is_relative_to(output_root)
        or report_root == input_root
        or report_root.is_relative_to(input_root)
        or input_root.is_relative_to(report_root)
    ):
        raise ValueError("Input bundle, output bundle, and report directory must not overlap")
    return optimize_bundle(
        input_root,
        output_root,
        lm_method="int8-dynamic-chunked",
        validate=True,
        report_dir=report_root,
        chunk_rows=chunk_rows,
        exclude_matmul_nodes=exclude_matmul_nodes,
        exclude_matmul_initializers=exclude_matmul_initializers,
    )


CANDIDATE_BLOCK_COUNTS = (4, 8, 14, 20)


class _PeakRssSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.peak_rss_kib = self._read_rss_kib()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    @staticmethod
    def _read_rss_kib() -> int | None:
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except OSError:
            return None
        return None

    def _sample(self) -> None:
        while not self._stop.wait(0.05):
            current = self._read_rss_kib()
            if current is not None:
                self.peak_rss_kib = max(self.peak_rss_kib or current, current)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()
        current = self._read_rss_kib()
        if current is not None:
            self.peak_rss_kib = max(self.peak_rss_kib or current, current)


def candidate_matrix(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for preserve_last_blocks in CANDIDATE_BLOCK_COUNTS:
        nodes, initializers = exclusions_for_candidate(
            inventory,
            preserve_last_blocks=preserve_last_blocks,
        )
        if inventory["lm_head"]["node"] not in nodes:
            raise ValueError("Every mixed-precision candidate must keep the LM head in FP32")
        candidates.append(
            {
                "name": f"lm-head-fp32-last-{preserve_last_blocks}-blocks-fp32",
                "preserve_last_blocks": preserve_last_blocks,
                "exclude_matmul_nodes": nodes,
                "exclude_matmul_initializers": initializers,
            }
        )
    return candidates


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _diagnostic_corpus(source: Path, destination: Path) -> Path:
    corpus = json.loads(source.read_text(encoding="utf-8"))
    cases = [case for case in corpus["cases"] if case["name"] == "very_short"]
    if not cases:
        raise ValueError(f"Corpus has no very_short diagnostic case: {source}")
    corpus["cases"] = cases
    _write_json(destination, corpus)
    return destination


def _quality_summary(report: dict[str, Any]) -> dict[str, Any]:
    case = next(case for case in report["cases"] if case["name"] == "very_short")
    return {
        "review_status": report["quality_review"]["status"],
        "first_divergence": report["first_divergence"],
        "token_divergence_step": case["token_divergence_step"],
        "logits_max_abs_error": case["logits_max_abs_error"],
        "logits_rmse": case["logits_rmse"],
        "logits_compared_steps": case["logits_compared_steps"],
        "waveform_finite": case["waveform_finite"],
        "audio": case["quality_audio"],
    }


def run_search(
    input_bundle: str | Path,
    output_root: str | Path,
    reports_root: str | Path,
    *,
    model_dir: str | Path,
    corpus_path: str | Path,
    full_corpus: bool = False,
    chunk_rows: int = 64,
) -> dict[str, Any]:
    from scripts.validate_e2e_corpus import validate_corpus

    input_root = Path(input_bundle).resolve()
    output_root = Path(output_root).resolve()
    reports_root = Path(reports_root).resolve()
    model_root = Path(model_dir).resolve()
    corpus_file = Path(corpus_path).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if reports_root.exists():
        raise FileExistsError(reports_root)
    if (
        output_root == input_root
        or output_root.is_relative_to(input_root)
        or input_root.is_relative_to(output_root)
        or reports_root == input_root
        or reports_root.is_relative_to(input_root)
        or input_root.is_relative_to(reports_root)
        or output_root == reports_root
        or output_root.is_relative_to(reports_root)
        or reports_root.is_relative_to(output_root)
    ):
        raise ValueError("Input, output, and report roots must not overlap")

    inventory = inventory_matmuls(input_root)
    candidates = candidate_matrix(inventory)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    reports_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    reports_root.mkdir()
    _write_json(output_root / "inventory.json", inventory)
    results: list[dict[str, Any]] = []
    results_path = output_root / "results.json"

    for candidate in candidates:
        candidate_output = output_root / candidate["name"]
        candidate_reports = reports_root / candidate["name"]
        started = time.perf_counter()
        with _PeakRssSampler() as rss:
            try:
                optimization = execute_candidate(
                    input_root,
                    candidate_output,
                    candidate_reports,
                    exclude_matmul_nodes=candidate["exclude_matmul_nodes"],
                    exclude_matmul_initializers=candidate["exclude_matmul_initializers"],
                    chunk_rows=chunk_rows,
                )
                candidate_reports.mkdir(parents=True, exist_ok=True)
                diagnostic_corpus = _diagnostic_corpus(
                    corpus_file,
                    candidate_reports / "very-short-corpus.json",
                )
                quality_report = validate_corpus(
                    model_root,
                    candidate_output,
                    diagnostic_corpus,
                    candidate_reports / "quality-report.json",
                    providers=["CPUExecutionProvider"],
                    quality_report=True,
                    reference_bundle_dir=input_root,
                )
                row: dict[str, Any] = {
                    **candidate,
                    "status": "diagnostic_complete",
                    "bundle_bytes": optimization["output_bundle_bytes"],
                    "transformed_matmul_count": optimization["quantization"].get(
                        "transformed_matmul_count"
                    ),
                    "quality": _quality_summary(quality_report),
                    "quality_report": str(candidate_reports / "quality-report.json"),
                }
                if full_corpus:
                    full_report_path = candidate_reports / "quality-full-corpus.json"
                    full_report = validate_corpus(
                        model_root,
                        candidate_output,
                        corpus_file,
                        full_report_path,
                        providers=["CPUExecutionProvider"],
                        quality_report=True,
                        reference_bundle_dir=input_root,
                    )
                    row["full_corpus_quality"] = {
                        "report": str(full_report_path),
                        "first_divergence": full_report["first_divergence"],
                        "review_status": full_report["quality_review"]["status"],
                    }
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                row = {
                    **candidate,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
        row["peak_rss_kib"] = rss.peak_rss_kib
        row["elapsed_seconds"] = time.perf_counter() - started
        results.append(row)
        _write_json(
            results_path,
            {
                "format": "kitten2-onnx-int8-mixed-search-v1",
                "source_bundle": str(input_root),
                "inventory": "inventory.json",
                "candidates": results,
                "quality_decision": "diagnostic_only_requires_manual_review",
            },
        )
    return json.loads(results_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("tests/fixtures/e2e_corpus.json"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reports-root", type=Path)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=64)
    args = parser.parse_args()
    if args.inventory_only:
        print(json.dumps(inventory_matmuls(args.bundle), indent=2, sort_keys=True))
        return
    if not args.model_dir or not args.output_root or not args.reports_root:
        parser.error("--model-dir, --output-root, and --reports-root are required for a search")
    result = run_search(
        args.bundle,
        args.output_root,
        args.reports_root,
        model_dir=args.model_dir,
        corpus_path=args.corpus,
        full_corpus=args.full_corpus,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
