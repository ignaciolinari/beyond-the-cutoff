#!/usr/bin/env python3
"""Run consolidated evaluation metrics (faithfulness, citations, retrieval)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from beyond_the_cutoff.config import ProjectConfig, load_config
from beyond_the_cutoff.evaluation.harness import compute_harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation harness with generation and retrieval metrics"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Project config path")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset JSONL path (defaults to config.evaluation.offline_dataset_path)",
    )
    parser.add_argument("--predictions", required=True, help="Predictions JSONL file")
    parser.add_argument(
        "--prediction-id-field",
        default="task_id",
        help="Field name carrying task identifiers in predictions",
    )
    parser.add_argument(
        "--prediction-field",
        default="model_answer",
        help="Field name containing model answers in predictions",
    )
    parser.add_argument(
        "--bert-lang",
        default="en",
        help="Language code to use for BERTScore computations",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Override retrieval index path (directory or index.faiss file)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Override retrieval mapping TSV path",
    )
    parser.add_argument(
        "--retrieval-topk",
        default="1,3,5,10",
        help="Comma separated list of K values for Hit@K (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of dataset examples",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to write aggregated summary",
    )
    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional JSONL file capturing per-example metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    dataset_path = _resolve_dataset_path(cfg, args.dataset)
    dataset_records = list(_iter_jsonl(dataset_path))
    if args.limit is not None:
        dataset_records = dataset_records[: max(args.limit, 0)]

    if not dataset_records:
        raise SystemExit("Dataset is empty; nothing to evaluate")

    predictions_path = Path(args.predictions).resolve()
    predictions = _load_predictions(
        predictions_path,
        id_field=args.prediction_id_field,
        prediction_field=args.prediction_field,
    )

    topk_values = _parse_topk(args.retrieval_topk)
    summary, details_rows = compute_harness(
        dataset_records=dataset_records,
        project_config=cfg,
        predictions=predictions,
        bert_lang=args.bert_lang,
        topk_values=topk_values,
        index_override=Path(args.index).resolve() if args.index else None,
        mapping_override=Path(args.mapping).resolve() if args.mapping else None,
    )

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.details_output:
        details_path = Path(args.details_output).resolve()
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for row in details_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_dataset_path(cfg: ProjectConfig, override: str | None) -> Path:
    if override:
        path = Path(override).resolve()
    else:
        path = cfg.evaluation.offline_dataset_path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def _parse_topk(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            number = int(stripped)
        except ValueError:
            raise ValueError(f"Invalid top-k value: {stripped!r}") from None
        if number < 1:
            raise ValueError(f"Top-k values must be >= 1 (got {number})")
        values.append(number)
    if not values:
        raise ValueError("At least one top-k value must be provided")
    return sorted(set(values))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            yield json.loads(payload)


def _load_predictions(
    path: Path,
    *,
    id_field: str,
    prediction_field: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in _iter_jsonl(path):
        identifier = row.get(id_field)
        prediction = row.get(prediction_field)
        if identifier is None or prediction is None:
            continue
        identifier_str = str(identifier).strip()
        if not identifier_str:
            continue
        mapping[identifier_str] = str(prediction).strip()
    return mapping


if __name__ == "__main__":
    main()
