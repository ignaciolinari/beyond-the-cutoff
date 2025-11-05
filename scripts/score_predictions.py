#!/usr/bin/env python3
"""Compute automatic metrics (factuality, citation accuracy, BLEU, BERTScore)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from beyond_the_cutoff.config import ProjectConfig, load_config
from beyond_the_cutoff.evaluation.scoring import score_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score predictions against evaluation splits")
    parser.add_argument("--config", default="configs/default.yaml", help="Project config path")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Evaluation dataset JSONL (defaults to config.evaluation.offline_dataset_path)",
    )
    parser.add_argument(
        "--predictions", required=True, help="JSONL predictions produced by evaluator"
    )
    parser.add_argument(
        "--prediction-id-field",
        default="task_id",
        help="Field name carrying the task identifier in the predictions file (default: task_id)",
    )
    parser.add_argument(
        "--prediction-field",
        default="model_answer",
        help="Field name storing the generated answer in the predictions file (default: model_answer)",
    )
    parser.add_argument(
        "--task-type",
        dest="task_types",
        action="append",
        default=None,
        help="Limit scoring to specific task types (can be passed multiple times)",
    )
    parser.add_argument(
        "--bert-lang",
        default="en",
        help="Language code used for BERTScore (default: en)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write summary JSON (always printed to stdout)",
    )
    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional path for per-example metrics JSONL",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            yield json.loads(payload)


def load_predictions(
    path: Path,
    *,
    id_field: str,
    prediction_field: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in iter_jsonl(path):
        identifier = row.get(id_field)
        prediction = row.get(prediction_field)
        if identifier is None or prediction is None:
            continue
        identifier_str = str(identifier).strip()
        if not identifier_str:
            continue
        mapping[identifier_str] = str(prediction).strip()
    return mapping


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    project_config: ProjectConfig = load_config(config_path)

    dataset_path = (
        Path(args.dataset) if args.dataset else project_config.evaluation.offline_dataset_path
    ).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    predictions_path = Path(args.predictions).resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    dataset_iter = list(iter_jsonl(dataset_path))
    predictions = load_predictions(
        predictions_path,
        id_field=args.prediction_id_field,
        prediction_field=args.prediction_field,
    )

    summary, per_example = score_predictions(
        dataset_iter,
        predictions,
        task_types=args.task_types,
        bert_lang=args.bert_lang,
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
            for row in per_example:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
