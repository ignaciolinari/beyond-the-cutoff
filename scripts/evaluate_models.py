#!/usr/bin/env python3
"""Evaluate assistant models on the offline dataset using judge prompts.

The script loads an offline dataset JSONL (produced by
``scripts/generate_offline_dataset.py``), runs the target model to generate
responses, evaluates each answer with a judge prompt, and summarises citation
metrics.

Usage example::

    python scripts/evaluate_models.py \
        --config configs/default.yaml \
        --dataset evaluation/datasets/offline_dataset.jsonl \
        --model-label rag_baseline_v1 \
        --judge-config configs/judges/scientific_default.yaml \
        --judge-inference configs/judges/ollama_qwen7b.yaml \
        --output evaluation/results/rag_baseline_v1/metrics.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_the_cutoff.config import ProjectConfig, load_config
from beyond_the_cutoff.evaluation.runner import (
    EvaluationResult,
    load_inference_from_yaml,
    run_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate assistant models on offline dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Project config path")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Offline dataset JSONL (defaults to config.evaluation.offline_dataset_path)",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional inference YAML overriding config.inference",
    )
    parser.add_argument(
        "--judge-config",
        default="configs/judges/scientific_default.yaml",
        help="Judge prompt configuration YAML",
    )
    parser.add_argument(
        "--judge-inference",
        default=None,
        help="Inference YAML describing the judge model backend",
    )
    parser.add_argument("--model-label", default=None, help="Label used in the output summary")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of examples to evaluate",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write metrics JSON (summary always printed to stdout)",
    )
    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional JSONL file to write per-example evaluation records",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional JSONL file capturing experiment metadata (defaults to evaluation/results/<label>/metadata.jsonl)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retry attempts when model calls fail (default: 2)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=15.0,
        help="Base delay in seconds between retry attempts (default: 15.0)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["rag", "instruction"],
        default="rag",
        help="Prompt mode: 'rag' uses RAG prompts with contexts, 'instruction' uses instruction-only prompts (default: rag)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    project_cfg: ProjectConfig = load_config(config_path)
    dataset_path = (
        Path(args.dataset) if args.dataset else project_cfg.evaluation.offline_dataset_path
    ).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {dataset_path}")

    model_cfg = (
        load_inference_from_yaml(Path(args.model_config).resolve())
        if args.model_config
        else project_cfg.inference
    )

    judge_prompt_path = Path(args.judge_config).resolve()

    model_config_path = Path(args.model_config).resolve() if args.model_config else None

    judge_inference_path = Path(args.judge_inference).resolve() if args.judge_inference else None
    if judge_inference_path:
        judge_inference_cfg = load_inference_from_yaml(judge_inference_path)
    else:
        judge_inference_cfg = model_cfg

    model_label = args.model_label or model_cfg.model

    output_path = Path(args.output).resolve() if args.output else None
    details_output_path = Path(args.details_output).resolve() if args.details_output else None
    metadata_output_path = Path(args.metadata_output).resolve() if args.metadata_output else None

    result: EvaluationResult = run_evaluation(
        project_config=project_cfg,
        dataset_path=dataset_path,
        model_cfg=model_cfg,
        judge_prompt_path=judge_prompt_path,
        judge_inference_cfg=judge_inference_cfg,
        model_label=model_label,
        config_path=config_path,
        model_config_path=model_config_path,
        judge_inference_path=judge_inference_path,
        limit=args.limit,
        output_path=output_path,
        details_output_path=details_output_path,
        metadata_output_path=metadata_output_path,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        prompt_mode=args.prompt_mode,
    )

    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
