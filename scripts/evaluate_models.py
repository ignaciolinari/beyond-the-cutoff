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
        --judge-inference configs/judges/ollama_qwen3b.yaml \
        --output evaluation/results/rag_baseline_v1/metrics.json

"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml  # type: ignore[import-untyped]

from beyond_the_cutoff.config import (
    InferenceConfig,
    ProjectConfig,
    load_config,
)
from beyond_the_cutoff.models import LLMClient, build_generation_client


@dataclass
class JudgePrompt:
    """Parsed judge configuration."""

    name: str
    prompt: str
    criteria: list[dict[str, Any]]
    output_format: dict[str, Any]
    references: dict[str, Any]


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
    return parser.parse_args()


def load_inference_from_yaml(path: Path) -> InferenceConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected judge inference YAML to be a mapping, got {type(data)!r}")
    return InferenceConfig.model_validate(data)


def load_judge_prompt(path: Path) -> JudgePrompt:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Judge config must be a mapping, got {type(payload)!r}")
    return JudgePrompt(
        name=str(payload.get("name", "judge")),
        prompt=str(payload.get("prompt", "")),
        criteria=list(payload.get("criteria", [])),
        output_format=dict(payload.get("format", {})),
        references=dict(payload.get("references", {})),
    )


def iter_dataset(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_judge_prompt(template: str, question: str, contexts: Iterable[Any], answer: str) -> str:
    numbered_contexts = _normalize_contexts(contexts)
    context_block = "\n\n".join(numbered_contexts)
    rendered = template.replace("QUESTION", question.strip())
    rendered = rendered.replace("ASSISTANT_RESPONSE", answer.strip())
    rendered = rendered.replace("CONTEXTS", context_block)
    return rendered


def parse_judge_output(payload: str) -> dict[str, Any]:
    payload = payload.strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    except json.JSONDecodeError:
        # Attempt to salvage JSON snippet
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = payload[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
                return {"raw": parsed}
            except json.JSONDecodeError:
                return {"raw": payload}
        return {"raw": payload}


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CONTEXT_NUMBER_PATTERN = re.compile(r"^\s*\[(\d+)\]")


def _normalize_contexts(contexts: Iterable[Any]) -> list[str]:
    processed: list[str] = [str(ctx).strip() for ctx in contexts if ctx is not None]
    if not processed:
        return []
    if all(_CONTEXT_NUMBER_PATTERN.match(ctx) for ctx in processed):
        return processed
    return [f"[{idx + 1}] {ctx}" for idx, ctx in enumerate(processed)]


def evaluate_citations(answer: str, contexts: list[str]) -> dict[str, Any]:
    marks = [int(match) for match in _CITATION_PATTERN.findall(answer)]
    unique_marks = sorted(set(marks))
    total = len(contexts)
    missing = [i for i in range(1, total + 1) if i not in unique_marks]
    extra = [i for i in unique_marks if i < 1 or i > total]

    answer_words = set(answer.lower().split())
    coverage: dict[int, float] = {}
    for idx in unique_marks:
        if idx < 1 or idx > total:
            continue
        context_words = [w for w in contexts[idx - 1].lower().split() if len(w) > 3]
        if not context_words:
            coverage[idx] = 0.0
            continue
        overlap = sum(1 for w in context_words if w in answer_words)
        coverage[idx] = overlap / max(len(context_words), 1)

    mean_coverage = mean(coverage.values()) if coverage else 0.0
    return {
        "referenced": unique_marks,
        "missing": missing,
        "extra": extra,
        "coverage": coverage,
        "mean_coverage": mean_coverage,
    }


def summarise_scores(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[float]] = defaultdict(list)
    coverage_values: list[float] = []
    for row in score_rows:
        scores = row.get("judge_scores", {})
        for key, value in scores.items():
            if isinstance(value, int | float):
                metrics[key].append(float(value))
        citation_metrics = row.get("citation_metrics", {})
        coverage = citation_metrics.get("mean_coverage")
        if isinstance(coverage, int | float):
            coverage_values.append(float(coverage))
    summary = {key: mean(values) if values else 0.0 for key, values in metrics.items()}
    summary["citation_mean_coverage"] = mean(coverage_values) if coverage_values else 0.0
    return summary


def main() -> None:
    args = parse_args()

    project_cfg: ProjectConfig = load_config(args.config)
    dataset_path = (
        Path(args.dataset) if args.dataset else project_cfg.evaluation.offline_dataset_path
    )
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {dataset_path}")

    model_cfg = (
        load_inference_from_yaml(Path(args.model_config).resolve())
        if args.model_config
        else project_cfg.inference
    )
    judge_prompt = load_judge_prompt(Path(args.judge_config).resolve())

    if args.judge_inference:
        judge_inference_cfg = load_inference_from_yaml(Path(args.judge_inference).resolve())
    else:
        judge_inference_cfg = model_cfg

    model_label = args.model_label or model_cfg.model

    model_client: LLMClient = build_generation_client(model_cfg)
    judge_client: LLMClient = build_generation_client(judge_inference_cfg)

    score_rows: list[dict[str, Any]] = []

    for example in iter_dataset(dataset_path, limit=args.limit):
        task_id = example.get("task_id")
        instruction = example.get("instruction", "")
        rag = example.get("rag", {})
        prompt = rag.get("prompt") or example.get("rag_prompt")
        contexts_raw = rag.get("contexts") or example.get("contexts") or []
        if not prompt:
            raise KeyError(f"Example {task_id} missing prompt field")

        model_response = model_client.generate(prompt)
        assistant_answer = str(model_response.get("response", "")).strip()

        contexts_numbered = _normalize_contexts(contexts_raw)
        citation_metrics = evaluate_citations(assistant_answer, contexts_numbered)
        judge_prompt_text = render_judge_prompt(
            judge_prompt.prompt,
            instruction,
            contexts_numbered,
            assistant_answer,
        )
        judge_response = judge_client.generate(judge_prompt_text)
        judge_payload = parse_judge_output(str(judge_response.get("response", "")))

        judge_scores = judge_payload.get("scores") if isinstance(judge_payload, dict) else {}

        score_rows.append(
            {
                "task_id": task_id,
                "task_type": example.get("task_type"),
                "model_answer": assistant_answer,
                "judge_scores": judge_scores if isinstance(judge_scores, dict) else {},
                "judge_verdict": judge_payload.get("verdict"),
                "citation_metrics": citation_metrics,
                "model_label": model_label,
            }
        )

    summary = summarise_scores(score_rows)
    summary.update(
        {
            "model_label": model_label,
            "dataset": str(dataset_path),
            "examples_evaluated": len(score_rows),
            "judge_name": judge_prompt.name,
            "judge_model": judge_inference_cfg.model,
        }
    )

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"summary": summary, "examples": score_rows}, indent=2), encoding="utf-8"
        )

    if args.details_output:
        details_path = Path(args.details_output).resolve()
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for row in score_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
