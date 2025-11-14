"""Utilities for executing evaluation runs across models and judges."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from beyond_the_cutoff.config import InferenceConfig, ProjectConfig
from beyond_the_cutoff.evaluation.metrics import evaluate_citations, normalize_contexts
from beyond_the_cutoff.models import LLMClient, build_generation_client
from beyond_the_cutoff.utils.experiment_logging import append_experiment_record


@dataclass
class JudgePrompt:
    """Parsed judge configuration."""

    name: str
    prompt: str
    criteria: list[dict[str, Any]]
    output_format: dict[str, Any]
    references: dict[str, Any]


@dataclass
class EvaluationResult:
    """Container summarising the outcome of an evaluation run."""

    summary: dict[str, Any]
    score_rows: list[dict[str, Any]]
    metrics_path: Path | None
    details_path: Path | None
    metadata_path: Path


@cache
def load_inference_from_yaml(path: Path) -> InferenceConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected inference YAML to be a mapping, got {type(data)!r}")
    return InferenceConfig.model_validate(data)


@cache
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


def render_judge_prompt(
    template: str, question: str, contexts: Iterable[Any], answer: str, *, has_contexts: bool = True
) -> str:
    numbered_contexts = normalize_contexts(contexts) if has_contexts else []
    context_block = (
        "\n\n".join(numbered_contexts) if numbered_contexts else "(No contexts provided)"
    )
    rendered = template.replace("QUESTION", question.strip())
    rendered = rendered.replace("ASSISTANT_RESPONSE", answer.strip())
    rendered = rendered.replace("CONTEXTS", context_block)
    return rendered


def _build_instruction_only_prompt(instruction: str) -> str:
    """Build a prompt for instruction-only mode (no RAG contexts)."""
    instruction_text = instruction.strip()
    if not instruction_text:
        raise ValueError("Instruction cannot be empty for instruction-only mode")
    return (
        "You are a research paper assistant. Answer the following question based on your knowledge.\n\n"
        f"Question: {instruction_text}\n\nAnswer:"
    )


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


def summarise_scores(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[float]] = defaultdict(list)
    coverage_values: list[float] = []
    for row in score_rows:
        scores = row.get("judge_scores", {})
        for key, value in scores.items():
            if isinstance(value, int | float):
                metrics[key].append(float(value))
        citation_metrics = row.get("citation_metrics", {})
        # Exclude citation metrics from instruction-only mode (not applicable)
        if citation_metrics.get("mode") != "instruction_only":
            coverage = citation_metrics.get("mean_coverage")
            if isinstance(coverage, int | float):
                coverage_values.append(float(coverage))
    summary = {key: mean(values) if values else 0.0 for key, values in metrics.items()}
    summary["citation_mean_coverage"] = mean(coverage_values) if coverage_values else 0.0
    return summary


def _call_with_retries(
    func: Callable[[], Any],
    *,
    stage: str,
    max_retries: int,
    retry_delay: float,
) -> tuple[Any | None, str | None]:
    attempts = max_retries + 1
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func(), None
        except KeyboardInterrupt:  # pragma: no cover - allow user aborts
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == attempts:
                break
            wait_seconds = max(retry_delay, 0.0) * attempt
            print(
                f"[warn] {stage} failed on attempt {attempt}/{attempts}: {last_error}. "
                f"Retrying in {wait_seconds:.1f}s...",
                file=sys.stderr,
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
    return None, last_error


def run_evaluation(
    *,
    project_config: ProjectConfig,
    dataset_path: Path,
    model_cfg: InferenceConfig,
    judge_prompt_path: Path,
    judge_inference_cfg: InferenceConfig,
    model_label: str,
    config_path: Path,
    model_config_path: Path | None = None,
    judge_inference_path: Path | None = None,
    limit: int | None = None,
    output_path: Path | None = None,
    details_output_path: Path | None = None,
    metadata_output_path: Path | None = None,
    max_retries: int = 2,
    retry_delay: float = 15.0,
    prompt_mode: str = "rag",
) -> EvaluationResult:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {dataset_path}")

    judge_prompt = load_judge_prompt(judge_prompt_path.resolve())

    # Warn if judge config might not match prompt mode
    judge_name_lower = judge_prompt.name.lower()
    is_instruction_judge = "instruction" in judge_name_lower and "rag" not in judge_name_lower
    is_rag_judge = "rag" in judge_name_lower and "instruction" not in judge_name_lower
    if prompt_mode == "instruction" and is_rag_judge:
        print(
            f"[warn] prompt_mode='instruction' but judge config '{judge_prompt.name}' appears to be for RAG evaluation. "
            "Consider using an instruction-only judge config.",
            file=sys.stderr,
        )
    elif prompt_mode == "rag" and is_instruction_judge:
        print(
            f"[warn] prompt_mode='rag' but judge config '{judge_prompt.name}' appears to be for instruction-only evaluation. "
            "Consider using a RAG judge config.",
            file=sys.stderr,
        )

    model_client: LLMClient = build_generation_client(model_cfg)
    judge_client: LLMClient = build_generation_client(judge_inference_cfg)

    if prompt_mode not in ("rag", "instruction"):
        raise ValueError(f"prompt_mode must be 'rag' or 'instruction', got {prompt_mode!r}")

    # Validate dataset structure before processing
    _validate_dataset_structure(dataset_path, prompt_mode, limit=limit)

    score_rows: list[dict[str, Any]] = []
    evaluated_task_ids: set[str] = set()

    # Count total examples for progress reporting
    total_examples = _count_dataset_examples(dataset_path, limit=limit)
    print(f"[info] Starting evaluation of {total_examples} example(s)", file=sys.stderr)

    for idx, example in enumerate(_iter_dataset(dataset_path, limit=limit), start=1):
        task_id = example.get("task_id")
        if task_id:
            evaluated_task_ids.add(str(task_id))
        instruction = example.get("instruction", "")
        rag = example.get("rag", {})

        # Determine prompt based on mode
        if prompt_mode == "instruction":
            if not instruction:
                raise KeyError(
                    f"Example {task_id} missing instruction field for instruction-only mode"
                )
            prompt_text = _build_instruction_only_prompt(instruction)
            contexts_raw: list[Any] = []  # No contexts for instruction-only mode
        else:  # prompt_mode == "rag"
            prompt = rag.get("prompt") or example.get("rag_prompt")
            if not prompt:
                raise KeyError(f"Example {task_id} missing prompt field for RAG mode")
            prompt_text = str(prompt)
            contexts_raw = rag.get("contexts") or example.get("contexts") or []
            # Warn if RAG mode but no contexts retrieved
            if not contexts_raw:
                print(
                    f"[warn] RAG mode enabled but example {task_id} has no contexts. "
                    "Citation metrics may not be meaningful.",
                    file=sys.stderr,
                )

        record_errors: dict[str, str] = {}

        # Time generation
        generation_start = time.time()
        model_response, generation_error = _call_with_retries(
            partial(model_client.generate, prompt_text),
            stage=f"generation (task {task_id or idx})",
            max_retries=max(max_retries, 0),
            retry_delay=retry_delay,
        )
        generation_time = time.time() - generation_start
        assistant_answer = ""
        if model_response is not None:
            assistant_answer = str(model_response.get("response", "")).strip()
        if generation_error:
            record_errors["generation"] = generation_error

        # Track empty responses
        has_empty_response = not assistant_answer.strip()
        if has_empty_response and not generation_error:
            record_errors.setdefault("generation", "Empty response received")

        contexts_numbered = normalize_contexts(contexts_raw)
        # has_contexts should be False for instruction mode, True for RAG mode only if contexts exist
        has_contexts = prompt_mode == "rag" and len(contexts_raw) > 0

        # For instruction-only mode, citation metrics are not meaningful
        # We still compute them but mark them as N/A
        citation_metrics = evaluate_citations(assistant_answer, contexts_numbered)
        if prompt_mode == "instruction":
            # Mark citation metrics as not applicable for instruction-only mode
            citation_metrics = {
                **citation_metrics,
                "mode": "instruction_only",
                "note": "Citation metrics not applicable - no contexts provided",
            }

        judge_prompt_text = render_judge_prompt(
            judge_prompt.prompt,
            instruction,
            contexts_numbered,
            assistant_answer,
            has_contexts=has_contexts,
        )
        judge_response = None
        judge_payload: dict[str, Any] = {}
        judge_time = 0.0

        if generation_error:
            record_errors.setdefault("judge", "Skipped due to generation failure")
        else:
            # Time judging
            judge_start = time.time()
            judge_response, judge_error = _call_with_retries(
                partial(judge_client.generate, judge_prompt_text),
                stage=f"judging (task {task_id or idx})",
                max_retries=max(max_retries, 0),
                retry_delay=retry_delay,
            )
            judge_time = time.time() - judge_start
            if judge_error:
                record_errors["judge"] = judge_error
            if judge_response is not None:
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
                "timing": {
                    "generation_seconds": generation_time,
                    "judge_seconds": judge_time,
                    "total_seconds": generation_time + judge_time,
                },
                **({"errors": record_errors} if record_errors else {}),
            }
        )

        # Progress reporting every 10 examples or at milestones
        if idx % 10 == 0 or idx == total_examples:
            error_count = sum(1 for row in score_rows if row.get("errors"))
            print(
                f"[info] Progress: {idx}/{total_examples} examples evaluated "
                f"({error_count} with errors)",
                file=sys.stderr,
            )

    summary = summarise_scores(score_rows)

    # Compute timing statistics
    timing_stats = _compute_timing_stats(score_rows)

    summary.update(
        {
            "model_label": model_label,
            "dataset": str(dataset_path),
            "examples_evaluated": len(score_rows),
            "judge_name": judge_prompt.name,
            "judge_model": judge_inference_cfg.model,
            "examples_with_errors": sum(1 for row in score_rows if row.get("errors")),
            "examples_with_empty_responses": sum(
                1
                for row in score_rows
                if not str(row.get("model_answer", "")).strip()
                and (not row.get("errors") or "generation" not in row.get("errors", {}))
            ),
            "error_rate": (
                sum(1 for row in score_rows if row.get("errors")) / len(score_rows)
                if score_rows
                else 0.0
            ),
            "prompt_mode": prompt_mode,
            "evaluated_task_ids": sorted(evaluated_task_ids),
            "timing": timing_stats,
        }
    )

    # Warn if error rate is high
    error_rate = summary.get("error_rate", 0.0)
    if error_rate > 0.1:  # More than 10% errors
        print(
            f"[warn] High error rate detected: {error_rate:.1%} of examples had errors. "
            "Consider reviewing error logs.",
            file=sys.stderr,
        )

    metrics_path: Path | None = None
    if output_path:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = {"summary": summary, "examples": score_rows}
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        metrics_path = output_path

    details_path_resolved: Path | None = None
    if details_output_path:
        details_path = details_output_path.resolve()
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for row in score_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        details_path_resolved = details_path

    metadata_path = _resolve_metadata_path(
        metadata_output_path,
        metrics_path,
        model_label,
    )

    append_experiment_record(
        metadata_path,
        project_config=project_config,
        dataset_path=dataset_path,
        model_config=model_cfg,
        judge_config=judge_inference_cfg,
        metrics=summary,
        score_rows=score_rows,
        model_label=model_label,
        config_path=config_path,
        model_config_path=model_config_path,
        judge_prompt_path=judge_prompt_path.resolve(),
        judge_inference_path=judge_inference_path.resolve() if judge_inference_path else None,
        details_path=details_path_resolved,
        metrics_path=metrics_path,
    )

    return EvaluationResult(
        summary=summary,
        score_rows=score_rows,
        metrics_path=metrics_path,
        details_path=details_path_resolved,
        metadata_path=metadata_path,
    )


def _validate_dataset_structure(
    dataset_path: Path, prompt_mode: str, *, limit: int | None = None
) -> None:
    """Validate that dataset has required fields for the given prompt_mode."""
    issues: list[str] = []
    sample_count = 0

    for idx, example in enumerate(_iter_dataset(dataset_path, limit=limit), start=1):
        sample_count += 1
        task_id = example.get("task_id", f"line_{idx}")

        if prompt_mode == "instruction":
            if not example.get("instruction"):
                issues.append(
                    f"Example {task_id} missing 'instruction' field (required for instruction mode)"
                )
        else:  # prompt_mode == "rag"
            rag = example.get("rag", {})
            if not rag.get("prompt") and not example.get("rag_prompt"):
                issues.append(
                    f"Example {task_id} missing 'rag.prompt' or 'rag_prompt' field (required for RAG mode)"
                )

    if issues:
        error_msg = f"Dataset validation failed for {prompt_mode} mode:\n" + "\n".join(issues[:10])
        if len(issues) > 10:
            error_msg += f"\n... and {len(issues) - 10} more issues"
        raise ValueError(error_msg)

    if sample_count == 0:
        raise ValueError(f"Dataset {dataset_path} appears to be empty")


def _compute_timing_stats(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute timing statistics from score rows."""
    generation_times: list[float] = []
    judge_times: list[float] = []
    total_times: list[float] = []

    for row in score_rows:
        timing = row.get("timing", {})
        if isinstance(timing, dict):
            gen_time = timing.get("generation_seconds")
            judge_time = timing.get("judge_seconds")
            total_time = timing.get("total_seconds")

            if isinstance(gen_time, int | float):
                generation_times.append(float(gen_time))
            if isinstance(judge_time, int | float):
                judge_times.append(float(judge_time))
            if isinstance(total_time, int | float):
                total_times.append(float(total_time))

    stats: dict[str, Any] = {}
    if generation_times:
        stats["generation"] = {
            "mean_seconds": mean(generation_times),
            "min_seconds": min(generation_times),
            "max_seconds": max(generation_times),
            "total_seconds": sum(generation_times),
        }
    if judge_times:
        stats["judge"] = {
            "mean_seconds": mean(judge_times),
            "min_seconds": min(judge_times),
            "max_seconds": max(judge_times),
            "total_seconds": sum(judge_times),
        }
    if total_times:
        stats["total"] = {
            "mean_seconds": mean(total_times),
            "min_seconds": min(total_times),
            "max_seconds": max(total_times),
            "total_seconds": sum(total_times),
        }

    return stats


def _count_dataset_examples(dataset_path: Path, *, limit: int | None = None) -> int:
    """Count total examples in dataset (for progress reporting)."""
    count = 0
    with dataset_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                count += 1
    return count


def _iter_dataset(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            payload = line.strip()
            if not payload:
                continue
            yield json.loads(payload)


def _resolve_metadata_path(
    requested: Path | None,
    metrics_path: Path | None,
    model_label: str,
) -> Path:
    if requested:
        metadata_path = requested.resolve()
    elif metrics_path is not None:
        metadata_path = metrics_path.with_name("metadata.jsonl")
    else:
        metadata_path = Path("evaluation/results") / model_label / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    return metadata_path
