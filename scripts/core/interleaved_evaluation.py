#!/usr/bin/env python3
"""Interleaved evaluation across all conditions.

Runs evaluation in an interleaved fashion:
  Example 1 → Condition 1, 2, 3, 4, 5, 6
  Example 2 → Condition 1, 2, 3, 4, 5, 6
  ...

Advantages:
- If interrupted, you have balanced data across all conditions
- Easy to resume from any point
- Can work with partial results immediately

Usage:
    # Start fresh
    python scripts/core/interleaved_evaluation.py \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --limit 30

    # Resume from where it left off
    python scripts/core/interleaved_evaluation.py \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --limit 30 --resume

    # Force restart (ignore existing progress)
    python scripts/core/interleaved_evaluation.py \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --limit 30 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from beyond_the_cutoff.config import ProjectConfig, load_config
from beyond_the_cutoff.evaluation.comparison import (
    ComparisonPlan,
    ComparisonRunSpec,
    PlanDefaults,
    load_comparison_plan,
)
from beyond_the_cutoff.evaluation.metrics import (
    evaluate_citations,
    normalize_contexts,
)
from beyond_the_cutoff.evaluation.runner import (
    JudgePrompt,
    _build_instruction_only_prompt,
    _build_rag_prompt_for_instruction_only_model,
    _call_with_retries,
    _detect_model_type,
    load_inference_from_yaml,
    load_judge_prompt,
    parse_judge_output,
    render_judge_prompt,
)
from beyond_the_cutoff.models import LLMClient, build_generation_client

# =============================================================================
# State Management
# =============================================================================


@dataclass
class EvaluationState:
    """Tracks progress of interleaved evaluation."""

    plan_path: str
    limit: int
    current_example_idx: int = 0  # 0-indexed
    current_condition_idx: int = 0  # 0-indexed
    completed_pairs: list[tuple[int, int]] = field(
        default_factory=list
    )  # (example_idx, condition_idx)
    started_at: str = ""
    last_updated: str = ""
    total_examples: int = 0
    total_conditions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": self.plan_path,
            "limit": self.limit,
            "current_example_idx": self.current_example_idx,
            "current_condition_idx": self.current_condition_idx,
            "completed_pairs": self.completed_pairs,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_examples": self.total_examples,
            "total_conditions": self.total_conditions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationState:
        return cls(
            plan_path=data["plan_path"],
            limit=data["limit"],
            current_example_idx=data.get("current_example_idx", 0),
            current_condition_idx=data.get("current_condition_idx", 0),
            completed_pairs=[tuple(p) for p in data.get("completed_pairs", [])],
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
            total_examples=data.get("total_examples", 0),
            total_conditions=data.get("total_conditions", 0),
        )

    def mark_completed(self, example_idx: int, condition_idx: int) -> None:
        """Mark an example-condition pair as completed."""
        pair = (example_idx, condition_idx)
        if pair not in self.completed_pairs:
            self.completed_pairs.append(pair)
        self.last_updated = datetime.now().isoformat()

    def is_completed(self, example_idx: int, condition_idx: int) -> bool:
        """Check if an example-condition pair is already completed."""
        return (example_idx, condition_idx) in self.completed_pairs

    def get_progress_string(self) -> str:
        """Get a human-readable progress string."""
        total_pairs = self.total_examples * self.total_conditions
        completed = len(self.completed_pairs)
        pct = (completed / total_pairs * 100) if total_pairs > 0 else 0
        return f"{completed}/{total_pairs} ({pct:.1f}%)"


def load_state(state_path: Path) -> EvaluationState | None:
    """Load evaluation state from file."""
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return EvaluationState.from_dict(data)
    except Exception as e:
        print(f"[warn] Could not load state from {state_path}: {e}", file=sys.stderr)
        return None


def save_state(state: EvaluationState, state_path: Path) -> None:
    """Save evaluation state to file."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================================================================
# Result Management
# =============================================================================


@dataclass
class ConditionResult:
    """Result for a single condition."""

    label: str
    results_path: Path
    details: list[dict[str, Any]] = field(default_factory=list)


def load_existing_results(results_path: Path) -> list[dict[str, Any]]:
    """Load existing results from a JSONL file."""
    if not results_path.exists():
        return []
    results = []
    try:
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except Exception as e:
        print(f"[warn] Could not load results from {results_path}: {e}", file=sys.stderr)
    return results


def append_result(results_path: Path, result: dict[str, Any]) -> None:
    """Append a single result to a JSONL file."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def get_evaluated_task_ids(results_path: Path) -> set[str]:
    """Get set of task IDs already evaluated for a condition."""
    results = load_existing_results(results_path)
    return {str(r.get("task_id")) for r in results if r.get("task_id")}


# =============================================================================
# Evaluation Logic
# =============================================================================


def evaluate_single_example(
    example: dict[str, Any],
    spec: ComparisonRunSpec,
    defaults: PlanDefaults,
    project_config: ProjectConfig,
    model_client: LLMClient,
    judge_client: LLMClient,
    judge_prompt: JudgePrompt,
    *,
    max_retries: int = 2,
    retry_delay: float = 15.0,
) -> dict[str, Any]:
    """Evaluate a single example for a single condition.

    Returns a result dict with all evaluation data.
    """
    task_id = example.get("task_id", "unknown")
    instruction = example.get("instruction", "")
    rag = example.get("rag", {})
    expected_response = example.get("expected_response", "")

    # Determine prompt mode
    prompt_mode = spec.prompt_mode or defaults.prompt_mode or "rag"

    # Get model config for type detection
    model_cfg = None
    if spec.model_config:
        model_cfg = load_inference_from_yaml(spec.model_config)

    # Build prompt based on mode
    if prompt_mode == "instruction":
        prompt_text = _build_instruction_only_prompt(
            instruction,
            model_config_path=spec.model_config,
            model_name=model_cfg.model if model_cfg else "",
            model_cfg=model_cfg,
        )
        contexts_raw: list[Any] = []
    else:  # RAG mode
        contexts_raw = rag.get("contexts") or example.get("contexts") or []

        # Detect model type for prompt building
        model_type = _detect_model_type(
            spec.model_config,
            model_cfg.model if model_cfg else "",
            model_cfg=model_cfg,
            warn_on_inference=False,
        )

        if model_type == "instruction_only" and contexts_raw:
            # Instruction-only model with RAG contexts (Condition 4)
            contexts_numbered = normalize_contexts(contexts_raw)
            prompt_text = _build_rag_prompt_for_instruction_only_model(
                instruction,
                contexts_numbered,
                model_config_path=spec.model_config,
                model_name=model_cfg.model if model_cfg else "",
            )
        else:
            # Standard RAG: use pre-built prompt
            prompt_text = rag.get("prompt") or example.get("rag_prompt") or ""
            if not prompt_text:
                # Fallback: build simple RAG prompt
                contexts_numbered = normalize_contexts(contexts_raw)
                context_block = "\n\n".join(contexts_numbered) if contexts_numbered else ""
                prompt_text = (
                    "Answer the question using the provided context. "
                    "Cite sources as [#].\n\n"
                    f"Context:\n{context_block}\n\n"
                    f"Question: {instruction}\nAnswer:"
                )

    record_errors: dict[str, str] = {}
    error_categories: dict[str, str] = {}

    # Generate response
    generation_start = time.time()
    from functools import partial

    model_response, generation_error, generation_error_category = _call_with_retries(
        partial(model_client.generate, prompt_text),
        stage=f"generation ({spec.label}:{task_id})",
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    generation_time = time.time() - generation_start

    assistant_answer = ""
    if model_response is not None:
        assistant_answer = str(model_response.get("response", "")).strip()
    if generation_error:
        record_errors["generation"] = generation_error
        if generation_error_category:
            error_categories["generation"] = generation_error_category

    # Citation metrics
    contexts_numbered = normalize_contexts(contexts_raw)
    has_contexts = prompt_mode == "rag" and len(contexts_raw) > 0
    citation_metrics = evaluate_citations(assistant_answer, contexts_numbered)

    if prompt_mode == "instruction":
        citation_metrics = {
            **citation_metrics,
            "mode": "instruction_only",
            "note": "Citation metrics not applicable",
            "validated": True,
        }
    else:
        citation_metrics["mode"] = "rag"
        citation_metrics["validated"] = True

    # Judge evaluation
    judge_prompt_text = render_judge_prompt(
        judge_prompt.prompt,
        instruction,
        contexts_numbered,
        assistant_answer,
        has_contexts=has_contexts,
        expected_response=expected_response,
    )

    judge_payload: dict[str, Any] = {}
    judge_time = 0.0

    if generation_error:
        record_errors.setdefault("judge", "Skipped due to generation failure")
        error_categories.setdefault("judge", "validation")
    else:
        judge_start = time.time()
        judge_response, judge_error, judge_error_category = _call_with_retries(
            partial(judge_client.generate, judge_prompt_text),
            stage=f"judging ({spec.label}:{task_id})",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        judge_time = time.time() - judge_start

        if judge_error:
            record_errors["judge"] = judge_error
            if judge_error_category:
                error_categories["judge"] = judge_error_category

        if judge_response is not None:
            judge_payload = parse_judge_output(str(judge_response.get("response", "")))

    judge_scores = judge_payload.get("scores") if isinstance(judge_payload, dict) else {}

    # Build result
    result: dict[str, Any] = {
        "task_id": task_id,
        "task_type": example.get("task_type"),
        "model_answer": assistant_answer,
        "expected_response": expected_response,
        "judge_scores": judge_scores if isinstance(judge_scores, dict) else {},
        "judge_verdict": judge_payload.get("verdict"),
        "citation_metrics": citation_metrics,
        "model_label": spec.label,
        "prompt_mode": prompt_mode,
        "response_length": {
            "word_count": len(assistant_answer.split()) if assistant_answer else 0,
            "char_count": len(assistant_answer) if assistant_answer else 0,
        },
        "timing": {
            "generation_seconds": generation_time,
            "judge_seconds": judge_time,
            "total_seconds": generation_time + judge_time,
        },
        "timestamp": datetime.now().isoformat(),
    }

    if record_errors:
        result["errors"] = record_errors
        if error_categories:
            result["error_categories"] = error_categories

    return result


# =============================================================================
# Main Pipeline
# =============================================================================


def run_interleaved_evaluation(
    plan: ComparisonPlan,
    project_config: ProjectConfig,
    config_path: Path,
    *,
    limit: int | None,
    output_dir: Path,
    state_path: Path,
    resume: bool = True,
    force: bool = False,
    max_retries: int = 2,
    retry_delay: float = 15.0,
) -> None:
    """Run interleaved evaluation across all conditions."""

    # Load or create state
    state: EvaluationState | None = None
    if resume and not force:
        state = load_state(state_path)
        if state:
            print(f"\n{'='*80}", file=sys.stderr)
            print("[info] RESUMING from previous run", file=sys.stderr)
            print(f"[info] Progress: {state.get_progress_string()}", file=sys.stderr)
            print(f"[info] Last updated: {state.last_updated}", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)

    # Load dataset
    dataset_path = plan.defaults.dataset or project_config.evaluation.offline_dataset_path
    if not dataset_path:
        raise ValueError("No dataset path specified in plan or config")
    dataset_path = Path(dataset_path).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load all examples (up to limit if specified)
    examples: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                examples.append(json.loads(line))

    # If no limit specified, use the actual count
    effective_limit = limit if limit is not None else len(examples)

    if state is None:
        state = EvaluationState(
            plan_path=str(plan.source_path),
            limit=effective_limit,
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_conditions=len(plan.runs),
        )

    state.total_examples = len(examples)
    state.total_conditions = len(plan.runs)

    print(f"\n{'='*80}", file=sys.stderr)
    print("[info] INTERLEAVED EVALUATION", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"[info] Dataset: {dataset_path}", file=sys.stderr)
    print(f"[info] Examples: {len(examples)}", file=sys.stderr)
    print(f"[info] Conditions: {len(plan.runs)}", file=sys.stderr)
    print(f"[info] Total evaluations: {len(examples) * len(plan.runs)}", file=sys.stderr)
    print(f"[info] Output directory: {output_dir}", file=sys.stderr)
    print(f"[info] State file: {state_path}", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)

    # Print condition labels
    print("[info] Conditions:", file=sys.stderr)
    for i, spec in enumerate(plan.runs):
        prompt_mode = spec.prompt_mode or plan.defaults.prompt_mode or "rag"
        print(f"  {i+1}. {spec.label} ({prompt_mode})", file=sys.stderr)
    print("", file=sys.stderr)

    # Build clients for each condition (lazy initialization)
    condition_clients: dict[str, tuple[LLMClient, LLMClient, JudgePrompt]] = {}

    def get_clients(spec: ComparisonRunSpec) -> tuple[LLMClient, LLMClient, JudgePrompt]:
        """Get or create clients for a condition."""
        if spec.label not in condition_clients:
            # Model client
            if spec.model_config:
                model_cfg = load_inference_from_yaml(spec.model_config)
            else:
                model_cfg = project_config.inference
            model_client = build_generation_client(model_cfg)

            # Judge client
            judge_inference_path = spec.judge_inference or plan.defaults.judge_inference
            if judge_inference_path:
                judge_cfg = load_inference_from_yaml(judge_inference_path)
            else:
                judge_cfg = model_cfg
            judge_client = build_generation_client(judge_cfg)

            # Judge prompt
            judge_config_path = spec.judge_config or plan.defaults.judge_config
            if not judge_config_path:
                raise ValueError(f"No judge config for {spec.label}")
            judge_prompt = load_judge_prompt(judge_config_path.resolve())

            condition_clients[spec.label] = (model_client, judge_client, judge_prompt)

        return condition_clients[spec.label]

    # Results paths for each condition
    results_paths: dict[str, Path] = {}
    for spec in plan.runs:
        results_paths[spec.label] = output_dir / spec.label / "details.jsonl"

    # Main evaluation loop: iterate by example, then by condition
    total_start = time.time()
    evaluations_done = 0

    try:
        for example_idx, example in enumerate(examples):
            task_id = example.get("task_id", f"example_{example_idx}")

            print(f"\n{'─'*80}", file=sys.stderr)
            print(
                f"[EXAMPLE {example_idx + 1}/{len(examples)}] Task: {task_id}",
                file=sys.stderr,
            )
            print(f"{'─'*80}", file=sys.stderr)

            for condition_idx, spec in enumerate(plan.runs):
                # Check if already completed
                if state.is_completed(example_idx, condition_idx):
                    print(
                        f"  [✓] Condition {condition_idx + 1}/{len(plan.runs)}: {spec.label} (already done)",
                        file=sys.stderr,
                    )
                    continue

                # Also check if task_id exists in results file (backup check)
                existing_ids = get_evaluated_task_ids(results_paths[spec.label])
                if task_id in existing_ids:
                    print(
                        f"  [✓] Condition {condition_idx + 1}/{len(plan.runs)}: {spec.label} (found in results)",
                        file=sys.stderr,
                    )
                    state.mark_completed(example_idx, condition_idx)
                    save_state(state, state_path)
                    continue

                # Run evaluation
                prompt_mode = spec.prompt_mode or plan.defaults.prompt_mode or "rag"
                print(
                    f"  [→] Condition {condition_idx + 1}/{len(plan.runs)}: {spec.label} ({prompt_mode})",
                    file=sys.stderr,
                    end="",
                )
                sys.stderr.flush()

                eval_start = time.time()

                try:
                    model_client, judge_client, judge_prompt = get_clients(spec)

                    result = evaluate_single_example(
                        example=example,
                        spec=spec,
                        defaults=plan.defaults,
                        project_config=project_config,
                        model_client=model_client,
                        judge_client=judge_client,
                        judge_prompt=judge_prompt,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                    )

                    # Append result
                    append_result(results_paths[spec.label], result)

                    # Mark completed
                    state.mark_completed(example_idx, condition_idx)
                    save_state(state, state_path)

                    eval_time = time.time() - eval_start
                    evaluations_done += 1

                    # Show result summary
                    scores = result.get("judge_scores", {})
                    if scores:
                        score_str = " | ".join(
                            f"{k[:4]}={v:.1f}"
                            for k, v in sorted(scores.items())
                            if isinstance(v, int | float)
                        )
                        print(f" → {score_str} ({eval_time:.1f}s)", file=sys.stderr)
                    else:
                        has_error = "errors" in result
                        status = "ERROR" if has_error else "OK"
                        print(f" → {status} ({eval_time:.1f}s)", file=sys.stderr)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f" → FAILED: {e}", file=sys.stderr)
                    # Save error result
                    error_result = {
                        "task_id": task_id,
                        "model_label": spec.label,
                        "errors": {"evaluation": str(e)},
                        "timestamp": datetime.now().isoformat(),
                    }
                    append_result(results_paths[spec.label], error_result)
                    state.mark_completed(example_idx, condition_idx)
                    save_state(state, state_path)

            # Progress summary after each example
            elapsed = time.time() - total_start
            progress = state.get_progress_string()
            rate = evaluations_done / elapsed if elapsed > 0 else 0
            remaining = (
                (state.total_examples * state.total_conditions - len(state.completed_pairs)) / rate
                if rate > 0
                else 0
            )
            print(
                f"\n  📊 Progress: {progress} | "
                f"Rate: {rate:.2f}/s | "
                f"ETA: {remaining/60:.1f}min",
                file=sys.stderr,
            )

    except KeyboardInterrupt:
        print(f"\n\n{'='*80}", file=sys.stderr)
        print("[info] INTERRUPTED - Progress saved!", file=sys.stderr)
        print(f"[info] Progress: {state.get_progress_string()}", file=sys.stderr)
        print("[info] Resume with: --resume flag", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        save_state(state, state_path)
        sys.exit(1)

    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'='*80}", file=sys.stderr)
    print("[info] EVALUATION COMPLETE", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"[info] Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", file=sys.stderr)
    print(f"[info] Evaluations: {len(state.completed_pairs)}", file=sys.stderr)
    print(f"[info] Results saved to: {output_dir}", file=sys.stderr)

    # Generate metrics files
    print("\n[info] Generating metrics summaries...", file=sys.stderr)
    for spec in plan.runs:
        results = load_existing_results(results_paths[spec.label])
        if results:
            metrics = compute_metrics_summary(results, spec.label)
            metrics_path = output_dir / spec.label / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [✓] {spec.label}: {len(results)} examples → {metrics_path}", file=sys.stderr)

    print(f"\n{'='*80}\n", file=sys.stderr)


def compute_metrics_summary(results: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Compute summary metrics from results."""
    from collections import defaultdict
    from statistics import mean

    metrics: dict[str, list[float]] = defaultdict(list)
    coverage_values: list[float] = []
    word_counts: list[int] = []
    errors_count = 0

    for row in results:
        if row.get("errors"):
            errors_count += 1

        scores = row.get("judge_scores", {})
        for key, value in scores.items():
            if isinstance(value, int | float):
                metrics[key].append(float(value))

        citation = row.get("citation_metrics", {})
        if citation.get("mode") == "rag":
            cov = citation.get("mean_coverage")
            if isinstance(cov, int | float):
                coverage_values.append(float(cov))

        response_length = row.get("response_length", {})
        wc = response_length.get("word_count", 0)
        if isinstance(wc, int):
            word_counts.append(wc)

    summary: dict[str, Any] = {
        key: mean(values) if values else 0.0 for key, values in metrics.items()
    }
    summary["citation_mean_coverage"] = mean(coverage_values) if coverage_values else 0.0
    summary["model_label"] = label
    summary["examples_evaluated"] = len(results)
    summary["examples_with_errors"] = errors_count
    summary["response_length"] = {
        "mean_word_count": mean(word_counts) if word_counts else 0.0,
        "min_word_count": min(word_counts) if word_counts else 0,
        "max_word_count": max(word_counts) if word_counts else 0,
    }
    summary["evaluated_task_ids"] = [str(r.get("task_id")) for r in results if r.get("task_id")]

    return summary


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interleaved evaluation across all conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start fresh evaluation
    python scripts/core/interleaved_evaluation.py \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --limit 30

    # Resume interrupted evaluation
    python scripts/core/interleaved_evaluation.py \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --limit 30 --resume

    # Force restart (ignore previous progress)
    python scripts/core/interleaved_evaluation.py \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --limit 30 --force
        """,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Project config path",
    )
    parser.add_argument(
        "--plan",
        required=True,
        help="Comparison plan YAML path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all examples in dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/interleaved"),
        help="Output directory for results (default: evaluation/results/interleaved)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress (default behavior)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restart, ignoring previous progress",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries for generation/judging (default: 2)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=15.0,
        help="Delay between retries in seconds (default: 15.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configs
    config_path = Path(args.config).resolve()
    project_config: ProjectConfig = load_config(config_path)

    plan_path = Path(args.plan).resolve()
    plan = load_comparison_plan(plan_path)

    # Output and state paths
    output_dir = args.output_dir.resolve()
    state_path = output_dir / "evaluation_state.json"

    # Run evaluation
    run_interleaved_evaluation(
        plan=plan,
        project_config=project_config,
        config_path=config_path,
        limit=args.limit,
        output_dir=output_dir,
        state_path=state_path,
        resume=not args.force,  # Resume by default unless --force
        force=args.force,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


if __name__ == "__main__":
    main()
