#!/usr/bin/env python3
"""Phase 1: Generate model responses for all conditions in a comparison plan.

This script generates responses for each model condition and saves them to JSONL files.
The responses can then be evaluated in Phase 2 using compare_models.py with --responses-dir.

Benefits of two-phase evaluation:
1. Responses are generated once and can be reused for multiple evaluation passes
2. Pairwise comparisons use the same responses for fair ELO ranking
3. If judge fails, you don't lose generated responses
4. Can parallelize generation across different machines

Usage:
    python scripts/generate_responses.py \
        --config configs/default.yaml \
        --plan configs/evaluation/six_condition_experiment.yaml \
        --output-dir evaluation/responses/

Output structure:
    evaluation/responses/
        base_baseline_0p5b.jsonl
        base_rag_0p5b.jsonl
        instruction_only_baseline_0p5b.jsonl
        ...
        generation_metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beyond_the_cutoff.config import InferenceConfig, load_config
from beyond_the_cutoff.evaluation.comparison import (
    ComparisonPlan,
    load_comparison_plan,
)
from beyond_the_cutoff.evaluation.metrics import normalize_contexts
from beyond_the_cutoff.evaluation.runner import (
    _build_instruction_only_prompt,
    _build_rag_prompt_for_instruction_only_model,
    _call_with_retries,
    _count_dataset_examples,
    _detect_model_type,
    _iter_dataset,
    _validate_dataset_structure,
    load_inference_from_yaml,
)
from beyond_the_cutoff.models import build_generation_client
from beyond_the_cutoff.utils.experiment_logging import compute_file_sha256


def generate_responses_for_condition(
    *,
    model_cfg: InferenceConfig,
    model_config_path: Path | None,
    dataset_path: Path,
    output_path: Path,
    model_label: str,
    prompt_mode: str,
    limit: int | None = None,
    max_retries: int = 2,
    retry_delay: float = 15.0,
    resume: bool = True,
) -> dict[str, Any]:
    """Generate responses for a single model condition.

    Returns metadata about the generation run.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Validate dataset structure
    _validate_dataset_structure(dataset_path, prompt_mode, limit=limit)

    # Build model client
    model_client = build_generation_client(model_cfg)

    # Detect model type for prompt building
    detected_model_type = _detect_model_type(
        model_config_path, model_cfg.model, model_cfg=model_cfg, warn_on_inference=True
    )

    # Count total examples
    total_examples = _count_dataset_examples(dataset_path, limit=limit)

    # Load existing responses if resuming
    existing_responses: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        print(f"[info] Loading existing responses from {output_path}", file=sys.stderr)
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    task_id = record.get("task_id")
                    if task_id:
                        existing_responses[task_id] = record
        print(f"[info] Found {len(existing_responses)} existing responses", file=sys.stderr)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start generation
    print("=" * 80, file=sys.stderr)
    print(f"[info] Generating responses for: {model_label}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"[info] Model: {model_cfg.model}", file=sys.stderr)
    print(f"[info] Prompt mode: {prompt_mode}", file=sys.stderr)
    print(f"[info] Detected model type: {detected_model_type}", file=sys.stderr)
    print(f"[info] Dataset: {dataset_path}", file=sys.stderr)
    print(f"[info] Total examples: {total_examples}", file=sys.stderr)
    print(f"[info] Output: {output_path}", file=sys.stderr)
    if existing_responses:
        print(f"[info] Resuming: {len(existing_responses)} already generated", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    start_time = time.time()
    generated_count = 0
    skipped_count = 0
    error_count = 0

    # Open output file in append mode if resuming, write mode otherwise
    mode = "a" if resume and existing_responses else "w"

    with open(output_path, mode, encoding="utf-8") as out_f:
        for idx, example in enumerate(_iter_dataset(dataset_path, limit=limit), start=1):
            task_id = str(example.get("task_id", f"line_{idx}"))

            # Skip if already generated
            if task_id in existing_responses:
                skipped_count += 1
                continue

            instruction = example.get("instruction", "")
            rag = example.get("rag", {})

            # Build prompt based on mode
            if prompt_mode == "instruction":
                if not instruction:
                    raise KeyError(f"Example {task_id} missing instruction field")
                prompt_text = _build_instruction_only_prompt(
                    instruction,
                    model_config_path=model_config_path,
                    model_name=model_cfg.model,
                    model_cfg=model_cfg,
                )
                contexts_raw: list[Any] = []
            else:  # rag mode
                contexts_raw = rag.get("contexts") or example.get("contexts") or []

                if detected_model_type == "instruction_only":
                    # Instruction-only model with RAG contexts (Condition 4)
                    if not instruction:
                        raise KeyError(f"Example {task_id} missing instruction field")
                    if not contexts_raw:
                        prompt_text = _build_instruction_only_prompt(
                            instruction,
                            model_config_path=model_config_path,
                            model_name=model_cfg.model,
                            model_cfg=model_cfg,
                        )
                    else:
                        contexts_numbered = normalize_contexts(contexts_raw)
                        prompt_text = _build_rag_prompt_for_instruction_only_model(
                            instruction,
                            contexts_numbered,
                            model_config_path=model_config_path,
                            model_name=model_cfg.model,
                        )
                else:
                    # Standard RAG mode
                    prompt = rag.get("prompt") or example.get("rag_prompt")
                    if not prompt:
                        raise KeyError(f"Example {task_id} missing prompt field for RAG mode")
                    prompt_text = str(prompt)

            # Generate response
            generation_start = time.time()
            response, error, error_category = _call_with_retries(
                partial(model_client.generate, prompt_text),
                stage=f"generation (task {task_id})",
                max_retries=max(max_retries, 0),
                retry_delay=retry_delay,
            )
            generation_time = time.time() - generation_start

            # Extract response text
            response_text = ""
            if response is not None:
                response_text = str(response.get("response", "")).strip()

            # Build record
            record = {
                "task_id": task_id,
                "task_type": example.get("task_type"),
                "instruction": instruction,
                "expected_response": example.get("expected_response", ""),  # Ground truth for judge
                "prompt_text": prompt_text,
                "response": response_text,
                "contexts": normalize_contexts(contexts_raw) if contexts_raw else [],
                "generation_time_seconds": generation_time,
                "model_label": model_label,
                "prompt_mode": prompt_mode,
                "timestamp": datetime.now().isoformat(),
            }

            if error:
                record["error"] = error
                record["error_category"] = error_category
                error_count += 1

            # Write to file
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()  # Ensure it's written immediately

            generated_count += 1

            # Progress update
            total_processed = generated_count + skipped_count
            elapsed = time.time() - start_time
            rate = generated_count / elapsed if elapsed > 0 else 0
            remaining = (total_examples - total_processed) / rate if rate > 0 else 0

            print(
                f"\r[{total_processed}/{total_examples}] "
                f"Generated: {generated_count}, Skipped: {skipped_count}, Errors: {error_count} "
                f"({rate:.1f}/s, ETA: {remaining:.0f}s)    ",
                end="",
                file=sys.stderr,
            )

    print(file=sys.stderr)  # Newline after progress

    total_time = time.time() - start_time

    print("-" * 80, file=sys.stderr)
    print(f"[info] Generation complete for {model_label}", file=sys.stderr)
    print(
        f"[info] Generated: {generated_count}, Skipped: {skipped_count}, Errors: {error_count}",
        file=sys.stderr,
    )
    print(f"[info] Total time: {total_time:.1f}s", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    return {
        "model_label": model_label,
        "model": model_cfg.model,
        "prompt_mode": prompt_mode,
        "detected_model_type": detected_model_type,
        "dataset_path": str(dataset_path),
        "dataset_sha256": compute_file_sha256(dataset_path),
        "output_path": str(output_path),
        "total_examples": total_examples,
        "generated_count": generated_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
    }


def execute_generation_plan(
    plan: ComparisonPlan,
    config_path: Path,
    output_dir: Path,
    *,
    limit_override: int | None = None,
    max_retries_override: int | None = None,
    retry_delay_override: float | None = None,
    resume: bool = True,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    """Execute generation for all conditions in a comparison plan.

    Args:
        plan: The comparison plan to execute
        config_path: Path to project config
        output_dir: Directory to save generated responses
        limit_override: Override limit for all runs
        max_retries_override: Override max retries
        retry_delay_override: Override retry delay
        resume: Whether to resume from existing responses
        conditions: Optional list of condition labels to run (runs all if None)

    Returns:
        Metadata dict with generation results for all conditions
    """
    project_config = load_config(config_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter runs if specific conditions requested
    runs_to_execute = plan.runs
    if conditions:
        runs_to_execute = [r for r in plan.runs if r.label in conditions]
        if not runs_to_execute:
            raise ValueError(
                f"No matching conditions found. Available: {[r.label for r in plan.runs]}"
            )
        print(
            f"[info] Running {len(runs_to_execute)} of {len(plan.runs)} conditions", file=sys.stderr
        )

    print("=" * 80, file=sys.stderr)
    print("[info] Starting response generation", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"[info] Conditions to generate: {len(runs_to_execute)}", file=sys.stderr)
    print(f"[info] Output directory: {output_dir}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    generation_results: list[dict[str, Any]] = []
    overall_start = time.time()

    for i, spec in enumerate(runs_to_execute, start=1):
        print(f"\n[info] Condition {i}/{len(runs_to_execute)}: {spec.label}", file=sys.stderr)

        # Resolve dataset
        dataset_source = (
            spec.dataset or plan.defaults.dataset or project_config.evaluation.offline_dataset_path
        )
        dataset_path = Path(dataset_source).resolve()

        # Resolve model config
        if spec.model_config:
            if not spec.model_config.exists():
                raise FileNotFoundError(f"Model config not found: {spec.model_config}")
            model_cfg = load_inference_from_yaml(spec.model_config)
            model_config_path = spec.model_config
        else:
            model_cfg = project_config.inference
            model_config_path = None

        # Resolve parameters
        limit = (
            limit_override
            if limit_override is not None
            else spec.limit
            if spec.limit is not None
            else plan.defaults.limit
        )
        max_retries = (
            max_retries_override
            if max_retries_override is not None
            else spec.max_retries
            if spec.max_retries is not None
            else plan.defaults.max_retries
        )
        retry_delay = (
            retry_delay_override
            if retry_delay_override is not None
            else spec.retry_delay
            if spec.retry_delay is not None
            else plan.defaults.retry_delay
        )
        prompt_mode = spec.prompt_mode or plan.defaults.prompt_mode or "rag"

        # Output path for this condition
        output_path = output_dir / f"{spec.label}.jsonl"

        # Generate responses
        result = generate_responses_for_condition(
            model_cfg=model_cfg,
            model_config_path=model_config_path,
            dataset_path=dataset_path,
            output_path=output_path,
            model_label=spec.label,
            prompt_mode=prompt_mode,
            limit=limit,
            max_retries=max_retries,
            retry_delay=retry_delay,
            resume=resume,
        )

        generation_results.append(result)

    overall_time = time.time() - overall_start

    # Save generation metadata
    metadata = {
        "plan_path": str(plan.source_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "conditions": [r["model_label"] for r in generation_results],
        "total_conditions": len(generation_results),
        "total_time_seconds": overall_time,
        "timestamp": datetime.now().isoformat(),
        "results": generation_results,
    }

    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80, file=sys.stderr)
    print("[info] All response generation complete!", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"[info] Total conditions: {len(generation_results)}", file=sys.stderr)
    print(f"[info] Total time: {overall_time:.1f}s ({overall_time/60:.1f} min)", file=sys.stderr)
    print(f"[info] Metadata saved to: {metadata_path}", file=sys.stderr)
    print(f"[info] Response files in: {output_dir}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    # Summary table
    print("\nGeneration Summary:", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    for r in generation_results:
        status = "✓" if r["error_count"] == 0 else f"WARNING:  ({r['error_count']} errors)"
        print(
            f"  {r['model_label']}: {r['generated_count']} generated, "
            f"{r['skipped_count']} skipped {status}",
            file=sys.stderr,
        )
    print("-" * 60, file=sys.stderr)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model responses for all conditions in a comparison plan (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate responses for all conditions
    python scripts/generate_responses.py \\
        --config configs/default.yaml \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --output-dir evaluation/responses/

    # Generate for specific conditions only
    python scripts/generate_responses.py \\
        --config configs/default.yaml \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --output-dir evaluation/responses/ \\
        --conditions base_baseline_0p5b base_rag_0p5b

    # Quick test with limit
    python scripts/generate_responses.py \\
        --config configs/default.yaml \\
        --plan configs/evaluation/six_condition_experiment.yaml \\
        --output-dir evaluation/responses/ \\
        --limit 5
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to project config file",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Path to comparison plan YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated responses",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to generate (for testing)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override max retries for generation",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=None,
        help="Override retry delay in seconds",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from existing responses",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Only generate for specific condition labels",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"[error] Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if not args.plan.exists():
        print(f"[error] Plan file not found: {args.plan}", file=sys.stderr)
        sys.exit(1)

    plan = load_comparison_plan(args.plan)

    execute_generation_plan(
        plan=plan,
        config_path=args.config,
        output_dir=args.output_dir,
        limit_override=args.limit,
        max_retries_override=args.max_retries,
        retry_delay_override=args.retry_delay,
        resume=not args.no_resume,
        conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
