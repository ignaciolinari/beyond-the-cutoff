#!/usr/bin/env python3
"""Run retrieval ablation study with live retrieval for ELO ranking.

This script tests different retrieval configurations with a fixed model,
generating responses that can be used for pairwise comparison and ELO ranking.

Supported configurations:
- Dense retrieval with different top_k values
- With/without cross-encoder reranking (BGE-Reranker-v2-M3 recommended)
- Retrieve-more-rerank-fewer pattern (e.g., retrieve 12, rerank to top 5)

Usage:
    # Run all retrieval conditions
    python scripts/run_retrieval_ablation.py \
        --config configs/default.yaml \
        --plan configs/evaluation/retrieval_ablation.yaml \
        --output-dir evaluation/results/retrieval_ablation/

    # Run specific conditions only
    python scripts/run_retrieval_ablation.py \
        --config configs/default.yaml \
        --plan configs/evaluation/retrieval_ablation.yaml \
        --output-dir evaluation/results/retrieval_ablation/ \
        --conditions dense_top4_baseline dense_top4_rerank

    # Quick test
    python scripts/run_retrieval_ablation.py \
        --config configs/default.yaml \
        --plan configs/evaluation/retrieval_ablation.yaml \
        --output-dir evaluation/results/retrieval_ablation/ \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.evaluation.metrics import evaluate_citations
from beyond_the_cutoff.evaluation.runner import (
    _call_with_retries,
    load_inference_from_yaml,
)
from beyond_the_cutoff.models import build_generation_client
from beyond_the_cutoff.retrieval.query import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCondition:
    """A retrieval configuration to test."""

    label: str
    top_k: int
    reranker: str | None = None
    rerank_top_k: int | None = None  # If set, keep only top N after reranking
    description: str = ""


@dataclass
class RerankerCache:
    """Cache for loaded reranker models to avoid reloading."""

    _cache: dict[str, Any] = field(default_factory=dict)

    def get_reranker(self, model_name: str | None) -> Any:
        """Get or load a reranker model."""
        if model_name is None:
            return None

        if model_name in self._cache:
            return self._cache[model_name]

        try:
            from sentence_transformers import CrossEncoder

            print(f"[info] Loading reranker: {model_name}", file=sys.stderr)
            reranker = CrossEncoder(model_name)
            # Set to eval mode for inference
            if hasattr(reranker, "model") and hasattr(reranker.model, "eval"):
                reranker.model.eval()
            self._cache[model_name] = reranker
            return reranker
        except Exception as e:
            print(f"[error] Failed to load reranker {model_name}: {e}", file=sys.stderr)
            return None


def load_ablation_plan(plan_path: Path) -> tuple[dict[str, Any], list[RetrievalCondition]]:
    """Load retrieval ablation plan from YAML."""
    with open(plan_path, encoding="utf-8") as f:
        plan = yaml.safe_load(f)

    conditions = []
    for cond in plan.get("conditions", []):
        conditions.append(
            RetrievalCondition(
                label=cond["label"],
                top_k=cond.get("top_k", 5),
                reranker=cond.get("reranker"),
                rerank_top_k=cond.get("rerank_top_k"),
                description=cond.get("description", ""),
            )
        )

    return plan, conditions


def load_questions(dataset_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load questions from evaluation dataset."""
    questions = []
    with open(dataset_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                questions.append(json.loads(line))
    return questions


def rerank_contexts(
    query: str,
    contexts: list[dict[str, Any]],
    reranker: Any,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank contexts using a cross-encoder and optionally truncate to top_k."""
    if reranker is None:
        return contexts

    try:
        import torch

        # Create query-context pairs
        pairs = [(query, ctx.get("text", ctx.get("rendered_context", ""))) for ctx in contexts]

        with torch.inference_mode():
            scores = reranker.predict(pairs)

        # Add reranker scores and sort
        for ctx, score in zip(contexts, scores, strict=False):
            ctx["reranker_score"] = float(score)

        contexts.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)

        # Truncate if rerank_top_k specified
        if top_k is not None and top_k < len(contexts):
            contexts = contexts[:top_k]

        return contexts
    except Exception as e:
        logger.warning(f"Reranking failed: {e}, using original order")
        return contexts


def run_retrieval_condition(
    *,
    condition: RetrievalCondition,
    pipeline: RAGPipeline,
    model_client: Any,
    questions: list[dict[str, Any]],
    output_path: Path,
    reranker_cache: RerankerCache,
    max_retries: int = 2,
    retry_delay: float = 15.0,
    resume: bool = True,
) -> dict[str, Any]:
    """Run evaluation for a single retrieval condition."""

    # Load existing if resuming
    existing: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    task_id = record.get("task_id")
                    if task_id:
                        existing[task_id] = record
        print(f"[info] Resuming: {len(existing)} already generated", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get reranker if specified
    reranker = reranker_cache.get_reranker(condition.reranker)
    final_top_k = condition.rerank_top_k or condition.top_k

    print(f"[info] Running condition: {condition.label}", file=sys.stderr)
    print(f"[info]   retrieve_k={condition.top_k}, final_k={final_top_k}", file=sys.stderr)
    print(f"[info]   reranker={condition.reranker or 'none'}", file=sys.stderr)

    results: list[dict[str, Any]] = []
    start_time = time.time()
    generated_count = 0
    skipped_count = 0

    # Open in append mode if resuming
    mode = "a" if resume and existing else "w"

    with open(output_path, mode, encoding="utf-8") as out_f:
        for idx, q in enumerate(questions, start=1):
            task_id = q.get("task_id", f"q_{idx}")
            instruction = q.get("instruction", "")

            # Skip if already done
            if task_id in existing:
                skipped_count += 1
                continue

            if not instruction:
                continue

            # Perform retrieval with this condition's settings
            retrieval_start = time.time()

            # Get more candidates if we're going to rerank
            retrieve_k = condition.top_k
            prepared = pipeline.prepare_prompt(
                instruction,
                top_k_override=retrieve_k,
            )

            # Apply reranking if configured
            retrieved_records = prepared.get("retrieved", [])
            if reranker and retrieved_records:
                retrieved_records = rerank_contexts(
                    query=instruction,
                    contexts=retrieved_records,
                    reranker=reranker,
                    top_k=condition.rerank_top_k,
                )
                # Rebuild contexts from reranked records
                contexts = [
                    rec.get("rendered_context", rec.get("text", "")) for rec in retrieved_records
                ]
            else:
                contexts = prepared["contexts"]
                # Truncate if needed (for non-reranked conditions with rerank_top_k)
                if condition.rerank_top_k and condition.rerank_top_k < len(contexts):
                    contexts = contexts[: condition.rerank_top_k]

            retrieval_time = time.time() - retrieval_start

            # Rebuild prompt with final contexts
            prompt_text = pipeline._build_prompt(
                instruction,
                contexts,
                pipeline.config.retrieval.max_context_chars,
            )

            # Generate response
            generation_start = time.time()
            response, error, error_cat = _call_with_retries(
                partial(model_client.generate, prompt_text),
                stage=f"generation ({task_id})",
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            generation_time = time.time() - generation_start

            response_text = ""
            if response:
                response_text = str(response.get("response", "")).strip()

            # Compute citation metrics
            citation_metrics = evaluate_citations(response_text, contexts)

            record = {
                "task_id": task_id,
                "instruction": instruction,
                "prompt_text": prompt_text,
                "response": response_text,
                "contexts": contexts,
                "retrieval_scores": [rec.get("score", 0) for rec in retrieved_records]
                if retrieved_records
                else [],
                "reranker_scores": [
                    rec.get("reranker_score")
                    for rec in retrieved_records
                    if rec.get("reranker_score")
                ]
                if reranker
                else [],
                "citation_metrics": citation_metrics,
                "condition_label": condition.label,
                "retrieval_config": {
                    "retrieve_k": condition.top_k,
                    "final_k": final_top_k,
                    "reranker": condition.reranker,
                    "rerank_top_k": condition.rerank_top_k,
                },
                "timing": {
                    "retrieval_seconds": retrieval_time,
                    "generation_seconds": generation_time,
                },
                "timestamp": datetime.now().isoformat(),
            }

            if error:
                record["error"] = error
                record["error_category"] = error_cat

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            results.append(record)
            generated_count += 1

            # Progress
            total = len(questions)
            processed = generated_count + skipped_count
            elapsed = time.time() - start_time
            rate = generated_count / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0

            print(
                f"\r  [{processed}/{total}] Gen: {generated_count}, Skip: {skipped_count}, "
                f"Rate: {rate:.1f}/s, ETA: {eta:.0f}s    ",
                end="",
                file=sys.stderr,
            )

    print(file=sys.stderr)
    total_time = time.time() - start_time

    return {
        "condition": condition.label,
        "generated_count": generated_count,
        "skipped_count": skipped_count,
        "total_time_seconds": total_time,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval ablation study for ELO ranking",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Project config file",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Retrieval ablation plan YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for responses",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Run only these conditions (by label)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit questions (for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries for generation",
    )

    args = parser.parse_args()

    # Load plan
    plan, all_conditions = load_ablation_plan(args.plan)

    # Filter conditions if specified
    if args.conditions:
        conditions = [c for c in all_conditions if c.label in args.conditions]
        if not conditions:
            print(
                f"[error] No matching conditions. Available: {[c.label for c in all_conditions]}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        conditions = all_conditions

    # Load config and initialize pipeline
    project_config = load_config(args.config)

    # Load model config
    model_config_path = Path(
        plan.get("model_config", "configs/lora_science_v1_rag_trained_ollama.yaml")
    )
    if not model_config_path.is_absolute():
        model_config_path = args.plan.parent / model_config_path

    model_cfg = load_inference_from_yaml(model_config_path)
    model_client = build_generation_client(model_cfg)

    # Initialize RAG pipeline
    index_dir = project_config.paths.processed_data / "faiss_index"
    if not index_dir.exists():
        index_dir = project_config.paths.external_data / "index"

    if not index_dir.exists():
        print(f"[error] FAISS index not found at {index_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Loading RAG pipeline from {index_dir}", file=sys.stderr)
    pipeline = RAGPipeline(
        project_config,
        index_path=index_dir / "index.faiss",
        mapping_path=index_dir / "mapping.tsv",
    )

    # Load questions
    dataset_path = Path(plan.get("dataset", "evaluation/datasets/eval_dataset.jsonl"))
    if not dataset_path.is_absolute():
        dataset_path = args.plan.parent / dataset_path

    questions = load_questions(dataset_path, limit=args.limit)

    print("=" * 80, file=sys.stderr)
    print("[info] Retrieval Ablation Study", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"[info] Model: {model_cfg.model}", file=sys.stderr)
    print(f"[info] Conditions: {len(conditions)}", file=sys.stderr)
    print(f"[info] Questions: {len(questions)}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    # Run each condition
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    reranker_cache = RerankerCache()

    for i, condition in enumerate(conditions, start=1):
        print(f"\n[info] Condition {i}/{len(conditions)}: {condition.label}", file=sys.stderr)

        output_path = args.output_dir / f"{condition.label}.jsonl"

        result = run_retrieval_condition(
            condition=condition,
            pipeline=pipeline,
            model_client=model_client,
            questions=questions,
            output_path=output_path,
            reranker_cache=reranker_cache,
            max_retries=args.max_retries,
            resume=not args.no_resume,
        )

        all_results.append(result)

    # Save metadata
    metadata = {
        "plan_path": str(args.plan),
        "model": model_cfg.model,
        "conditions": [c.label for c in conditions],
        "questions_count": len(questions),
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    metadata_path = args.output_dir / "ablation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80, file=sys.stderr)
    print("[info] Ablation study complete!", file=sys.stderr)
    print(f"[info] Results saved to: {args.output_dir}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print("\nNext steps:", file=sys.stderr)
    print("1. Run pairwise evaluation:", file=sys.stderr)
    print(
        "   python scripts/run_pairwise_evaluation.py --results ...",
        file=sys.stderr,
    )
    print("2. Compute ELO rankings:", file=sys.stderr)
    print(
        "   python scripts/compute_elo_rankings.py --comparisons ...",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
