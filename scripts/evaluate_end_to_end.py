#!/usr/bin/env python3
"""End-to-end evaluation with live retrieval.

This script performs full RAG pipeline evaluation:
1. Takes questions from evaluation dataset
2. Performs LIVE retrieval (not using pre-computed contexts)
3. Generates responses using the specified model
4. Evaluates responses with the judge

This is different from the main evaluation which uses pre-computed contexts.
Use this for:
- Validating the full pipeline works correctly
- Testing different retrieval configurations
- Final end-to-end benchmarking

Usage:
    # Basic end-to-end evaluation
    python scripts/evaluate_end_to_end.py \
        --config configs/default.yaml \
        --model-config configs/lora_science_v1_rag_trained_ollama.yaml \
        --output evaluation/results/end_to_end/

    # With custom retrieval settings
    python scripts/evaluate_end_to_end.py \
        --config configs/default.yaml \
        --model-config configs/lora_science_v1_rag_trained_ollama.yaml \
        --top-k 8 \
        --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --output evaluation/results/end_to_end_rerank/

    # Compare with pre-computed contexts
    python scripts/evaluate_end_to_end.py \
        --config configs/default.yaml \
        --model-config configs/lora_science_v1_rag_trained_ollama.yaml \
        --compare-precomputed \
        --output evaluation/results/end_to_end/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.evaluation.metrics import evaluate_citations, normalize_contexts
from beyond_the_cutoff.evaluation.runner import (
    _call_with_retries,
    load_inference_from_yaml,
    load_judge_prompt,
    parse_judge_output,
    render_judge_prompt,
)
from beyond_the_cutoff.models import build_generation_client
from beyond_the_cutoff.retrieval.query import RAGPipeline


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""

    top_k: int = 5
    reranker: str | None = None
    rerank_top_k: int | None = None  # If set, retrieve more then rerank to this


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


def compute_context_overlap(
    live_contexts: list[str],
    precomputed_contexts: list[str],
) -> dict[str, Any]:
    """Compute overlap between live and precomputed contexts."""
    # Normalize contexts for comparison
    live_set = {c.strip().lower()[:200] for c in live_contexts}
    precomputed_set = {c.strip().lower()[:200] for c in precomputed_contexts}

    intersection = live_set & precomputed_set
    union = live_set | precomputed_set

    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        "live_count": len(live_contexts),
        "precomputed_count": len(precomputed_contexts),
        "overlap_count": len(intersection),
        "jaccard_similarity": jaccard,
    }


def run_end_to_end_evaluation(
    *,
    config_path: Path,
    model_config_path: Path,
    judge_config_path: Path,
    judge_inference_path: Path,
    dataset_path: Path,
    output_dir: Path,
    retrieval_config: RetrievalConfig,
    limit: int | None = None,
    max_retries: int = 2,
    retry_delay: float = 15.0,
    compare_precomputed: bool = False,
) -> dict[str, Any]:
    """Run end-to-end evaluation with live retrieval."""

    # Load configurations
    project_config = load_config(config_path)
    model_cfg = load_inference_from_yaml(model_config_path)
    judge_inference_cfg = load_inference_from_yaml(judge_inference_path)
    judge_prompt = load_judge_prompt(judge_config_path)

    # Override retrieval config if specified
    if retrieval_config.top_k != project_config.retrieval.top_k:
        project_config.retrieval.top_k = retrieval_config.top_k
    if retrieval_config.reranker:
        project_config.retrieval.reranker_model = retrieval_config.reranker

    # Initialize RAG pipeline
    index_dir = project_config.paths.processed_data / "faiss_index"
    if not index_dir.exists():
        # Try alternate location
        index_dir = project_config.paths.external_data / "index"

    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found. Tried:\n"
            f"  - {project_config.paths.processed_data / 'faiss_index'}\n"
            f"  - {project_config.paths.external_data / 'index'}\n"
            "Run scripts/ingest_and_index.py first."
        )

    print(f"[info] Loading RAG pipeline from {index_dir}", file=sys.stderr)
    pipeline = RAGPipeline(
        project_config,
        index_path=index_dir / "index.faiss",
        mapping_path=index_dir / "mapping.tsv",
    )

    # Build clients
    model_client = build_generation_client(model_cfg)
    judge_client = build_generation_client(judge_inference_cfg)

    # Load questions
    questions = load_questions(dataset_path, limit=limit)
    total_questions = len(questions)

    print("=" * 80, file=sys.stderr)
    print("[info] End-to-End Evaluation with Live Retrieval", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"[info] Model: {model_cfg.model}", file=sys.stderr)
    print(f"[info] Retrieval top_k: {retrieval_config.top_k}", file=sys.stderr)
    if retrieval_config.reranker:
        print(f"[info] Reranker: {retrieval_config.reranker}", file=sys.stderr)
    print(f"[info] Questions: {total_questions}", file=sys.stderr)
    print(f"[info] Index: {index_dir}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    start_time = time.time()

    for idx, question_data in enumerate(questions, start=1):
        task_id = question_data.get("task_id", f"q_{idx}")
        instruction = question_data.get("instruction", "")

        if not instruction:
            print(f"[warn] Skipping {task_id}: no instruction", file=sys.stderr)
            continue

        # Perform live retrieval
        retrieval_start = time.time()
        prepared = pipeline.prepare_prompt(
            instruction,
            top_k_override=retrieval_config.top_k,
        )
        retrieval_time = time.time() - retrieval_start

        live_contexts = prepared["contexts"]
        prompt_text = prepared["prompt"]

        # Compare with precomputed if requested
        context_comparison = None
        if compare_precomputed:
            precomputed = question_data.get("rag", {}).get("contexts", [])
            precomputed_normalized = normalize_contexts(precomputed)
            context_comparison = compute_context_overlap(live_contexts, precomputed_normalized)

        # Generate response
        generation_start = time.time()
        response, gen_error, gen_error_cat = _call_with_retries(
            partial(model_client.generate, prompt_text),
            stage=f"generation ({task_id})",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        generation_time = time.time() - generation_start

        response_text = ""
        if response:
            response_text = str(response.get("response", "")).strip()

        # Evaluate with judge
        judge_start = time.time()
        has_contexts = len(live_contexts) > 0

        judge_prompt_text = render_judge_prompt(
            judge_prompt.prompt,
            instruction,
            live_contexts,
            response_text,
            has_contexts=has_contexts,
        )

        judge_response, judge_error, judge_error_cat = _call_with_retries(
            partial(judge_client.generate, judge_prompt_text),
            stage=f"judging ({task_id})",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        judge_time = time.time() - judge_start

        judge_payload = {}
        if judge_response:
            judge_payload = parse_judge_output(str(judge_response.get("response", "")))

        # Compute citation metrics
        citation_metrics = evaluate_citations(response_text, live_contexts)

        # Build result record
        result = {
            "task_id": task_id,
            "instruction": instruction,
            "response": response_text,
            "contexts": live_contexts,
            "retrieval_scores": prepared.get("scores", []),
            "judge_scores": judge_payload.get("scores", {}),
            "judge_verdict": judge_payload.get("verdict"),
            "citation_metrics": citation_metrics,
            "timing": {
                "retrieval_seconds": retrieval_time,
                "generation_seconds": generation_time,
                "judge_seconds": judge_time,
                "total_seconds": retrieval_time + generation_time + judge_time,
            },
        }

        if context_comparison:
            result["context_comparison"] = context_comparison

        if gen_error:
            result["generation_error"] = gen_error
        if judge_error:
            result["judge_error"] = judge_error

        results.append(result)

        # Progress
        elapsed = time.time() - start_time
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (total_questions - idx) / rate if rate > 0 else 0

        print(
            f"\r[{idx}/{total_questions}] " f"Rate: {rate:.2f}/s, ETA: {eta:.0f}s    ",
            end="",
            file=sys.stderr,
        )

    print(file=sys.stderr)
    total_time = time.time() - start_time

    # Compute summary statistics
    judge_scores_list = [r["judge_scores"] for r in results if r.get("judge_scores")]

    summary = {
        "model": model_cfg.model,
        "retrieval_top_k": retrieval_config.top_k,
        "reranker": retrieval_config.reranker,
        "total_questions": total_questions,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
    }

    # Aggregate judge scores
    if judge_scores_list:
        score_keys = set()
        for scores in judge_scores_list:
            score_keys.update(scores.keys())

        for key in score_keys:
            values = [s.get(key) for s in judge_scores_list if s.get(key) is not None]
            if values:
                summary[f"mean_{key}"] = sum(values) / len(values)

    # Aggregate timing
    timing_keys = ["retrieval_seconds", "generation_seconds", "judge_seconds", "total_seconds"]
    for key in timing_keys:
        values = [r["timing"].get(key, 0) for r in results]
        if values:
            summary[f"mean_{key}"] = sum(values) / len(values)

    # Context comparison stats
    if compare_precomputed:
        jaccard_values = [
            r["context_comparison"]["jaccard_similarity"]
            for r in results
            if r.get("context_comparison")
        ]
        if jaccard_values:
            summary["mean_context_jaccard"] = sum(jaccard_values) / len(jaccard_values)

    # Save results
    details_path = output_dir / "details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("-" * 80, file=sys.stderr)
    print("[info] Evaluation complete", file=sys.stderr)
    print(f"[info] Total time: {total_time:.1f}s", file=sys.stderr)
    print(f"[info] Results saved to: {output_dir}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation with live retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Project config file",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Model inference config",
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=Path("configs/judges/scientific_default_rag.yaml"),
        help="Judge prompt config",
    )
    parser.add_argument(
        "--judge-inference",
        type=Path,
        default=Path("configs/judges/dataset_quality_judge.yaml"),
        help="Judge inference config",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/eval_dataset.jsonl"),
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of contexts to retrieve",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=None,
        help="Cross-encoder reranker model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    parser.add_argument(
        "--compare-precomputed",
        action="store_true",
        help="Compare live retrieval with precomputed contexts",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries for API calls",
    )

    args = parser.parse_args()

    retrieval_config = RetrievalConfig(
        top_k=args.top_k,
        reranker=args.reranker,
    )

    run_end_to_end_evaluation(
        config_path=args.config,
        model_config_path=args.model_config,
        judge_config_path=args.judge_config,
        judge_inference_path=args.judge_inference,
        dataset_path=args.dataset,
        output_dir=args.output,
        retrieval_config=retrieval_config,
        limit=args.limit,
        max_retries=args.max_retries,
        compare_precomputed=args.compare_precomputed,
    )


if __name__ == "__main__":
    main()
