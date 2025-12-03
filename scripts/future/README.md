# Future Work Scripts

This directory contains scripts for planned features that were **not executed** as part of the main Qwen 2.5 0.5B experiment. These are fully implemented and ready to use for future experiments.

## Why These Are "Future Work"

The main experiment focused on:
1. **Six-condition comparison** → Completed ✅
2. **Pairwise comparison of best candidates** (Base+RAG vs FT+RAG) → Completed ✅
3. **Quantization impact analysis** (Q4_K_M vs F16) → Completed ✅

The scripts in this directory extend beyond that scope to enable more sophisticated evaluation workflows.

---

## Scripts Overview

### Evaluation Orchestration

| Script | Purpose | Status |
|--------|---------|--------|
| `run_evaluation_pipeline.py` | Unified orchestrator for full-comparison, quantization, retrieval-ablation, and end-to-end workflows | Ready, not executed |
| `evaluate_end_to_end.py` | Validates full RAG pipeline with **live retrieval** (not pre-computed contexts) | Ready, not executed |
| `run_retrieval_ablation.py` | Tests different retrieval configurations (top_k, rerankers) | Ready, not executed |

### ELO Ranking & Tournaments

| Script | Purpose | Status |
|--------|---------|--------|
| `compute_elo_rankings.py` | Computes ELO rankings from pairwise comparisons with bootstrap confidence intervals | Ready, not executed |
| `run_pairwise_evaluation.py` | Runs multi-judge pairwise evaluation for ELO tournaments | Ready, not executed |
| `pairwise_tournament.py` | Full tournament between all model pairs | Ready, not executed |

### Dataset Quality

| Script | Purpose | Status |
|--------|---------|--------|
| `evaluate_dataset_quality.py` | LLM-based quality validation of generated QA pairs | **Used during dataset generation** |
| `evaluation_harness.py` | Computes automatic metrics (BERTScore, BLEU, etc.) | Ready |

### Legacy/Deprecated

| Script | Purpose | Status |
|--------|---------|--------|
| `evaluate_models.py` | Single-model evaluation (replaced by `compare_models.py`) | Deprecated |
| `judge_comparison_experiment.py` | Compare different judge models | Ready, not executed |

---

## How to Use These Scripts

### End-to-End Validation

After identifying the best model from the six-condition experiment, validate with live retrieval:

```bash
python scripts/future/evaluate_end_to_end.py \
    --config configs/default.yaml \
    --model-config configs/models/lora_rag_trained.yaml \
    --output evaluation/results/end_to_end/
```

### Retrieval Ablation

Optimize retrieval configuration using ELO ranking:

```bash
python scripts/future/run_retrieval_ablation.py \
    --config configs/default.yaml \
    --plan configs/future/retrieval_ablation.yaml \
    --output-dir evaluation/results/retrieval_ablation/
```

### ELO Rankings

Compute rankings from existing pairwise comparisons:

```bash
python scripts/future/compute_elo_rankings.py \
    --comparisons evaluation/results/pairwise/comparisons.jsonl \
    --output evaluation/results/elo_leaderboard.json
```

---

## Why These Weren't Executed

1. **Scope**: The main experiment answered the core research question (RAG vs Fine-tuning for small models)
2. **Resources**: Live retrieval evaluation is computationally expensive
3. **Diminishing returns**: With the null result on pairwise comparison (p=0.35), further refinement was deprioritized
4. **Future work**: These are better suited for larger model experiments where fine-tuning may show more benefit

---

## Related Documentation

- [ELO & Human Evaluation](../../docs/future/elo_and_human_eval.md) - Detailed methodology
- [Methodology](../../docs/experiment/methodology.md) - Evaluation phases including future work
- [Pipeline Reference](../../docs/reference/pipeline.md) - Full pipeline documentation
