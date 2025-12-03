# Future Configuration Files

This directory contains configuration files for planned experiments that were **not executed** as part of the main Qwen 2.5 0.5B experiment.

## Configurations

| Config | Purpose | Status |
|--------|---------|--------|
| `end_to_end.yaml` | Live retrieval validation (not pre-computed contexts) | Ready, not executed |
| `pairwise_evaluation_plan.yaml` | Full ELO tournament between all model pairs | Ready, not executed |
| `retrieval_ablation.yaml` | Optimization of retrieval parameters (top_k, rerankers) | Ready, not executed |
| `quantization_comparison.yaml` | **Duplicate** - see `configs/evaluation/quantization_comparison.yaml` | To be removed |

## Why These Weren't Executed

The main experiment focused on the core research question:

> **Does fine-tuning improve RAG performance for post-cutoff knowledge in small language models?**

The answer was **marginal improvement** (+1.3% pass rate, p=0.35), which deprioritized further optimization studies. These configs remain available for:

1. **Larger model experiments** where fine-tuning may show more benefit
2. **Production deployment** where retrieval optimization matters
3. **Ablation studies** for academic rigor

## How to Use

### End-to-End Evaluation

```bash
python scripts/future/evaluate_end_to_end.py \
    --plan configs/future/end_to_end.yaml \
    --output-dir evaluation/results/end_to_end/
```

### Retrieval Ablation

```bash
python scripts/future/run_retrieval_ablation.py \
    --plan configs/future/retrieval_ablation.yaml \
    --output-dir evaluation/results/retrieval_ablation/
```

### ELO Tournament

```bash
python scripts/future/run_pairwise_evaluation.py \
    --plan configs/future/pairwise_evaluation_plan.yaml \
    --output evaluation/results/elo_rankings/
```

## Related Documentation

- [ELO & Human Evaluation](../../docs/future/elo_and_human_eval.md)
- [Methodology - Future Phases](../../docs/experiment/methodology.md)
