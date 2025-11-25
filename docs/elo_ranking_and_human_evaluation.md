# ELO Ranking & Human Evaluation System

This document describes the ELO-based model ranking system, automated pairwise evaluation, and human evaluation protocol implemented in this project.

## Overview

The system provides three complementary evaluation mechanisms:

1. **ELO Ranking**: A chess-inspired rating system for comparing models based on pairwise matchups
2. **Automated Pairwise Evaluation**: Multi-judge LLM-based comparison for fully automated evaluation
3. **Human Evaluation Protocol**: Tools for collecting human judgments and validating judge model reliability

## ELO Ranking System

### How It Works

The ELO system rates models based on head-to-head comparisons:

1. All models start with an initial rating (default: 1500)
2. When Model A defeats Model B, A gains points and B loses points
3. The magnitude of rating change depends on the **expected outcome**:
   - Upsets (lower-rated beats higher-rated) cause larger rating changes
   - Expected wins cause smaller changes
4. Ties split the difference (each gets 0.5 score)

### Key Parameters

- **K-Factor** (default: 32): Controls rating volatility. Higher = more change per game
- **Initial Rating** (default: 1500): Starting rating for new models
- **Bootstrap Samples** (default: 1000): Number of samples for confidence intervals

### Usage

#### From Python

```python
from beyond_the_cutoff.evaluation.elo_ranking import (
    PairwiseComparison,
    compute_elo_rankings,
    head_to_head_matrix,
)

# Create comparisons (from judge evaluations or human annotations)
comparisons = [
    PairwiseComparison(
        model_a="qwen-base",
        model_b="qwen-rag-trained",
        outcome="win_b",  # "win_a", "win_b", or "tie"
        task_id="task_001",
        annotator="judge_7b",
    ),
    # ... more comparisons
]

# Compute rankings with confidence intervals
leaderboard, metadata = compute_elo_rankings(
    comparisons,
    k_factor=32,
    bootstrap_samples=1000,
)

# Print results
for i, rating in enumerate(leaderboard):
    print(f"{i+1}. {rating.model}: {rating.rating:.0f} "
          f"[{rating.confidence_lower:.0f}-{rating.confidence_upper:.0f}]")
```

#### From CLI

```bash
# From JSONL comparisons file
python scripts/compute_elo_rankings.py \
    --comparisons evaluation/comparisons.jsonl \
    --output evaluation/results/leaderboard.json

# From human annotation batches
python scripts/compute_elo_rankings.py \
    --annotations-dir evaluation/human_annotations \
    --output evaluation/results/human_leaderboard.json \
    --format markdown

# From evaluation result files (creates synthetic comparisons)
python scripts/compute_elo_rankings.py \
    --eval-results results/model_a.json results/model_b.json results/model_c.json \
    --output evaluation/results/auto_leaderboard.json \
    --head-to-head evaluation/results/h2h_matrix.json
```

### Output Format

```json
{
  "leaderboard": [
    {
      "model": "qwen-rag-trained-ft+rag",
      "rating": 1623.4,
      "games_played": 150,
      "wins": 85,
      "losses": 45,
      "ties": 20,
      "win_rate": 0.633,
      "confidence_lower": 1589.2,
      "confidence_upper": 1658.1
    }
  ],
  "metadata": {
    "n_comparisons": 450,
    "n_models": 6,
    "k_factor": 32,
    "bootstrap_samples": 1000,
    "confidence_level": 0.95
  }
}
```

## Automated Pairwise Evaluation

### Overview

For fully automated evaluation without human annotation, the system supports multi-judge pairwise comparison using capable locally-runnable LLMs.

### Recommended Judge Models

The following 8B parameter models provide strong judging capabilities while remaining locally runnable:

| Model | Size | Strengths |
|-------|------|-----------|
| `qwen2.5:7b-instruct-q4_K_M` | 4.4GB | Fast inference, good instruction following |
| `qwen3:8b` | 5.2GB | Latest generation, enhanced reasoning, Apache 2.0 |
| `llama3.1:8b` | 4.9GB | 128K context, state-of-the-art reasoning |

### Usage

#### From CLI

```bash
# Run with evaluation plan (recommended)
python scripts/run_pairwise_evaluation.py \
    --plan configs/evaluation/pairwise_evaluation_plan.yaml \
    --output evaluation/results/pairwise_rankings

# Ad-hoc comparison between models using result directories
python scripts/run_pairwise_evaluation.py \
    --results base=evaluation/results/base_baseline_0p5b \
    --results rag=evaluation/results/rag_baseline_0p5b \
    --judge configs/judges/pairwise_qwen3_8b.yaml \
    --judge configs/judges/pairwise_llama31_8b.yaml \
    --output evaluation/results/pairwise_rankings

# Use single judge configuration
python scripts/run_pairwise_evaluation.py \
    --results baseline=evaluation/results/base_baseline_0p5b \
    --results finetuned=evaluation/results/lora_science_0p5b_ft_only \
    --judge configs/judges/pairwise_qwen3_8b.yaml \
    --output evaluation/results/pairwise_rankings
```

#### From Python

```python
from beyond_the_cutoff.evaluation.pairwise_judge import (
    PairwiseJudge,
    MultiJudgeEvaluator,
    PairwiseJudgeConfig,
    compute_consensus,
)
from pathlib import Path

# Load judge configuration
config = PairwiseJudgeConfig.from_yaml("configs/judges/pairwise_qwen3_8b.yaml")
judge = PairwiseJudge(config)

# Single judge comparison
result = judge.compare(
    question="What are the key findings of the 2025 climate report?",
    response_a="Model A's answer...",
    response_b="Model B's answer...",
)
print(f"Verdict: {result.verdict}, Confidence: {result.confidence}")

# Multi-judge evaluation with consensus
# Note: Using 8B judges only to avoid self-preference bias with Qwen 2.5 7B generator
evaluator = MultiJudgeEvaluator.from_yaml_files([
    Path("configs/judges/pairwise_qwen3_8b.yaml"),
    Path("configs/judges/pairwise_llama31_8b.yaml"),
])

result = evaluator.evaluate_pair(
    task_id="task_001",
    model_a="model_base",
    model_b="model_finetuned",
    question="...",
    response_a="...",
    response_b="...",
)
print(f"Consensus: {result.consensus_verdict} ({result.agreement_rate:.0%} agreement)")
```
```

### Key Features

- **Multi-judge consensus**: Uses multiple judge models with majority voting for reliability
- **Position debiasing**: Automatically swaps response positions to avoid order bias
- **Structured output**: JSON-based judge responses with fallback keyword detection
- **Configurable retries**: Handles transient Ollama API failures gracefully

### Judge Configuration

Judge configs are stored in `configs/judges/`:

```yaml
# configs/judges/pairwise_qwen3_8b.yaml
name: pairwise_qwen3_8b
description: Qwen3 8B judge with enhanced reasoning

inference:
  provider: ollama
  model: qwen3:8b
  base_url: http://localhost:11434
  temperature: 0.1
  max_tokens: 1024
  timeout: 120

max_retries: 3
retry_delay: 2.0
```

## Human Evaluation Protocol

### Purpose

Human evaluation serves to:
1. **Validate judge model reliability** by comparing human vs judge verdicts
2. **Provide ground truth** for ambiguous cases
3. **Calculate inter-annotator agreement** to ensure annotation quality

### Workflow

```
1. Sample tasks for annotation
      ↓
2. Create annotation batches per annotator
      ↓
3. Annotators complete tasks via UI
      ↓
4. Compute inter-annotator agreement
      ↓
5. Compare human vs judge verdicts
      ↓
6. Export to ELO system
```

### Creating Annotation Tasks

```python
from beyond_the_cutoff.evaluation.human_evaluation import (
    create_pairwise_tasks,
    sample_for_annotation,
    SamplingStrategy,
)

# Load model predictions
model_predictions = {
    "model_a": [...],  # List of prediction dicts with task_id, model_answer, question
    "model_b": [...],
    "model_c": [...],
}

# Create pairwise comparison tasks
tasks = create_pairwise_tasks(
    model_predictions,
    n_per_pair=50,  # 50 comparisons per model pair
    seed=42,
)

# Or sample from existing predictions
sampled = sample_for_annotation(
    all_predictions,
    n_samples=100,
    strategy=SamplingStrategy.STRATIFIED,  # Equal samples per task_type
)
```

### Running the Annotation UI

```bash
# Start the Streamlit annotation app
streamlit run apps/human_annotation.py

# Or specify custom paths
streamlit run apps/human_annotation.py -- \
    --batch-dir evaluation/human_annotations \
    --tasks-file evaluation/datasets/pairwise_tasks.jsonl
```

### Inter-Annotator Agreement Metrics

The system computes:

- **Cohen's Kappa**: Agreement between two annotators (corrected for chance)
- **Fleiss' Kappa**: Agreement among multiple annotators
- **Raw Agreement**: Percentage of unanimous decisions

```python
from beyond_the_cutoff.evaluation.human_evaluation import (
    compute_agreement_stats,
    human_judge_correlation,
)

# Compute agreement between human annotators
stats = compute_agreement_stats(annotations_by_task)
print(f"Fleiss' Kappa: {stats['fleiss_kappa']:.3f}")
print(f"Raw Agreement: {stats['raw_agreement']:.1%}")

# Compare human judgments to judge model
correlation = human_judge_correlation(human_annotations, judge_verdicts)
print(f"Human-Judge Agreement: {correlation['agreement_rate']:.1%}")
print(f"Cohen's Kappa: {correlation['cohens_kappa']:.3f}")
```

### Interpretation Guidelines

| Kappa Value | Interpretation |
|-------------|----------------|
| < 0.0       | Poor agreement (worse than chance) |
| 0.0 - 0.20  | Slight agreement |
| 0.21 - 0.40 | Fair agreement |
| 0.41 - 0.60 | Moderate agreement |
| 0.61 - 0.80 | Substantial agreement |
| 0.81 - 1.00 | Almost perfect agreement |

## Integration Example

Complete workflow for model comparison:

```bash
# 1. Run evaluations to get predictions
python scripts/evaluate_models.py --config configs/default.yaml

# 2. Create pairwise comparison tasks
python -c "
from beyond_the_cutoff.evaluation.human_evaluation import create_pairwise_tasks
import json

# Load predictions from different models
model_preds = {...}
tasks = create_pairwise_tasks(model_preds, n_per_pair=30)

with open('evaluation/datasets/pairwise_tasks.jsonl', 'w') as f:
    for task in tasks:
        f.write(json.dumps(task.as_dict()) + '\n')
"

# 3. Run human annotation UI
streamlit run apps/human_annotation.py

# 4. Compute ELO rankings from annotations
python scripts/compute_elo_rankings.py \
    --annotations-dir evaluation/human_annotations \
    --output evaluation/results/final_leaderboard.json \
    --format markdown

# 5. View results
cat evaluation/results/final_leaderboard.json
```

## Files Reference

| File | Purpose |
|------|---------|
| `src/beyond_the_cutoff/evaluation/elo_ranking.py` | Core ELO algorithm and data structures |
| `src/beyond_the_cutoff/evaluation/pairwise_judge.py` | Automated pairwise evaluation with LLM judges |
| `src/beyond_the_cutoff/evaluation/human_evaluation.py` | Human evaluation protocol and agreement metrics |
| `apps/human_annotation.py` | Streamlit UI for human annotators |
| `scripts/compute_elo_rankings.py` | CLI for computing rankings |
| `scripts/run_pairwise_evaluation.py` | CLI for automated pairwise evaluation |
| `configs/judges/pairwise_qwen3_8b.yaml` | Qwen3 8B judge configuration |
| `configs/judges/pairwise_llama31_8b.yaml` | Llama 3.1 8B judge configuration |
| `configs/judges/pairwise_qwen7b.yaml` | Qwen 2.5 7B judge (excluded from main experiment to avoid self-preference bias) |
| `configs/evaluation/pairwise_evaluation_plan.yaml` | Full evaluation plan with all model pairs |
| `tests/test_elo_ranking.py` | Tests for ELO system |
| `tests/test_pairwise_judge.py` | Tests for pairwise judge system |
| `tests/test_human_evaluation.py` | Tests for human evaluation |
