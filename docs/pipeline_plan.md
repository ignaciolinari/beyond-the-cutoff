
# Complete Pipeline Plan: From Setup to Results

This document provides a step-by-step guide with all commands needed to run the complete pipeline from initial setup through final evaluation and visualization.

Note: The offline dataset generation system has been refactored into a modular architecture with separate components for improved maintainability. All existing commands and interfaces remain backward compatible.

## Table of Contents

1. [Initial Setup](#1-initial-setup)
2. [Data Collection](#2-data-collection)
3. [Data Ingestion & Indexing](#3-data-ingestion--indexing)
4. [Offline Dataset Generation](#4-offline-dataset-generation)
5. [Fine-Tuning](#5-fine-tuning)
6. [Model Deployment](#6-model-deployment)
7. [Evaluation](#7-evaluation)
   - [7.5 Two-Phase Evaluation Pipeline](#75-two-phase-evaluation-pipeline)
   - [7.6 Unified Pipeline Orchestrator](#76-unified-pipeline-orchestrator)
8. [Results Analysis](#8-results-analysis)
   - [8.4 ELO Rankings (Automated)](#84-compute-elo-rankings-automated)
   - [8.5 Human Evaluation (Optional)](#85-human-evaluation-optional)
9. [Post-Experiment Optimization](#9-post-experiment-optimization)
   - [9.1 Quantization Analysis](#91-quantization-analysis)
   - [9.2 Retrieval Optimization](#92-retrieval-optimization)
   - [9.3 End-to-End Validation](#93-end-to-end-validation)
10. [Verification & Testing](#10-verification--testing)

---

## 1. Initial Setup

### 1.1 Environment Bootstrap

```bash
# Navigate to project root
cd /Users/ignacio/Repos/beyond-the-cutoff

# Bootstrap Python environment (installs dependencies, sets up pre-commit hooks)
python scripts/bootstrap_env.py

# Activate virtual environment
source .venv/bin/activate

# Verify installation
python -c "from beyond_the_cutoff import load_config; print('Setup successful!')"
```

### 1.2 Ollama Setup

```bash
# Ensure Ollama is installed and running
# Check if Ollama daemon is running
curl http://localhost:11434/api/tags

# Pull base models for inference
ollama pull qwen2.5:0.5b-instruct
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M

# Verify models are available
ollama list
```

### 1.3 Optional: Prefetch Hugging Face Models

```bash
# Prefetch base models for fine-tuning (optional, speeds up Colab/Kaggle sync)
python scripts/prefetch_models.py \
  --cache-dir .cache/huggingface \
  Qwen/Qwen2.5-0.5B-Instruct \
  Qwen/Qwen2.5-3B-Instruct
```

### 1.4 Verify Configuration

```bash
# Test configuration loading
python -c "
from beyond_the_cutoff import load_config
config = load_config('configs/default.yaml')
print(f'Base model: {config.fine_tuning.base_model}')
print(f'Inference model: {config.inference.model}')
print(f'Generator model: {config.dataset_generation.generator.model}')
"
```

---

## 2. Data Collection

### 2.1 Fetch arXiv Papers

```bash
# Download 100+ papers from arXiv (2025, post-cutoff)
python scripts/fetch_arxiv_corpus.py \
  --contact-email your.email@example.com \
  --total 100 \
  --output-dir data/raw/arxiv_2025

# Optional: Filter by specific categories
python scripts/fetch_arxiv_corpus.py \
  --contact-email your.email@example.com \
  --total 120 \
  --category cs.AI \
  --category cs.CL \
  --category cs.LG \
  --category stat.ML \
  --output-dir data/raw/arxiv_2025

# Verify downloaded papers
ls -lh data/raw/arxiv_2025/*.pdf | wc -l
```

### 2.2 Verify Data Quality

```bash
# Check downloaded metadata
head -n 5 data/raw/arxiv_2025/metadata.jsonl

# Inspect manifest
cat data/raw/arxiv_2025/manifest.json | jq '.document_count'
```

---

## 3. Data Ingestion & Indexing

### 3.1 Ingest PDFs and Build Index

```bash
# Process PDFs, extract text, build FAISS index
python scripts/ingest_and_index.py \
  --config configs/default.yaml

# Note: PDF conversion now includes automatic extraction quality analysis.
# Each converted text file will have an accompanying .quality.json file
# with metrics like extraction success rate, content volume, structural
# integrity, and a confidence score (0.0-1.0).

# Optional: Specify custom source directory
python scripts/ingest_and_index.py \
  --config configs/default.yaml \
  --source data/raw/arxiv_2025

# Optional: Skip per-page JSONL sidecars (faster, less detailed)
python scripts/ingest_and_index.py \
  --config configs/default.yaml \
  --no-page-sidecars
```

### 3.2 Verify Ingestion Results

```bash
# Check processed manifest
cat data/processed/manifest.json | jq '.document_count, .total_chunks'

# Check metadata catalog
head -n 5 data/processed/metadata_catalog.csv

# Verify FAISS index exists
ls -lh data/external/index/

# Inspect processed text samples
head -n 3 data/processed/*.jsonl
```

### 3.3 Rebuild Metadata Catalog (if needed)

```bash
# Rebuild catalog from existing manifest
python scripts/build_metadata_catalog.py \
  --config configs/default.yaml

# With custom output prefix
python scripts/build_metadata_catalog.py \
  --config configs/default.yaml \
  --output-prefix data/processed/custom_catalog
```

---

## 4. Offline Dataset Generation

The offline dataset generation system uses a modular architecture with separate components for parsing, validation, citation enforcement, and document metadata management. All existing commands remain backward compatible.

### 4.1 Generate Offline Tasks and Dataset

```bash
# Generate QA/summary/citation tasks using 7B generator model
python scripts/generate_offline_dataset.py \
  --config configs/default.yaml \
  --output evaluation/datasets/offline_dataset.jsonl \
  --raw-tasks evaluation/datasets/offline_tasks.jsonl

# Optional: Limit number of documents for testing
python scripts/generate_offline_dataset.py \
  --config configs/default.yaml \
  --max-docs 10 \
  --output evaluation/datasets/offline_dataset_sample.jsonl

# Optional: Use custom index directory
python scripts/generate_offline_dataset.py \
  --config configs/default.yaml \
  --index-dir data/external/index
```

### 4.2 Verify Generated Dataset

```bash
# Check dataset statistics
python scripts/utility/analyze_offline_dataset.py \
  evaluation/datasets/offline_dataset.jsonl

# Count tasks by type
cat evaluation/datasets/offline_dataset.jsonl | \
  jq -r '.task_type' | sort | uniq -c

# Inspect sample tasks
head -n 1 evaluation/datasets/offline_dataset.jsonl | jq '.'

# View raw generator output
head -n 1 evaluation/datasets/offline_tasks.jsonl | jq '.'
```

### 4.3 Quality Validation with LLM Judge

**Before proceeding to fine-tuning**, validate dataset quality using an LLM judge. This catches semantic issues that structural validation cannot detect:

```bash
# Quick quality check on a sample of 50 examples
python scripts/evaluate_dataset_quality.py \
  --dataset evaluation/datasets/offline_dataset.jsonl \
  --sample-size 50

# Full evaluation with detailed output
python scripts/evaluate_dataset_quality.py \
  --dataset evaluation/datasets/offline_dataset.jsonl \
  --output evaluation/quality_report.json \
  --include-verdicts

# Filter to specific task types (e.g., focus on citations)
python scripts/evaluate_dataset_quality.py \
  --dataset evaluation/datasets/offline_dataset.jsonl \
  --task-type citations \
  --sample-size 30
```

**Quality criteria evaluated:**
| Criterion | Description |
|-----------|-------------|
| Answerability | Can the question be answered from the provided contexts? |
| Correctness | Is the gold answer factually accurate given the contexts? |
| Clarity | Is the instruction clear and unambiguous? |
| Coherence | Does the expected response appropriately address the instruction? |

**Pass criteria:** All scores ≥ 0.6 AND (answerability + correctness) ≥ 1.4

**Target pass rate:** ≥75% before proceeding to fine-tuning.

**Important:** The judge model (Qwen 3 8B) is intentionally different from the generator model (Qwen 2.5 7B) to avoid self-preference bias.

If the pass rate is below 75%, review `quality_report.json` for common issues and consider:
- Adjusting generator prompts
- Increasing citation coverage thresholds
- Re-generating problematic examples

### 4.4 Split Dataset for Train/Eval Holdout

**Critical step:** Split the dataset into training and evaluation sets before fine-tuning. This creates a **question-level holdout** where:
- Questions from each paper are distributed across train and eval
- Fine-tuned models see paper content but NOT the exact eval questions
- Prevents data leakage while testing true knowledge generalization

```bash
# Default split: 70% train, 30% eval
python scripts/split_dataset.py \
  --input evaluation/datasets/offline_dataset.jsonl \
  --train-output evaluation/datasets/train_dataset.jsonl \
  --eval-output evaluation/datasets/eval_dataset.jsonl

# Custom split with 25% eval
python scripts/split_dataset.py \
  --input evaluation/datasets/offline_dataset.jsonl \
  --train-output evaluation/datasets/train_dataset.jsonl \
  --eval-output evaluation/datasets/eval_dataset.jsonl \
  --eval-ratio 0.25

# Ensure at least 2 eval examples per paper
python scripts/split_dataset.py \
  --input evaluation/datasets/offline_dataset.jsonl \
  --train-output evaluation/datasets/train_dataset.jsonl \
  --eval-output evaluation/datasets/eval_dataset.jsonl \
  --min-eval-per-paper 2

# Preview split without writing files
python scripts/split_dataset.py \
  --input evaluation/datasets/offline_dataset.jsonl \
  --train-output evaluation/datasets/train_dataset.jsonl \
  --eval-output evaluation/datasets/eval_dataset.jsonl \
  --dry-run
```

**Output files:**
| File | Purpose | Used By |
|------|---------|---------|
| `train_dataset.jsonl` | Fine-tuning training data | Colab/Kaggle notebooks |
| `eval_dataset.jsonl` | Final experiment evaluation | `evaluate_models.py` |

**Why question-level holdout?**
- Models see paper content (via training questions) but not exact eval questions
- Tests whether models learned underlying knowledge vs. memorized Q&A pairs
- All 6 conditions use the SAME eval questions → valid comparison
- RAG models retrieve from same papers, non-RAG models rely on learned knowledge

### 4.5 Inspect Offline Tasks (Optional)

```bash
# Launch Streamlit viewer for offline tasks
streamlit run apps/offline_task_viewer.py

# Or use command-line inspection
python scripts/utility/inspect_offline_tasks.py \
  evaluation/datasets/offline_dataset.jsonl \
  --limit 5
```

---

## 5. Fine-Tuning

### 5.1 Prepare Fine-Tuning Data

Use the **training split** (`evaluation/datasets/train_dataset.jsonl`) created in step 4.4. This ensures models don't see eval questions during training.

**Important:** Upload `train_dataset.jsonl` (not `offline_dataset.jsonl`) to your Colab/Kaggle environment.

**Note:** The notebooks will perform their own internal train/val/test split from this file for training purposes. This is separate from the train/eval holdout — the internal split is for training validation, while the external `eval_dataset.jsonl` is for final experiment evaluation.

### 5.2 Train Instruction-Only Model (Condition 3-4)

**In Google Colab or Kaggle:**

1. Upload `notebooks/finetuning/lora_science_v1_instruction_only.ipynb`
2. Upload `evaluation/datasets/train_dataset.jsonl` ← Use training split!
3. Upload `configs/models/lora_instruction_only.yaml` (for reference)
4. Run the notebook to train WITHOUT RAG contexts
5. Export adapter weights and merged checkpoint

**Expected outputs:**
- `outputs/lora_science_v1_instruction_only/adapter.safetensors`
- `outputs/lora_science_v1_instruction_only/merged_full_model/`

### 5.3 Train RAG-Trained Model (Condition 5-6)

**In Google Colab or Kaggle:**

1. Upload `notebooks/finetuning/lora_science_v1.ipynb`
2. Upload `evaluation/datasets/train_dataset.jsonl` ← Use training split!
3. Upload `configs/models/lora_rag_trained.yaml` (for reference)
4. Run the notebook to train WITH RAG contexts
5. Export adapter weights and merged checkpoint

**Expected outputs:**
- `outputs/lora_science_v1/adapter.safetensors`
- `outputs/lora_science_v1/merged_full_model/`

### 5.4 Sync Checkpoints to Local Machine

```bash
# Download checkpoints from Colab/Kaggle (adjust paths as needed)
# Option 1: Using Google Drive sync
# Option 2: Using Hugging Face Hub
# Option 3: Direct download

# Verify checkpoints are present
ls -lh outputs/lora_science_v1_instruction_only/
ls -lh outputs/lora_science_v1/
```

---

## 6. Model Deployment

### 6.1 Convert to GGUF Format

**Prerequisites:** Install `llama.cpp` tools

```bash
# Convert instruction-only model to GGUF
python /path/to/llama.cpp/convert-hf-to-gguf.py \
  --model-dir outputs/lora_science_v1_instruction_only/merged_full_model \
  --outfile outputs/lora_science_v1_instruction_only/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf \
  --data-type Q4_K_M

# Convert RAG-trained model to GGUF
python /path/to/llama.cpp/convert-hf-to-gguf.py \
  --model-dir outputs/lora_science_v1/merged_full_model \
  --outfile outputs/lora_science_v1/Qwen2.5-0.5B-lora_science_v1.Q4_K_M.gguf \
  --data-type Q4_K_M

# Optional: Quantize further if needed
# /path/to/llama.cpp/quantize \
#   outputs/lora_science_v1_instruction_only/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf \
#   outputs/lora_science_v1_instruction_only/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q2_K.gguf \
#   Q2_K
```

### 6.2 Register Models with Ollama

```bash
# Update Modelfile paths to point to your GGUF files
# Edit ollama/Modelfile.instruction_only and ollama/Modelfile.rag_trained
# Update the FROM line to point to your GGUF file path

# Register instruction-only model
ollama create lora_science_0p5_instruction_only \
  -f ollama/Modelfile.instruction_only

# Register RAG-trained model
ollama create lora_science_0p5 \
  -f ollama/Modelfile.rag_trained

# Verify models are registered
ollama list | grep lora_science

# Test model loading
ollama show lora_science_0p5_instruction_only
ollama show lora_science_0p5
```

### 6.3 Verify Model Configurations

```bash
# Test instruction-only config
python -c "
from beyond_the_cutoff import load_config
config = load_config('configs/models/lora_instruction_only.yaml')
print(f'Model: {config.inference.model}')
"

# Test RAG-trained config
python -c "
from beyond_the_cutoff import load_config
config = load_config('configs/models/lora_rag_trained.yaml')
print(f'Model: {config.inference.model}')
"
```

---

## 7. Evaluation

**Important:** Use `eval_dataset.jsonl` (the held-out evaluation set) for all final evaluations. This ensures fair comparison across all 6 conditions with no data leakage from training.

### 7.1 Validate Experiment Setup

```bash
# Validate configuration files
python scripts/validate_experiment.py \
  --config configs/default.yaml \
  --model-config configs/models/base_ollama.yaml \
  --judge-config configs/judges/rag.yaml \
  --prompt-mode rag

# Validate eval dataset versioning (use eval_dataset.jsonl!)
python scripts/validate_experiment.py \
  --dataset evaluation/datasets/eval_dataset.jsonl \
  --dataset evaluation/results/rag_baseline_0p5b/details.jsonl

# Validate experiment metadata (after first run)
python scripts/validate_experiment.py \
  --metadata evaluation/results/rag_baseline_0p5b/metadata.jsonl
```

### 7.2 Run Single Model Evaluation

**Note:** All commands use `--dataset evaluation/datasets/eval_dataset.jsonl` to evaluate on held-out questions.

```bash
# Evaluate base baseline (condition 1)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --dataset evaluation/datasets/eval_dataset.jsonl \
  --model-config configs/models/base_ollama.yaml \
  --judge-config configs/judges/instruction.yaml \
  --prompt-mode instruction \
  --output evaluation/results/base_baseline_0p5b/

# Evaluate RAG baseline (condition 2)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --dataset evaluation/datasets/eval_dataset.jsonl \
  --model-config configs/models/base_ollama.yaml \
  --judge-config configs/judges/rag.yaml \
  --prompt-mode rag \
  --output evaluation/results/rag_baseline_0p5b/

# Evaluate instruction-only FT-only (condition 3)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --dataset evaluation/datasets/eval_dataset.jsonl \
  --model-config configs/models/lora_instruction_only.yaml \
  --judge-config configs/judges/instruction.yaml \
  --prompt-mode instruction \
  --output evaluation/results/lora_science_0p5b_ft_only/

# Evaluate instruction-only FT+RAG (condition 4)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --dataset evaluation/datasets/eval_dataset.jsonl \
  --model-config configs/models/lora_instruction_only.yaml \
  --judge-config configs/judges/rag.yaml \
  --prompt-mode rag \
  --output evaluation/results/hybrid_science_0p5b_instruction_only/

# Evaluate RAG-trained FT-only (condition 5)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --dataset evaluation/datasets/eval_dataset.jsonl \
  --model-config configs/models/lora_rag_trained.yaml \
  --judge-config configs/judges/instruction.yaml \
  --prompt-mode instruction \
  --output evaluation/results/lora_science_0p5b_rag_trained_ft_only/

# Evaluate RAG-trained FT+RAG (condition 6)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --dataset evaluation/datasets/eval_dataset.jsonl \
  --model-config configs/models/lora_rag_trained.yaml \
  --judge-config configs/judges/rag.yaml \
  --prompt-mode rag \
  --output evaluation/results/hybrid_science_0p5b_rag_trained/
```

### 7.3 Run Complete 6-Condition Comparison

```bash
# Run all 6 conditions automatically (uses eval_dataset.jsonl)
python scripts/core/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --dataset evaluation/datasets/eval_dataset.jsonl

# This generates:
# - evaluation/results/comparison_report.json
# - Individual results for each condition in evaluation/results/<label>/
```

### 7.4 Compute Automatic Metrics

```bash
# Compute metrics for a single model's predictions (use eval_dataset.jsonl!)
python scripts/evaluation_harness.py \
  --dataset evaluation/datasets/eval_dataset.jsonl \
  --predictions evaluation/results/rag_baseline_0p5b/details.jsonl \
  --output evaluation/results/rag_baseline_0p5b/automatic_metrics.json \
  --details-output evaluation/results/rag_baseline_0p5b/automatic_metrics_details.jsonl

# Or use Makefile shortcut
make score \
  SCORE_DATASET=evaluation/datasets/eval_dataset.jsonl \
  SCORE_PREDICTIONS=evaluation/results/rag_baseline_0p5b/details.jsonl \
  SCORE_OUTPUT=evaluation/results/rag_baseline_0p5b/automatic_metrics.json \
  SCORE_DETAILS=evaluation/results/rag_baseline_0p5b/automatic_metrics_details.jsonl
```

### 7.5 Two-Phase Evaluation Pipeline

For efficient evaluation of multiple conditions, the pipeline separates response generation from judging. This allows:
- Generating all responses first (parallelizable across conditions)
- Running expensive judging once on all responses
- Re-running judging with different judges without regenerating responses

**Phase 1: Generate Responses**

```bash
# Generate responses for all 6 conditions
python scripts/core/generate_responses.py \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output-dir evaluation/responses/

# Generate for specific conditions only
python scripts/core/generate_responses.py \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output-dir evaluation/responses/ \
  --conditions rag_baseline_0p5b hybrid_science_0p5b_rag_trained

# Quick test with limited examples
python scripts/core/generate_responses.py \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output-dir evaluation/responses/ \
  --limit 5

# Resume interrupted generation
python scripts/core/generate_responses.py \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output-dir evaluation/responses/ \
  --resume
```

**Phase 2: Evaluate Pre-generated Responses**

```bash
# Evaluate all pre-generated responses with judge
python scripts/core/compare_models.py \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --responses-dir evaluation/responses/ \
  --output evaluation/results/comparison_results.jsonl

# This runs judging on existing responses without regenerating them
```

**Benefits of two-phase approach:**
| Aspect | Single-Phase | Two-Phase |
|--------|--------------|-----------|
| Time for 6 conditions | ~6x generation + judging | Generation once, then judging |
| Resume capability | Limited | Full resume support per phase |
| Re-judging | Must regenerate | Use existing responses |
| Parallelization | Sequential | Can parallelize generation |

### 7.6 Unified Pipeline Orchestrator

For complete evaluation workflows, use the unified orchestrator script:

```bash
# Full 6-condition model comparison (generation + judging + visualization)
python scripts/run_evaluation_pipeline.py full-comparison \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output-dir evaluation/results/six_condition/

# Skip generation if responses already exist
python scripts/run_evaluation_pipeline.py full-comparison \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output-dir evaluation/results/six_condition/ \
  --skip-generation

# Skip evaluation to only generate responses
python scripts/run_evaluation_pipeline.py full-comparison \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output-dir evaluation/results/six_condition/ \
  --skip-evaluation
```

**Available workflows:**

| Workflow | Command | Purpose |
|----------|---------|---------|
| `full-comparison` | Model comparison | 6-condition experiment |
| `quantization` | Q4_K_M vs F16 | Quantization impact analysis |
| `retrieval-ablation` | Retrieval optimization | ELO tournament for retrieval configs |
| `end-to-end` | Live retrieval validation | Full pipeline validation |

See Section 9 for post-experiment optimization workflows.

---

## 8. Results Analysis

### 8.1 Validate Evaluation Results

```bash
# Validate metrics and details
python scripts/validate_experiment_results.py \
  --metrics evaluation/results/rag_baseline_0p5b/metrics.json \
  --details evaluation/results/rag_baseline_0p5b/details.jsonl

# Validate comparison report
python scripts/validate_experiment_results.py \
  --comparison evaluation/results/comparison_report.json
```

### 8.2 Generate Visualizations

```bash
# Visualize from comparison report
python scripts/core/visualize_comparison.py \
  --report evaluation/results/comparison_report.json \
  --output evaluation/results/visualizations/

# Visualize from individual metrics files
python scripts/core/visualize_comparison.py \
  --metrics evaluation/results/rag_baseline_0p5b/metrics.json \
  --metrics evaluation/results/lora_science_0p5b_ft_only/metrics.json \
  --metrics evaluation/results/hybrid_science_0p5b_rag_trained/metrics.json \
  --output evaluation/results/visualizations/

# Generate specific visualizations only
python scripts/core/visualize_comparison.py \
  --report evaluation/results/comparison_report.json \
  --output evaluation/results/visualizations/ \
  --only metrics error-rates citations
```

### 8.3 Inspect Results

```bash
# View comparison report summary
cat evaluation/results/comparison_report.json | jq '.summary'

# View individual model metrics
cat evaluation/results/rag_baseline_0p5b/metrics.json | jq '.'

# View sample predictions
head -n 3 evaluation/results/rag_baseline_0p5b/details.jsonl | jq '.'

# Check error rates
cat evaluation/results/comparison_report.json | \
  jq '.runs[] | {label: .label, error_rate: .metrics.error_rate}'
```

### 8.4 Compute ELO Rankings (Automated)

Use judge models to automatically compare model outputs head-to-head and compute ELO rankings with confidence intervals.

```bash
# Run pairwise evaluation using 8B judges for consensus
# Note: Qwen 7B excluded to avoid self-preference bias with the generator
python scripts/run_pairwise_evaluation.py \
  --results base_baseline=evaluation/results/base_baseline_0p5b \
  --results rag_baseline=evaluation/results/rag_baseline_0p5b \
  --results ft_only=evaluation/results/lora_science_0p5b_ft_only \
  --results ft_rag=evaluation/results/hybrid_science_0p5b_instruction_only \
  --results ft_rag_trained=evaluation/results/hybrid_science_0p5b_rag_trained \
  --judge configs/judges/pairwise_qwen3_8b.yaml \
  --judge configs/judges/pairwise_llama31_8b.yaml \
  --comparisons-per-pair 50 \
  --output evaluation/results/elo_rankings

# Or use a predefined evaluation plan
python scripts/run_pairwise_evaluation.py \
  --plan configs/evaluation/pairwise_evaluation_plan.yaml

# Compute ELO from existing comparison data
python scripts/compute_elo_rankings.py \
  --comparisons evaluation/results/elo_rankings/all_comparisons.jsonl \
  --output evaluation/results/elo_rankings/leaderboard.json \
  --head-to-head evaluation/results/elo_rankings/h2h_matrix.json
```

**Output files:**
- `all_comparisons.jsonl` — All pairwise judgments
- `leaderboard.json` — ELO ratings with confidence intervals
- `h2h_matrix.json` — Head-to-head win rates

### 8.5 Human Evaluation (Optional)

For validating judge reliability or collecting additional annotations:

```bash
# Launch human annotation UI
streamlit run apps/human_annotation.py

# Compute ELO from human annotations
python scripts/compute_elo_rankings.py \
  --annotations-dir evaluation/human_annotations \
  --output evaluation/results/human_leaderboard.json

# Compare human vs judge agreement
python -c "
from beyond_the_cutoff.evaluation.human_evaluation import (
    human_judge_correlation, load_annotation_batch
)
from pathlib import Path
import json

# Load human annotations
batches = [load_annotation_batch(p) for p in Path('evaluation/human_annotations').glob('*.json')]
human_annots = [a for b in batches for a in b.annotations]

# Load judge verdicts
with open('evaluation/results/elo_rankings/all_comparisons.jsonl') as f:
    judge_verdicts = {json.loads(l)['task_id']: json.loads(l)['outcome'] for l in f}

# Compute correlation
stats = human_judge_correlation(human_annots, judge_verdicts)
print(f\"Human-Judge Agreement: {stats['agreement_rate']:.1%}\")
print(f\"Cohen's Kappa: {stats['cohens_kappa']:.3f}\")
"
```

---

## 9. Post-Experiment Optimization

After completing the 6-condition experiment and identifying the best model, proceed with these optimization stages.

### 9.1 Quantization Analysis

Compare Q4_K_M (current deployment) vs F16 (full precision) to determine if quantization significantly impacts quality.

**Step 1: Register F16 Model with Ollama**

```bash
# Create F16 Modelfile (already created at ollama/Modelfile.rag_trained_f16)
# Register with Ollama
ollama create lora_science_0p5_f16 -f ollama/Modelfile.rag_trained_f16

# Verify registration
ollama list | grep lora_science
```

**Step 2: Run Quantization Comparison**

```bash
# Using unified pipeline
python scripts/run_evaluation_pipeline.py quantization \
  --plan configs/evaluation/quantization_comparison.yaml \
  --output-dir evaluation/results/quantization/ \
  --register-f16

# Or manually with two-phase approach
python scripts/core/generate_responses.py \
  --plan configs/evaluation/quantization_comparison.yaml \
  --output-dir evaluation/results/quantization/responses/

python scripts/core/compare_models.py \
  --plan configs/evaluation/quantization_comparison.yaml \
  --responses-dir evaluation/results/quantization/responses/
```

**Expected outcome:** For 0.5B models, Q4_K_M typically shows minimal degradation (<2-3% on judge scores). If acceptable, prefer Q4_K_M for:
- 2.5x smaller memory footprint
- Faster inference on CPU
- Fair comparison with base model quantization

### 9.2 Retrieval Optimization

Optimize retrieval configuration using ELO tournament with different top_k values and reranker settings.

**Retrieval conditions to test:**

| Condition | Retrieve K | Final K | Reranker | Description |
|-----------|------------|---------|----------|-------------|
| dense_top4_baseline | 4 | 4 | None | Current production config |
| dense_top3 | 3 | 3 | None | Minimal context |
| dense_top6 | 6 | 6 | None | More context |
| dense_top4_rerank | 4 | 4 | BGE-v2-M3 | Same k + reranking |
| dense_top8_rerank4 | 8 | 4 | BGE-v2-M3 | Wider net, rerank to 4 |
| dense_top12_rerank5 | 12 | 5 | BGE-v2-M3 | Widest net, rerank to 5 |

**Step 1: Run Retrieval Ablation**

```bash
# Using unified pipeline (runs all phases)
python scripts/run_evaluation_pipeline.py retrieval-ablation \
  --plan configs/evaluation/retrieval_ablation.yaml \
  --output-dir evaluation/results/retrieval_ablation/

# Or run generation phase separately
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
  --limit 10
```

**Step 2: Run Pairwise Comparisons**

```bash
# Run pairwise evaluation on retrieval ablation results
python scripts/run_pairwise_evaluation.py \
  --results evaluation/results/retrieval_ablation/*.jsonl \
  --judge configs/judges/pairwise_qwen3_8b.yaml \
  --output evaluation/results/retrieval_ablation/pairwise/
```

**Step 3: Compute ELO Rankings**

```bash
# Compute ELO from pairwise comparisons
python scripts/compute_elo_rankings.py \
  --comparisons evaluation/results/retrieval_ablation/pairwise/pairwise_results.jsonl \
  --output evaluation/results/retrieval_ablation/elo_rankings.json \
  --bootstrap-samples 1000
```

**Reranker selection:**

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| ms-marco-MiniLM-L-6-v2 | 22M | Good | Fast | Quick experiments |
| ms-marco-MiniLM-L-12-v2 | 33M | Better | Medium | Balanced |
| BAAI/bge-reranker-v2-m3 | 568M | Best | Slower | Production (recommended) |

### 9.3 End-to-End Validation

Validate the complete pipeline with live retrieval (not pre-computed contexts) using the best model and optimal retrieval configuration.

```bash
# Using unified pipeline
python scripts/run_evaluation_pipeline.py end-to-end \
  --plan configs/evaluation/end_to_end.yaml \
  --output-dir evaluation/results/end_to_end/

# Or run directly with custom retrieval settings
python scripts/evaluate_end_to_end.py \
  --config configs/default.yaml \
  --model-config configs/models/lora_rag_trained.yaml \
  --output evaluation/results/end_to_end/

# With custom retrieval parameters
python scripts/evaluate_end_to_end.py \
  --config configs/default.yaml \
  --model-config configs/models/lora_rag_trained.yaml \
  --top-k 8 \
  --reranker BAAI/bge-reranker-v2-m3 \
  --output evaluation/results/end_to_end_optimized/

# Compare with pre-computed contexts
python scripts/evaluate_end_to_end.py \
  --config configs/default.yaml \
  --model-config configs/models/lora_rag_trained.yaml \
  --compare-precomputed \
  --output evaluation/results/end_to_end/
```

**Metrics collected:**
- Judge scores (same as offline evaluation)
- Retrieval overlap (Jaccard similarity between live and pre-computed contexts)
- Latency (total time including retrieval)
- Citation metrics with live contexts

---

## 10. Verification & Testing

### 10.1 Run Test Suite

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_config.py
pytest tests/test_rag_pipeline.py
pytest tests/test_evaluation_harness.py
pytest tests/test_offline_dataset.py

# Run with coverage
pytest --cov=src/beyond_the_cutoff --cov-report=html
```

### 10.2 Check Pipeline Status

```bash
# Check overall pipeline status
python scripts/utility/check_pipeline_status.py

# Check processed corpus
python scripts/utility/check_processed_corpus.py \
  --manifest data/processed/manifest.json

# Verify index artifacts
python scripts/validate_index_artifacts.py \
  --index-dir data/external/index
```

### 10.3 Manual Testing

```bash
# Test RAG pipeline interactively
python scripts/ask.py "What are the main contributions of recent papers on large language models?"

# Test API server (in separate terminal)
uvicorn beyond_the_cutoff.api.server:app --reload --port 8000

# Test API endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings in recent NLP papers?"}'
```

---

## Quick Reference: Complete Pipeline in One Go

For experienced users, here's the minimal command sequence:

```bash
# 1. Setup
python scripts/bootstrap_env.py
source .venv/bin/activate
ollama pull qwen2.5:0.5b-instruct qwen2.5:7b-instruct-q4_K_M

# 2. Data collection
python scripts/fetch_arxiv_corpus.py \
  --contact-email your.email@example.com \
  --total 100 \
  --output-dir data/raw/arxiv_2025

# 3. Ingestion
python scripts/ingest_and_index.py --config configs/default.yaml

# 4. Dataset generation
python scripts/generate_offline_dataset.py --config configs/default.yaml

# 5. Fine-tuning (in Colab/Kaggle - see section 5)
# Upload notebooks/finetuning/*.ipynb and run

# 6. Model deployment (after syncing checkpoints)
# Convert to GGUF and register with Ollama (see section 6)

# 7. Evaluation (two-phase for efficiency)
python scripts/core/generate_responses.py \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output-dir evaluation/responses/

python scripts/core/compare_models.py \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --responses-dir evaluation/responses/

# 8. Visualization
python scripts/core/visualize_comparison.py \
  --report evaluation/results/comparison_report.json \
  --output evaluation/results/visualizations/

# 9. Post-experiment optimization (after identifying best model)
python scripts/run_evaluation_pipeline.py quantization \
  --plan configs/evaluation/quantization_comparison.yaml \
  --output-dir evaluation/results/quantization/

python scripts/run_evaluation_pipeline.py retrieval-ablation \
  --plan configs/evaluation/retrieval_ablation.yaml \
  --output-dir evaluation/results/retrieval_ablation/

python scripts/run_evaluation_pipeline.py end-to-end \
  --plan configs/evaluation/end_to_end.yaml \
  --output-dir evaluation/results/end_to_end/
```

---

## Troubleshooting

### Common Issues

1. **Ollama connection errors**: Ensure Ollama daemon is running (`ollama serve` or check system service)
2. **Model not found**: Verify model tags match config files (`ollama list`)
3. **Index not found**: Re-run `ingest_and_index.py` to rebuild index
4. **Dataset errors**: Check dataset format with `analyze_offline_dataset.py`
5. **Memory issues**: Reduce batch sizes in config or use smaller models

### Getting Help

- Check logs in `evaluation/results/*/` directories
- Review configuration files in `configs/`
- Inspect intermediate outputs (manifests, catalogs, raw tasks)
- Run validation scripts to identify issues

---

## Next Steps

After completing the pipeline:

1. **Scale up**: Repeat with 3B models (`qwen2.5:3b-instruct-q4_K_M`)
2. **Expand dataset**: Increase paper count or add new categories
3. **Human evaluation**: Add manual review of judge outputs
4. **Ablation studies**: Test different chunk sizes, top-k values, etc.
5. **Documentation**: Update results in `docs/` and create analysis reports

---

## Appendix: File Structure Reference

```
beyond-the-cutoff/
├── configs/                    # Configuration files
│   ├── default.yaml            # Main config
│   ├── base_ollama.yaml
│   ├── models/lora_instruction_only.yaml
│   ├── models/lora_rag_trained.yaml
│   └── evaluation/
│       ├── six_condition_experiment.yaml    # 6-condition experiment
│       ├── quantization_comparison.yaml     # Q4_K_M vs F16
│       ├── retrieval_ablation.yaml          # Retrieval optimization
│       ├── end_to_end.yaml                  # Live retrieval validation
│       └── lora_science_v1_rag_trained_f16_ollama.yaml
├── data/
│   ├── raw/                    # Downloaded PDFs
│   ├── processed/              # Processed text + manifests
│   └── external/               # FAISS index
├── evaluation/
│   ├── datasets/               # Offline tasks and dataset
│   ├── responses/              # Generated responses (Phase 1)
│   └── results/                # Evaluation outputs
├── outputs/                    # Fine-tuned checkpoints
│   ├── lora_science_v1_instruction_only/
│   └── lora_science_v1/
├── scripts/
│   ├── generate_responses.py           # Phase 1: Response generation
│   ├── compare_models.py               # Phase 2: Evaluation with judge
│   ├── run_evaluation_pipeline.py      # Unified pipeline orchestrator
│   ├── run_retrieval_ablation.py       # Retrieval optimization
│   ├── evaluate_end_to_end.py          # Live retrieval evaluation
│   ├── run_pairwise_evaluation.py      # Pairwise comparisons
│   ├── compute_elo_rankings.py         # ELO ranking computation
│   └── visualize_comparison.py         # Result visualization
├── ollama/
│   ├── Modelfile.instruction_only
│   ├── Modelfile.rag_trained
│   └── Modelfile.rag_trained_f16       # F16 version for quantization study
├── notebooks/finetuning/               # Training notebooks
└── docs/
    ├── pipeline_plan.md                # This document
    └── evaluation_methodology.md       # Detailed evaluation methodology
```

---

_Last updated: 2025-11-28_
