
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
8. [Results Analysis](#8-results-analysis)
   - [8.4 ELO Rankings (Automated)](#84-compute-elo-rankings-automated)
   - [8.5 Human Evaluation (Optional)](#85-human-evaluation-optional)
9. [Verification & Testing](#9-verification--testing)

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

### 4.3 Inspect Offline Tasks (Optional)

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

The offline dataset (`evaluation/datasets/offline_dataset.jsonl`) is already in the correct format for fine-tuning. Ensure it's accessible from your Colab/Kaggle environment (upload to Drive or download directly).

### 5.2 Train Instruction-Only Model (Condition 3-4)

**In Google Colab or Kaggle:**

1. Upload `notebooks/finetuning/lora_science_v1_instruction_only.ipynb`
2. Upload `evaluation/datasets/offline_dataset.jsonl`
3. Upload `configs/lora_science_v1_instruction_only_ollama.yaml` (for reference)
4. Run the notebook to train WITHOUT RAG contexts
5. Export adapter weights and merged checkpoint

**Expected outputs:**
- `outputs/lora_science_v1_instruction_only/adapter.safetensors`
- `outputs/lora_science_v1_instruction_only/merged_full_model/`

### 5.3 Train RAG-Trained Model (Condition 5-6)

**In Google Colab or Kaggle:**

1. Upload `notebooks/finetuning/lora_science_v1.ipynb`
2. Upload `evaluation/datasets/offline_dataset.jsonl`
3. Upload `configs/lora_science_v1_rag_trained_ollama.yaml` (for reference)
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
config = load_config('configs/lora_science_v1_instruction_only_ollama.yaml')
print(f'Model: {config.inference.model}')
"

# Test RAG-trained config
python -c "
from beyond_the_cutoff import load_config
config = load_config('configs/lora_science_v1_rag_trained_ollama.yaml')
print(f'Model: {config.inference.model}')
"
```

---

## 7. Evaluation

### 7.1 Validate Experiment Setup

```bash
# Validate configuration files
python scripts/validate_experiment.py \
  --config configs/default.yaml \
  --model-config configs/rag_baseline_ollama.yaml \
  --judge-config configs/judges/scientific_default_rag.yaml \
  --prompt-mode rag

# Validate dataset versioning
python scripts/validate_experiment.py \
  --dataset evaluation/datasets/offline_dataset.jsonl \
  --dataset evaluation/results/rag_baseline_0p5b/details.jsonl

# Validate experiment metadata (after first run)
python scripts/validate_experiment.py \
  --metadata evaluation/results/rag_baseline_0p5b/metadata.jsonl
```

### 7.2 Run Single Model Evaluation

```bash
# Evaluate base baseline (condition 1)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --model-config configs/rag_baseline_ollama.yaml \
  --judge-config configs/judges/scientific_default_instruction.yaml \
  --prompt-mode instruction \
  --output evaluation/results/base_baseline_0p5b/

# Evaluate RAG baseline (condition 2)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --model-config configs/rag_baseline_ollama.yaml \
  --judge-config configs/judges/scientific_default_rag.yaml \
  --prompt-mode rag \
  --output evaluation/results/rag_baseline_0p5b/

# Evaluate instruction-only FT-only (condition 3)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --model-config configs/lora_science_v1_instruction_only_ollama.yaml \
  --judge-config configs/judges/scientific_default_instruction.yaml \
  --prompt-mode instruction \
  --output evaluation/results/lora_science_0p5b_ft_only/

# Evaluate instruction-only FT+RAG (condition 4)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --model-config configs/lora_science_v1_instruction_only_ollama.yaml \
  --judge-config configs/judges/scientific_default_rag.yaml \
  --prompt-mode rag \
  --output evaluation/results/hybrid_science_0p5b_instruction_only/

# Evaluate RAG-trained FT-only (condition 5)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --model-config configs/lora_science_v1_rag_trained_ollama.yaml \
  --judge-config configs/judges/scientific_default_instruction.yaml \
  --prompt-mode instruction \
  --output evaluation/results/lora_science_0p5b_rag_trained_ft_only/

# Evaluate RAG-trained FT+RAG (condition 6)
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --model-config configs/lora_science_v1_rag_trained_ollama.yaml \
  --judge-config configs/judges/scientific_default_rag.yaml \
  --prompt-mode rag \
  --output evaluation/results/hybrid_science_0p5b_rag_trained/
```

### 7.3 Run Complete 6-Condition Comparison

```bash
# Run all 6 conditions automatically
python scripts/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/compare_0p5b_experiments.yaml

# This generates:
# - evaluation/results/comparison_report.json
# - Individual results for each condition in evaluation/results/<label>/
```

### 7.4 Compute Automatic Metrics

```bash
# Compute metrics for a single model's predictions
python scripts/evaluation_harness.py \
  --dataset evaluation/datasets/offline_dataset.jsonl \
  --predictions evaluation/results/rag_baseline_0p5b/details.jsonl \
  --output evaluation/results/rag_baseline_0p5b/automatic_metrics.json \
  --details-output evaluation/results/rag_baseline_0p5b/automatic_metrics_details.jsonl

# Or use Makefile shortcut
make score \
  SCORE_DATASET=evaluation/datasets/offline_dataset.jsonl \
  SCORE_PREDICTIONS=evaluation/results/rag_baseline_0p5b/details.jsonl \
  SCORE_OUTPUT=evaluation/results/rag_baseline_0p5b/automatic_metrics.json \
  SCORE_DETAILS=evaluation/results/rag_baseline_0p5b/automatic_metrics_details.jsonl
```

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
python scripts/visualize_comparison.py \
  --report evaluation/results/comparison_report.json \
  --output evaluation/results/visualizations/

# Visualize from individual metrics files
python scripts/visualize_comparison.py \
  --metrics evaluation/results/rag_baseline_0p5b/metrics.json \
  --metrics evaluation/results/lora_science_0p5b_ft_only/metrics.json \
  --metrics evaluation/results/hybrid_science_0p5b_rag_trained/metrics.json \
  --output evaluation/results/visualizations/

# Generate specific visualizations only
python scripts/visualize_comparison.py \
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
# Run pairwise evaluation using 7B and 3B judges for consensus
python scripts/run_pairwise_evaluation.py \
  --results base_baseline=evaluation/results/base_baseline_0p5b \
  --results rag_baseline=evaluation/results/rag_baseline_0p5b \
  --results ft_only=evaluation/results/lora_science_0p5b_ft_only \
  --results ft_rag=evaluation/results/hybrid_science_0p5b_instruction_only \
  --results ft_rag_trained=evaluation/results/hybrid_science_0p5b_rag_trained \
  --judge configs/judges/pairwise_qwen7b.yaml \
  --judge configs/judges/pairwise_qwen3b.yaml \
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

## 9. Verification & Testing

### 9.1 Run Test Suite

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

### 9.2 Check Pipeline Status

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

### 9.3 Manual Testing

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

# 7. Evaluation
python scripts/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/compare_0p5b_experiments.yaml

# 8. Visualization
python scripts/visualize_comparison.py \
  --report evaluation/results/comparison_report.json \
  --output evaluation/results/visualizations/
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
│   ├── rag_baseline_ollama.yaml
│   ├── lora_science_v1_instruction_only_ollama.yaml
│   ├── lora_science_v1_rag_trained_ollama.yaml
│   └── evaluation/
│       └── compare_0p5b_experiments.yaml
├── data/
│   ├── raw/                    # Downloaded PDFs
│   ├── processed/              # Processed text + manifests
│   └── external/              # FAISS index
├── evaluation/
│   ├── datasets/              # Offline tasks and dataset
│   └── results/               # Evaluation outputs
├── outputs/                    # Fine-tuned checkpoints
│   ├── lora_science_v1_instruction_only/
│   └── lora_science_v1/
├── notebooks/finetuning/       # Training notebooks
├── scripts/                    # Pipeline scripts
└── ollama/                     # Ollama Modelfiles
```

---

_Last updated: 2025-11-23_
