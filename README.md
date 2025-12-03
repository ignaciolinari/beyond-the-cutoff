# Beyond the Cutoff

A reproducible pipeline for evaluating how language models acquire knowledge beyond their training data cutoff, comparing **fine-tuning** and **retrieval-augmented generation (RAG)** approaches.

> **📊 Experiment Status: COMPLETED** — This pipeline was tested on Qwen 2.5 0.5B (a small language model). Results and methodology are documented below. The pipeline is model-agnostic and can be adapted for larger models.

## Overview

This project provides an end-to-end framework for investigating how LLMs can acquire new scientific knowledge published after their training cutoff. We compare two adaptation strategies:

| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **Fine-tuning** | Update model weights with new data | Internalizes knowledge | Expensive, may forget |
| **RAG** | Retrieve relevant context at inference | No retraining needed | Depends on retrieval quality |

The benchmark uses **2025 arXiv papers** (post-cutoff for current open-weight models) as the knowledge source.

### Completed Experiment

| Aspect | Configuration |
|--------|---------------|
| **Subject Model** | Qwen 2.5 0.5B Instruct (small language model, Q4_K_M quantized) |
| **Judge Model** | Qwen 3 8B (thinking mode) + Gemini 3 Pro (pairwise) |
| **Dataset** | 120 arXiv papers (2025), 154 evaluation QA pairs |
| **Quantization** | Q4_K_M (4-bit) vs F16 comparison |

> ⚠️ **Scope**: The findings below apply to **Qwen 2.5 0.5B** — a small language model with limited capacity. Larger models (3B, 7B, 14B+) may exhibit different patterns, particularly regarding fine-tuning benefits. This pipeline is designed to test such hypotheses.

---

## Experimental Results (Qwen 2.5 0.5B)

### 1. Six-Condition Experiment

We evaluated 6 conditions crossing training approach (none, instruction-only FT, RAG-trained FT) with inference mode (with/without RAG):

| Finding | Evidence |
|---------|----------|
| **RAG provides substantial improvement** | 4-5x improvement in pass rate (22% vs 4%) |
| **Fine-tuning provides marginal gains** | +1.3% when training format matches inference |
| **Training format affects transfer** | RAG-trained models perform better with RAG at inference |
| **Fine-tuning alone insufficient** | FT-only models (no RAG) perform similarly to base (4-6%) |

### 2. Pairwise Comparison (Top 2 Candidates)

| Comparison | Result |
|------------|--------|
| **Base+RAG vs FT+RAG** | 54.9% vs 45.1% win rate (p=0.35, **not statistically significant**) |
| **Judge** | Gemini 3 pro (62 FT wins, 51 Base wins, 41 ties) |
| **Interpretation** | Insufficient evidence that fine-tuning improves RAG performance for this model size |

### 3. Quantization Impact (Q4_K_M vs F16)

| Metric | Result |
|--------|--------|
| **Win rate** | 48.4% vs 51.6% (p=0.84, **no significant difference**) |
| **Size reduction** | 60% smaller (397MB vs 994MB) |
| **Conclusion** | Q4_K_M quantization does not degrade quality for this model |

### Summary

For **Qwen 2.5 0.5B** on post-cutoff scientific QA:

1. **RAG is essential** — provides the primary performance gain
2. **Fine-tuning has minimal impact** — Base+RAG achieves ~95% of best performance
3. **Q4_K_M is viable** — no measurable quality loss vs F16
4. **Open question** — larger models may show different fine-tuning benefits

📖 **Detailed reports**: [Six-Condition Results](docs/reports/six_condition_experiment_results.md) | [Pairwise Evaluation](docs/reports/pairwise_evaluation_results.md) | [Quantization Analysis](docs/reports/quantization_comparison_results.md)

---

## Pipeline Overview

The project consists of 5 main stages:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. FETCH   │───▶│  2. INGEST  │───▶│ 3. DATASET  │───▶│ 4. FINETUNE │───▶│ 5. EVALUATE │
│   Papers    │    │   & Index   │    │  Generate   │    │   Models    │    │   Compare   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
    arXiv           PDF→Text           QA/Summary         LoRA/PEFT          6 Conditions
    Harvest         FAISS Index        Train/Eval         2 Models           LLM Judge
```

### Stage 1: Fetch Papers
Download recent arXiv papers that are beyond model training cutoffs.

```bash
python scripts/data/fetch_arxiv_corpus.py \
    --contact-email you@example.com \
    --total 100 \
    --output-dir data/raw/arxiv_2025
```

### Stage 2: Ingest & Index
Convert PDFs to text, chunk documents, and build a FAISS retrieval index.

```bash
python scripts/data/ingest_and_index.py --config configs/default.yaml
```

Output: `data/processed/` (text), `data/external/index/` (FAISS)

### Stage 3: Generate Dataset
Create QA pairs, summaries, and citation tasks using a strong generator model.

```bash
# Generate offline dataset
python scripts/data/generate_offline_dataset.py --config configs/default.yaml

# Validate quality with LLM judge
python scripts/data/evaluate_dataset_quality.py \
    --dataset evaluation/datasets/offline_dataset.jsonl

# Split into train/eval sets
python scripts/data/split_dataset.py \
    --input evaluation/datasets/offline_dataset.jsonl \
    --train-output evaluation/datasets/train_dataset.jsonl \
    --eval-output evaluation/datasets/eval_dataset.jsonl
```

### Stage 4: Fine-tune Models
Train two LoRA models in cloud notebooks (Colab/Kaggle):

| Model | Notebook | Training Data |
|-------|----------|---------------|
| **Instruction-only** | `notebooks/finetuning/lora_science_v1_instruction_only.ipynb` | Without RAG contexts |
| **RAG-trained** | `notebooks/finetuning/lora_science_v1.ipynb` | With RAG contexts |

Register with Ollama after training:
```bash
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
ollama create lora_science_0p5 -f ollama/Modelfile.rag_trained
```

### Stage 5: Evaluate (6-Condition Experiment)

| # | Condition | Training | RAG at Eval | Purpose |
|---|-----------|----------|-------------|---------|
| 1 | Base Baseline | None | ❌ | Lower bound |
| 2 | RAG Baseline | None | ✅ | RAG-only benefit |
| 3 | FT Only (instruction) | Instruction-only | ❌ | FT-only benefit |
| 4 | FT+RAG (instruction) | Instruction-only | ✅ | Transfer learning test |
| 5 | FT Only (RAG-trained) | RAG-trained | ❌ | Degradation test |
| 6 | FT+RAG (RAG-trained) | RAG-trained | ✅ | Optimal setup |

```bash
python scripts/core/compare_models.py --plan configs/evaluation/six_condition_experiment.yaml
```

📖 See [docs/experiment/](docs/experiment/) for detailed methodology.

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [Ollama](https://ollama.com/) for local inference
- macOS (Apple Silicon recommended) or Linux

### Installation

```bash
git clone https://github.com/ignaciolinari/beyond-the-cutoff.git
cd beyond-the-cutoff
python scripts/bootstrap_env.py
source .venv/bin/activate

# Pull required models
ollama pull qwen2.5:0.5b-instruct   # Base model
ollama pull qwen3:8b                 # Judge model
```

### Interactive RAG Assistant

Once you have papers ingested, ask questions:

```bash
python scripts/ask.py "What are the main contributions of paper X?"
```

## Project Structure

```
beyond-the-cutoff/
├── apps/                      # Interactive Streamlit applications
│   ├── human_annotation.py    # Human evaluation annotation UI
│   └── offline_task_viewer.py # Browse generated QA tasks
├── artifacts/                 # Training artifacts and checkpoints
├── configs/
│   ├── default.yaml          # Main pipeline configuration
│   ├── models/               # Model configurations (base, fine-tuned)
│   ├── evaluation/           # Experiment plans (6-condition, pairwise, etc.)
│   └── judges/               # LLM judge configurations
├── data/
│   ├── raw/                  # Downloaded PDFs from arXiv
│   ├── processed/            # Extracted text and manifests
│   └── external/             # FAISS index and embeddings
├── docs/
│   ├── experiment/           # Experiment design and methodology
│   ├── reports/              # Analysis results and findings
│   ├── reference/            # Technical documentation
│   ├── scaling/              # Guide for larger models
│   └── future/               # Planned features
├── evaluation/
│   ├── datasets/             # Train/eval JSONL datasets
│   ├── responses/            # Pre-generated model responses
│   ├── results/              # Evaluation metrics and details
│   └── exports/              # Batch files for external judges
├── notebooks/
│   ├── finetuning/           # LoRA training notebooks (Colab/Kaggle)
│   └── data_quality/         # Data analysis notebooks
├── ollama/                   # Modelfiles for Ollama registration
├── outputs/                  # Fine-tuned model weights (GGUF)
├── prompts/                  # Prompt templates
├── scripts/
│   ├── ask.py                # Interactive RAG assistant
│   ├── bootstrap_env.py      # Environment setup
│   ├── core/                 # Main evaluation pipeline
│   │   ├── compare_models.py
│   │   ├── generate_responses.py
│   │   └── interleaved_evaluation.py
│   ├── data/                 # Data processing scripts
│   │   ├── fetch_arxiv_corpus.py
│   │   ├── ingest_and_index.py
│   │   ├── generate_offline_dataset.py
│   │   └── split_dataset.py
│   ├── future/               # Advanced evaluation (ELO, retrieval ablation)
│   ├── utility/              # Analysis and inspection tools
│   └── validation/           # Experiment validation scripts
├── src/beyond_the_cutoff/    # Core Python library
│   ├── data/                 # PDF loading, chunking, extraction
│   ├── retrieval/            # FAISS indexing and querying
│   ├── evaluation/           # Judges, metrics, scoring
│   ├── datasets/             # Dataset generation
│   └── models/               # Model interfaces
├── tests/                    # Pytest test suite
└── vintage/                  # Archived configs and scripts
```

## Documentation

| Section | Description |
|---------|-------------|
| [📋 Experiment Setup](docs/experiment/setup.md) | 6-condition design |
| [📊 Methodology](docs/experiment/methodology.md) | Evaluation metrics |
| [🔧 Pipeline Reference](docs/reference/pipeline.md) | Full technical details |

See [docs/README.md](docs/README.md) for the complete documentation index.

## Models

| Model | Purpose | Ollama Tag |
|-------|---------|------------|
| Qwen 2.5 0.5B | Base model for experiments | `qwen2.5:0.5b-instruct` |
| Qwen 3 8B | LLM judge | `qwen3:8b` |
| Fine-tuned (instruction) | Trained without RAG contexts | `lora_science_0p5_instruction_only` |
| Fine-tuned (RAG-trained) | Trained with RAG contexts | `lora_science_0p5` |

## Development

```bash
pytest tests/              # Run tests
pre-commit run --all-files # Linting & formatting
mypy src/                  # Type checking
```

---

## Adapting the Pipeline

This pipeline is **model-agnostic**. To test with different models:

### Changing the Subject Model

| Model Size | Ollama Tag | Notes |
|------------|------------|-------|
| **3B** | `qwen2.5:3b-instruct` | Moderate capacity increase |
| **7B** | `qwen2.5:7b-instruct-q4_K_M` | ~11GB VRAM |
| **14B+** | `qwen2.5:14b-instruct-q4_K_M` | ~16GB VRAM |

1. Update `configs/models/base_ollama.yaml` with the new model tag
2. Fine-tune using the notebooks (adjust for model size)
3. Re-run the evaluation pipeline

### Hypotheses for Larger Models

| Aspect | 0.5B (Observed) | Larger Models (Hypothesis) |
|--------|-----------------|----------------------------|
| **FT benefit** | Marginal (+1.3%) | May increase with capacity |
| **Knowledge retention** | Limited | Better with more parameters |
| **RAG dependence** | Critical | May decrease |

📖 See [docs/scaling/](docs/scaling/) for detailed instructions.

---

## Future Work

Features implemented but not executed in this experiment:

| Feature | Script | Purpose |
|---------|--------|---------|
| **Live retrieval evaluation** | `scripts/future/evaluate_end_to_end.py` | Test with dynamic retrieval |
| **Retrieval ablation** | `scripts/future/run_retrieval_ablation.py` | Optimize top_k and rerankers |
| **ELO tournament** | `scripts/future/compute_elo_rankings.py` | Multi-model ranking |

See [docs/future/](docs/future/) and [scripts/future/README.md](scripts/future/README.md).

## License

MIT License. See [LICENSE](LICENSE) for details.
