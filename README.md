# Beyond the Cutoff

Evaluating how large language models update knowledge beyond their training data cutoff by comparing **fine-tuning** and **retrieval-augmented generation (RAG)** on recent scientific papers.

## Overview

This project investigates how LLMs can acquire new scientific knowledge published after their training cutoff. We compare two adaptation strategies:

| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **Fine-tuning** | Update model weights with new data | Internalizes knowledge | Expensive, may forget |
| **RAG** | Retrieve relevant context at inference | No retraining needed | Depends on retrieval quality |

The benchmark uses **2025 arXiv papers** that are out-of-distribution for current open-weight LLMs.

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
├── configs/
│   ├── models/           # Model configurations
│   ├── evaluation/       # Experiment plans
│   └── judges/           # LLM judge configs
├── scripts/
│   ├── core/             # Main pipeline (compare_models, generate_responses)
│   ├── data/             # Data processing (fetch, ingest, split, generate)
│   └── validation/       # Experiment validation
├── src/beyond_the_cutoff/  # Core library
├── evaluation/
│   ├── datasets/         # Train/eval datasets
│   ├── responses/        # Pre-generated model responses
│   └── results/          # Evaluation results
├── notebooks/finetuning/ # Cloud training notebooks
└── docs/                 # Documentation
```

## Documentation

| Section | Description |
|---------|-------------|
| [📋 Experiment Setup](docs/experiment/setup.md) | 6-condition design |
| [📊 Methodology](docs/experiment/methodology.md) | Evaluation metrics |
| [🔧 Pipeline Reference](docs/reference/pipeline.md) | Full technical details |
| [📖 Detailed Usage](docs/reference/detailed_usage.md) | Complete command reference |

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

## License

MIT License. See [LICENSE](LICENSE) for details.
