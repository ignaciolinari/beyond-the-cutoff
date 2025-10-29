# Beyond the Cutoff

Evaluating how large language models update knowledge beyond their training data cutoff by comparing fine-tuning and retrieval-augmented generation (RAG) on new scientific papers for research-assistant applications.

## Overview

Beyond the Cutoff investigates how large language models (LLMs) can acquire and integrate new scientific knowledge that was not available during their original training. The project compares two main adaptation strategies:

1. **Fine-tuning** — updating model parameters directly using new textual data.
2. **Retrieval-Augmented Generation (RAG)** — enriching model context dynamically through external document retrieval.

We evaluate each approach in research-assistant tasks, such as:

- Scientific question answering (QA)
- Summarization and synthesis of findings
- Citation-grounded reasoning and factual accuracy

The benchmark focuses on recent (2025) peer-reviewed papers that are certainly out-of-distribution for current open-weight LLMs. The ultimate goal is to measure how effectively different methods allow a model to update its factual and reasoning capabilities beyond its pretraining cutoff.

## Objectives

- Assess the temporal generalization of LLMs when exposed to unseen, post-training data.
- Compare fine-tuning and RAG in terms of factual accuracy, coherence, citation consistency, and efficiency.
- Provide a reproducible evaluation framework for testing LLM knowledge updating.
- Explore implications for building adaptive research assistants capable of staying current with new literature.

## Methodology

### 1. Data Collection

- Curate a corpus of 2025 scientific papers from open-access repositories (arXiv, bioRxiv, medRxiv, etc.).
- Extract metadata (title, abstract, DOI, field) automatically.
- Convert documents to clean text and JSONL format for fine-tuning and retrieval pipelines.

### 2. Model Adaptation

- Fine-tuning: lightweight instruction tuning via LoRA / PEFT. Training runs on cloud notebooks (e.g., Kaggle, Colab) using GPUs.
- RAG: local retrieval pipeline with FAISS or Chroma backends.
- Candidate models: prioritize lightweight checkpoints (≤4B parameters) such as Phi-3 mini / Phi-3.5 mini, Qwen 2.5 3B, or similar Apple-optimized MLX builds. Larger 7B–8B models (e.g., Llama 3 Instruct, Mistral) should only be used when available in 4-bit quantized form and with ample memory headroom.
- Fine-tuned checkpoints synchronized back to the local environment for evaluation.

### 3. Evaluation Framework

- Create synthetic and human-curated QA pairs derived from the 2025 corpus.
- Score factuality, citation correctness, and coherence using local evaluator models.
- Track quantitative metrics (BLEU, BERTScore, factual consistency) and qualitative feedback from human raters.

## Local Development Setup

### Prerequisites

- macOS (Apple Silicon recommended)
- Python ≥ 3.10
- [Apple MLX](https://github.com/ml-explore/mlx) for accelerated inference
- Sufficient free memory for 3B–4B parameter models (prefer 4-bit quantization); avoid swap usage on 8 GB devices
- [Ollama](https://ollama.com/) (optional) for downloading/serving macOS-ready quantized models

### Quickstart

```bash
python scripts/bootstrap_env.py
source .venv/bin/activate
brew install ollama # or follow https://ollama.com/download
ollama pull phi3:mini
```

The bootstrap script installs both runtime and development dependencies in editable mode and wires up the `pre-commit` hook so formatting and linting run automatically. Re-run the script at any time to pick up dependency updates (pass `--no-dev` or `--no-pre-commit` if you want to opt out).

### Local Inference (Ollama)

1. Start the Ollama service: `ollama serve` (or rely on auto-start via `ollama run`).
2. Generate text with the packaged helper:

```python
from beyond_the_cutoff import OllamaClient, load_config

config = load_config()
client = OllamaClient(model=config.inference.model, host=config.inference.host, port=config.inference.port)
response = client.generate("Summarise the latest findings on generative retrieval.")
print(response["response"])
```

The default configuration assumes the daemon listens on `http://localhost:11434` and that the `phi3:mini` tag is available. Update `configs/default.yaml` or provide an alternative config file to point at a different tag or host.

### Project Structure

```
├── configs/             # YAML/JSON configs for data, training, evaluation
├── data/
│   ├── raw/             # Original downloaded papers
│   ├── processed/       # Cleaned/plain text versions
│   └── external/        # External resources (embeddings, metrics)
├── evaluation/          # Evaluation scripts, reports, and results
├── notebooks/           # Exploratory analysis and visualization
├── scripts/             # Utility scripts (data prep, training orchestrators)
└── src/beyond_the_cutoff/
    ├── __init__.py
    ├── data/            # Data loading and preprocessing modules
    ├── models/          # Fine-tuning wrappers, RAG components
    ├── evaluation/      # Metric calculations, scoring tools
    └── utils/           # Shared helpers
```

## Tooling

- `pyproject.toml` defines dependencies and tooling (ruff, mypy, pytest, etc.).
- Pre-commit hooks enforce formatting and linting.
- Configurable evaluation pipelines with Hydra or pydantic settings.
- Ollama streamlines downloading and running lightweight (≤4B) models in Q4/Q8 formats optimized for MLX.

### Configuration

- Primary settings live in `configs/default.yaml` and are validated by `beyond_the_cutoff.load_config()` (default base model: `microsoft/Phi-3-mini-4k-instruct`).
- Paths in the config resolve relative to the repository root so you can keep environment-specific overrides minimal.
- Provide alternate configuration files per experiment and pass their paths to `load_config` when needed.

### Model Handling

- Keep local models lightweight (Phi-3 mini, Qwen 2.5 3B) via Ollama (`ollama pull qwen2.5:3b`), Hugging Face MLX checkpoints, or GGUF conversions.
- For 7B–8B models rely on quantized (Q4) variants and load them only when you have spare memory; close heavy processes before inference.
- Sync fine-tuned checkpoints from Colab/Kaggle back into the local `models/` directory; keep quantized copies tailored to the local machine.
- Manage caches under `~/.cache/beyond-the-cutoff/` or within Ollama to avoid repeated downloads.

### Data & Checkpoints Sync

- Scripts in `scripts/` should handle downloading, cleaning, and converting recent papers to JSONL; keep inputs in `data/raw/` and processed assets in `data/processed/`.
- Use cloud storage (Drive, S3, Hugging Face Hub) to move LoRA/PEFT checkpoints trained on remote notebooks; document sync commands (e.g., `scripts/sync_checkpoints.py`, to be created).
- Track versioning and quantization metadata inside each checkpoint to map evaluations back to the corresponding model.

## Roadmap

1. Implement reproducible data ingestion pipeline for 2025 papers.
2. Build QA generation scripts and evaluation dataset.
3. Develop RAG baseline with FAISS/Chroma + local inference via MLX.
4. Orchestrate fine-tuning workflows on cloud notebooks (LoRA/PEFT) and sync checkpoints.
5. Run comparative evaluation suite; summarize findings with visualizations.

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
