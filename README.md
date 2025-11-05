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

- Curate a 2025 corpus with at least 100 arXiv papers post-training-cutoff (see `docs/data_sourcing_plan.md` for category mix and scheduling).
- Extract metadata (title, abstract, authors, categories) automatically via the arXiv export API.
- Convert downloaded PDFs to clean text and JSONL format for fine-tuning and retrieval pipelines.
- Track manifest entries so downstream experiments can version the dataset alongside model checkpoints. A consolidated `manifest.json` plus metadata catalog exports (`metadata_catalog.csv` / `.parquet` and `corpus.jsonl`) are produced during ingestion so every run captures the exact document set.

#### arXiv Harvest Quickstart

```bash
python scripts/fetch_arxiv_corpus.py \
  --contact-email you@example.com \
  --total 100 \
  --output-dir data/raw/arxiv_2025
```

The CLI respects arXiv rate limits, writes metadata to JSONL/CSV, persists a manifest, and downloads PDFs with configurable retries/backoff. Use `--category` flags to adjust subject coverage or increase `--total` (for example to 200 when expanding into other fields).

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
- Sufficient free memory for 4B-8B parameter models (prefer 4-bit quantization); avoid swap usage on 8 GB devices
- [Ollama](https://ollama.com/) for downloading/serving the default quantized models
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) aligned with the installed `transformers` version.

### Quickstart

```bash
python scripts/bootstrap_env.py
source .venv/bin/activate
# Optional: seed a local Hugging Face cache for Colab/Kaggle syncs
python scripts/prefetch_models.py --cache-dir .cache/huggingface Qwen/Qwen2-0.5B-Instruct
# Pull the Ollama baselines (0.5B assistant, 1.5B generator, 3B judge) for pipeline testing
ollama pull qwen2:0.5b-instruct-q4_0
ollama pull qwen2:1.5b-instruct-q4_0
ollama pull qwen2.5:3b-instruct-q4_K_M
# Build/refresh the LoRA assistant alias used by the default config
ollama create qwen2-lora-science -f ollama/Modelfile
# Additional models once the pipeline hardens
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull phi4-mini:latest
```

The bootstrap script installs both runtime and development dependencies in editable mode and wires up the `pre-commit` hook so formatting and linting run automatically. Re-run the script at any time to pick up dependency updates (pass `--no-dev` or `--no-pre-commit` if you want to opt out).

### Local Inference (Ollama by default)

With Ollama running locally, the default configuration calls the `qwen2-lora-science:latest` tag for retrieval-augmented answering:

```python
from beyond_the_cutoff import load_config
from beyond_the_cutoff.models import build_generation_client

config = load_config()
client = build_generation_client(config.inference)
response = client.generate("Summarise the latest findings on generative retrieval.")
print(response["response"])
```

The default configuration connects to the Ollama daemon at `http://localhost:11434` and queries `qwen2-lora-science:latest`. Override `configs/default.yaml` (or pass a custom config such as the original baseline) to toggle providers, sampling parameters, or model tags.

## Pipeline Workflow

- **Ingestion/indexing**: run `python scripts/ingest_and_index.py --config configs/default.yaml` to turn the downloaded PDFs into text chunks and rebuild the FAISS index under `data/external/index`. Each run refreshes `data/processed/manifest.json` and writes metadata catalog exports under `data/processed/metadata_catalog*` for downstream analysis/versioning.
- **Offline tasks**: once the index exists, call `python scripts/generate_offline_dataset.py --config configs/default.yaml` so the `qwen2:1.5b-instruct-q4_0` generator can produce QA/summaries/citation tasks backed by those chunks.
- **Fine-tuning**: take the resulting JSONL plus your assistant prompts into Colab/Kaggle, fine-tune `Qwen/Qwen2-0.5B-Instruct` with LoRA, export the adapter/full weights, and keep the safetensors checkpoints.
- **Deployment**: convert that tuned checkpoint to GGUF (e.g., `llama.cpp convert` + `quantize`) and recreate the Ollama alias via `ollama create qwen2-lora-science -f ollama/Modelfile`; pull the 1.5B/3B Qwen tags locally via `ollama pull` for generation/judging.
- **Evaluation**: reuse `python scripts/ingest_and_index.py` results plus the evaluation datasets with the 3B judge (`qwen2.5:3b-instruct-q4_K_M`) or a cloud grader by flipping the provider/model in the config.
- **Verification**: after each stage, run `pytest tests/test_config.py` (and the broader suite once the pipeline is populated) to ensure the configuration and adapters remain wired correctly.

## Paper Assistant (RAG) Quickstart

Build a local retrieval index over your PDFs and ask questions grounded in the papers.

1) Ingest PDFs and build index (FAISS + sentence-transformers):

```bash
python scripts/ingest_and_index.py --config configs/default.yaml
# add --no-page-sidecars to skip writing per-page JSONL sidecars
```

Place your PDFs under `data/raw/` (or pass `--source PATH`). Processed text will be written to `data/processed/`, the processed manifest and metadata catalog will be regenerated under the same directory, and the FAISS index will be updated in `data/external/index/`.

Need to rebuild just the catalog? Use:

```bash
python scripts/build_metadata_catalog.py --config configs/default.yaml
```

Pass `--manifest` to reuse an existing manifest or `--output-prefix` to direct the CSV/Parquet/corpus files elsewhere.

2) Ask a question from the command line:

```bash
python scripts/ask.py "What are the main contributions of paper X?"
```

The CLI now surfaces inline citations with optional section headings and page numbers, plus a short snippet for quick scanning.

3) Optional: start a minimal API server:

```bash
uvicorn beyond_the_cutoff.api.server:app --reload --port 8000
```

POST to `/ask` with a JSON body like `{ "question": "..." }`.

Configuration knobs:
- `retrieval.chunk_size` / `retrieval.chunk_overlap`: text chunking
- `retrieval.top_k`: how many chunks to retrieve
- `retrieval.max_context_chars`: max context passed into the prompt
- `retrieval.chunking_strategy`: `words` (fast) or `sentences` (section-friendlier)
- `retrieval.embedding_model`: defaults to `BAAI/bge-small-en-v1.5`
- `retrieval.reranker_model`: optional cross-encoder for reranking (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)

Notes:
- Uses `BAAI/bge-small-en-v1.5` for embeddings by default; adjust for quality/speed.
- If a reranker is configured, top-k is reranked before prompting (with warnings logged when the reranker fails).
- Answers are generated via the configured backend (default: Ollama with `qwen2-lora-science:latest`).
- API responses include a `citations` array with `{id, source_path, page, token_start, token_end, score, excerpt}`.
- Responses also expose `citation_verification` metadata summarising which inline markers were found and their lexical overlap with retrieved context.

## Offline Prompt Pipeline

Pre-compute task instructions, prompts, and gold answers once so you can evaluate fine-tuning, plain RAG, and hybrid RAG + fine-tuning on identical retrieval outputs.

```bash
python scripts/generate_offline_dataset.py \
  --config configs/default.yaml \
  --output evaluation/datasets/offline_dataset.jsonl \
  --raw-tasks evaluation/datasets/offline_tasks.jsonl
```

The script orchestrates three steps:

- samples chunks from each indexed paper and asks the configured **dataset_generation.generator** (e.g., GPT-4-class endpoint or a strong local model) to author QA, summary, and citation-check instructions;
- runs the local `RAGPipeline.prepare_prompt` to attach the exact contexts, sources, and prompt that your assistant will see at inference time;
- stores both the curated task bank (`offline_tasks.jsonl`) and the expanded RAG-ready dataset (`offline_dataset.jsonl`) with per-example metadata (chunk ids, citation requirements, generator provenance).

Tune behaviour via `configs/default.yaml` → `dataset_generation` (counts per document, chunk limits, RNG seed, generator backend). Override paths or document caps with CLI flags like `--index-dir`, `--max-docs`, or `--output` when experimenting.

## Fine-Tuning Workflow

- Launch training from Colab/Kaggle notebooks that load `evaluation/datasets/offline_dataset.jsonl`, apply LoRA/PEFT using the `fine_tuning` config block, and export adapter weights (e.g., `adapter.safetensors`).
- Store notebooks under `notebooks/finetuning/` so runs are reproducible; checkpoint artefacts should be synced back into `outputs/adapters/` (or another path declared in `fine_tuning.adapter_output_dir`).
- Keep a manifest per run (model tag, dataset version, seeds, hyperparameters) so evaluations map cleanly back to the generated weights.

## Evaluation Strategy

- Use the offline dataset to compare three systems: baseline RAG, fine-tuned model without retrieval, and the hybrid fine-tuned+RAG assistant.
- For automated scoring, rely on a stronger judge model (cloud API or high-quality local checkpoint) to grade factuality, citation adherence, and summaries; log judge prompts/responses for reproducibility.
- Complement automatic grading with targeted human spot checks, prioritising disagreements or low-confidence judge outputs.
- Track results in `evaluation/results/` so trends over time (different checkpoints or datasets) remain auditable.
- Automate comparative sweeps with `python scripts/compare_models.py --plan configs/evaluation/compare_default.yaml` to evaluate multiple assistants and emit a consolidated JSON report.

### Automated Metrics Harness

- Run generation and retrieval metrics together via `python scripts/evaluation_harness.py --predictions <predictions.jsonl>`. The command defaults to the offline dataset from your loaded config and prints an aggregated JSON summary.
- Provide `--output` and `--details-output` paths to persist the overall metrics JSON and per-example JSONL rows. Override retrieval assets with `--index`/`--mapping`, and adjust Hit@K calculation with `--retrieval-topk`.
- The Makefile exposes `make score`, which wraps the harness under `BTC_USE_FAISS_STUB=1` so CI environments can execute without native FAISS. Tweak the `SCORE_*` variables at the top of the Makefile to point at different datasets or prediction files.

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
    ├── retrieval/       # Index builder and query-time RAG pipeline
    ├── api/             # Optional FastAPI server for /ask
    ├── evaluation/      # Metric calculations, scoring tools
    └── utils/           # Shared helpers
```

## Tooling

- `pyproject.toml` defines dependencies and tooling (ruff, mypy, pytest, etc.).
- Pre-commit hooks enforce formatting and linting.
- Configurable evaluation pipelines with Hydra or pydantic settings.
- Ollama streamlines downloading and running lightweight (≤4B) models in Q4/Q8 formats optimized for MLX when you switch to larger checkpoints.

### Configuration

- Primary settings live in `configs/default.yaml` and are validated by `beyond_the_cutoff.load_config()` (default fine-tuning base: `Qwen/Qwen2-0.5B-Instruct`).
- Paths in the config resolve relative to the repository root so you can keep environment-specific overrides minimal.
- The evaluation block exposes `offline_tasks_path` and `offline_dataset_path` to keep generated task banks and prompt/answer corpora alongside QA and summary sets.
- Provide alternate configuration files per experiment and pass their paths to `load_config` when needed.

### Model Handling

- Start with compact checkpoints such as `Qwen/Qwen2-0.5B-Instruct` for LoRA (default assistant), then rely on Ollama tags like `qwen2:1.5b-instruct-q4_0` for task generation and `qwen2.5:3b-instruct-q4_K_M` (or a cloud API) for judging.
- Sync fine-tuned checkpoints from Colab/Kaggle back into the local `models/` directory; keep quantized copies (GGUF) tailored to the local machine.
- Register custom quantized builds with Ollama via `ollama create` or `Modelfile` definitions so they can drop in for inference.

### Data & Checkpoints Sync

- Scripts in `scripts/` handle downloading, cleaning, and converting recent papers to JSONL; keep inputs in `data/raw/` and processed assets in `data/processed/`.
- Use cloud storage (Drive, S3, Hugging Face Hub) to move LoRA/PEFT checkpoints trained on remote notebooks; document sync commands (e.g., `scripts/sync_checkpoints.py`, to be created).
- Track versioning and quantization metadata inside each checkpoint to map evaluations back to the corresponding model.

## Roadmap

1. Implement reproducible data ingestion pipeline for 2025 papers.
2. Build QA generation scripts and evaluation dataset.
3. Develop RAG baseline with FAISS/Chroma + local inference via Ollama-backed models.
4. Orchestrate fine-tuning workflows on cloud notebooks (LoRA/PEFT) and sync checkpoints.
5. Run comparative evaluation suite; summarize findings with visualizations.

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
