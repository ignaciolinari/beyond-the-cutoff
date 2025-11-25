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
- Convert downloaded PDFs to clean text and JSONL format for fine-tuning and retrieval pipelines. PDF extraction includes automatic quality analysis with 17 metrics covering parse success rate, content volume, structural integrity, text composition, and overall confidence scores.
- Track manifest entries so downstream experiments can version the dataset alongside model checkpoints. A consolidated `manifest.json` plus metadata catalog exports (`metadata_catalog.csv` / `.parquet` and `corpus.jsonl`) are produced during ingestion so every run captures the exact document set. Extraction quality metrics are saved as `.quality.json` sidecars alongside each converted text file.

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
- Candidate models: prioritize the Qwen2.5 family (`Qwen/Qwen2.5-0.5B-Instruct` for LoRA experiments, `Qwen/Qwen2.5-3B-Instruct` for higher-quality assistants) and their matching Ollama builds. Reserve the 7B quantized tag (`qwen2.5:7b-instruct-q4_K_M`) for generation and judging when additional headroom is needed.
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
python scripts/prefetch_models.py --cache-dir .cache/huggingface \
  Qwen/Qwen2.5-0.5B-Instruct \
  Qwen/Qwen2.5-3B-Instruct
# Pull the Ollama baselines (0.5B assistant, 3B assistant, 7B generator/judge)
ollama pull qwen2.5:0.5b-instruct
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M
```

The bootstrap script installs both runtime and development dependencies in editable mode and wires up the `pre-commit` hook so formatting and linting run automatically. Re-run the script at any time to pick up dependency updates (pass `--no-dev` or `--no-pre-commit` if you want to opt out).

### Local Inference (Ollama by default)

With Ollama running locally, the default configuration calls the `qwen2.5:0.5b-instruct` tag for retrieval-augmented answering during the first experiment sequence:

```python
from beyond_the_cutoff import load_config
from beyond_the_cutoff.models import build_generation_client

config = load_config()
client = build_generation_client(config.inference)
response = client.generate("Summarise the latest findings on generative retrieval.")
print(response["response"])
```

The default configuration connects to the Ollama daemon at `http://localhost:11434` and queries `qwen2.5:0.5b-instruct`. Override `configs/default.yaml` (or pass a custom config such as the original baseline) to toggle providers, sampling parameters, or model tags. Swap the `inference.model` to `qwen2.5:3b-instruct-q4_K_M` once the 0.5B 6-condition experiments finish so you can rerun the full comparison at 3B capacity.

## Pipeline Workflow

- **Ingestion/indexing**: run `python scripts/ingest_and_index.py --config configs/default.yaml` to turn the downloaded PDFs into text chunks and rebuild the FAISS index under `data/external/index`. Each run refreshes `data/processed/manifest.json` and writes metadata catalog exports under `data/processed/metadata_catalog*` for downstream analysis/versioning.
- **Offline tasks**: once the index exists, call `python scripts/generate_offline_dataset.py --config configs/default.yaml` so the `qwen2.5:7b-instruct-q4_K_M` generator can produce QA/summaries/citation tasks backed by those chunks. The offline dataset generation system uses a modular architecture with separate components for parsing, validation, citation enforcement, and document metadata management.
- **Fine-tuning**: take the resulting JSONL into Colab/Kaggle notebooks (`notebooks/finetuning/`). For the 6-condition experiment, train TWO models: (1) `lora_science_v1_instruction_only.ipynb` trains WITHOUT RAG contexts, (2) `lora_science_v1.ipynb` trains WITH RAG contexts. Export adapter/full weights and keep the safetensors checkpoints.
- **Deployment**: convert tuned checkpoints to GGUF (e.g., `llama.cpp convert` + `quantize`) and, if desired, register custom Ollama aliases; the default pipeline now calls the stock Qwen2.5 tags directly.
- **Evaluation**: reuse `python scripts/ingest_and_index.py` results plus the evaluation datasets with the 7B judge (`qwen2.5:7b-instruct-q4_K_M`) or a cloud grader by flipping the provider/model in the config.
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
- `retrieval.embedding_model`: defaults to `BAAI/bge-m3`
- `retrieval.reranker_model`: optional cross-encoder for reranking (default `BAAI/bge-reranker-v2-m3`)

Notes:
- Uses `BAAI/bge-m3` for embeddings by default; adjust for quality/speed.
- If a reranker is configured, top-k is reranked before prompting (with warnings logged when the reranker fails).
- Answers are generated via the configured backend (default: Ollama with `qwen2.5:0.5b-instruct`; upgrade to the 3B tag when starting the second experiment series).
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

### Offline Task Viewer

Launch a Streamlit app to inspect the generated tasks next to their source documents. It reads both the curated dataset and the raw generator output so status/errors are visible while spot-checking.

```bash
streamlit run apps/offline_task_viewer.py
```

The sidebar exposes text boxes for the curated dataset and raw tasks JSONL paths (defaults to the evaluation artefacts under `evaluation/datasets/`). Use the document search to jump to a paper, filter by task type, and expand tasks to review prompts, responses, retrieved contexts, and metadata. The document column previews the corresponding paper text (first *n* characters, adjustable).

## Fine-Tuning Workflow

For the 6-condition experiment, **two fine-tuned models are required**:

1. **Instruction-only model** (`lora_science_v1_instruction_only.ipynb`): Trains WITHOUT RAG contexts
   - Used for: FT-only (condition 3) and FT+RAG instruction-only (condition 4)
   - Register with Ollama: `ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only`

2. **RAG-trained model** (`lora_science_v1.ipynb`): Trains WITH RAG contexts
   - Used for: RAG-trained FT-only (condition 5) and RAG-trained FT+RAG (condition 6)
   - Register with Ollama: `ollama create lora_science_0p5 -f ollama/Modelfile.rag_trained`

Both notebooks load `evaluation/datasets/offline_dataset.jsonl`, apply LoRA/PEFT using the `fine_tuning` config block, and export adapter weights (e.g., `adapter.safetensors`). Store notebooks under `notebooks/finetuning/` so runs are reproducible; checkpoint artefacts should be synced back into `outputs/adapters/` (or another path declared in `fine_tuning.adapter_output_dir`). Keep a manifest per run (model tag, dataset version, seeds, hyperparameters) so evaluations map cleanly back to the generated weights.

## Evaluation Strategy

- Use the offline dataset to compare six conditions: base baseline, RAG baseline, FT-only (instruction-only), FT+RAG (instruction-only), RAG-trained FT-only, and RAG-trained FT+RAG. See `docs/six_condition_experiment_setup.md` for details.
- For automated scoring, rely on a stronger judge model (cloud API or high-quality local checkpoint) to grade factuality, citation adherence, and summaries; log judge prompts/responses for reproducibility.
- Complement automatic grading with targeted human spot checks, prioritising disagreements or low-confidence judge outputs.
- Track results in `evaluation/results/` so trends over time (different checkpoints or datasets) remain auditable.
- Automate comparative sweeps with `python scripts/compare_models.py --plan configs/evaluation/compare_0p5b_experiments.yaml` to evaluate multiple assistants and emit a consolidated JSON report.

### Automated Metrics Harness

- Run generation and retrieval metrics together via `python scripts/evaluation_harness.py --predictions <predictions.jsonl>`. The command defaults to the offline dataset from your loaded config and prints an aggregated JSON summary.
- Provide `--output` and `--details-output` paths to persist the overall metrics JSON and per-example JSONL rows. Override retrieval assets with `--index`/`--mapping`, and adjust Hit@K calculation with `--retrieval-topk`.
- The Makefile exposes `make score`, which wraps the harness under `BTC_USE_FAISS_STUB=1` so CI environments can execute without native FAISS. Tweak the `SCORE_*` variables at the top of the Makefile to point at different datasets or prediction files.

### Experiment Validation

The evaluation pipeline includes automatic validation to ensure reproducibility and correctness:

- **Configuration validation**: Checks config files exist, are valid, and don't have conflicts (e.g., prompt mode vs judge config mismatch)
- **Dataset versioning**: Ensures all runs in a comparison use the same dataset version (via SHA256 hashing)
- **Reproducibility checks**: Validates that experiment metadata includes all required fields
- **Evaluation sanity checks**: Detects common issues like high error rates, missing predictions, empty responses

Validation runs automatically during evaluation, but you can also run it independently:

```bash
# Validate configuration before running evaluation
python scripts/validate_experiment.py \
    --config configs/default.yaml \
    --model-config configs/rag_baseline_ollama.yaml \
    --judge-config configs/judges/scientific_default_rag.yaml \
    --prompt-mode rag

# Validate dataset versioning across runs
python scripts/validate_experiment.py \
    --dataset evaluation/datasets/offline_dataset.jsonl \
    --dataset evaluation/results/rag_baseline_0p5b/details.jsonl

# Validate experiment reproducibility
python scripts/validate_experiment.py \
    --metadata evaluation/results/rag_baseline_0p5b/metadata.jsonl

# Validate evaluation results
python scripts/validate_experiment.py \
    --metrics evaluation/results/rag_baseline_0p5b/metrics.json \
    --details evaluation/results/rag_baseline_0p5b/details.jsonl
```

### Visualization Tools

Generate visualizations from evaluation results to compare models:

```bash
# Visualize from comparison report JSON
python scripts/visualize_comparison.py \
    --report evaluation/results/comparison_report.json \
    --output evaluation/results/visualizations/

# Visualize from individual metrics files
python scripts/visualize_comparison.py \
    --metrics evaluation/results/rag_baseline_0p5b/metrics.json \
    --metrics evaluation/results/lora_science_0p5b_ft_only/metrics.json \
    --output evaluation/results/visualizations/

# Generate specific visualizations only
python scripts/visualize_comparison.py \
    --report evaluation/results/comparison_report.json \
    --output evaluation/results/visualizations/ \
    --only metrics error-rates citations
```

The visualization tool generates:
- **Metrics comparison**: Bar charts comparing judge scores (factuality, grounding, completeness, communication) across models
- **Error rates**: Comparison of error rates and examples with errors
- **Citation metrics**: Citation coverage, precision, and recall for RAG models
- **Timing comparison**: Generation, judge, and total timing metrics
- **Prompt mode comparison**: RAG vs instruction-only mode performance comparison
- **Task type breakdown**: Distribution of examples across task types

### ELO Ranking System

The project includes an ELO-based ranking system for comparing model performance through pairwise comparisons:

```bash
# Compute ELO rankings from pairwise comparison results
python scripts/compute_elo_rankings.py evaluation/results/pairwise_comparisons.jsonl \
    --output evaluation/results/elo_rankings.json \
    --k-factor 32 \
    --bootstrap-samples 1000

# View rankings summary
python scripts/compute_elo_rankings.py evaluation/results/pairwise_comparisons.jsonl --summary
```

Key features:
- **Bootstrap confidence intervals**: Statistical significance testing with configurable sample sizes
- **Configurable K-factor**: Control rating volatility (default: 32)
- **Multiple comparison sources**: Works with human annotations or automated judge outputs

### Automated Pairwise Evaluation

For fully automated evaluation without human annotation, the project supports multi-judge pairwise comparison:

```bash
# Run automated pairwise evaluation using evaluation plan
python scripts/run_pairwise_evaluation.py \
    --plan configs/evaluation/pairwise_evaluation_plan.yaml \
    --output evaluation/results/pairwise_rankings

# Ad-hoc comparison between models using result directories
python scripts/run_pairwise_evaluation.py \
    --results base=evaluation/results/base_baseline_0p5b \
    --results rag=evaluation/results/rag_baseline_0p5b \
    --judge configs/judges/pairwise_qwen7b.yaml \
    --judge configs/judges/pairwise_qwen3_8b.yaml \
    --judge configs/judges/pairwise_llama31_8b.yaml \
    --output evaluation/results/pairwise_rankings

# Use specific judge configuration
python scripts/run_pairwise_evaluation.py \
    --results baseline=evaluation/results/base_baseline_0p5b \
    --results finetuned=evaluation/results/lora_science_0p5b_ft_only \
    --judge configs/judges/pairwise_qwen7b.yaml \
    --output evaluation/results/pairwise_rankings
```

Features:
- **Multi-judge consensus**: Uses multiple judge models (Qwen 2.5 7B, Qwen3 8B, Llama 3.1 8B) with majority voting
- **Position debiasing**: Automatically swaps response positions to avoid order bias
- **Structured output**: JSON-based judge responses with fallback keyword detection
- **Configurable retries**: Handles transient API failures gracefully

Judge configurations are stored in `configs/judges/` (e.g., `pairwise_qwen7b.yaml`, `pairwise_qwen3_8b.yaml`, `pairwise_llama31_8b.yaml`).

### Human Evaluation (Optional)

For validation studies requiring human judgment:

```bash
# Launch annotation interface
streamlit run apps/human_annotation.py

# Generate annotation tasks from evaluation results
python -c "from beyond_the_cutoff.evaluation.human_evaluation import sample_for_annotation; ..."
```

The human evaluation module supports:
- **Stratified sampling**: Balance tasks across categories and difficulty levels
- **Inter-annotator agreement**: Cohen's Kappa (2 annotators) and Fleiss' Kappa (3+ annotators)
- **Annotation batching**: Manageable task sets with progress tracking

See `docs/elo_ranking_and_human_evaluation.md` for detailed documentation.

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

- Primary settings live in `configs/default.yaml` and are validated by `beyond_the_cutoff.load_config()` (default fine-tuning base: `Qwen/Qwen2.5-0.5B-Instruct`).
- Paths in the config resolve relative to the repository root so you can keep environment-specific overrides minimal.
- The evaluation block exposes `offline_tasks_path` and `offline_dataset_path` to keep generated task banks and prompt/answer corpora alongside QA and summary sets.
- Provide alternate configuration files per experiment and pass their paths to `load_config` when needed.

### Model Handling

- Start with compact checkpoints such as `Qwen/Qwen2.5-0.5B-Instruct` for LoRA (default assistant), then rely on Ollama tags like `qwen2.5:7b-instruct-q4_K_M` for task generation and judging, and promote to `qwen2.5:3b-instruct-q4_K_M` when repeating the experiments at the larger model size.
- Sync fine-tuned checkpoints from Colab/Kaggle back into the local `models/` directory; keep quantized copies (GGUF) tailored to the local machine.
- Register custom quantized builds with Ollama via `ollama create` or `Modelfile` definitions so they can drop in for inference.

### Data & Checkpoints Sync

- Scripts in `scripts/` handle downloading, cleaning, and converting recent papers to JSONL; keep inputs in `data/raw/` and processed assets in `data/processed/`.
- Use cloud storage (Drive, S3, Hugging Face Hub) to move LoRA/PEFT checkpoints trained on remote notebooks. See `vintage/scripts/sync_checkpoints.py` for reference implementation.
- Track versioning and quantization metadata inside each checkpoint to map evaluations back to the corresponding model.

## Roadmap

1. Implement reproducible data ingestion pipeline for 2025 papers.
2. Build QA generation scripts and evaluation dataset.
3. Develop RAG baseline with FAISS/Chroma + local inference via Ollama-backed models.
4. Orchestrate fine-tuning workflows on cloud notebooks (LoRA/PEFT) and sync checkpoints (see `vintage/scripts/sync_checkpoints.py`).
5. Run comparative evaluation suite; summarize findings with visualizations.

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
