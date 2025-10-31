# Beyond the Cutoff – Lightweight Project Tracker

_Last updated: 2025-10-31_

## Status Legend
- [ ] Backlog
- [~] In Progress (partial completion)
- [x] Complete

## Foundation
- [x] Document environment setup script (`scripts/bootstrap_env.py`) and usage in README quickstart
- [x] Add `pre-commit` configuration and CI hook to enforce formatting
- [x] Populate `src/beyond_the_cutoff/__init__.py` with package exports once modules exist
- [x] Define project-wide configuration schema (Hydra or Pydantic) under `configs/`

## Data Pipeline
- [x] Draft data sourcing plan (target venues, scraping approach, rate limits)
- [x] Implement arXiv downloader with rate limiting and retries to fetch 2025 papers into `data/raw/`
- [~] Build cleaner/normalizer to produce JSONL/text in `data/processed/` (partial: PDF → text)
- [ ] Capture metadata catalog (CSV/Parquet) with title, authors, DOI/arXiv ID, subjects, publication date
- [ ] Extract page numbers and section titles during parsing for downstream citation grounding
- [ ] Ingest downloaded PDFs and rebuild the FAISS index via `python scripts/ingest_and_index.py --config configs/default.yaml`
- [ ] Generate offline QA/summary/citation tasks with `python scripts/generate_offline_dataset.py --config configs/default.yaml`
- [ ] Add validation pass for offline dataset outputs (duplicate prompts, empty answers, citation coverage)
- [ ] Write data quality checks and unit tests covering edge cases
- [ ] Fine-tune `Qwen/Qwen2-0.5B-Instruct` (LoRA) in Colab/Kaggle using the offline JSONL outputs; export adapters and full checkpoints
- [ ] Quantize the tuned checkpoint to GGUF (`llama.cpp convert` + `quantize`) and register it with Ollama; update `configs/default.yaml` with the new model tag
- [ ] Evaluate with the 3B judge using the refreshed datasets (or swap to a cloud judge/generator as needed)
- [ ] Run `pytest tests/test_config.py` after each stage to confirm configuration wiring remains consistent

## Model Adaptation
- [ ] Set baseline model shortlist and document hardware requirements
- [x] Implement RAG pipeline (index build + query API)
- [x] Add cross-encoder re-ranker after dense retrieval (e.g., bge-reranker-small)
- [~] Implement section-aware chunking using headings/page boundaries
- [~] Include page/section metadata in retrieval mapping and surface inline citations in answers
- [ ] Implement fine-tuning harness (LoRA/PEFT) with configuration options
- [ ] Create Colab/Kaggle notebook that trains from `offline_dataset.jsonl`, logs hyperparameters, and exports adapter weights to `outputs/adapters/`
- [ ] Create scripts to sync fine-tuned checkpoints between local and cloud environments
- [ ] Track experiment metadata (model version, dataset version, hyperparameters)

## Evaluation Suite
- [ ] Design evaluation dataset (QA pairs, summaries) and storage format
- [ ] Implement automatic scoring metrics (factuality, citation accuracy, BLEU/BERTScore)
- [ ] Build comparative evaluation harness for baseline RAG vs fine-tuned vs hybrid setups
- [ ] Integrate large-model judge (cloud or local) for automated evaluations and capture prompt/response logs
- [ ] Introduce a small evaluation harness (faithfulness, citation accuracy, retrieval hit@k/MRR)
- [ ] Integrate human evaluation protocol
- [ ] Provide CLI/report generator for comparative analysis
- [ ] Add regression tests ensuring evaluation metrics run end-to-end

## Documentation & Reporting
- [ ] Add architecture overview doc in `docs/`
- [ ] Maintain changelog (`CHANGELOG.md`) once first milestone ships
- [ ] Create contribution guidelines and coding standards
- [ ] Summarize ongoing experiments in `evaluation/results/`
- [ ] Optional: Streamlit mini app that calls the FastAPI `/ask` endpoint
- [ ] Establish dataset manifest/versioning guidelines for generated offline corpora

## Inference
- [x] Provide Ollama client wrapper and default config wiring
- [ ] Add streaming support and higher-level RAG integration tests
- [x] Expose minimal FastAPI `/ask` endpoint backed by RAG pipeline

## Milestones
1. **MVP Data + RAG Baseline** – data ingestion, simple retriever, evaluation on QA subset
	- Status: In progress (RAG pipeline done; downloader/eval subset pending)
2. **Fine-Tuning Integration** – LoRA training loop, checkpoint sync, comparative results
3. **Complete Evaluation Release** – polished scripts, documentation, replicable benchmarks
