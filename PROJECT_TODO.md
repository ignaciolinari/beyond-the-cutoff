# Beyond the Cutoff – Lightweight Project Tracker

_Last updated: 2025-11-13_

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
- [x] Build cleaner/normalizer to produce JSONL/text in `data/processed/` (partial complete: JSONL + text)
- [x] Capture metadata catalog (CSV/Parquet) with title, authors, DOI/arXiv ID, subjects, publication date
- [x] Extract page numbers and section titles during parsing for downstream citation grounding (page indices + heuristic section titles in place; evaluate accuracy)
- [x] Ingest downloaded PDFs and rebuild the FAISS index via `python scripts/ingest_and_index.py --config configs/default.yaml`
- [x] Publish generated manifests for raw and processed corpora (document schema + storage location)
- [x] Automate manifest regeneration/version bump whenever ingestion reruns
- [x] Generate offline QA/summary/citation tasks with `python scripts/generate_offline_dataset.py --config configs/default.yaml`
- [x] Add validation pass for offline dataset outputs (drops duplicate prompts, empty responses, and citation failures)
- [x] Write data quality checks and unit tests covering edge cases (chunk ordering regression added; expanded coverage)
- [x] Offline task viewer for human QA review
- [x] Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` (LoRA) in Colab/Kaggle using the offline JSONL outputs; export adapters and full checkpoints (artifacts in `outputs/lora_science_v1_instruction_only/`)
- [x] Quantize the tuned checkpoint to GGUF (`llama.cpp convert` + `quantize`) and register it with Ollama; update `configs/default.yaml` with the new model tag (GGUF + Ollama tag live)
- [x] Evaluate with the 3B judge using the refreshed datasets (or swap to a cloud judge/generator as needed) (metrics logged)

## Model Adaptation
- [x] Implement RAG pipeline (index build + query API)
- [x] Add cross-encoder re-ranker after dense retrieval (e.g., bge-reranker-small)
- [x] Implement section-aware chunking using headings/page boundaries
- [x] Include page/section metadata in retrieval mapping and surface inline citations in answers
- [x] Implement fine-tuning harness (LoRA/PEFT) with configuration options
- [x] Create Colab/Kaggle notebook that trains from `offline_dataset.jsonl`, logs hyperparameters, and exports adapter weights to `outputs/adapters/`
- [x] Create scripts to sync fine-tuned checkpoints between local and cloud environments (see `vintage/scripts/sync_checkpoints.py`)
- [x] Track experiment metadata (model version, dataset version, hyperparameters)

## Evaluation Suite
- [x] Design evaluation dataset (QA pairs, summaries) and storage format
- [x] Implement automatic scoring metrics (factuality, citation accuracy, BLEU/BERTScore) via `scripts/evaluation_harness.py` (uses `score_predictions()` function from `src/beyond_the_cutoff/evaluation/scoring.py`)
- [x] Build comparative evaluation harness for baseline RAG vs fine-tuned vs hybrid setups (`scripts/compare_models.py` + plan)
- [~] Integrate large-model judge (cloud or local) for automated evaluations and capture prompt/response logs (started with 3B judge scaled up to 7B... might scale up to cloud later)
- [x] Introduce a small evaluation harness (faithfulness, citation accuracy, retrieval hit@k/MRR)
- [x] Document the consolidated evaluation harness workflow (README + Makefile usage)
- [x] Provide CLI/report generator for comparative analysis (`compare_models.py` aggregates JSON report)
- [x] Add regression tests ensuring evaluation metrics run end-to-end (CLI harness smoke test)
- [x] Scale offline dataset/evaluations to >100 tasks covering new papers
- [ ] Integrate human evaluation protocol

## Documentation & Reporting
- [ ] Add architecture overview doc in `docs/`
- [~] Summarize ongoing experiments in `evaluation/results/`
- [ ] Establish dataset manifest/versioning guidelines for generated offline corpora

## Inference
- [x] Provide Ollama client wrapper and default config wiring
- [ ] Add streaming support and higher-level RAG integration tests
- [x] Expose minimal FastAPI `/ask` endpoint backed by RAG pipeline
- [ ] Harden Ollama automation (model alias creation + timeout mitigation) for unattended runs

## Milestones
1. **MVP Data + RAG Baseline** – data ingestion, simple retriever, evaluation on QA subset
	- Status: Done
2. **Fine-Tuning Integration** – LoRA training loop, checkpoint sync, comparative results
	- Status: Done
3. **Complete Evaluation Release** – polished scripts, documentation, replicable benchmarks
	- Status: In progress
