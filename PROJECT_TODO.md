# Beyond the Cutoff – Lightweight Project Tracker

_Last updated: 2025-10-30_

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
- [ ] Draft data sourcing plan (target venues, scraping approach, rate limits)
- [ ] Implement arXiv downloader with rate limiting and retries to fetch 2025 papers into `data/raw/`
- [~] Build cleaner/normalizer to produce JSONL/text in `data/processed/` (partial: PDF → text)
- [ ] Capture metadata catalog (CSV/Parquet) with title, authors, DOI/arXiv ID, subjects, publication date
- [ ] Extract page numbers and section titles during parsing for downstream citation grounding
- [ ] Write data quality checks and unit tests covering edge cases

## Model Adaptation
- [ ] Set baseline model shortlist and document hardware requirements
- [x] Implement RAG pipeline (index build + query API)
- [x] Add cross-encoder re-ranker after dense retrieval (e.g., bge-reranker-small)
- [~] Implement section-aware chunking using headings/page boundaries
- [~] Include page/section metadata in retrieval mapping and surface inline citations in answers
- [ ] Implement fine-tuning harness (LoRA/PEFT) with configuration options
- [ ] Create scripts to sync fine-tuned checkpoints between local and cloud environments
- [ ] Track experiment metadata (model version, dataset version, hyperparameters)

## Evaluation Suite
- [ ] Design evaluation dataset (QA pairs, summaries) and storage format
- [ ] Implement automatic scoring metrics (factuality, citation accuracy, BLEU/BERTScore)
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

## Inference
- [x] Provide Ollama client wrapper and default config wiring
- [ ] Add streaming support and higher-level RAG integration tests
- [x] Expose minimal FastAPI `/ask` endpoint backed by RAG pipeline

## Milestones
1. **MVP Data + RAG Baseline** – data ingestion, simple retriever, evaluation on QA subset
	- Status: In progress (RAG pipeline done; downloader/eval subset pending)
2. **Fine-Tuning Integration** – LoRA training loop, checkpoint sync, comparative results
3. **Complete Evaluation Release** – polished scripts, documentation, replicable benchmarks
