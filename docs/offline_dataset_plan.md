# Offline Dataset Configuration & QA Plan

_Last updated: 2025-11-01_

## Goals
- Produce reusable QA, summary, and citation-check examples tailored to post-cutoff scientific papers.
- Guarantee each generated example includes citation coverage metadata and reproducible generation parameters.
- Filter out noisy or formula-heavy chunks that degrade downstream fine-tuning quality.

## Configuration Baseline
- Generator model: `qwen2:1.5b-instruct-q4_0` (override via CLI flag when experimenting).
- Default options to record per run: temperature, top_p, max_new_tokens, RNG seed.
- Output paths: inherit from `dataset_generation` config (`evaluation/datasets/offline_dataset.jsonl`, `evaluation/datasets/offline_tasks.jsonl`).

## Prompt Design
- Maintain prompt templates in `configs/dataset_generation/` (to be created). Each template should document:
  - Target scientific tone (emphasise methods, results, limitations).
  - Citation expectations (e.g., require inline `[n]` markers referencing supplied chunks).
  - Reject instructions referencing figures/tables when text context is insufficient.

## Filtering Rules
- Drop chunks where 40%+ of characters are mathematical symbols or references (regex-based heuristic).
- Omit contexts dominated by bibliography sections (detect via keywords: "References", "Bibliography").
- Flag and review outputs when the generated answer contains fewer than 3 sentences for QA tasks.

## Validation Reports
- Generate `evaluation/datasets/reports/offline_dataset_report_<date>.md` covering:
  - Counts per task type.
  - Percentage of examples meeting citation coverage threshold (≥ 0.35 as computed via `RAGPipeline.verify_citations`).
  - Duplicate instruction detection (case-insensitive comparison across tasks).
  - Top 10 documents contributing tasks; ensure no single paper exceeds 8% of the dataset.

## Logging & Audit Trail
- For each run, append a line to `evaluation/datasets/raw_tasks/run_log.md` with:
  - Timestamp.
  - Generator model tag.
  - Seed.
  - Prompt template ID.
  - Total tasks generated and filtered counts.
- Store raw generator responses in `evaluation/datasets/raw_tasks/<run_id>.jsonl` (one JSON blob per document).
- Capture the CLI command (or Hydra config) used for reproducibility.

## Next Steps
- Create prompt templates and wire CLI arguments to select them.
- Implement filtering heuristics in `OfflineDatasetGenerator` (tracked separately).
- Add pytest coverage to ensure dataset exports contain citation metadata and non-empty answers.
