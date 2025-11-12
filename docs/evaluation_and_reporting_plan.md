# Evaluation & Reporting Plan

_Last updated: 2025-11-11_

## Metric Capture
- Implement CLI `python scripts/evaluate_models.py` to:
  - Load judge config (`configs/judges/scientific_default.yaml`).
  - Compute factuality, grounding, completeness, and clarity scores.
  - Aggregate citation coverage statistics from `RAGPipeline.verify_citations`.
  - Record latency (seconds per sample) and cost estimates (for cloud judges/generators).

## Automation Strategy
- Nightly GitHub Action (to be added) running:
  1. `pytest tests/` with focus on ingestion, chunking, retrieval.
  2. Evaluation CLI on latest dataset snapshot using baseline models.
  3. Notebook smoke test `notebooks/evaluation/sanity_checks.ipynb` executed via `papermill`.
- Store logs and artifacts under `evaluation/results/nightly/<YYYYMMDD>/`.

## Judge Selection & Calibration
- Default local judge: `qwen2.5:7b-instruct-q4_K_M` via Ollama (no API cost).
- Calibration judge: GPT-4o (API) sampled on 25% of evaluation set.
- Calibration protocol:
  1. Run both judges on identical subset.
  2. Compute correlation and disagreement stats.
  3. Adjust local judge thresholds if drift > 10% relative to GPT-4o.
- Document calibration outcomes in `evaluation/results/judge_calibration_<date>.md`.

## Reporting Templates
- Markdown summary (`evaluation/results/<experiment_id>/report.md`) containing:
  - Experiment metadata (dataset tag, model versions, judge config).
  - Metric table (with deltas vs `rag_baseline_v1`).
  - Citation coverage histogram image.
  - Latency and cost breakdown.
- CSV export (`evaluation/results/<experiment_id>/metrics.csv`) for downstream plotting.

## Alerting & Thresholds
- Trigger alert when:
  - Factuality drops >3 percentage points vs previous run.
  - Citation coverage <0.30 for any task type.
  - Judge disagreement (local vs GPT-4o) >15 percentage points.
- Alerts logged to `evaluation/results/alerts.md` and shared in the evaluation Slack channel.

## Data Dependencies
- Index manifest path and dataset tag must be referenced in every report to ensure traceability.
- QA outputs from `evaluation/results/data_quality/` feed into reporting (e.g., token distributions for context).

## Future Enhancements
- Integrate streaming support for evaluating latency improvements.
- Add human evaluation sampling workflow (track in separate doc).
- Build dashboard (Streamlit or Observable) sourcing from CSV metrics for executive summary.
