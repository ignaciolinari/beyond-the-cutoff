# Regression Test Plan

_Last updated: 2025-11-04_

## Objectives
- Prevent regressions across ingestion, retrieval, reasoning, and scoring pipelines as evaluation tooling evolves.
- Provide quick signal (<= 30 minutes) during CI for common failure modes.
- Establish weekly/nightly extended suites covering larger datasets and judge integrations.

## Scope
- **Offline dataset build** (`scripts/generate_offline_dataset.py`) and manifest integrity checks.
- **Retrieval stack** (index creation + query execution) with FAISS stub and real FAISS (when available).
- **Evaluation CLI** (`scripts/evaluation_harness.py`, `scripts/evaluate_models.py`) including metric aggregation and artifact emission.
- **Fine-tuning checkpoints** loading/sanity inference round-trip (subset only) to ensure adapters remain compatible.
- Excludes notebook smoke tests (tracked separately in `evaluation_and_reporting_plan.md`).

## Test Matrix
| Suite | Trigger | Estimated Runtime | Coverage Highlights |
| --- | --- | --- | --- |
| `pytest -m "not slow"` | Every PR | ~8 min | Unit tests for ingestion, chunking, retrieval, metrics.
| `make score` (FAISS stub) | Every PR touching evaluation | ~3 min | CLI wiring, JSONL outputs, metric aggregation sanity.
| `pytest tests/test_rag_pipeline.py::TestOfflineRAGEndToEnd` | Nightly | ~12 min | Offline dataset + retrieval + generation pipeline regression.
| `scripts/evaluate_models.py --dataset subset20` | Nightly | ~20 min | Judge scoring integration, artifact structure, cost logging.
| ~~`scripts/run_lora_finetune.py --smoke --epochs 1`~~ | ~~Weekly~~ | ~~~25 min (GPU)~~ | ~~Adapter training entry point & checkpoint sync.~~ **Deprecated**: Moved to `vintage/scripts/`. Fine-tuning now handled via Colab notebooks (`notebooks/finetuning/`).
| `make score BTC_USE_FAISS_STUB=0` | Weekly (lab machine) | ~6 min | Real FAISS path to catch native index regressions.

## Tooling & Automation
- **CI Integration**: Extend GitHub Actions to run PR suite (`pytest -m "not slow"`, `make score`).
- **Nightly Workflow**: Reuse planned evaluation workflow; add artifact upload for regression logs and metrics.
- **Reporting**: Append suite outcomes to `evaluation/results/alerts.md` when thresholds breached.

## Data Requirements
- Maintain small deterministic fixture set (`data/subset20/`) for PR pipelines.
- Snapshot larger offline dataset monthly; store in `evaluation/datasets/offline_dataset_ft_only.jsonl` for nightly runs.
- Cache baseline prediction files per experiment (`evaluation/results/<baseline>/predictions.jsonl`).

## Environment & Dependencies
- Default to FAISS stub via `BTC_USE_FAISS_STUB=1` for CI to avoid native builds.
- Provide documented path to native FAISS install for weekly lab runs.
- Require `bert-score>=0.3` (already in `pyproject.toml`).
- GPU availability needed for fine-tune smoke (skip automatically when unavailable).

## Risks & Mitigations
- **Flaky judge responses**: Cache seeds and use deterministic subsets; fall back to recorded transcripts when API unavailable.
- **Long runtimes**: Gate expensive suites behind nightly/weekly schedules; keep PR suite lean.
- **Data drift**: Version datasets and include manifest hash checks in regression scripts.

## Immediate Next Steps
1. Add `pytest -m "not slow"` + `make score` to CI workflow (`.github/workflows/ci.yaml`).
2. Draft nightly GitHub Action orchestrating evaluation + regression suites with artifact upload.
3. Document native FAISS setup and guard weekly job to skip when unavailable.
