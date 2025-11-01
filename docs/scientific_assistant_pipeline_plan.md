# Scientific Assistant Pipeline Plan

_Last updated: 2025-11-01_

## 1. Scope Alignment & Ownership
- **Objective**: Adapt the "RAG vs Fine-tuning" pipeline to a research-assistant focused on post-cutoff scientific literature.
- **Owners**: Assign leads per stage (Data, Task Generation, Model Adaptation, Evaluation). Capture owner, contact, and status in `PROJECT_TODO.md`.
- **Deliverable**: Written mapping of the paper's stages to repository components (README excerpt + this plan).

## 2. Corpus Refresh & Diagnostics
### 2.1 Data Targets
- Select 120–150 arXiv papers submitted after the global LLM cutoff (focus on cs.AI, cs.CL, cs.LG, stat.ML) using seed `20251101` for stratified sampling.
- Document selection heuristics (query windows, dedupe strategy) and seed logging in `docs/data_sourcing_plan.md`.

### 2.2 Ingestion & Indexing
- Run `python scripts/ingest_and_index.py --config configs/default.yaml` after each corpus refresh.
- Record manifest version, ingestion timestamp, and paper counts in `evaluation/results/data_quality/ingestion_log.md`.

### 2.3 Quality Checks
- Build notebooks or scripts to validate:
  - Per-paper token counts and empty pages.
  - Presence/accuracy of `.pages.jsonl` sidecars (section titles, page numbering).
  - Duplicate canonical IDs or missing metadata entries.
- Summaries of checks go into `evaluation/results/data_quality/qa_report_<date>.md`.

### 2.4 Temporal Drift Monitoring
- Add submission dates to manifests (no code change yet; specify fields).
- Plan dashboard spec (e.g., monthly paper counts) for later implementation.

## 3. Task & Response Generation
### 3.1 Offline Dataset Configuration
- Finalize `OfflineDatasetGenerator` usage for three task families: QA, summaries, citation checks.
- Define generator hyperparameters (model tag, temperature, max tokens) and log them with each run.

### 3.2 Prompt Calibration
- Draft scientific-style generator prompts emphasizing method/result grounding and citation requirements.
- Store prompt variants in `configs/dataset_generation/` (to be created later) for traceability.

### 3.3 Validation & Logging
- Design heuristics for:
  - Removing duplicate tasks or near-identical prompts.
  - Rejecting responses with empty or malformed citations.
  - Flagging formula/figure-heavy chunks that degrade generator output.
- Plan logging format for raw generator responses with metadata (model tag, seed, timestamp) in `evaluation/datasets/raw_tasks/*.jsonl`.

### 3.4 Citation Coverage Metrics
- Outline metrics leveraging `RAGPipeline.verify_citations` (coverage percentage, missing references).
- Document acceptance thresholds (e.g., coverage ≥ 0.35, no extra citation markers) for inclusion in offline datasets.

## 4. Model Adaptation Portfolio
### 4.1 Experiment Matrix
- Define baseline configurations:
  1. RAG-only (current pipeline).
  2. Fine-tune-only (`Qwen/Qwen2-0.5B-Instruct` with LoRA adapters).
  3. Hybrid (fine-tuned + RAG retrieval).
  4. Optional larger baselines (Phi-3, Mistral, or cloud API) for comparison.

### 4.2 Data Splits & Governance
- Specify train/validation/test splits drawn from offline dataset (e.g., 70/15/15 by paper or task ID).
- Maintain split manifests in `evaluation/datasets/splits/` with versioned date stamps.

### 4.3 Training Workflow Planning
- Document Colab/Kaggle notebook template requirements (LoRA config, logging, checkpoint export).
- Plan adapter export locations (`outputs/adapters/<experiment_id>/`).
- Define GGUF quantization workflow and Ollama `Modelfile` registration steps for tuned models.

### 4.4 Artifact Management
- Establish checksum recording and sync steps for cloud→local transfers.
- Draft experiment README template capturing hyperparameters, dataset version, and evaluation metrics.

## 5. Evaluation & Benchmarking
### 5.1 Metric Suite
- Metrics inspired by the paper: factual accuracy, citation correctness, temporal freshness proxy (recency awareness), answer coherence.
- Plan implementation using:
  - `RAGPipeline.verify_citations` results.
  - Judge models (see 5.2) for qualitative scoring.

### 5.2 Judge Strategy
- Choose default local judge (e.g., `qwen2:3b-instruct-q4_0`) and optional cloud judge (GPT-4o) for calibration.
- Draft evaluation prompts emphasizing scientific grounding and citation verification.
- Store judge configuration specs under `configs/judges/`.

### 5.3 Automation & Testing
- Schedule nightly `pytest` runs focusing on ingestion, chunking, and RAG pipeline tests.
- Plan notebook or CLI scripts to compute metrics for each model variant, saving outputs to `evaluation/results/<experiment_id>/metrics.json`.
- Define alert thresholds (e.g., regression alerts when factual accuracy drops >3 p.p.).

### 5.4 Comparative Reporting
- Produce markdown summaries per experiment comparing all baseline/hybrid models.
- Include latency, cost estimates, and citation coverage charts alongside accuracy metrics.

## 6. Research Assistant Readiness
### 6.1 Scenario Alignment
- Enumerate key assistant tasks: literature QA, method comparison, summary synthesis, citation tracing.
- For each task, specify required pipeline outputs (retrieval settings, prompt templates, evaluation metrics).

### 6.2 API & UX Planning
- Outline minimal FastAPI enhancements (e.g., endpoint for hybrid model selection, citation verification payloads).
- Plan retrieval configuration presets (chunk size, overlap, top_k) optimized for scientific papers.

### 6.3 Governance & Iteration
- Establish milestone checkpoints: data refresh complete, dataset generated, fine-tuning run, evaluation report ready.
- Maintain changelog entries summarizing decisions and performance deltas (future `CHANGELOG.md`).
- Schedule regular review meetings to decide on promoting models or adjusting data strategies.

## 7. Open Questions & Follow-Ups
- Do we need additional domains (e.g., biomedical, physics) beyond the core categories for broader coverage?
- What level of human evaluation is required to validate automated judge outputs?
- Should we integrate an online retrieval source (e.g., Semantic Scholar) for "beyond cutoff" real-time updates?
- How will we handle math-heavy papers where text extraction quality is poor?
