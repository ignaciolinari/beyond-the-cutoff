## arXiv 2025 Corpus Plan

### Objectives
- Assemble a **120–150 paper** corpus of post-cutoff research (submitted after **2025-07-01**) to drive downstream RAG vs fine-tuning comparisons for the scientific assistant.
- Ensure topical diversity across core LLM-adjacent areas while keeping pipeline throughput manageable on local hardware.
- Capture rich metadata and artefacts (PDF, plain text, per-page JSONL) to support chunking, evaluation, citation grounding, and temporal-drift analysis.

### Target Mix (120–150 papers total)
- **cs.AI** — 30–35 papers focused on general AI/agent frameworks.
- **cs.CL** — 30–35 papers emphasizing language models and NLP.
- **cs.LG** — 30–35 papers covering learning theory and optimisation advances.
- **stat.ML** — 30–35 papers for methodological breadth and overlap with ML theory.

> Use **stratified sampling** with a reproducible RNG seed (`DATASET_SEED=20251101`) so repeat harvests yield comparable distributions. Store the sampled IDs and seed in `data/raw/arxiv_2025/selection_log.jsonl` (to be created when the next harvest runs).

> Rationale: the four categories overlap with our evaluation tasks and provide robust coverage of new techniques (architectures, alignment, retrieval, evaluation). Each batch is capped at the most recent submissions to balance recency with topic diversity.

### Selection Heuristics
- Query window: `submittedDate:[20250701000000 TO 20251231235959]`.
- Sort by `submittedDate` descending to capture the latest work.
- Deduplicate on canonical arXiv ID (strip version suffix) across categories.
- Persist a `selection_seed` field and the category draw order in the manifest for reproducibility.
- Filter out withdrawals or replacements lacking PDFs.
- Allow manual overrides via `configs/data/arxiv_overrides.yaml` (to be added later) for high-priority papers outside the default categories.

### Pipeline Stages
1. **Metadata Harvest**
   - Use the arXiv export API via `httpx` with a configurable rate limit (default 1 request / 3 seconds).
   - Persist metadata to `data/raw/arxiv_2025/metadata.jsonl` with fields: `arxiv_id`, `version`, `title`, `summary`, `authors`, `categories`, `primary_category`, `published`, `updated`, `pdf_url`.
   - Emit a companion CSV for quick inspection (`metadata.csv`).

2. **PDF Download**
   - Stream PDFs to `data/raw/arxiv_2025/papers/{arxiv_id}.pdf`.
   - Retry up to 3 times with exponential backoff on network failures.
   - Record failures in `data/raw/arxiv_2025/download_failures.jsonl` for manual follow-up.

3. **Text Normalisation**
   - Reuse `PDFIngestor` to produce plain-text and per-page JSONL under `data/processed/arxiv_2025/`.
   - Capture derived metadata (page count, token estimates, average tokens per page) in the `catalog.parquet` step.
   - Enrich manifests with `submission_date` (from arXiv `submittedDate`) so temporal drift dashboards can bucket documents by month.

   ### Randomisation & Reproducibility
   - Maintain a `rng_state.json` artifact under `data/raw/arxiv_2025/` capturing:
      - Seed (`20251101` by default)
      - Python version
      - Selection timestamp
      - Category order and counts at selection time
   - When expanding beyond 150 papers, increment `selection_batch` and record the new seed; never overwrite prior batches.

   ### Versioning & Naming
   - Use semantic dataset tags: `arxiv-2025.<mmdd>-v<iteration>` and log them in the processed manifest.
   - Update `evaluation/results/data_quality/ingestion_log.md` after each run with totals per category, min/max submission dates, and count of missing artefacts.

### Scheduling & Ownership
- **T0 (today)**: Implement metadata harvester + downloader CLI (`scripts/fetch_arxiv_corpus.py`).
- **T0 + 1 day**: Run initial 100-paper harvest, verify completeness, resolve failed downloads.
- **T0 + 2 days**: Trigger normalisation pipeline and add quality checks (duplicate detection, empty text, page counts).
- **Weekly**: Re-run harvester with `--incremental` flag (to be added later) to append newly submitted papers.

### Data Management Notes
- Store raw responses (Atom XML) in `.cache/arxiv/` for reproducibility.
- Maintain a manifest (`data/raw/arxiv_2025/manifest.json`) detailing fetch parameters (timestamp, query, category mix, operator).
- Version dataset outputs via semantic tags (e.g., `arxiv-2025.10.30-v1`) recorded in metadata.
- Respect arXiv TOS: throttle requests, include a contact email in the `User-Agent`, and avoid aggressive parallel downloads.

### Open Questions
- Do we need additional categories (e.g., `cs.IR`, `cs.CV`) for retrieval benchmarks?
- Should we blend in peer-reviewed venues (OpenReview, ACL Anthology) once the arXiv pipeline stabilises?
