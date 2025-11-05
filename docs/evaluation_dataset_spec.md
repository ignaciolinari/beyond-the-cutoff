# Evaluation Dataset Specification

## Objectives
- Provide a reproducible evaluation set for RAG and fine-tuned models covering QA, summarisation, and citation behaviours.
- Standardise record structure so automatic metrics and judge pipelines can consume the data without per-run adapters.
- Capture provenance, retrieval context, and task metadata to enable regression tracking and reproducibility.

## Target Artifacts
- `evaluation/datasets/qa_pairs.jsonl` – canonical question answering split.
- `evaluation/datasets/summaries.jsonl` – abstractive summarisation split.
- `evaluation/datasets/manifest.json` – manifest capturing dataset version, counts, and source provenance.
- (Existing) `evaluation/datasets/offline_dataset.jsonl` – raw generator output acting as the single source of truth.

All paths are already surfaced in `EvaluationConfig` and resolve relative to the project root by default.

## Record Schemas
### Shared Envelope
Each record carries the following common fields:

| Field | Type | Description |
| --- | --- | --- |
| `task_id` | `str` | Stable UUID identifying the source example. |
| `task_type` | `str` | Original task type (`qa`, `summaries`, `citations`, ...). |
| `rag_prompt` | `str` | Prompt issued to the target assistant (post-templating). |
| `contexts` | `list[str]` | Ordered retrieval snippets shown to the model (numbered where applicable). |
| `sources` | `list[str]` | Source file paths or URIs aligned with `contexts`. |
| `retrieved` | `list[dict]` | Raw retrieval metadata (chunk indices, scores, etc.). |
| `citations` | `list[dict]` | Citation spans emitted during offline generation for traceability. |
| `metadata` | `dict[str, Any]` | Generator metadata (document info, run ids, tags). |

### QA Split (`qa_pairs.jsonl`)
Adds QA-specific payload on top of the shared envelope:

| Field | Type | Description |
| --- | --- | --- |
| `question` | `str` | User-facing question/instruction. |
| `reference_answer` | `str` | Reference answer expected from the assistant (ground truth). |
| `answer_citations` | `list[dict]` | Optional curated citation targets for the reference answer (mirrors `citations`). |

Example record:

```json
{
  "task_id": "75bcdabf-4562-4886-b86e-ea183d451921",
  "task_type": "qa",
  "question": "What are the assumptions made in this paper?",
  "reference_answer": "The paper assumes ...",
  "rag_prompt": "You are a research paper assistant...",
  "contexts": ["[1] ...", "[2] ..."],
  "sources": ["data/processed/.../2510.25529.txt#page=28", "..."],
  "retrieved": [...],
  "citations": [...],
  "answer_citations": [...],
  "metadata": {"document": "2510.25529.txt", "run_id": "..."}
}
```

### Summaries Split (`summaries.jsonl`)
Adds summary-specific payload:

| Field | Type | Description |
| --- | --- | --- |
| `instruction` | `str` | Summarisation instruction or template prompt. |
| `reference_summary` | `str` | Expected structured or narrative summary text. |

Example record:

```json
{
  "task_id": "f0cbd8f2-e8e7-4b4e-9548-c0b9dd499c69",
  "task_type": "summaries",
  "instruction": "Draft a structured abstract for the following paper...",
  "reference_summary": "Summary: ...",
  "rag_prompt": "You are a research paper assistant...",
  "contexts": [...],
  "sources": [...],
  "retrieved": [...],
  "citations": [...],
  "metadata": {"document": "..."}
}
```

### Handling Additional Task Types
Offline generation can emit specialised tasks (e.g. `citations`). These remain in `offline_dataset.jsonl` for now. The builder script records counts for unsupported task types and we can introduce dedicated splits later if required.

## File Layout & Versioning
- Default split files live directly under `evaluation/datasets/` to maintain backwards compatibility with the existing configuration.
- `manifest.json` captures the dataset `version`, `source_offline_dataset`, timestamp, counts per split, and optional build parameters. Example stub:

```json
{
  "version": "v1",
  "generated_at_utc": "2025-11-04T18:30:00Z",
  "source_offline_dataset": "evaluation/datasets/offline_dataset.jsonl",
  "filters": {},
  "splits": {
    "qa": {"path": "evaluation/datasets/qa_pairs.jsonl", "count": 120},
    "summaries": {"path": "evaluation/datasets/summaries.jsonl", "count": 45}
  },
  "excluded_task_types": {"citations": 12}
}
```

Version identifiers follow `vN` (semantic-like once we add major changes). Updating the version requires regenerating splits and refreshing the manifest.

## Generation Process
1. Ensure the latest offline dataset is present (`scripts/generate_offline_dataset.py`).
2. Run the new builder: `python scripts/build_evaluation_dataset.py --config configs/default.yaml`.
   - Optional flags:
     - `--offline-dataset` to point at alternate raw data.
     - `--qa-output` / `--summary-output` for custom destinations.
     - `--manifest-output` to override manifest location.
     - `--version` to stamp manifest entries (default `v1`).
     - `--limit` for smoke checks.
     - `--force` to overwrite existing split files.
3. Inspect stdout summary and the generated manifest to confirm record counts.

The script streams records to avoid loading the entire offline dataset in memory. It normalises context formatting, preserves retrieval metadata, and warns about unsupported task types.

## Validation & Regression Checks
- **Schema sanity**: Run `scripts/check_processed_corpus.py` (or add a dedicated validator) to ensure each record includes mandatory fields and non-empty reference answers.
- **Counts & coverage**: Compare QA and summary counts against the manifest and historical baselines. Trigger alerts if counts drop unexpectedly (>10%).
- **Content spot checks**: Randomly sample records from each split to verify references align with contexts.
- **Compatibility tests**: Extend unit tests to load the new splits and ensure `scripts/evaluate_models.py` can iterate over them without modification. Add regression tests in `tests/test_offline_dataset.py` for schema compliance.

## Next Steps
- Hook the new split files into the evaluation CLI (allow passing `--dataset-type qa|summaries`).
- Add unit tests asserting that builder output validates against expected schema samples.
- Expand support for specialised tasks (citation-only, retrieval diagnostics) when metric implementations land.
- Integrate dataset version info into experiment logging so result reports capture the exact evaluation snapshot.
