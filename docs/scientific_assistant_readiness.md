# Scientific Assistant Readiness Guide

_Last updated: 2025-11-01_

## Key User Scenarios
1. **Literature QA**: Answer targeted questions about post-cutoff findings with citations.
2. **Method Comparison**: Compare methodologies across papers, highlighting strengths, weaknesses, and recency.
3. **Summary Synthesis**: Produce structured summaries (motivation, method, results, limitations) for a single paper or small cohort.
4. **Citation Tracing**: Identify which chunks support a claim and expose page-level metadata for verification.

## Pipeline Touchpoints per Scenario
| Scenario | Retrieval Settings | Prompt Template | Evaluation Metrics |
| --- | --- | --- | --- |
| Literature QA | `chunk_size=512`, `top_k=6` | `prompt_templates/qa_citation.txt` | Factuality, citation coverage |
| Method Comparison | `chunk_size=384`, `top_k=8`, reranker on | `prompt_templates/method_compare.txt` | Completeness, clarity |
| Summary Synthesis | `chunk_size=600`, `top_k=4` | `prompt_templates/summary_structured.txt` | Coherence, coverage |
| Citation Tracing | `chunk_size=256`, `top_k=10` | `prompt_templates/citation_trace.txt` | Grounding, latency |

## API & UX Requirements
- Extend FastAPI server (`beyond_the_cutoff.api.server`) with endpoints:
  - `POST /ask`: existing; ensure response includes `citation_verification` payload.
  - `POST /compare`: accepts two paper IDs and comparison axis (methods/results).
  - `GET /papers/{paper_id}/metadata`: surfaces processed manifest entries for front-end display.
- Response schema needs explicit `model_tag`, `dataset_tag`, and `retrieval_config_id` fields for auditing.

## Retrieval Presets
- Define YAML presets under `configs/retrieval_presets/` (to be created) mapping scenario → retrieval parameters.
- Each preset should specify:
  - Embedding model.
  - `chunk_size`, `chunk_overlap`, `top_k`.
  - Whether reranker is enabled and which model.

## Demo Readiness Checklist
- [ ] Latest corpus ingested and QA report published.
- [ ] Offline dataset refreshed and validated.
- [ ] Fine-tuned model registered with Ollama (`qwen2-science-lora`).
- [ ] Evaluation report shows promoted model surpassing thresholds.
- [ ] API endpoints tested via `scripts/ask.py` and curl samples.
- [ ] README quickstart updated with new workflow instructions.

## Governance & Iteration
- Milestone reviews (bi-weekly):
  1. **Data Refresh** — confirm ingestion + QA complete.
  2. **Task Generation** — review offline dataset report and audit log.
  3. **Model Training** — inspect adapter metadata and training logs.
  4. **Evaluation** — approve or reject promotion based on metrics.
- Maintain decisions in `docs/governance_decisions.md` (to be created) capturing rationale, date, and stakeholders.
- Future change requests should reference the relevant report and milestone review.
