# Judge Prompt Library

_Last updated: 2025-11-01_

## Purpose
Provide reusable evaluation prompts for automated judges that assess scientific assistant outputs across factuality, citation correctness, and reasoning quality.

## File Naming
- `scientific_default.yaml` — primary rubric for post-cutoff scientific QA, summaries, and citation checks.
- `scientific_lite.yaml` — lightweight rubric for local models when latency is critical (to be added later).
- Additional files should use the pattern `<domain>_<tier>.yaml`.

## Usage Pattern
1. Load the YAML file in evaluation scripts to obtain:
   - `prompt`: system prompt for the judge model.
   - `criteria`: list of scoring rubrics with weights.
   - `format`: expected output schema (JSON instructions).
   - `references`: hints for inline citation parsing.
2. Inject dynamic fields (question, answer, retrieved contexts) before sending to the judge model.
3. Persist judge responses alongside raw prompts under `evaluation/results/<experiment_id>/judgements/`.

## Maintenance Notes
- Keep prompts model-agnostic; avoid hard-coding model names.
- Document breaking changes in `CHANGELOG.md` once it exists.
- Update owners in `PROJECT_TODO.md` if judge responsibilities shift.
