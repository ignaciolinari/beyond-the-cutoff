# Judge Prompt Library

_Last updated: 2025-01-13_

## Purpose
Provide reusable evaluation prompts for automated judges that assess scientific assistant outputs across factuality, citation correctness, and reasoning quality.

## Dual Judge System

For fair evaluation across different experimental conditions, we use **two specialized judge prompts**:

### 1. `scientific_default_rag.yaml` — For RAG Conditions
**Use for:** Conditions that include RAG contexts (conditions 2, 4, 6)
- Evaluates citation grounding and correctness
- Checks that citations match provided contexts
- Scoring weights: 40% factuality, 30% grounding, 20% completeness, 10% communication
- Requires inline citations `[1]`, `[2]`, etc.

### 2. `scientific_default_instruction.yaml` — For Instruction-Only Conditions
**Use for:** Conditions without RAG contexts (conditions 1, 3, 5)
- Evaluates factuality based on general knowledge (not context grounding)
- **Skips citation checks** (not applicable)
- Scoring weights: 50% factuality, 0% grounding, 30% completeness, 20% communication
- Grounding score always set to 0.0 (not applicable)

**Why two judges?** Instruction-only responses shouldn't be penalized for missing citations when no contexts are provided. Using separate judges ensures fair comparison across all experimental conditions.

## File Naming
- `scientific_default_rag.yaml` — rubric for RAG-augmented answers (with contexts)
- `scientific_default_instruction.yaml` — rubric for instruction-only answers (no contexts)
- `scientific_lite.yaml` — lightweight rubric for local models when latency is critical (to be added later)
- Additional files should use the pattern `<domain>_<tier>[_rag|_instruction].yaml`.

## Usage Pattern
1. Select the appropriate judge prompt based on evaluation mode:
   - RAG conditions → `scientific_default_rag.yaml`
   - Instruction-only conditions → `scientific_default_instruction.yaml`
2. Load the YAML file in evaluation scripts to obtain:
   - `prompt`: system prompt for the judge model.
   - `criteria`: list of scoring rubrics with weights.
   - `format`: expected output schema (JSON instructions).
   - `references`: hints for inline citation parsing.
3. Inject dynamic fields (question, answer, retrieved contexts) before sending to the judge model.
4. Persist judge responses alongside raw prompts under `evaluation/results/<experiment_id>/judgements/`.

## Configuration

The comparison plan (`configs/evaluation/compare_0p5b_experiments.yaml`) automatically selects the appropriate judge:
- Conditions 1, 3, 5 (instruction-only): `judge_config: ../judges/scientific_default_instruction.yaml`
- Conditions 2, 4, 6 (RAG): `judge_config: ../judges/scientific_default_rag.yaml`

## Maintenance Notes
- Keep prompts model-agnostic; avoid hard-coding model names.
- When adding new judge prompts, ensure they're appropriate for their evaluation mode (RAG vs instruction-only).
- Document breaking changes in `CHANGELOG.md` once it exists.
- Update owners in `PROJECT_TODO.md` if judge responsibilities shift.
