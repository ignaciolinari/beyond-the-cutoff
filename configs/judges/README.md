# Judge Prompt Library

_Last updated: 2025-11-24_

## Purpose
Provide reusable evaluation prompts for automated judges that assess scientific assistant outputs across factuality, citation correctness, and reasoning quality.

## Judge Types

### Evaluation Judges (Model Output Assessment)

For fair evaluation across different experimental conditions, we use **two specialized judge prompts**:

#### 1. `scientific_default_rag.yaml` — For RAG Conditions
**Use for:** Conditions that include RAG contexts (conditions 2, 4, 6)
- Evaluates citation grounding and correctness
- Checks that citations match provided contexts
- Scoring weights: 40% factuality, 30% grounding, 20% completeness, 10% communication
- Requires inline citations `[1]`, `[2]`, etc.

#### 2. `scientific_default_instruction.yaml` — For Instruction-Only Conditions
**Use for:** Conditions without RAG contexts (conditions 1, 3, 5)
- Evaluates factuality based on general knowledge (not context grounding)
- **Skips citation checks** (not applicable)
- Scoring weights: 50% factuality, 0% grounding, 30% completeness, 20% communication
- Grounding score always set to 0.0 (not applicable)

**Why two judges?** Instruction-only responses shouldn't be penalized for missing citations when no contexts are provided. Using separate judges ensures fair comparison across all experimental conditions.

### Dataset Quality Judge (Pre-Training Validation)

#### `dataset_quality_judge.yaml` — For Dataset Validation
**Use for:** Validating generated training examples BEFORE fine-tuning
- Evaluates: answerability, correctness, clarity, coherence
- **Uses Qwen 3 8B** (different from generator's Qwen 2.5 7B to avoid self-preference bias)
- Pass criteria: All scores ≥ 0.6 AND (answerability + correctness) ≥ 1.4

```bash
python scripts/evaluate_dataset_quality.py \
  --dataset evaluation/datasets/offline_dataset.jsonl \
  --judge-inference configs/judges/dataset_quality_judge.yaml \
  --sample-size 50
```

### Pairwise Comparison Judges

For automated model ranking via pairwise comparison:
- `pairwise_qwen7b.yaml` — Qwen 2.5 7B judge
- `pairwise_qwen3_8b.yaml` — Qwen 3 8B judge
- `pairwise_llama31_8b.yaml` — Llama 3.1 8B judge

Multi-judge consensus uses all three for position-debiased majority voting.

## File Naming
- `scientific_default_rag.yaml` — rubric for RAG-augmented answers (with contexts)
- `scientific_default_instruction.yaml` — rubric for instruction-only answers (no contexts)
- `dataset_quality_judge.yaml` — inference config for dataset quality validation
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
