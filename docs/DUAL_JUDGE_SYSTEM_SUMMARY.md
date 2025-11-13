# Dual Judge System Implementation Summary

**Date:** 2025-01-XX
**Status:** ✅ **IMPLEMENTED**

---

## Problem Solved

Previously, all experimental conditions were evaluated using a single judge prompt designed for RAG evaluation. This unfairly penalized instruction-only conditions (1, 3, 5) because:

- The judge checked for citations even when no contexts were provided
- Scoring weighted "grounding" (30%) which requires citations
- Instruction-only responses were penalized for missing citations they couldn't provide

**Result:** Unfair comparison between instruction-only and RAG conditions.

---

## Solution: Dual Judge System

We now use **two specialized judge prompts**:

### 1. `scientific_default_rag.yaml` — For RAG Conditions (2, 4, 6)
- **Scoring:** 40% factuality, 30% grounding, 20% completeness, 10% communication
- **Checks:** Citations, grounding against contexts, citation correctness
- **Use when:** RAG contexts are provided in the evaluation

### 2. `scientific_default_instruction.yaml` — For Instruction-Only Conditions (1, 3, 5)
- **Scoring:** 50% factuality, 0% grounding, 30% completeness, 20% communication
- **Checks:** Factuality based on general knowledge (no citation requirements)
- **Use when:** No RAG contexts are provided

---

## Files Created/Modified

### ✅ Created
- `configs/judges/scientific_default_instruction.yaml` — Instruction-only judge prompt

### ✅ Renamed
- `configs/judges/scientific_default.yaml` → `configs/judges/scientific_default_rag.yaml`

### ✅ Updated
- `configs/evaluation/compare_0p5b_experiments.yaml` — Each condition now specifies its judge
- `configs/judges/README.md` — Documented dual judge system
- `docs/EVALUATION_FAIRNESS_ANALYSIS.md` — Implementation status updated

---

## Configuration Changes

### Before
```yaml
defaults:
  judge_config: ../judges/scientific_default.yaml  # Single judge for all
runs:
  - label: base_baseline_0p5b
    prompt_mode: instruction
    # Uses default judge (RAG-focused) ❌
```

### After
```yaml
defaults:
  judge_config: ../judges/scientific_default_rag.yaml  # Default for RAG
runs:
  - label: base_baseline_0p5b
    prompt_mode: instruction
    judge_config: ../judges/scientific_default_instruction.yaml  # ✅ Override
```

---

## Condition-to-Judge Mapping

| Condition | Label | Prompt Mode | Judge Prompt |
|-----------|-------|-------------|--------------|
| 1 | `base_baseline_0p5b` | instruction | `scientific_default_instruction.yaml` |
| 2 | `rag_baseline_0p5b` | rag | `scientific_default_rag.yaml` |
| 3 | `lora_science_0p5b_ft_only` | instruction | `scientific_default_instruction.yaml` |
| 4 | `hybrid_science_0p5b_instruction_only` | rag | `scientific_default_rag.yaml` |
| 5 | `lora_science_0p5b_rag_trained_ft_only` | instruction | `scientific_default_instruction.yaml` |
| 6 | `hybrid_science_0p5b_rag_trained` | rag | `scientific_default_rag.yaml` |

---

## Key Differences Between Judges

| Aspect | RAG Judge | Instruction-Only Judge |
|--------|-----------|------------------------|
| **Citation Checks** | ✅ Required | ❌ Skipped |
| **Grounding Score** | 30% weight | 0% weight (always 0.0) |
| **Factuality Basis** | Context-grounded | General knowledge |
| **Citation Arrays** | Populated | Always empty `[]` |
| **Overall Score Formula** | 0.40×fact + 0.30×ground + 0.20×comp + 0.10×comm | 0.50×fact + 0.30×comp + 0.20×comm |

---

## Verification

The comparison runner (`src/beyond_the_cutoff/evaluation/comparison.py`) already supports per-run judge_config overrides:

```python
judge_prompt_path = spec.judge_config or plan.defaults.judge_config
```

This means:
- ✅ Each run can specify its own judge_config
- ✅ Falls back to default if not specified
- ✅ No code changes needed

---

## Next Steps

1. **Re-run evaluation** with the new judge prompts:
   ```bash
   python scripts/compare_models.py --plan configs/evaluation/compare_0p5b_experiments.yaml
   ```

2. **Verify results**:
   - Instruction-only conditions should no longer be unfairly penalized
   - Scores should reflect fair comparison across all 6 conditions
   - Citation metrics should be marked N/A for instruction-only conditions

3. **Compare metrics**:
   - Check that instruction-only conditions have grounding = 0.0
   - Verify that factuality scores are comparable (knowledge-based vs context-grounded)
   - Ensure overall scores reflect appropriate weighting

---

## Benefits

✅ **Fair Comparison:** Instruction-only conditions evaluated appropriately
✅ **Clear Separation:** Different evaluation criteria for different modes
✅ **Reproducible:** Explicit judge selection per condition
✅ **Maintainable:** Well-documented dual judge system

---

**Implementation Complete:** 2025-01-XX
**Ready for Testing:** ✅
