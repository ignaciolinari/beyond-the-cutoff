# Experiment Analysis Guide

## Overview

This document explains how to interpret the results of the 6-condition experiment comparing fine-tuning and RAG approaches for post-cutoff knowledge.

## The 6 Experimental Conditions

| # | Condition | Training | Evaluation | Config |
|---|-----------|----------|------------|--------|
| 1 | Base Baseline | None | No RAG | `base_ollama.yaml` + instruction mode |
| 2 | RAG Baseline | None | With RAG | `base_ollama.yaml` + rag mode |
| 3 | FT Only (instruction) | Without contexts | No RAG | `lora_instruction_only.yaml` |
| 4 | FT+RAG (instruction) | Without contexts | With RAG | `lora_instruction_only.yaml` + rag mode |
| 5 | RAG-trained FT Only | With contexts | No RAG | `lora_rag_trained.yaml` + instruction mode |
| 6 | RAG-trained FT+RAG | With contexts | With RAG | `lora_rag_trained.yaml` |

## Evaluation Strategy

### Two Judge Configurations

We use different judges depending on whether RAG contexts are provided:

**`instruction.yaml`** (Conditions 1, 3, 5):
- No citation checking (model has no contexts to cite)
- Weights: 50% factuality + 30% completeness + 20% communication
- Grounding score = 0.0 (not applicable)

**`rag.yaml`** (Conditions 2, 4, 6):
- Full citation checking (model should cite provided contexts)
- Weights: 40% factuality + 30% grounding + 20% completeness + 10% communication
- Checks for missing/hallucinated citations

### Fair Comparison Considerations

#### ✅ Fair Comparisons (Same Evaluation Mode)

**Within instruction-only group (1, 3, 5):**
- All use same judge, same metrics, no contexts provided
- Direct comparison on factuality, completeness, communication

**Within RAG group (2, 4, 6):**
- All use same judge, same metrics, contexts provided
- Direct comparison on all metrics including citation grounding

#### ⚠️ Cross-Group Comparisons (Different Evaluation Modes)

When comparing across groups (e.g., 3 vs 4), be aware:

| Metric | Cross-Group Comparable? | Notes |
|--------|------------------------|-------|
| Factuality | ✅ Yes | Both judges evaluate factual accuracy |
| Completeness | ✅ Yes | Both judges evaluate answer coverage |
| Communication | ✅ Yes | Both judges evaluate clarity |
| Grounding/Citations | ❌ No | Only applicable to RAG conditions |
| **Weighted Total** | ❌ **NO** | **Different weighting schemes - NOT comparable** |

#### 🚨 Critical: Weighted Totals Are NOT Comparable Across Groups

The two judges use different weighting schemes:

**Instruction-only judge** (`instruction.yaml`):
- 50% factuality + 30% completeness + 20% communication = **100%**

**RAG judge** (`rag.yaml`):
- 40% factuality + 30% grounding + 20% completeness + 10% communication = **100%**

**The problem**: Even if two models perform identically on factuality, completeness, and communication:
- Instruction-only model: Gets 100% of score from these three dimensions
- RAG model: Gets only 70% from these three dimensions (30% allocated to grounding)

**This creates an inherent bias**: Instruction-only models will appear to score higher on weighted totals even with identical underlying performance on the common dimensions.

**Solution - Compare only raw dimension scores across groups**:

```
WRONG (cross-group):
  FT-only weighted_total: 7.2
  RAG weighted_total: 6.8
  Conclusion: FT-only is better  ← INVALID COMPARISON

RIGHT (cross-group):
  FT-only:  factuality=7.5, completeness=7.0, communication=7.0
  RAG:      factuality=7.5, completeness=7.0, communication=7.0
  Conclusion: Equivalent performance on comparable metrics
```

**Recommendation**: For cross-group comparisons, focus on:
- **Raw dimension scores only** (factuality, completeness, communication)
- Pairwise judge comparisons (head-to-head, same judge evaluates both)
- Qualitative analysis of responses

## Key Research Questions & Comparisons

### 1. How much does RAG help the base model?

**Comparison**: Condition 1 vs Condition 2

```
Base Baseline (no RAG) vs RAG Baseline (with RAG)
```

- Expected: Condition 2 significantly outperforms Condition 1
- This establishes the value of RAG for post-cutoff knowledge

### 2. How much does fine-tuning alone help?

**Comparison**: Condition 1 vs Condition 3

```
Base Baseline vs FT-only (instruction-trained)
```

- Shows whether fine-tuning can "memorize" domain knowledge
- If 3 >> 1: Fine-tuning successfully internalized knowledge
- If 3 ≈ 1: Fine-tuning didn't help without RAG

### 3. Which is better: RAG or Fine-tuning?

**Comparison**: Condition 2 vs Condition 3

```
RAG Baseline vs FT-only
```

- Direct comparison of the two approaches
- Key finding for practitioners: invest in RAG or fine-tuning?

### 4. Does adding RAG help instruction-trained models?

**Comparison**: Condition 3 vs Condition 4

```
FT-only (instruction) vs FT+RAG (instruction)
```

- Tests transfer learning: can a model trained WITHOUT contexts benefit from RAG at inference?
- If 4 >> 3: Yes, RAG helps even instruction-trained models
- If 4 ≈ 3: Model doesn't know how to use contexts it wasn't trained with

### 5. Does adding RAG help RAG-trained models?

**Comparison**: Condition 5 vs Condition 6

```
RAG-trained FT-only vs RAG-trained FT+RAG
```

- Expected: 6 >> 5 (model trained with contexts benefits from having them)
- This is the "matched" condition — training matches evaluation

### 6. Does training WITH contexts improve RAG performance?

**Comparison**: Condition 4 vs Condition 6

```
Instruction-FT+RAG vs RAG-FT+RAG
```

- Both evaluated WITH RAG contexts
- If 6 >> 4: Training mode matters — train with contexts for best RAG performance
- If 6 ≈ 4: Training mode doesn't matter as long as RAG is available

### 7. Does training mode affect non-RAG performance?

**Comparison**: Condition 3 vs Condition 5

```
Instruction-FT (no RAG) vs RAG-FT (no RAG)
```

- Both evaluated WITHOUT RAG contexts
- If 3 >> 5: Training with contexts hurts when contexts unavailable (overfitting to format)
- If 5 >> 3: RAG training provides better knowledge retention
- If 3 ≈ 5: Training mode doesn't matter without RAG

## Expected Results Visualization

```
Performance
    ▲
    │                                    ┌─────────────────┐
    │                                    │ 6. RAG-FT+RAG   │ ← Expected best
    │                              ┌─────┴─────────────────┘
    │                              │ 4. Instruction-FT+RAG
    │                    ┌─────────┴───┐
    │                    │ 2. RAG      │
    │              ┌─────┴─────────────┘
    │              │ 5. RAG-FT only (mismatch?)
    │        ┌─────┴───┐
    │        │ 3. FT   │
    │  ┌─────┴─────────┘
    │  │ 1. Base
    └──┴────────────────────────────────────────────────────▶
       No RAG                                          With RAG
```

### Expected Outcomes

1. **Best overall**: Condition 6 (RAG-trained + RAG at inference)
   - Training matches evaluation → optimal performance

2. **Best without RAG**: Likely Condition 3 (instruction-trained)
   - Trained without contexts, evaluated without contexts

3. **Potential surprises**:
   - Condition 5 might underperform 3 (trained with contexts but tested without)
   - Condition 4 might perform well despite training mismatch

## Analysis Workflow

### Step 1: Run All Evaluations

Use the comparison script with the experiment plan:

```bash
# Run all 6 conditions via comparison plan
python scripts/core/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --dataset evaluation/datasets/eval_dataset.jsonl
```

Or use the interleaved evaluation for balanced progress:

```bash
# Interleaved evaluation across all conditions
python scripts/core/interleaved_evaluation.py \
  --plan configs/evaluation/six_condition_experiment.yaml
```

### Step 2: Collect Metrics

For each condition, extract from `metrics.json`:
- `factuality` (comparable across all)
- `completeness` (comparable across all)
- `communication` (comparable across all)
- `grounding` (RAG conditions only)
- `overall` (use with caution for cross-group)

### Step 3: Create Comparison Tables

**Within-Group Comparisons (fully comparable):**

| Metric | Cond 1 | Cond 3 | Cond 5 |
|--------|--------|--------|--------|
| Factuality | | | |
| Completeness | | | |
| Communication | | | |
| Overall | | | |

| Metric | Cond 2 | Cond 4 | Cond 6 |
|--------|--------|--------|--------|
| Factuality | | | |
| Completeness | | | |
| Communication | | | |
| Grounding | | | |
| Overall | | | |

**Cross-Group Comparisons (raw dimension scores ONLY - no weighted totals):**

| Condition | Factuality | Completeness | Communication | Has RAG? | Training Mode |
|-----------|------------|--------------|---------------|----------|---------------|
| 1. Base | | | | No | None |
| 2. RAG | | | | Yes | None |
| 3. FT-instr | | | | No | Instruction |
| 4. FT-instr+RAG | | | | Yes | Instruction |
| 5. FT-rag | | | | No | RAG |
| 6. FT-rag+RAG | | | | Yes | RAG |

> ⚠️ **Important**: Do NOT include weighted totals in cross-group comparisons. The different weighting schemes make them incomparable.

### Step 4: Statistical Analysis (Optional)

For pairwise comparisons, use:
- Paired t-test or Wilcoxon signed-rank test on per-example scores
- Bootstrap confidence intervals for aggregate metrics
- ELO rankings from pairwise judge comparisons

### Step 5: Interpret Results

Answer the key research questions using **raw dimension scores**:

1. **RAG value** (within instruction group): 2.factuality - 1.factuality, etc.
2. **FT value** (within instruction group): 3.factuality - 1.factuality, etc.
3. **RAG vs FT**: Compare raw dimensions (2 vs 3), not weighted totals
4. **FT+RAG synergy**: Compare 6's raw scores to 2 and 3's raw scores
5. **Training alignment**: 6.factuality - 4.factuality (both have RAG, same judge)

> 🎯 **Key principle**: Within-group can use weighted totals. Cross-group must use raw dimensions.

## Limitations & Caveats

### 1. Citation Metrics Not Cross-Comparable
- Don't compare grounding scores between RAG and non-RAG conditions
- Models without contexts literally cannot cite

### 2. ⚠️ Weighted Totals NOT Cross-Comparable
- **RAG judge**: 40% factuality + 30% grounding + 20% completeness + 10% communication
- **Instruction judge**: 50% factuality + 30% completeness + 20% communication
- The instruction-only models get **100% of their score from fact/comp/comm**
- The RAG models only get **70% from those same dimensions**
- **This inherently inflates instruction-only weighted totals**
- **Solution**: Only compare raw dimension scores across groups

### 3. Small Dataset Considerations
- With limited eval examples, differences may not be statistically significant
- Report confidence intervals where possible

### 4. Judge Model Bias
- Same judge model evaluates all conditions (fair)
- But judge may have inherent preferences (e.g., longer answers)
- Pairwise comparisons can help validate automatic scores

## Quick Reference: Which Comparisons Answer What

| Research Question | Compare | Metric Focus |
|-------------------|---------|--------------|
| Does RAG help base model? | 1 vs 2 | Overall (RAG judge) |
| Does FT help without RAG? | 1 vs 3 | Overall (instruction judge) |
| RAG vs FT alone? | 2 vs 3 | Factuality only |
| Does RAG help instruction-FT? | 3 vs 4 | Factuality, overall |
| Does RAG help RAG-FT? | 5 vs 6 | Factuality, overall |
| Best RAG setup? | 4 vs 6 | Overall (RAG judge) |
| Best non-RAG setup? | 3 vs 5 | Overall (instruction judge) |
| Training/eval mismatch penalty? | 4 vs 6, 3 vs 5 | Factuality, completeness |

## Conclusion

This experiment provides a comprehensive view of how fine-tuning and RAG interact for post-cutoff knowledge. The key insight is not just "which is best" but understanding:

1. **When to use RAG**: Always helps, but especially for models not fine-tuned on domain data
2. **When to fine-tune**: Valuable even without RAG, but best when combined
3. **How to fine-tune for RAG**: Training with contexts may improve RAG performance
4. **Transfer learning**: Can instruction-trained models benefit from RAG at inference?

The answers to these questions inform practical decisions about building knowledge-augmented LLM systems.
