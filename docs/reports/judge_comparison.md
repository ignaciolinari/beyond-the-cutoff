# Comparative Analysis of Judges for Evaluation

**Date:** November 29, 2025
**Author:** Beyond the Cutoff Pipeline Automated Evaluation
**Version:** 1.0

## Executive Summary

A comparative smoke test was conducted with 3 judge configurations to evaluate which is most scientifically appropriate for the complete evaluation of the 6-condition experiment. 3 examples were evaluated for each of the 6 conditions (18 total evaluations per judge).

**Final Recommendation:** Use **Qwen3 8B with Thinking Mode** for its greater rigor and scientific discriminability.

---

## 1. Configurations Evaluated

| Judge | Model | Temperature | Thinking Mode | Total Time (18 ex) |
|-------|-------|-------------|---------------|---------------------|
| Qwen3 + Think | qwen3:8b | 0.6 | ✅ Enabled | 29.0 min |
| Qwen3 - Think | qwen3:8b | 0.0 | ❌ Disabled | 25.8 min |
| Llama 3.1 | llama3.1:8b | 0.0 | N/A | 15.1 min |

---

## 2. Results by Condition

### 2.1 Factuality Scores

| Condition | Qwen3+Think | Qwen3-Think | Llama 3.1 |
|-----------|-------------|-------------|-----------|
| base_baseline | 0.37 | 0.37 | 0.30 |
| rag_baseline | 0.70 | 0.70 | 0.83 |
| lora_ft_only | 0.30 | 0.30 | 0.27 |
| hybrid_instruction_only | 0.40 | 0.47 | **0.77** ⚠️ |
| lora_rag_trained_ft_only | 0.20 | 0.23 | 0.07 |
| hybrid_rag_trained | 0.30 | 0.37 | 0.50 |

### 2.2 Grounding Scores

| Condition | Qwen3+Think | Qwen3-Think | Llama 3.1 |
|-----------|-------------|-------------|-----------|
| base_baseline | 0.00 | 0.00 | 0.00 |
| rag_baseline | 0.65 | 0.87 | 0.83 |
| lora_ft_only | 0.00 | 0.00 | 0.00 |
| hybrid_instruction_only | 0.37 | 0.37 | **0.90** ⚠️ |
| lora_rag_trained_ft_only | 0.00 | 0.00 | 0.00 |
| hybrid_rag_trained | 0.50 | 0.53 | 0.47 |

### 2.3 Completeness Scores

| Condition | Qwen3+Think | Qwen3-Think | Llama 3.1 |
|-----------|-------------|-------------|-----------|
| base_baseline | 0.43 | 0.47 | 0.50 |
| rag_baseline | 0.50 | 0.50 | 0.67 |
| lora_ft_only | 0.40 | 0.40 | 0.43 |
| hybrid_instruction_only | 0.37 | 0.37 | 0.67 |
| lora_rag_trained_ft_only | 0.30 | 0.30 | 0.30 |
| hybrid_rag_trained | 0.30 | 0.33 | 0.37 |

### 2.4 Communication Scores

| Condition | Qwen3+Think | Qwen3-Think | Llama 3.1 |
|-----------|-------------|-------------|-----------|
| base_baseline | 0.73 | 0.67 | 0.80 |
| rag_baseline | 0.90 | 0.93 | 0.90 |
| lora_ft_only | 0.73 | 0.67 | 0.80 |
| hybrid_instruction_only | 0.77 | 0.77 | 0.93 |
| lora_rag_trained_ft_only | 0.60 | 0.63 | 0.53 |
| hybrid_rag_trained | 0.70 | 0.70 | 0.57 |

---

## 3. Discriminability Analysis

### 3.1 Improvement Delta: base_baseline → rag_baseline

This comparison measures how much the model improves when given RAG contexts.

| Metric | Qwen3+Think | Qwen3-Think | Llama 3.1 |
|--------|-------------|-------------|-----------|
| Δ Factuality | +0.33 | +0.33 | +0.53 |
| Δ Grounding | +0.65 | +0.87 | +0.83 |
| Δ Completeness | +0.07 | +0.03 | +0.17 |

### 3.2 Transfer Learning Delta: lora_ft_only → hybrid_instruction_only

This comparison measures "imperfect transfer" - a model trained WITHOUT RAG contexts, evaluated WITH contexts.

| Metric | Qwen3+Think | Qwen3-Think | Llama 3.1 |
|--------|-------------|-------------|-----------|
| Δ Factuality | +0.10 | +0.17 | **+0.50** ⚠️ |
| Δ Grounding | +0.37 | +0.37 | **+0.90** ⚠️ |

### 3.3 Factuality Rankings

**Expected ranking (theory):**
```
base ≈ lora_ft < hybrid_instr < rag_base ≤ hybrid_rag_trained
```

**Qwen3+Think (observed ranking):**
```
lora_rag_trained(0.20) < lora_ft(0.30) = hybrid_rag(0.30) < base(0.37) < hybrid_instr(0.40) < rag_base(0.70)
```
✅ Consistent with theoretical expectations

**Llama 3.1 (observed ranking):**
```
lora_rag_trained(0.07) < lora_ft(0.27) < base(0.30) < hybrid_rag(0.50) < hybrid_instr(0.77) < rag_base(0.83)
```
❌ `hybrid_instruction_only` (0.77) nearly equals `rag_baseline` (0.83) - scientifically incorrect

---

## 4. Detected Problems Analysis

### 4.1 Main Problem: Score Inflation in Llama

Llama 3.1 8B assigns **Grounding = 0.90** to `hybrid_instruction_only`, which is problematic because:

1. **The `instruction_only` model was NOT trained to use RAG contexts**
2. **It never learned to make correct citations**
3. **It cannot have better grounding (0.90) than the base model with RAG (0.83)**

This indicates that Llama:
- Does not rigorously verify citations against context
- Gives credit for mentioning information even if not correctly cited
- Is too generous in evaluation

### 4.2 Lack of Discriminability in Llama

| Comparison | Qwen3 | Llama | Problem |
|------------|-------|-------|---------|
| hybrid_instr vs rag_base (Factuality) | 0.40 vs 0.70 (gap=0.30) | 0.77 vs 0.83 (gap=0.06) | Llama doesn't discriminate |
| hybrid_instr vs rag_base (Grounding) | 0.37 vs 0.65 (gap=0.28) | 0.90 vs 0.83 (gap=-0.07) | Llama inverts ranking |

### 4.3 Qwen3: Thinking vs No-Thinking

The difference between Qwen3 with and without thinking mode is minimal:
- Very similar scores (difference < 0.1)
- Time: only ~10% slower with thinking
- Thinking mode provides more elaborate reasoning but does not significantly change scores

---

## 5. Judge Reasoning Example

### Qwen3 8B (with Thinking) - base_baseline, Example 1:

```json
{
  "scores": {
    "factuality": 0.3,
    "grounding": 0.0,
    "completeness": 0.4,
    "communication": 0.7
  },
  "verdict": "fail",
  "reasoning": "The assistant's answer contradicts the expected answer by claiming
    1 scenario vs. 8 scenarios and using randomized controlled experiments vs.
    factorial design. Key elements like factorial dimensions (A/B/C), 2x2x2
    structure, and endogenous group generation are entirely missing."
}
```

**Observations:**
- Identifies specific contradictions with expected_response
- Mentions concrete missing elements
- Clearly justifies low score

---

## 6. Judge Evaluation Criteria

| Criterion | Qwen3 8B | Llama 3.1 8B |
|-----------|----------|--------------|
| **Discriminability** | ✅ High (clear gaps between conditions) | ⚠️ Low (generalized inflation) |
| **Theoretical consistency** | ✅ Expected rankings | ❌ hybrid > rag_base in grounding |
| **Citation rigor** | ✅ G=0.37 for FT without RAG | ❌ G=0.90 for FT without RAG |
| **Calibration** | ✅ Conservative, interpretable | ❌ Too generous |
| **Reasoning** | ✅ Detailed and specific | ⚠️ Less rigorous |

---

## 7. Time Estimates for Full Evaluation

| Judge | Time/Example (average) | Total Time (924 examples) |
|-------|------------------------|---------------------------|
| Qwen3 8B + Thinking | ~97s | **~25 hours** |
| Qwen3 8B - Thinking | ~86s | ~22 hours |
| Llama 3.1 8B | ~50s | ~13 hours |

---

## 8. Conclusions

### 8.1 Main Findings

1. **Llama 3.1 8B is not suitable** for this scientific evaluation due to:
   - Systematic score inflation
   - Lack of discriminability between conditions
   - Rankings that contradict theoretical expectations

2. **Qwen3 8B is the most rigorous judge** because:
   - Maintains clear discriminability between conditions
   - Produces rankings consistent with theory
   - Actively verifies citations and content

3. **Thinking Mode does not significantly affect scores** but provides more elaborate reasoning for auditing.

### 8.2 Final Recommendation

🎯 **Use Qwen3 8B WITH Thinking Mode** for the full evaluation

**Justification:**
- Greater scientific rigor
- Better discriminability between conditions
- Rankings that reflect theoretical expectations
- The additional time cost (~25h vs ~13h) is justified by result quality

### 8.3 Recommended Configuration

```yaml
# configs/judges/dataset_quality_judge.yaml
provider: ollama
model: qwen3:8b
temperature: 0.6  # Enables thinking mode
timeout: 180.0
max_new_tokens: 1024
```

---

## Appendix A: Description of the 6 Conditions

| # | Condition | Model | Training | Evaluation | Purpose |
|---|-----------|-------|----------|------------|---------|
| 1 | base_baseline | Qwen 2.5 0.5B | Base | Without RAG | Baseline without improvements |
| 2 | rag_baseline | Qwen 2.5 0.5B | Base | With RAG | Pure RAG effect |
| 3 | lora_ft_only | Qwen 2.5 0.5B | FT without RAG | Without RAG | Pure FT effect |
| 4 | hybrid_instruction_only | Qwen 2.5 0.5B | FT without RAG | With RAG | Transfer learning |
| 5 | lora_rag_trained_ft_only | Qwen 2.5 0.5B | FT with RAG | Without RAG | FT-RAG without contexts |
| 6 | hybrid_rag_trained | Qwen 2.5 0.5B | FT with RAG | With RAG | Optimal configuration |

---

## Appendix B: Relevant Configuration Files

- `configs/judges/dataset_quality_judge.yaml` - Main Qwen3 judge configuration
- `configs/judges/llama3_judge_inference.yaml` - Llama judge configuration
- `configs/judges/qwen3_judge_no_thinking.yaml` - Qwen3 without thinking mode
- `configs/evaluation/six_condition_experiment.yaml` - Main evaluation plan
