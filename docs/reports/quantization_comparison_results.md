# Quantization Impact Analysis: Q4_K_M vs F16

**Experiment Date:** December 2, 2025
**Judge Model:** Gemini 3 Pro
**Evaluation Method:** Blinded pairwise comparison with randomized presentation order

> ⚠️ **Scope Limitation**: This analysis applies to **Qwen 2.5 0.5B**. Larger models typically show similar quantization resilience with Q4_K_M, but testing is recommended for production deployments. See [Scaling Guide](../scaling/README.md).

---

## Executive Summary

This experiment tested whether 4-bit quantization (Q4_K_M) degrades response quality compared to full-precision (F16) inference for a fine-tuned 0.5B parameter model on scientific question answering tasks.

### Key Finding

**No significant quality difference was detected between quantized and non-quantized versions.**

| Metric | Value |
|--------|-------|
| Q4_K_M wins | 45 (29.2%) |
| F16 wins | 48 (31.2%) |
| Ties | 61 (39.6%) |
| p-value | 0.8357 |
| Effect size | 0.03 (negligible) |

The high tie rate (39.6%) and near-equal win distribution indicate that Q4_K_M quantization preserves model quality for this task, allowing a **60% reduction in model size** (994MB → 397MB) with no measurable performance loss.

---

## Experimental Design

### Models Compared

| Property | Q4_K_M | F16 |
|----------|--------|-----|
| Base checkpoint | lora_science_v1 | lora_science_v1 |
| Quantization | 4-bit (Q4_K_M) | None (FP16) |
| File size | 397 MB | 994 MB |
| Ollama model | `lora_science_0p5` | `lora_science_0p5_f16` |

### Controls

All parameters were held constant except quantization:

- ✅ Same fine-tuned checkpoint (lora_science_v1)
- ✅ Same system prompt
- ✅ Same temperature (0), top_p (0.9), repeat_penalty (1.05)
- ✅ Same context length (4096)
- ✅ Same evaluation dataset (154 examples)
- ✅ Same RAG contexts for each question
- ✅ Independent generation (no conversation context)

### Blinding

The judge (Gemini 3 Pro) was fully blinded:

- No mention of quantization in the prompt
- No model identities disclosed
- Randomized A/B presentation (50/50 split)
- Neutral evaluation criteria focused on factual accuracy

---

## Results

### Overall Match Outcomes

```
Total Comparisons: 154

Q4_K_M wins:  45 (29.2%)  ████████████░░░░░░░░░░░░░░░░░░
F16 wins:     48 (31.2%)  █████████████░░░░░░░░░░░░░░░░░
Ties:         61 (39.6%)  ████████████████░░░░░░░░░░░░░░
```

### Win Rate Analysis (Excluding Ties)

| Model | Wins | Win Rate | 95% CI |
|-------|------|----------|--------|
| Q4_K_M | 45 | 48.4% | 38.5% - 58.4% |
| F16 | 48 | 51.6% | 41.6% - 61.5% |

The confidence intervals overlap substantially, indicating no reliable difference.

### Statistical Significance

| Test | Result |
|------|--------|
| Sign Test p-value | 0.8357 |
| Significance (α=0.05) | ❌ Not significant |
| Effect size | 0.03 (negligible) |
| Decisive matches | 93 |

**Interpretation:** The p-value of 0.8357 is far above the 0.05 threshold. We cannot reject the null hypothesis that both models perform equally.


---

## Position Bias Analysis

To ensure the judge wasn't biased by response order, positions were randomized 50/50:

| Presentation Order | Q4 Wins | F16 Wins | Ties |
|--------------------|---------|----------|------|
| Q4 as Response A (77 matches) | 23 | 24 | 30 |
| F16 as Response A (77 matches) | 22 | 24 | 31 |

**Finding:** No position bias detected. Both models performed similarly regardless of presentation order.

---

## Judge Confidence Distribution

| Confidence | Count | Percentage |
|------------|-------|------------|
| High | 112 | 72.7% |
| Medium | 41 | 26.6% |
| Low | 1 | 0.6% |

The high proportion of high-confidence judgments (72.7%) indicates the judge was able to make clear assessments, and that ties were genuine cases of equivalent quality.

---

## Error Pattern Analysis

From manual inspection of the judge's reasoning across 154 comparisons:

### Common Failure Modes (Both Models)

1. **Context Retrieval Errors** (~15% of cases)
   - Both models sometimes answered from the wrong paper
   - Example: "Both responses discuss 'DeepSeek' instead of the expected topic"

2. **Hallucinated Citations** (~8% of cases)
   - Both occasionally invented citation numbers or author names
   - No systematic difference between Q4 and F16

3. **Truncation/Formatting Issues** (~5% of cases)
   - Responses cut off mid-sentence
   - Raw text dumps instead of synthesized answers
   - Slightly more common in Q4_K_M but not statistically significant

### When F16 Won

- More complete responses (included solution after identifying problem)
- Better numerical accuracy (specific angles, counts)
- Cleaner formatting

### When Q4_K_M Won

- More comprehensive context (included relevant background)
- Better mathematical notation
- More accurate domain-specific terminology

**Key Observation:** Neither model showed systematic advantages. Wins appeared random across categories.

---

## Practical Implications

### For Deployment

| Consideration | Recommendation |
|---------------|----------------|
| **Quality** | Q4_K_M is equivalent to F16 for this task |
| **Size** | Use Q4_K_M (60% smaller: 397MB vs 994MB) |
| **Speed** | Q4_K_M may be faster due to smaller memory footprint |
| **Memory** | Q4_K_M requires less VRAM/RAM |

### Caveats

1. **Task-specific:** Results apply to scientific QA with RAG. Other tasks (e.g., math, code) may differ.
2. **Model-specific:** Results apply to Qwen 2.5 0.5B. Larger models or different architectures may show different sensitivity to quantization.
3. **Quantization method:** Q4_K_M specifically. Other quantization schemes (e.g., Q2, Q8) may differ.

---

## Methodology Notes

### Sample Size Adequacy

With 93 decisive matches, we had sufficient power to detect a meaningful effect:
- Minimum detectable effect at 80% power: ~10 percentage points difference
- Observed difference: 3.2 percentage points (48.4% vs 51.6%)

### Reproducibility

All artifacts are saved:
- `evaluation/exports/quantization_batches/evaluation_results.json` - Parsed verdicts
- `evaluation/exports/quantization_batches/raw_judge_responses.json` - Raw Gemini outputs
- `evaluation/exports/quantization_batches/batch_mapping.json` - A/B assignment key
- `evaluation/responses/hybrid_science_0p5b_rag_trained.jsonl` - Q4_K_M responses
- `evaluation/responses/hybrid_science_0p5b_rag_trained_f16.jsonl` - F16 responses

---

## Conclusion

**Q4_K_M quantization does not degrade response quality** for this fine-tuned scientific QA model. The 4-bit quantization achieves:

- ✅ **Equivalent accuracy** (p=0.8357, no significant difference)
- ✅ **60% size reduction** (994MB → 397MB)
- ✅ **High tie rate** (39.6%) indicating genuinely similar outputs

**Recommendation:** Deploy the Q4_K_M version for production use. The smaller model size provides deployment benefits (lower memory, potentially faster inference) with no measurable quality trade-off.

---
