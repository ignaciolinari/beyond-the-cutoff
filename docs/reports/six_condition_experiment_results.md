# Six-Condition Experiment Results

**Date:** November-December 2025
**Evaluator:** Qwen3 8B (local LLM judge via Ollama, thinking mode, temp=0.6)
**Method:** Interleaved evaluation with multi-dimensional scoring
**Dataset:** 154 scientific Q&A pairs from papers published after model training cutoff

> ⚠️ **Scope Limitation**: These results apply to **Qwen 2.5 0.5B** only. Larger models with more parameter capacity may show different patterns, particularly for fine-tuning benefits. See [Scaling Guide](../scaling/README.md) for testing with larger models.

---

## Executive Summary

We conducted a comparative evaluation of **6 model configurations** on scientific Q&A tasks using papers published after the model's training cutoff. Each model answered ~72 questions, and responses were scored by an LLM judge on four dimensions plus a binary pass/fail verdict.

### Key Findings

1. **RAG at inference is critical:** Models with RAG achieve 19-24% pass rates vs 4-6% without RAG — a **4-5x improvement**
2. **Fine-tuning provides marginal gains:** FT-RAG+RAG (24.1%) only slightly outperforms Base+RAG (22.8%)
3. **Training format matters:** Fine-tuning on RAG-formatted data yields better results than instruction-only fine-tuning
4. **Fine-tuning cannot replace retrieval:** LoRA-only models (no RAG at inference) perform as poorly as base model

> 📊 **Metrics Note:** Pass rates and LLM judge scores are the primary valid metrics for cross-condition comparison. Reference-based metrics (BERTScore, BLEU) have limited validity for instruction-only conditions due to reference answer methodology — see [Limitations](#limitations).

---

## Experimental Design

### Model Configurations

This experiment tests three key questions:
1. **Does RAG help?** (Base vs Base+RAG)
2. **Does fine-tuning help?** (Base+RAG vs FT variants)
3. **What training format is best?** (Instruction-only vs RAG-trained)

| Condition ID | Description | Training | Inference Mode | Judge Type |
|--------------|-------------|----------|----------------|------------|
| `base_baseline_0p5b` | Base model, no RAG (instruction mode) | None | Instruction | Instruction |
| `rag_baseline_0p5b` | Base model + RAG | None | RAG | RAG |
| `lora_science_0p5b_ft_only` | FT instruction-only, no RAG | LoRA (instruction data) | Instruction | Instruction |
| `hybrid_science_0p5b_instruction_only` | FT instruction-only + RAG (**transfer test**) | LoRA (instruction data) | RAG | RAG |
| `lora_science_0p5b_rag_trained_ft_only` | FT RAG-trained, no RAG (**degradation test**) | LoRA (RAG data) | Instruction | Instruction |
| `hybrid_science_0p5b_rag_trained` | FT RAG-trained + RAG (**optimal**) | LoRA (RAG data) | RAG | RAG |

### Judge Configuration

- **RAG conditions** (`rag_baseline`, `hybrid_instruction_only`, `hybrid_rag_trained`): Use `judges/rag.yaml` which evaluates citation quality (grounding, citation metrics)
- **Instruction conditions** (`base_baseline`, `lora_ft_only`, `lora_rag_trained_ft_only`): Use `judges/instruction.yaml` which evaluates without citation requirements

### Reference Answer Generation

> ⚠️ **Methodological Note:** Reference answers (`expected_response`) were generated using **RAG with retrieved context** and contain inline citations (e.g., `[1] Section: ... | Page N`). This has implications for metrics interpretation — see [Reference-Based Metrics](#reference-based-metrics) section.

The reference answers are grounded in specific retrieved passages and contain:
- Factual claims derived from post-cutoff paper content
- Inline citations to numbered context chunks
- Information that is **inaccessible to instruction-only models** at inference time

---

## Overall Results

### Pass Rate Comparison

| Model | N | Pass Rate | Description |
|-------|---|-----------|-------------|
| `hybrid_science_0p5b_rag_trained` | 72 | **24.1%** | FT RAG-trained + RAG (**optimal**) |
| `rag_baseline_0p5b` | 72 | **22.8%** | Base model + RAG |
| `hybrid_science_0p5b_instruction_only` | 72 | 19.7% | FT instruction-only + RAG (**transfer test**) |
| `lora_science_0p5b_rag_trained_ft_only` | 72 | 5.6% | FT RAG-trained, no RAG (**degradation test**) |
| `lora_science_0p5b_ft_only` | 72 | 4.3% | FT instruction-only, no RAG |
| `base_baseline_0p5b` | 73 | 4.2% | Base model, no RAG |

### Pass Rate Visualization

```
hybrid_rag_trained     ████████████████████████░  24.1%  (optimal)
rag_baseline           ███████████████████████░░  22.8%  (base+RAG)
hybrid_instruction     ████████████████████░░░░░  19.7%  (transfer test)
lora_rag_trained_ft    ██░░░░░░░░░░░░░░░░░░░░░░░   5.6%  (degradation test)
lora_ft_only           █░░░░░░░░░░░░░░░░░░░░░░░░   4.3%  (FT only)
base_baseline          █░░░░░░░░░░░░░░░░░░░░░░░░   4.2%  (base only)
```

---

## Multi-Metric Analysis

### Judge Dimension Scores (0.0 - 1.0)

| Model | Factuality | Grounding | Completeness | Communication |
|-------|------------|-----------|--------------|---------------|
| `hybrid_rag_trained` | 0.43 | **0.44** | **0.48** | **0.76** |
| `rag_baseline` | **0.46** | **0.45** | 0.46 | 0.72 |
| `hybrid_instruction_only` | 0.38 | 0.38 | 0.41 | 0.70 |
| `lora_ft_only` | 0.33 | 0.00 | 0.36 | 0.74 |
| `base_baseline` | 0.29 | 0.00 | 0.35 | 0.70 |
| `lora_rag_trained_ft_only` | 0.29 | 0.00 | 0.35 | 0.69 |

### Reference-Based Metrics

> ⚠️ **Interpretability Warning:** Reference answers were generated with RAG context and contain citations. Comparing instruction-only model outputs (which lack context access) against these references is **methodologically problematic** — the models are being evaluated against answers they cannot possibly produce. These metrics are provided for completeness but should be interpreted with caution.

#### RAG Conditions (Valid Comparison)

These models receive the same retrieved context used to generate reference answers:

| Model | BERTScore F1 | BLEU | Interpretation |
|-------|--------------|------|----------------|
| `hybrid_rag_trained` | **0.873** | **0.135** | Best semantic and lexical overlap |
| `rag_baseline` | 0.863 | 0.106 | Strong baseline |
| `hybrid_instruction_only` | 0.861 | 0.093 | Transfer shows slight degradation |

#### Instruction-Only Conditions (Limited Validity)

These models **do not receive retrieved context** but are compared against context-grounded references. Low scores are expected and do not reflect model quality per se:

| Model | BERTScore F1 | BLEU | Interpretation |
|-------|--------------|------|----------------|
| `lora_ft_only` | 0.845 | 0.052 | Cannot match context-specific content |
| `base_baseline` | 0.821 | 0.018 | No access to post-cutoff knowledge |
| `lora_rag_trained_ft_only` | 0.820 | 0.013 | Training on RAG format doesn't transfer |

**Why instruction-only scores are low:**
- **BLEU (0.01-0.05):** Near-zero lexical overlap because models cannot cite `[1] Section: X | Page N` without seeing the context
- **BERTScore (0.82-0.85):** Some semantic overlap from topically related (but potentially hallucinated) content

### Citation Metrics (RAG conditions only)

| Model | Citation Coverage | Citation Precision | Citation Recall | Grounded Fraction |
|-------|-------------------|--------------------|-----------------|--------------------|
| `hybrid_rag_trained` | 0.40 | 0.35 | 0.38 | 0.41 |
| `rag_baseline` | 0.42 | 0.37 | 0.40 | 0.43 |
| `hybrid_instruction_only` | 0.36 | 0.31 | 0.34 | 0.37 |

*Note: Citation metrics are only computed for RAG conditions that have access to retrieved context.*

---

## Comparative Analysis

### RAG vs Non-RAG Gap

| Metric | RAG Models (avg) | Non-RAG Models (avg) | Gap | Validity |
|--------|------------------|----------------------|-----|----------|
| Pass Rate | 22.2% | 4.7% | **4.7x** | ✅ Valid |
| Factuality | 0.42 | 0.30 | +40% | ✅ Valid |
| Completeness | 0.45 | 0.35 | +29% | ✅ Valid |
| BERTScore F1 | 0.87 | 0.83 | +5% | ⚠️ Limited* |
| BLEU | 0.11 | 0.03 | **3.7x** | ⚠️ Limited* |

*Reference-based metrics (BERTScore, BLEU) compare against RAG-generated references. The gap partially reflects access to context rather than model quality.

**Key Insight:** RAG provides massive improvements in pass rate (valid metric), with corresponding gains in judge-assessed factuality and completeness. Reference-based metric gaps are confounded by context access differences.

### Fine-Tuning Impact (with RAG)

| Comparison | Model A | Model B | Pass Rate Δ |
|------------|---------|---------|-------------|
| RAG-trained FT vs Base | `hybrid_rag_trained` | `rag_baseline` | +1.3% |
| Instruction-only FT vs Base | `hybrid_instruction_only` | `rag_baseline` | -3.1% |

**Key Insight:** Fine-tuning on RAG-formatted examples provides a small benefit (+1.3%), but instruction-only fine-tuning actually *degrades* performance (-3.1%) when used with RAG. This suggests training format matters.

### Transfer Test: Instruction-Only FT + RAG

The `hybrid_science_0p5b_instruction_only` condition tests whether fine-tuning on instruction-only data transfers to RAG inference:

- **Result:** 19.7% pass rate (lower than base+RAG at 22.8%)
- **Interpretation:** Fine-tuning on one format doesn't transfer well to another format at inference time

### Degradation Test: RAG-Trained FT without RAG

The `lora_science_0p5b_rag_trained_ft_only` condition tests whether RAG-trained models work without RAG:

- **Result:** 5.6% pass rate (essentially same as base model at 4.2%)
- **Interpretation:** Fine-tuning on RAG data doesn't encode the knowledge — the model still needs retrieval at inference time

---

## Dimensional Analysis

### Grounding Scores

#### RAG Conditions (Metric Applicable)

| Model | Grounding Score |
|-------|-----------------|
| `rag_baseline` | 0.45 |
| `hybrid_rag_trained` | 0.44 |
| `hybrid_instruction_only` | 0.38 |

#### Instruction-Only Conditions (Metric Not Applicable)

| Model | Grounding Score | Note |
|-------|-----------------|------|
| `lora_ft_only` | 0.00 | No context to cite |
| `lora_rag_trained_ft_only` | 0.00 | No context to cite |
| `base_baseline` | 0.00 | No context to cite |

**Key Insight:** Grounding requires RAG at inference — models without access to retrieved context cannot cite sources.

> **Note:** The 0.0 grounding scores for instruction-only models are **by design**, not a model failure. These conditions use `judges/instruction.yaml` which does not require citations, and the models have no context to cite. This metric is only meaningful for RAG conditions.

### Communication Scores

| Model | Communication Score |
|-------|---------------------|
| `hybrid_rag_trained` | 0.76 |
| `lora_ft_only` | 0.74 |
| `rag_baseline` | 0.72 |
| `base_baseline` | 0.70 |
| `hybrid_instruction_only` | 0.70 |
| `lora_rag_trained_ft_only` | 0.69 |

**Key Insight:** Communication scores are remarkably stable across all models (range: 0.69-0.76, Δ=0.07). This narrow spread suggests fluency is not the differentiating factor — all models communicate at roughly the same quality level regardless of RAG or fine-tuning.

---

## Response Characteristics

| Model | Avg Generation Time | Notes |
|-------|---------------------|-------|
| `base_baseline` | ~2.5s | Fast, but low quality |
| `rag_baseline` | ~3.2s | Additional context processing |
| `lora_ft_only` | ~2.4s | LoRA efficient inference |
| `hybrid_instruction_only` | ~3.1s | Combined overhead |
| `lora_rag_trained_ft_only` | ~2.4s | LoRA efficient inference |
| `hybrid_rag_trained` | ~3.3s | Highest quality, moderate speed |

---

## Statistical Considerations

### Sample Size Warning

With ~72 examples per model, statistical power is limited:

- **Pass rate confidence:** ±5-8% at 95% confidence
- **Dimension scores:** More stable due to continuous scale
- **Pairwise comparisons:** Would need ~200+ examples for reliable p<0.05

### Effect Size Analysis

| Comparison | Effect | Cohen's d (approx) | Interpretation |
|------------|--------|-------------------|----------------|
| RAG vs Non-RAG | +17.5% pass rate | ~0.9 | Large effect |
| FT-RAG vs Base+RAG | +1.3% pass rate | ~0.1 | Negligible effect |
| RAG-trained vs Instruction-only | +4.4% pass rate | ~0.3 | Small effect |

---

## Conclusions

### Research Questions Answered

1. **Does RAG help?** ✅ **Yes, dramatically.** 4-5x improvement in pass rate (valid metric).

2. **Does fine-tuning help?** ⚠️ **Marginally.** Only +1.3% when training format matches inference format.

3. **What training format is best?** ✅ **RAG-formatted training.** Instruction-only training actually hurts RAG inference.

### Validity of Conclusions

These conclusions are primarily supported by:
- **LLM judge pass rates** (valid across all conditions)
- **Judge dimension scores** (factuality, completeness, communication)

They are **not** well-supported by reference-based metrics (BERTScore, BLEU) for instruction-only conditions, due to the confound that reference answers contain context-derived content unavailable to those models.

### Recommendations

| Use Case | Recommendation | Expected Performance |
|----------|----------------|---------------------|
| Production deployment | `hybrid_rag_trained` | 24% pass rate |
| Quick baseline | `rag_baseline` | 22% pass rate |
| Resource-constrained | `rag_baseline` | 22% pass rate (no FT needed) |
| No retrieval available | None recommended | <6% pass rate |

### Key Takeaways

1. ✅ **Always use RAG for post-cutoff knowledge tasks** — the improvement is dramatic and fine-tuning cannot compensate
2. ✅ **If fine-tuning, match training format to inference format** — instruction-only FT hurts RAG performance
3. ✅ **Fine-tuning is optional** — Base+RAG achieves 95% of the best model's performance
4. ❌ **Don't expect fine-tuning to encode new knowledge** — retrieval is still required at inference time

---

## Limitations

### Methodological Limitations

1. **Reference answer mismatch:** Reference answers were generated with RAG context, making reference-based metrics (BERTScore, BLEU) invalid for instruction-only conditions. These models lack access to the information needed to produce matching answers.

2. **Confounded comparisons:** The RAG vs non-RAG comparison conflates two factors: (a) access to retrieved context at inference, and (b) ability to match reference answers that themselves contain context-derived content.

### Statistical Limitations

3. **Sample size:** ~72 examples per model provides limited statistical power (95% CI: ±5-8% for pass rates)

4. **Single judge:** Qwen3 8B may have systematic biases; results may vary with different judges

5. **No human validation:** LLM judge scores not validated against human judgments

### Scope Limitations

6. **Experiment incomplete:** Evaluation was stopped before full dataset coverage

7. **Single model family:** Results specific to Qwen 2.5 0.5B; larger models may show different patterns for fine-tuning benefits

8. **Domain specificity:** Results based on scientific Q&A from arXiv papers; generalization to other domains is unknown

---

## Related Work

See also:
- [Pairwise Evaluation Results](pairwise_evaluation_results.md): Head-to-head comparison using Gemini 3 Pro judge
- `configs/evaluation/six_condition_experiment.yaml`: Full experimental configuration

---

## Files

- **Evaluation config:** `configs/evaluation/six_condition_experiment.yaml`
- **Results directory:** `evaluation/results/interleaved/`
- **Per-model details:** `evaluation/results/interleaved/{model}/details.jsonl`
- **Per-model metrics:** `evaluation/results/interleaved/{model}/metrics.json`
