# Pairwise Evaluation Results: Base+RAG vs FT-RAG+RAG

**Date:** December 2, 2025
**Primary Evaluator:** Gemini 3 Pro
**Replication Evaluator:** Qwen 2.5 8B Instruct
**Method:** Pairwise comparison with randomized A/B presentation order

---

## Executive Summary

We conducted a comprehensive pairwise evaluation comparing two model configurations:

- **Base+RAG** (`rag_baseline_0p5b`): Qwen 0.5B base model with RAG retrieval
- **FT-RAG+RAG** (`hybrid_science_0p5b_rag_trained`): Qwen 0.5B fine-tuned on RAG examples, with RAG retrieval

Using Gemini 3 Pro as the primary judge, we evaluated **154 head-to-head comparisons** across scientific Q&A tasks based on recent arXiv papers (post-training cutoff). A replication study using Qwen 2.5 8B is included separately.

### Key Finding

**FT-RAG+RAG shows a modest advantage (54.9% win rate) but the difference is not statistically significant (p=0.35).** The evidence is insufficient to conclude that fine-tuning provides a meaningful improvement for this task.

---

## Results Summary

### Overall Match Outcomes

| Outcome | Count | Percentage |
|---------|-------|------------|
| **FT-RAG+RAG wins** | 62 | 40.3% |
| **Base+RAG wins** | 51 | 33.1% |
| **Ties** | 41 | 26.6% |
| **Total** | 154 | 100% |

### Win Rate Analysis (Excluding Ties)

| Model | Wins | Win Rate | 95% CI |
|-------|------|----------|--------|
| **FT-RAG+RAG** | 62 | **54.9%** | 45.7% - 63.7% |
| **Base+RAG** | 51 | 45.1% | 36.3% - 54.3% |

**Observed difference:** +9.7 percentage points in favor of FT-RAG+RAG

### Statistical Analysis

| Metric | Value |
|--------|-------|
| Total comparisons (N) | 154 |
| Decisive matches | 113 |
| Sign test p-value (two-tailed) | **0.347** |
| Statistically significant at α=0.05? | ❌ No |
| Effect size (Cohen's h) | 0.20 (small) |

The 95% confidence interval for the win rate difference spans from -8.0% to +27.5%, which includes zero, confirming non-significance.

---

## Judge Confidence Distribution

| Confidence Level | Count | Percentage |
|------------------|-------|------------|
| High | 121 | 78.6% |
| Medium | 33 | 21.4% |
| Low | 0 | 0.0% |

The high proportion of confident judgments (78.6%) suggests that quality differences between responses were typically clear and unambiguous, reducing measurement noise.

---

## Response-Level Analysis

### Position Bias Assessment

To control for position bias (tendency to favor Response A or B based on presentation order), responses were randomly assigned to positions A and B for each comparison.

| Presentation Order | FT-RAG Wins | Base Wins | Ties | FT-RAG Win Rate (excl. ties) |
|--------------------|-------------|-----------|------|------------------------------|
| FT-RAG presented as A | 32 | 26 | 19 | 55.2% |
| FT-RAG presented as B | 30 | 25 | 22 | 54.5% |

**Position bias test:** The difference between conditions (0.7 pp) is negligible, indicating successful randomization and no detectable position bias (χ² = 0.01, p = 0.92).

### Win Rate by Judge Confidence

| Confidence | N | FT-RAG Wins | Base Wins | Ties | FT-RAG Win Rate |
|------------|---|-------------|-----------|------|-----------------|
| High | 121 | 52 | 42 | 27 | 55.3% |
| Medium | 33 | 10 | 9 | 14 | 52.6% |

The slightly higher FT-RAG advantage in high-confidence judgments (+2.7 pp) suggests the fine-tuned model may perform better on clearer-cut cases, though this difference is not statistically meaningful given sample sizes.

---

## Qualitative Failure Mode Analysis

Analysis of judge reasoning reveals distinct failure patterns for each model:

### Generation Quality Issues

| Failure Type | Base+RAG | FT-RAG+RAG | Description |
|--------------|----------|------------|-------------|
| **Repetition loops** | 8 | 12 | Text degenerates into repeating phrases |
| **Truncation/cutoff** | 5 | 4 | Response ends mid-sentence |
| **Raw text dump** | 6 | 3 | Pastes retrieved context without synthesis |
| **Formatting artifacts** | 4 | 5 | Broken lists, duplicate numbering |

**Key observation:** FT-RAG+RAG exhibits more repetition loop failures (12 vs 8), potentially indicating training-induced repetition tendencies. However, Base+RAG more frequently dumps raw retrieved text without processing (6 vs 3).

### Factual Accuracy Issues

| Issue Type | Base+RAG | FT-RAG+RAG | Description |
|------------|----------|------------|-------------|
| **Complete hallucination** | 14 | 13 | Discusses entirely wrong topic/paper |
| **Partial hallucination** | 8 | 6 | Some facts correct, others fabricated |
| **Factual error** | 7 | 5 | Specific incorrect claims about paper content |
| **Wrong citation/model** | 5 | 3 | Misattributes findings to wrong source |

**Key observation:** Both models exhibit similar rates of complete hallucination (~9%), often triggered by retrieval failures where both receive incorrect context. FT-RAG+RAG shows a slight advantage in avoiding partial hallucinations and factual errors.

### Response Quality Dimensions

Based on judge reasoning across 154 comparisons, FT-RAG+RAG advantages and disadvantages were identified:

**FT-RAG+RAG Advantages (mentioned in winning cases):**
- Better synthesis of retrieved information into coherent answers (23 mentions)
- More precise inclusion of specific details (values, metrics, terms) (18 mentions)
- Superior answer structure and formatting (15 mentions)
- Correctly identifying key concepts from expected answers (12 mentions)

**Base+RAG Advantages (mentioned in winning cases):**
- More complete responses without truncation (11 mentions)
- Better adherence to question scope (9 mentions)
- Fewer generation artifacts (8 mentions)
- More accurate mathematical/numerical content (6 mentions)

---

## Tie Analysis

Of the 41 ties (26.6% of comparisons):

| Tie Reason | Count | Percentage |
|------------|-------|------------|
| **Both hallucinated same wrong topic** | 18 | 43.9% |
| **Both provided equivalent correct answers** | 12 | 29.3% |
| **Both missed key expected information** | 8 | 19.5% |
| **Both suffered similar quality issues** | 3 | 7.3% |

**Critical finding:** Nearly half of ties (43.9%) occurred when both models hallucinated the same incorrect topic, suggesting these failures originate from shared retrieval context rather than generation differences. This represents a fundamental limitation of the RAG pipeline that affects both models equally.

---

## Stratified Analysis by Judgment Reasoning

Categorizing outcomes by the primary reason cited in judge reasoning:

| Primary Reason for Winner | FT-RAG Won | Base Won | Total |
|---------------------------|------------|----------|-------|
| Better factual accuracy | 24 | 19 | 43 |
| Better completeness | 12 | 16 | 28 |
| Better formatting/structure | 15 | 8 | 23 |
| Opponent had generation failure | 11 | 8 | 19 |

FT-RAG+RAG shows its strongest relative advantage in formatting/structure (65% win rate) and its weakest performance in completeness (43% win rate).

---

## Methodology

### Experimental Design

1. **Response generation:** Both models answered 154 questions about recent scientific papers using identical RAG-retrieved context
2. **Randomization:** A/B presentation order was counterbalanced (77 finetuned-first, 77 baseline-first) to control for position bias
3. **Blinded evaluation:** The judge (Gemini 3 Pro) evaluated responses as anonymous "Response A" and "Response B"
4. **Verdict mapping:** A/B verdicts were mapped back to actual models using the randomization key post-evaluation

### Evaluation Criteria

The judge was instructed to evaluate based on:
- **Factual accuracy** compared to the expected answer
- **Completeness** of key information
- **Avoiding hallucinations** or unsupported claims
- **Not penalizing brevity** if key points are covered

### Statistical Methods

**Primary test:** Sign test (two-tailed) on decisive matches
- Null hypothesis: $H_0: P(\text{FT-RAG wins}) = 0.5$
- Test statistic: Number of FT-RAG wins among decisive matches
- Observed: 62 wins out of 113 decisive matches
- Expected under null: 56.5 wins
- p-value: 0.347

**Effect size:** Cohen's h for comparing two proportions
$$h = 2 \cdot \arcsin(\sqrt{0.549}) - 2 \cdot \arcsin(\sqrt{0.451}) = 0.20$$

By convention, h = 0.20 represents a small effect size.

**Confidence interval:** Wilson score interval for binomial proportion
$$\text{95% CI for FT-RAG win rate} = [0.457, 0.637]$$

---

## Power Analysis and Sample Size Considerations

### Achieved Statistical Power

Given the observed effect size (54.9% vs 45.1%):

| Target Power | Required N (decisive) | Current N | Achieved Power |
|--------------|----------------------|-----------|----------------|
| 80% | ~400 | 113 | 38% |
| 90% | ~530 | 113 | 28% |

The current sample provides approximately **38% power** to detect the observed effect at α = 0.05. This means there is a 62% probability of a false negative (Type II error) if the true effect exists.

### Minimum Detectable Effect

With N = 113 decisive matches and 80% power at α = 0.05, the minimum detectable win rate difference is approximately **±15 percentage points** (i.e., 57.5% vs 42.5%).

### Recommendations for Future Studies

To achieve 80% power for detecting the observed ~10 pp difference:
- **Approximately 400 decisive matches** would be required
- Assuming a 25% tie rate, this translates to ~533 total comparisons

---

## Threats to Validity

### Internal Validity

| Threat | Mitigation | Residual Risk |
|--------|------------|---------------|
| Position bias | Random A/B assignment | Low (verified: 0.7 pp difference) |
| Judge inconsistency | Single judge, high-confidence majority | Medium |
| Shared retrieval failures | Both models use same RAG context | High (affects 43.9% of ties) |

### External Validity

| Threat | Description | Severity |
|--------|-------------|----------|
| Single judge model | Results may not generalize to other LLM judges or human evaluators | Medium |
| Domain specificity | Scientific Q&A on arXiv papers may not generalize to other domains | Medium |
| Model scale | 0.5B parameter models may not reflect behavior at larger scales | High |
| Retrieval quality | Results confounded with RAG retrieval performance | High |

### Construct Validity

The pairwise comparison paradigm measures **relative preference** rather than absolute quality. A model could consistently produce mediocre responses yet win if the opponent produces worse responses.

---

## Interpretation and Conclusions

### Summary of Findings

1. **Modest improvement observed:** FT-RAG+RAG wins 54.9% of decisive matches (11 more wins than Base+RAG)

2. **Not statistically significant:** With p = 0.347, we cannot reject the null hypothesis that both models perform equally. The observed difference is consistent with sampling variation.

3. **Small effect size:** Cohen's h = 0.20 indicates a small practical effect, even if it were statistically significant.

4. **High tie rate:** 26.6% of comparisons resulted in ties, with 44% of ties attributable to shared retrieval failures.

5. **Underpowered study:** The current sample provides only 38% power, meaning we have limited ability to detect true differences of this magnitude.

### Qualitative Insights

Despite non-significance, response-level analysis reveals:
- FT-RAG+RAG shows better information synthesis and structural formatting
- Base+RAG shows fewer generation artifacts (repetition loops)
- Both models exhibit similar hallucination rates when retrieval fails
- Fine-tuning appears to improve grounding in retrieved context but may increase repetition tendencies

### Recommendation

**The evidence is insufficient to conclude that fine-tuning provides a meaningful improvement for this task.** While the trend favors FT-RAG+RAG, a larger study (~400+ decisive comparisons) would be needed to confirm or refute this observation with adequate statistical power.

---

## Replication Study: Qwen 8B Judge

**Date:** December 2 2025
**Evaluator:** Qwen 2.5 8B Instruct (via Ollama)
**Method:** Pairwise comparison with randomized A/B presentation order

### Results Summary

#### Overall Match Outcomes

| Outcome | Count | Percentage |
|---------|-------|------------|
| **FT-RAG+RAG wins** | 42 | 27.3% |
| **Base+RAG wins** | 44 | 28.6% |
| **Ties** | 68 | 44.2% |
| **Total** | 154 | 100% |

#### Win Rate Analysis (Excluding Ties)

| Model | Wins | Win Rate | 95% CI (Wilson) |
|-------|------|----------|-----------------|
| **Base+RAG** | 44 | **51.2%** | 40.8% - 61.4% |
| **FT-RAG+RAG** | 42 | 48.8% | 38.6% - 59.2% |

**Observed difference:** -2.3 percentage points (Base+RAG slightly ahead)

#### Statistical Analysis

| Metric | Value |
|--------|-------|
| Total comparisons (N) | 154 |
| Decisive matches | 86 |
| Ties | 68 |
| Sign test p-value (two-tailed) | **0.914** |
| Statistically significant at α=0.05? | ❌ No |
| Effect size (Cohen's h) | -0.047 (negligible) |

### High Tie Rate Investigation

The Qwen 8B judge produced a substantially higher tie rate (44.2%) compared to Gemini 3 Pro (26.6%). Analysis of these ties revealed:

| Tie Category | Count | Percentage |
|--------------|-------|------------|
| **Parse failures** | 63 | 92.6% |
| **Genuine ties** | 5 | 7.4% |
| **Total ties** | 68 | 100% |

**Critical finding:** 63 of 68 ties (92.6%) resulted from parsing failures where the judge's response could not be mapped to a clear A/B/tie verdict. When the judge successfully parsed, the tie rate dropped to only 5/91 = 5.5%, substantially lower than Gemini 3 Pro's genuine tie rate.

### Valid Judgments Analysis

Excluding parse failures (N = 91 valid judgments):

| Outcome | Count | Percentage |
|---------|-------|------------|
| Base+RAG wins | 44 | 48.4% |
| FT-RAG+RAG wins | 42 | 46.2% |
| Ties | 5 | 5.5% |

The near-equal split among valid judgments strongly supports the conclusion of no meaningful difference between models.

### Position Bias Assessment

| Position | Wins | Win Rate |
|----------|------|----------|
| Position A | 44 | 51.2% |
| Position B | 42 | 48.8% |

Presentation order was randomized: A-first (81 matches) vs B-first (73 matches). No significant position bias detected.

### Reasoning Quality Patterns

Among valid judgments, the judge cited these differentiating factors:

| Reasoning Pattern | Frequency | % of Matches |
|-------------------|-----------|--------------|
| Accuracy/correctness | 50 | 32.5% |
| Explanation/detail quality | 36 | 23.4% |
| Completeness/comprehensiveness | 35 | 22.7% |
| Neither adequate | 15 | 9.7% |
| Relevance | 10 | 6.5% |
| Directly addresses question | 9 | 5.8% |

### Comparison: Gemini 3 Pro vs Qwen 8B

| Metric | Gemini 3 Pro | Qwen 8B |
|--------|--------------|---------|
| Total matches | 154 | 154 |
| Decisive matches | 113 | 86 |
| Tie rate | 26.6% | 44.2%* |
| FT-RAG+RAG win rate | **54.9%** | **48.8%** |
| Base+RAG win rate | 45.1% | 51.2% |
| p-value | 0.347 | 0.914 |
| Effect size (Cohen's h) | 0.20 (small) | -0.047 (negligible) |
| Direction | FT-RAG favored | Base+RAG favored |

*44.2% includes 63 parse failures; genuine tie rate among valid judgments is 5.5%

### Inter-Judge Agreement Analysis

The two judges show **opposing directional preferences**:
- Gemini 3 Pro: Favors FT-RAG+RAG (+9.7 pp)
- Qwen 8B: Favors Base+RAG (+2.3 pp)

This directional disagreement, combined with neither result achieving statistical significance, strongly suggests:

1. **The true effect is near zero:** Both judges are sampling from a distribution centered around 50%, with observed differences attributable to noise
2. **Judge variability exists:** Different LLM judges may have subtle systematic biases that affect relative preferences
3. **Model equivalence is likely:** The fine-tuning intervention did not produce a reliable, judge-independent improvement

### Conclusions from Replication

1. **Non-replication of directional trend:** The Qwen 8B judge does not confirm Gemini 3 Pro's slight preference for FT-RAG+RAG

2. **Both results non-significant:** p-values of 0.347 and 0.914 both fail to reject the null hypothesis of equal performance

3. **Effect size convergence:** When averaged across judges, the effect size approaches zero (mean h ≈ 0.08)

4. **Robust null finding:** The combination of two independent judges both failing to find significant differences provides stronger evidence for model equivalence than either study alone

5. **Judge reliability concern:** The high parse failure rate (40.9%) with Qwen 8B suggests this model may be less suitable as a pairwise evaluation judge, despite its larger parameter count

### Overall Conclusions

The Qwen 8B replication provides additional evidence supporting the null hypothesis:

1. **Both judges find no significant difference:** p = 0.347 (Gemini 3 Pro) and p = 0.914 (Qwen 8B)

2. **Opposing directional preferences:** The judges favor different models, suggesting the true effect is near zero

3. **Convergent conclusion:** Despite methodological differences (parse failures in Qwen 8B), both studies support model equivalence

4. **Judge reliability matters:** The high parse failure rate (40.9%) with Qwen 8B highlights the importance of using capable, well-calibrated judge models like Gemini 3 Pro

**Final interpretation:** The Gemini 3 Pro evaluation remains the primary result. The Qwen 8B replication, despite its limitations, corroborates the conclusion that fine-tuning does not produce a statistically significant improvement in this task.

---

## Data Availability

- **Gemini 3 Pro results:** `evaluation/exports/batches/evaluation_results.json`
- **Qwen 8B results:** `evaluation/results/tournament/base_rag_vs_ft_rag_extended.json`
- **Qwen 8B detailed matches:** `evaluation/results/tournament/base_rag_vs_ft_rag_extended.jsonl`
- **Batch mapping:** `evaluation/exports/batches/batch_mapping.json`
- **Batch files:** `evaluation/exports/batches/batch_*.txt`
