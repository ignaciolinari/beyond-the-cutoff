# Evaluation Methodology

This document explains the evaluation strategy used in this project, clarifying what is being measured at each stage and outlining plans for end-to-end evaluation.

## Current Evaluation: Context Utilization

### What We Are Measuring

The current 6-condition experiment evaluates **how well models utilize provided contexts to generate grounded responses**. This is distinct from evaluating retrieval quality.

```
Current Pipeline:
Question + Pre-retrieved Contexts --> Model --> Response --> Judge
```

The retrieval step is performed offline during dataset generation (`generate_offline_dataset.py`), and the same contexts are used for all model conditions. This design choice is intentional:

1. **Isolation of variables**: By fixing the contexts, we isolate the model's generation capability from retrieval variance
2. **Fair comparison**: All models see identical contexts for each question
3. **Reproducibility**: Retrieval can be non-deterministic; pre-computing ensures consistent evaluation

### What We Are NOT Measuring

- Retrieval quality (precision, recall of retrieved documents)
- End-to-end latency including retrieval
- Model behavior with dynamically retrieved (potentially different) contexts

### Justification

This approach aligns with standard practices in RAG research. Papers such as REALM, RAG, and RETRO typically evaluate generation quality with fixed retrieved contexts to isolate the contribution of the language model from the retriever.

For this thesis, the primary hypothesis concerns **fine-tuning the LLM for RAG format**, not improving retrieval. The retriever (embedding model + FAISS index) remains constant across all conditions.

---

## 6-Condition Experiment Design

| Condition | Model | Prompt Mode | Description |
|-----------|-------|-------------|-------------|
| 1. base_baseline_0p5b | qwen2.5:0.5b-instruct | instruction | Base model, no contexts |
| 2. rag_baseline_0p5b | qwen2.5:0.5b-instruct | rag | Base model with contexts |
| 3. lora_science_0p5b_ft_only | lora_science_0p5_instruction_only | instruction | FT (instruction-only), no contexts |
| 4. hybrid_science_0p5b_instruction_only | lora_science_0p5_instruction_only | rag | FT (instruction-only) with contexts |
| 5. lora_science_0p5b_rag_trained_ft_only | lora_science_0p5 | instruction | FT (RAG format), no contexts |
| 6. hybrid_science_0p5b_rag_trained | lora_science_0p5 | rag | FT (RAG format) with contexts |

### Expected Outcomes

Based on the hypothesis that fine-tuning with RAG-formatted data improves context utilization:

- Condition 6 should outperform Condition 2 (both use RAG, but 6 is fine-tuned for it)
- Condition 4 may show distribution shift (trained on simple format, evaluated with RAG format)
- Condition 5 may show inverse distribution shift (trained on RAG format, evaluated without contexts)

---

## Phase 2: End-to-End Evaluation Plan

After identifying the best-performing model from the 6-condition experiment, we will conduct end-to-end evaluation with live retrieval.

### Objective

Validate that the best model performs well in a realistic deployment scenario where retrieval happens at inference time.

### Methodology

```
End-to-End Pipeline:
Question --> Retriever (live) --> Contexts --> Model --> Response --> Judge
```

### Implementation Plan

#### Step 1: Select Best Model

After completing the 6-condition evaluation, identify the best model based on:
- Judge scores (factuality, grounding, completeness, communication)
- Citation metrics (coverage, accuracy)
- Aggregated ranking

Expected winner: `lora_science_0p5` (Condition 6: FT + RAG)

#### Step 2: Create End-to-End Evaluation Script

Create `scripts/evaluate_end_to_end.py` that:

1. Loads the retrieval index from `data/processed/faiss_index/`
2. For each evaluation question:
   - Performs live retrieval (top-k contexts)
   - Constructs RAG prompt dynamically
   - Generates response
   - Evaluates with judge
3. Outputs metrics comparable to the offline evaluation

#### Step 3: Evaluation Dataset

Use the same 154 evaluation questions from `evaluation/datasets/eval_dataset.jsonl`, but:
- Ignore the pre-computed `rag.contexts` and `rag.prompt`
- Perform fresh retrieval for each question
- Compare retrieved contexts with pre-computed ones (optional analysis)

#### Step 4: Metrics to Collect

| Metric | Description |
|--------|-------------|
| Judge scores | Same as offline evaluation |
| Retrieval overlap | Jaccard similarity between live and pre-computed contexts |
| Latency | Total time including retrieval |
| Context relevance | Optional: human evaluation of retrieved contexts |

#### Step 5: Comparison Points

Run end-to-end evaluation for:
1. Best fine-tuned model (expected: `lora_science_0p5`)
2. Base model (`qwen2.5:0.5b-instruct`) as baseline

This validates that the fine-tuned model maintains its advantage in realistic conditions.

---

## Phase 3: Quantization Impact Analysis

### Objective

Determine whether Q4_K_M quantization significantly degrades model quality compared to F16 (full precision).

### Background

Current deployment uses Q4_K_M quantized models (~397 MB) for fair comparison with the base model. The F16 versions (~988 MB) are preserved in:
- `outputs/lora_science_v1/merged_full_model/Qwen2.5-0.5B-lora_science_v1.gguf`
- `outputs/lora_science_v1_instruction_only/merged_full_model/Qwen2.5-0.5B-lora_science_v1_instruction_only.gguf`

### Implementation Plan

#### Step 1: Register F16 Models with Ollama

```bash
# Create Modelfiles for F16 versions
# Modelfile.rag_trained_f16
FROM outputs/lora_science_v1/merged_full_model/Qwen2.5-0.5B-lora_science_v1.gguf
# ... (same parameters as Q4_K_M version)

ollama create lora_science_0p5_f16 -f ollama/Modelfile.rag_trained_f16
```

#### Step 2: Create Comparison Plan

Add two new conditions to the evaluation:

| Condition | Model | Quantization | Size |
|-----------|-------|--------------|------|
| Q4_K_M (current) | lora_science_0p5 | Q4_K_M | ~397 MB |
| F16 (new) | lora_science_0p5_f16 | F16 | ~988 MB |

#### Step 3: Evaluation

Run the same evaluation (offline with pre-computed contexts) for both quantization levels:

```bash
# Generate responses for F16 model
python scripts/generate_responses.py \
    --plan configs/evaluation/quantization_comparison.yaml \
    --output-dir evaluation/responses_quantization/

# Evaluate
python scripts/compare_models.py \
    --plan configs/evaluation/quantization_comparison.yaml \
    --responses-dir evaluation/responses_quantization/
```

#### Step 4: Analysis

Compare:
- Judge scores (is there significant degradation?)
- Response length and style differences
- Specific failure cases in Q4_K_M that F16 handles correctly

### Expected Outcome

For 0.5B models, quantization impact is typically minimal because:
- Small models have less redundancy to compress
- Q4_K_M is a high-quality quantization method
- The model was already trained at reduced precision (likely BF16)

If the difference is negligible (less than 2-3% on judge scores), Q4_K_M is preferred for deployment due to:
- 2.5x smaller memory footprint
- Faster inference on CPU
- Matches base model quantization for fair comparison

---

## Execution Timeline

### Completed

1. 6-condition experiment setup
2. Two-phase evaluation pipeline (generate_responses.py + compare_models.py)
3. Model registration with Ollama (Q4_K_M)

### In Progress

1. Response generation for 6 conditions (154 examples each)
2. Judge evaluation of generated responses

### Planned

1. Analysis of 6-condition results
2. End-to-end evaluation with live retrieval (best model only)
3. Quantization impact analysis (F16 vs Q4_K_M)
4. Retrieval optimization with ELO ranking

---

## Phase 4: Retrieval Optimization with ELO Ranking

### Objective

Once the best model is identified (expected: `lora_science_0p5`), optimize the retrieval configuration using pairwise comparisons and ELO ranking.

### Rationale

The 6-condition experiment fixes retrieval to isolate model quality. However, in production, retrieval configuration significantly impacts end-to-end performance. A tournament-style evaluation allows systematic comparison of retrieval strategies.

### Retrieval Parameters to Optimize

| Parameter | Options to Test | Impact |
|-----------|-----------------|--------|
| Top-k | 3, 4, 6, 8, 12 | More contexts provide more information but increase noise and prompt length |
| Reranking | None, BGE-Reranker-v2-M3 | Rerankers refine initial retrieval using cross-encoder models |
| Retrieve-Rerank Pattern | Same k, or retrieve more + rerank fewer | Wider initial net can improve final quality |

### Implemented Ablation Study

Fix the model (`lora_science_0p5`) and vary only retrieval configuration:

| Condition | Retrieve K | Final K | Reranker | Notes |
|-----------|------------|---------|----------|-------|
| dense_top4_baseline | 4 | 4 | None | Current production config |
| dense_top3 | 3 | 3 | None | Minimal context (less noise) |
| dense_top6 | 6 | 6 | None | More context |
| dense_top4_rerank | 4 | 4 | BGE-v2-M3 | Same k + reranking |
| dense_top8_rerank4 | 8 | 4 | BGE-v2-M3 | Wider net, rerank to 4 |
| dense_top12_rerank5 | 12 | 5 | BGE-v2-M3 | Widest net, rerank to 5 |

### Reranker Selection

We use `BAAI/bge-reranker-v2-m3` (568M parameters) as the state-of-the-art open-source reranker:

| Model | Size | Quality | Speed | Recommendation |
|-------|------|---------|-------|----------------|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | Good | Fast | Quick experiments |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 33M | Better | Medium | Balanced |
| BAAI/bge-reranker-v2-m3 | 568M | Best | Slower | **Production** |

### Running the Ablation

```bash
# Generate responses for all retrieval conditions
python scripts/run_retrieval_ablation.py \
    --config configs/default.yaml \
    --plan configs/evaluation/retrieval_ablation.yaml \
    --output-dir evaluation/results/retrieval_ablation/

# Run specific conditions only
python scripts/run_retrieval_ablation.py \
    --config configs/default.yaml \
    --plan configs/evaluation/retrieval_ablation.yaml \
    --output-dir evaluation/results/retrieval_ablation/ \
    --conditions dense_top4_baseline dense_top4_rerank

# Quick test with limit
python scripts/run_retrieval_ablation.py \
    --config configs/default.yaml \
    --plan configs/evaluation/retrieval_ablation.yaml \
    --output-dir evaluation/results/retrieval_ablation/ \
    --limit 10
```

### Compute ELO Rankings

After generating responses for all conditions:

```bash
# Run pairwise comparisons
python scripts/run_pairwise_evaluation.py \
    --results evaluation/results/retrieval_ablation/*.jsonl \
    --output evaluation/results/retrieval_ablation/pairwise/

# Compute ELO rankings
python scripts/compute_elo_rankings.py \
    --comparisons evaluation/results/retrieval_ablation/pairwise/pairwise_results.jsonl \
    --output evaluation/results/retrieval_ablation/elo_rankings.json
```

### Future Work: BM25 and Hybrid Search

The current implementation only supports **dense retrieval** (FAISS + sentence-transformers). To add BM25/hybrid search:

1. **BM25 Implementation**: Use `rank_bm25` library for sparse lexical retrieval
2. **Hybrid Combination**: Weighted combination of dense and sparse scores
   - Formula: `α * dense_score + (1-α) * bm25_score`
   - Typical α: 0.7 (favor dense) to 0.3 (favor BM25)
3. **Query Expansion**: Use LLM to expand queries before retrieval

These would require modifications to `src/beyond_the_cutoff/retrieval/query.py`.

### Configuration File

See `configs/evaluation/retrieval_ablation.yaml` for the full configuration including:
- 6 retrieval conditions
- Pairwise evaluation settings
- ELO computation parameters

---

## Unified Evaluation Pipeline

### Overview

The `run_evaluation_pipeline.py` script orchestrates complete evaluation workflows:

```bash
# Full 6-condition comparison
python scripts/run_evaluation_pipeline.py full-comparison \
    --plan configs/evaluation/compare_0p5b_experiments.yaml \
    --output-dir evaluation/results/six_condition/

# Quantization comparison (Q4_K_M vs F16)
python scripts/run_evaluation_pipeline.py quantization \
    --plan configs/evaluation/quantization_comparison.yaml \
    --output-dir evaluation/results/quantization/ \
    --register-f16  # Optional: registers F16 model first

# Retrieval ablation with ELO ranking
python scripts/run_evaluation_pipeline.py retrieval-ablation \
    --plan configs/evaluation/retrieval_ablation.yaml \
    --output-dir evaluation/results/retrieval_ablation/

# End-to-end validation with live retrieval
python scripts/run_evaluation_pipeline.py end-to-end \
    --plan configs/evaluation/end_to_end.yaml \
    --output-dir evaluation/results/end_to_end/
```

### Workflow Options

| Workflow | Purpose | Outputs |
|----------|---------|---------|
| `full-comparison` | 6-condition model comparison | Responses + judge results + visualization |
| `quantization` | Q4_K_M vs F16 comparison | Quantization impact analysis |
| `retrieval-ablation` | Retrieval config optimization | ELO rankings of retrieval configs |
| `end-to-end` | Full pipeline validation | Live retrieval + generation + evaluation |

---

## Complete Evaluation Roadmap

### Stage 1: Model Selection (Current)
- 6-condition experiment with fixed retrieval
- Identify best model (expected: FT + RAG)
- Deliverable: Model recommendation with statistical significance

### Stage 2: Quantization Analysis
- Compare Q4_K_M vs F16 for best model
- Determine if quantization impact is acceptable
- Deliverable: Deployment recommendation (quantization level)

### Stage 3: Retrieval Optimization
- ELO tournament with 6 retrieval configurations
- Test top_k variants and BGE-Reranker-v2-M3
- Deliverable: Optimal retrieval configuration

### Stage 4: End-to-End Validation
- Full pipeline evaluation with best model + best retrieval
- Compare against baseline (base model + default retrieval)
- Deliverable: Final system performance metrics

### Stage 5: Production Readiness (Optional)
- Latency benchmarking under load
- Memory profiling
- Error handling and fallback strategies

---

## File References

### Scripts

| Purpose | Path |
|---------|------|
| Response generation (Phase 1) | `scripts/generate_responses.py` |
| Comparison evaluation (Phase 2) | `scripts/compare_models.py` |
| Unified pipeline orchestrator | `scripts/run_evaluation_pipeline.py` |
| Retrieval ablation | `scripts/run_retrieval_ablation.py` |
| End-to-end evaluation | `scripts/evaluate_end_to_end.py` |
| Pairwise evaluation | `scripts/run_pairwise_evaluation.py` |
| ELO ranking computation | `scripts/compute_elo_rankings.py` |
| Result visualization | `scripts/visualize_comparison.py` |

### Configuration Files

| Purpose | Path |
|---------|------|
| 6-condition experiment plan | `configs/evaluation/compare_0p5b_experiments.yaml` |
| Quantization comparison | `configs/evaluation/quantization_comparison.yaml` |
| Retrieval ablation | `configs/evaluation/retrieval_ablation.yaml` |
| End-to-end evaluation | `configs/evaluation/end_to_end.yaml` |
| RAG-trained model (Q4_K_M) | `configs/lora_science_v1_rag_trained_ollama.yaml` |
| RAG-trained model (F16) | `configs/evaluation/lora_science_v1_rag_trained_f16_ollama.yaml` |
| Instruction-only model | `configs/lora_science_v1_instruction_only_ollama.yaml` |

### Data Files

| Purpose | Path |
|---------|------|
| Evaluation dataset | `evaluation/datasets/eval_dataset.jsonl` |
| Training dataset | `evaluation/datasets/train_dataset.jsonl` |
| Generated responses | `evaluation/responses/` |
| Evaluation results | `evaluation/results/` |

### Model Files

| Purpose | Path |
|---------|------|
| F16 GGUF (RAG-trained) | `outputs/lora_science_v1/merged_full_model/Qwen2.5-0.5B-lora_science_v1.gguf` |
| F16 GGUF (instruction-only) | `outputs/lora_science_v1_instruction_only/merged_full_model/Qwen2.5-0.5B-lora_science_v1_instruction_only.gguf` |
| Q4_K_M GGUF (RAG-trained) | `outputs/lora_science_v1/merged_full_model/Qwen2.5-0.5B-lora_science_v1.Q4_K_M.gguf` |
| Q4_K_M GGUF (instruction-only) | `outputs/lora_science_v1_instruction_only/merged_full_model/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf` |

### Ollama Modelfiles

| Purpose | Path |
|---------|------|
| RAG-trained (Q4_K_M) | `ollama/Modelfile.rag_trained` |
| RAG-trained (F16) | `ollama/Modelfile.rag_trained_f16` |
| Instruction-only | `ollama/Modelfile.instruction_only` |
