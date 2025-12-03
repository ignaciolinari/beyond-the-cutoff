# Six-Condition Experiment Setup

## Overview

This experiment compares **six conditions** to understand the interaction between fine-tuning and RAG. This creates a complete 2x2 matrix (training with/without contexts × evaluation with/without contexts) plus two baselines:

1. **Base Baseline** - Base model WITHOUT RAG contexts (control condition)
2. **RAG Baseline** - Base model + RAG contexts
3. **FT Only (instruction-only)** - Instruction-only trained model WITHOUT RAG contexts
4. **FT+RAG (instruction-only)** - Instruction-only trained model WITH RAG contexts
5. **RAG-trained FT Only** - RAG-trained model WITHOUT RAG contexts
6. **RAG-trained FT+RAG** - RAG-trained model WITH RAG contexts

## Experimental Conditions

### Condition 1: Base Baseline
- **Model**: Base `qwen2.5:0.5b-instruct`
- **Training**: None (base model)
- **Evaluation**: WITHOUT RAG contexts
- **Config**: `base_ollama.yaml` (same model, instruction mode)
- **Label**: `base_baseline_0p5b`
- **What it tests**: Baseline performance without fine-tuning or RAG (control condition)
- **Why it's important**: Establishes the starting point to measure improvements from fine-tuning and/or RAG

### Condition 2: RAG Baseline
- **Model**: Base `qwen2.5:0.5b-instruct`
- **Training**: None (base model)
- **Evaluation**: WITH RAG contexts
- **Config**: `base_ollama.yaml`
- **Label**: `rag_baseline_0p5b`
- **What it tests**: Baseline RAG performance (how much does RAG help the base model?)

### Condition 3: FT Only
- **Model**: `lora_science_0p5_instruction_only` (instruction-only trained)
- **Training**: WITHOUT RAG contexts
- **Evaluation**: WITHOUT RAG contexts
- **Config**: `lora_instruction_only.yaml`
- **Label**: `lora_science_0p5b_ft_only`
- **What it tests**: Can fine-tuning alone memorize domain knowledge?

### Condition 4: FT+RAG (instruction-only)
- **Model**: `lora_science_0p5_instruction_only` (same as condition 3)
- **Training**: WITHOUT RAG contexts
- **Evaluation**: WITH RAG contexts
- **Config**: `lora_instruction_only.yaml` (same model, different prompt mode)
- **Label**: `hybrid_science_0p5b_instruction_only`
- **What it tests**: Can a model trained without contexts benefit from RAG at inference? (Transfer learning)
- **Note**: This may underperform because the model wasn't trained to use contexts
- **Prompt Format**: Uses the **standard RAG prompt format** (same as Conditions 2 and 6—the other RAG evaluation conditions) to ensure fair comparison. The model experiences distribution shift because it was trained on simple `"Question: ... Answer:"` format but now receives the full RAG prompt with contexts. This is the intended test—we isolate the training mode variable (with/without contexts) while keeping prompt format constant across all RAG evaluation conditions (2, 4, 6). Note: Condition 5 uses instruction mode (no RAG contexts). See `src/beyond_the_cutoff/evaluation/runner.py::_build_rag_prompt_for_instruction_only_model()`.

### Condition 5: RAG-trained FT Only
- **Model**: `lora_science_0p5` (RAG-trained)
- **Training**: WITH RAG contexts
- **Evaluation**: WITHOUT RAG contexts
- **Config**: `lora_rag_trained.yaml`
- **Label**: `lora_science_0p5b_rag_trained_ft_only`
- **What it tests**: Does training with contexts hurt performance when contexts aren't available?
- **Note**: Tests the inverse of condition #3 - training/evaluation mismatch in opposite direction

### Condition 6: RAG-trained FT+RAG
- **Model**: `lora_science_0p5` (RAG-trained)
- **Training**: WITH RAG contexts
- **Evaluation**: WITH RAG contexts
- **Config**: `lora_rag_trained.yaml`
- **Label**: `hybrid_science_0p5b_rag_trained`
- **What it tests**: Optimal RAG+FT performance (matches Microsoft paper approach)
- **Note**: This should perform best for RAG+FT because training matches evaluation

## Comparison Matrix

| Condition | Model | Training Contexts | Eval Contexts | Config File | Label |
|-----------|-------|-------------------|---------------|-------------|-------|
| Base Baseline | Base | N/A | ❌ No | `base_ollama.yaml` | `base_baseline_0p5b` |
| RAG Baseline | Base | N/A | ✅ Yes | `base_ollama.yaml` | `rag_baseline_0p5b` |
| FT Only (instruction-only) | Instruction-only | ❌ No | ❌ No | `lora_instruction_only.yaml` | `lora_science_0p5b_ft_only` |
| FT+RAG (instruction-only) | Instruction-only | ❌ No | ✅ Yes | `lora_instruction_only.yaml` | `hybrid_science_0p5b_instruction_only` |
| RAG-trained FT Only | RAG-trained | ✅ Yes | ❌ No | `lora_rag_trained.yaml` | `lora_science_0p5b_rag_trained_ft_only` |
| RAG-trained FT+RAG | RAG-trained | ✅ Yes | ✅ Yes | `lora_rag_trained.yaml` | `hybrid_science_0p5b_rag_trained` |

## Complete 2x2 Matrix + Baselines

This creates a complete experimental design:

| | **Eval WITHOUT Contexts** | **Eval WITH Contexts** |
|---|---|---|
| **No Training (Base Model)** | Condition 1: Base Baseline | Condition 2: RAG Baseline |
| **Train WITHOUT Contexts** | Condition 3: FT Only (instruction-only) | Condition 4: FT+RAG (instruction-only) |
| **Train WITH Contexts** | Condition 5: RAG-trained FT Only | Condition 6: RAG-trained FT+RAG |

## Expected Results

Based on training/evaluation alignment:

1. **Base Baseline**: Lowest performance - no fine-tuning, no RAG (control condition)
2. **RAG Baseline**: Should outperform base baseline - RAG helps base model
3. **FT Only (instruction-only)**: Should outperform base baseline if fine-tuning memorized knowledge
4. **FT+RAG (instruction-only)**: May underperform - model doesn't know how to use contexts
5. **RAG-trained FT Only**: May underperform - model trained to expect contexts, evaluated without them
6. **RAG-trained FT+RAG**: Should perform best - model trained to use contexts, evaluated with contexts

## Key Comparisons

### Baseline Comparisons
- **Base Baseline** vs **RAG Baseline**: How much does RAG help the base model?
- **Base Baseline** vs **FT Only**: How much does fine-tuning help without RAG?

### RAG vs FT
- **RAG Baseline** vs **FT Only**: Which approach works better without the other?

### Transfer Learning Test
- **FT Only** vs **FT+RAG (instruction-only)**: Can fine-tuning help even when contexts are added at inference?

### Optimal RAG+FT
- **RAG Baseline** vs **RAG-trained FT+RAG**: Does fine-tuning improve RAG performance when model is trained to use contexts?

### Training Alignment
- **FT+RAG (instruction-only)** vs **RAG-trained FT+RAG**: Does training with contexts improve RAG+FT performance?

### Training/Eval Mismatch (Both Directions)
- **FT Only (instruction-only)** vs **FT+RAG (instruction-only)**: Adding contexts to model trained without them
- **RAG-trained FT Only** vs **RAG-trained FT+RAG**: Adding contexts to model trained with them (should help more)
- **FT Only (instruction-only)** vs **RAG-trained FT Only**: Does training with contexts hurt performance without contexts?

## Design Choices

### Question-Level Holdout (Not Document-Level)

This experiment uses **question-level holdout**, meaning:
- The same papers appear in both training and evaluation splits
- Different *questions* about those papers are held out for evaluation
- Models see paper content (via training questions) but not the exact eval questions

**Rationale:**
1. **Tests knowledge acquisition, not memorization**: We want to know if models learned the underlying concepts vs. just memorized Q&A pairs
2. **Realistic deployment scenario**: In practice, users ask new questions about known information—this mirrors real-world usage where a RAG system or fine-tuned model must generalize to novel queries
3. **Fairer RAG comparison**: RAG models retrieve from the same papers during evaluation; if we used document-level holdout, RAG would be tested on completely unseen documents while fine-tuned models would be tested on their trained knowledge—an apples-to-oranges comparison
4. **Valid cross-condition comparison**: All 6 conditions use the SAME eval questions, ensuring differences reflect model capabilities rather than dataset artifacts

**Trade-off acknowledged:**
- This design does NOT test "Can the model handle completely new documents?"—that would require document-level holdout
- For document-level generalization testing, a separate evaluation with unseen papers would be needed

**Alternative design (not used):**
- Document-level holdout would test a different question: "Can the model handle entirely unseen content?"
- This tests retrieval and knowledge transfer to new domains rather than question generalization on known content

The question-level holdout is implemented in `scripts/data/split_dataset.py` via stratified sampling that maintains paper representation across splits.

## Setup Requirements

### Models Needed

1. **Base model**: `qwen2.5:0.5b-instruct` (already available via Ollama)
2. **Instruction-only model**: `lora_science_0p5_instruction_only`
   - Train with: `notebooks/finetuning/lora_science_v1_instruction_only.ipynb`
   - Register with: `ollama/Modelfile.instruction_only`
3. **RAG-trained model**: `lora_science_0p5`
   - Train with: `notebooks/finetuning/lora_science_v1.ipynb`
   - Register with: `ollama/Modelfile.rag_trained`

### Configuration Files

All configs are in `configs/`:
- `base_ollama.yaml`
- `lora_instruction_only.yaml`
- `lora_rag_trained.yaml`
- `hybrid_science_v1_ollama.yaml` (legacy, moved to `vintage/configs/` - not used in current experiment)

### Comparison Plan

The comparison plan is in `configs/evaluation/six_condition_experiment.yaml` and includes all 6 conditions.

## Running the Experiment

```bash
python scripts/core/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/six_condition_experiment.yaml
```

This will run all 6 conditions and generate comparative results.

## Related Documentation

- **[Experiment Analysis Guide](analysis_guide.md)**: Detailed guide for interpreting results, understanding which metrics are comparable across conditions, and expected findings
- **[Pipeline Plan](../reference/pipeline.md)**: Step-by-step pipeline for running the experiment, including dataset splitting
