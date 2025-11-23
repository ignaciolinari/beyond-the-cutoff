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
- **Config**: `rag_baseline_ollama.yaml` (same model, instruction mode)
- **Label**: `base_baseline_0p5b`
- **What it tests**: Baseline performance without fine-tuning or RAG (control condition)
- **Why it's important**: Establishes the starting point to measure improvements from fine-tuning and/or RAG

### Condition 2: RAG Baseline
- **Model**: Base `qwen2.5:0.5b-instruct`
- **Training**: None (base model)
- **Evaluation**: WITH RAG contexts
- **Config**: `rag_baseline_ollama.yaml`
- **Label**: `rag_baseline_0p5b`
- **What it tests**: Baseline RAG performance (how much does RAG help the base model?)

### Condition 3: FT Only
- **Model**: `lora_science_0p5_instruction_only` (instruction-only trained)
- **Training**: WITHOUT RAG contexts
- **Evaluation**: WITHOUT RAG contexts
- **Config**: `lora_science_v1_instruction_only_ollama.yaml`
- **Label**: `lora_science_0p5b_ft_only`
- **What it tests**: Can fine-tuning alone memorize domain knowledge?

### Condition 4: FT+RAG (instruction-only)
- **Model**: `lora_science_0p5_instruction_only` (same as condition 3)
- **Training**: WITHOUT RAG contexts
- **Evaluation**: WITH RAG contexts
- **Config**: `lora_science_v1_instruction_only_ollama.yaml` (same model, different prompt mode)
- **Label**: `hybrid_science_0p5b_instruction_only`
- **What it tests**: Can a model trained without contexts benefit from RAG at inference? (Transfer learning)
- **Note**: This may underperform because the model wasn't trained to use contexts
- **Prompt Format**: Uses a hybrid prompt format that preserves the training structure (`"Question: ... Answer:"`) while adding RAG contexts between them. This intentional distribution shift tests whether the model can adapt to contexts at inference time despite being trained without them. The evaluation runner automatically detects instruction-only models and applies this hybrid format (see `src/beyond_the_cutoff/evaluation/runner.py::_build_rag_prompt_for_instruction_only_model()`).

### Condition 5: RAG-trained FT Only
- **Model**: `lora_science_0p5` (RAG-trained)
- **Training**: WITH RAG contexts
- **Evaluation**: WITHOUT RAG contexts
- **Config**: `lora_science_v1_rag_trained_ollama.yaml`
- **Label**: `lora_science_0p5b_rag_trained_ft_only`
- **What it tests**: Does training with contexts hurt performance when contexts aren't available?
- **Note**: Tests the inverse of condition #3 - training/evaluation mismatch in opposite direction

### Condition 6: RAG-trained FT+RAG
- **Model**: `lora_science_0p5` (RAG-trained)
- **Training**: WITH RAG contexts
- **Evaluation**: WITH RAG contexts
- **Config**: `lora_science_v1_rag_trained_ollama.yaml`
- **Label**: `hybrid_science_0p5b_rag_trained`
- **What it tests**: Optimal RAG+FT performance (matches Microsoft paper approach)
- **Note**: This should perform best for RAG+FT because training matches evaluation

## Comparison Matrix

| Condition | Model | Training Contexts | Eval Contexts | Config File | Label |
|-----------|-------|-------------------|---------------|-------------|-------|
| Base Baseline | Base | N/A | ❌ No | `rag_baseline_ollama.yaml` | `base_baseline_0p5b` |
| RAG Baseline | Base | N/A | ✅ Yes | `rag_baseline_ollama.yaml` | `rag_baseline_0p5b` |
| FT Only (instruction-only) | Instruction-only | ❌ No | ❌ No | `lora_science_v1_instruction_only_ollama.yaml` | `lora_science_0p5b_ft_only` |
| FT+RAG (instruction-only) | Instruction-only | ❌ No | ✅ Yes | `lora_science_v1_instruction_only_ollama.yaml` | `hybrid_science_0p5b_instruction_only` |
| RAG-trained FT Only | RAG-trained | ✅ Yes | ❌ No | `lora_science_v1_rag_trained_ollama.yaml` | `lora_science_0p5b_rag_trained_ft_only` |
| RAG-trained FT+RAG | RAG-trained | ✅ Yes | ✅ Yes | `lora_science_v1_rag_trained_ollama.yaml` | `hybrid_science_0p5b_rag_trained` |

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
- `rag_baseline_ollama.yaml`
- `lora_science_v1_instruction_only_ollama.yaml`
- `lora_science_v1_rag_trained_ollama.yaml`
- `hybrid_science_v1_ollama.yaml` (legacy, moved to `vintage/configs/` - not used in current experiment)

### Comparison Plan

The comparison plan is in `configs/evaluation/compare_0p5b_experiments.yaml` and includes all 6 conditions.

## Running the Experiment

```bash
python scripts/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/compare_0p5b_experiments.yaml
```

This will run all 6 conditions and generate comparative results.

## Relationship to Microsoft Paper

The Microsoft paper likely used:
- **RAG Baseline**: Base model + RAG
- **FT Only**: Fine-tuned model (trained with contexts, evaluated without) - *matches our condition #5 (RAG-trained FT Only)*
- **RAG-trained FT+RAG**: Fine-tuned model (trained with contexts, evaluated with contexts) - **matches our condition #6**

**Key Difference**: Our experiment includes additional conditions that the Microsoft paper did not:
- **Condition 3 (FT Only)**: Trained WITHOUT contexts, evaluated WITHOUT contexts - Tests if fine-tuning can memorize knowledge without RAG
- **Condition 4 (FT+RAG instruction-only)**: Trained WITHOUT contexts, evaluated WITH contexts - Tests transfer learning (can a model trained without contexts benefit from RAG?)

This creates a complete 2×2 matrix that allows us to isolate the effects of training mode vs evaluation mode, providing more comprehensive insights than the Microsoft paper's setup.
