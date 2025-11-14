# Fine-Tuning Setup Summary

## Overview

For the 6-condition 0.5B experiment, **TWO fine-tuning notebooks** are required:

1. **`lora_science_v1_instruction_only.ipynb`** - Trains WITHOUT RAG contexts
   - Used for: FT-only (condition 3) and FT+RAG instruction-only (condition 4)

2. **`lora_science_v1.ipynb`** - Trains WITH RAG contexts
   - Used for: RAG-trained FT-only (condition 5) and RAG-trained FT+RAG (condition 6)

## Model Training Modes

### Instruction-Only Model (`lora_science_v1_instruction_only.ipynb`)

**Training Distribution:**
- Input: `instruction` only (no contexts)
- Output: Reference answers
- Matches: FT-only evaluation mode

**Use Cases:**
- **Condition 3 - FT Only**: Fine-tuned model without RAG contexts
- **Condition 4 - FT+RAG (instruction-only)**: Same fine-tuned model WITH RAG contexts

**Ollama Tag:** `lora_science_0p5_instruction_only`
**Config:** `configs/lora_science_v1_instruction_only_ollama.yaml`

### RAG-Trained Model (`lora_science_v1.ipynb`)

**Training Distribution:**
- Input: `instruction` + RAG contexts
- Output: Reference answers
- Matches: RAG-trained FT+RAG evaluation mode (optimal setup)

**Use Cases:**
- **Condition 5 - RAG-trained FT Only**: Model trained with contexts, evaluated without
- **Condition 6 - RAG-trained FT+RAG**: Model trained with contexts, evaluated with contexts (optimal)

**Ollama Tag:** `lora_science_0p5`
**Config:** `configs/lora_science_v1_rag_trained_ollama.yaml`

## Experiment Configuration

### 6-Condition Comparison Plan

The comparison plan (`configs/evaluation/compare_0p5b_experiments.yaml`) includes:

1. **Base Baseline** (Condition 1): Base model WITHOUT RAG contexts
   - Model: `rag_baseline_ollama.yaml`
   - Prompt mode: `instruction`

2. **RAG Baseline** (Condition 2): Base model + RAG contexts
   - Model: `rag_baseline_ollama.yaml`
   - Prompt mode: `rag`

3. **FT Only** (Condition 3): Instruction-only trained model WITHOUT RAG contexts
   - Model: `lora_science_v1_instruction_only_ollama.yaml`
   - Prompt mode: `instruction`
   - ✅ **Fair comparison**: Training matches evaluation (no contexts)

4. **FT+RAG (instruction-only)** (Condition 4): Instruction-only trained model WITH RAG contexts
   - Model: `lora_science_v1_instruction_only_ollama.yaml` (same as condition 3)
   - Prompt mode: `rag`
   - ✅ **Tests**: Transfer learning - can model trained without contexts benefit from RAG?

5. **RAG-trained FT Only** (Condition 5): RAG-trained model WITHOUT RAG contexts
   - Model: `lora_science_v1_rag_trained_ollama.yaml`
   - Prompt mode: `instruction`
   - ✅ **Tests**: Does training with contexts hurt performance when contexts aren't available?

6. **RAG-trained FT+RAG** (Condition 6): RAG-trained model WITH RAG contexts
   - Model: `lora_science_v1_rag_trained_ollama.yaml` (same as condition 5)
   - Prompt mode: `rag`
   - ✅ **Optimal setup**: Training matches evaluation (with contexts)

## Key Point

**TWO fine-tuned models are needed** for the complete 6-condition experiment:

| Condition | Model | Training Contexts | Eval Contexts | Prompt Mode |
|-----------|-------|-------------------|---------------|-------------|
| 1. Base Baseline | Base | N/A | ❌ No | `instruction` |
| 2. RAG Baseline | Base | N/A | ✅ Yes | `rag` |
| 3. FT Only | Instruction-only | ❌ No | ❌ No | `instruction` |
| 4. FT+RAG (instruction-only) | Instruction-only | ❌ No | ✅ Yes | `rag` |
| 5. RAG-trained FT Only | RAG-trained | ✅ Yes | ❌ No | `instruction` |
| 6. RAG-trained FT+RAG | RAG-trained | ✅ Yes | ✅ Yes | `rag` |

## Workflow

### 1. Train Models

**In Google Colab:**

1. Run `lora_science_v1_instruction_only.ipynb` → produces instruction-only model
   - Used for conditions 3-4

2. Run `lora_science_v1.ipynb` → produces RAG-trained model
   - Used for conditions 5-6

### 2. Export and Quantize

**Locally:**

```bash
# Convert instruction-only model to GGUF
python /path/to/llama.cpp/convert-hf-to-gguf.py \
  --model-dir outputs/lora_science_v1_instruction_only/merged_full_model \
  --outfile outputs/lora_science_v1_instruction_only/merged_full_model/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf \
  --data-type Q4_K_M

# Convert RAG-trained model to GGUF
python /path/to/llama.cpp/convert-hf-to-gguf.py \
  --model-dir outputs/lora_science_v1/merged_full_model \
  --outfile outputs/lora_science_v1/merged_full_model/Qwen2.5-0.5B-lora_science_v1.Q4_K_M.gguf \
  --data-type Q4_K_M

# Register both models with Ollama
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
ollama create lora_science_0p5 -f ollama/Modelfile.rag_trained
```

### 3. Run Comparison

```bash
python scripts/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/compare_0p5b_experiments.yaml \
  --limit 100  # Start with subset
```

This will run all 6 conditions and generate comparative results.

## Why This Design?

The 6-condition experiment creates a complete 2×2 matrix (training with/without contexts × evaluation with/without contexts) plus two baselines:

**Research Questions:**
- Does fine-tuning help without RAG? (Condition 3 vs 1)
- Does RAG help without fine-tuning? (Condition 2 vs 1)
- Does fine-tuning help even when RAG is available? (Condition 4 vs 2, Condition 6 vs 2)
- Does training with contexts improve RAG+FT performance? (Condition 6 vs 4)
- Does training with contexts hurt performance when contexts aren't available? (Condition 5 vs 3)

This ensures:
- ✅ Complete experimental design: Tests all combinations
- ✅ Scientifically rigorous: Isolates effects of training and evaluation modes
- ✅ Comparable to research literature: Matches Microsoft paper approach (condition 6)

## Files Created/Updated

- ✅ `notebooks/finetuning/lora_science_v1_instruction_only.ipynb` - Instruction-only training
- ✅ `notebooks/finetuning/lora_science_v1.ipynb` - RAG-trained training
- ✅ `configs/lora_science_v1_instruction_only_ollama.yaml` - For conditions 3-4
- ✅ `configs/lora_science_v1_rag_trained_ollama.yaml` - For conditions 5-6
- ✅ `configs/evaluation/compare_0p5b_experiments.yaml` - 6-condition comparison plan
- ✅ `ollama/Modelfile.instruction_only` - For instruction-only model
- ✅ `ollama/Modelfile.rag_trained` - For RAG-trained model

## Next Steps

1. Train TWO models in Colab (instruction-only and RAG-trained)
2. Export and quantize both models
3. Register both models with Ollama
4. Run 6-condition comparison evaluation
5. Analyze results across all conditions

## Complete Experimental Design

**Before**: Simplified 3-condition experiment
**After**: Complete 6-condition experiment with full 2×2 matrix + baselines

This provides comprehensive insights into the interaction between fine-tuning and RAG!
