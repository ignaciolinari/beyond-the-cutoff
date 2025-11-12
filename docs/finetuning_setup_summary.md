# Fine-Tuning Setup Summary

## Overview

For the 0.5B experiment, we only need **ONE fine-tuning notebook**:

1. **`lora_science_v1_instruction_only.ipynb`** - Trains WITHOUT RAG contexts
   - Used for BOTH FT-only AND RAG+FT evaluation

**Note**: `lora_science_v1.ipynb` (RAG-trained) exists but is NOT needed for this experiment. It's kept for reference or future experiments.

## Model Training Mode

### Instruction-Only Model (`lora_science_v1_instruction_only.ipynb`)

**Training Distribution:**
- Input: `instruction` only (no contexts)
- Output: Reference answers
- Matches: FT-only evaluation mode

**Use Cases:**
- **FT-only evaluation**: Fine-tuned model without RAG contexts
- **RAG+FT evaluation**: Same fine-tuned model WITH RAG contexts

**Ollama Tag:** `lora_science_0p5_instruction_only`
**Configs:**
- `configs/lora_science_v1_instruction_only_ollama.yaml` (FT-only)
- `configs/hybrid_science_v1_ollama.yaml` (RAG+FT - same model!)

## Experiment Configuration

### Comparison Plan

The comparison plan (`configs/evaluation/compare_0p5b_experiments.yaml`) uses:

1. **RAG Baseline**: Base model + RAG contexts
   - Model: `rag_baseline_ollama.yaml`
   - Prompt mode: `rag`

2. **FT Only**: Fine-tuned model + instruction-only prompts
   - Model: `lora_science_v1_instruction_only_ollama.yaml`
   - Prompt mode: `instruction`
   - ✅ **Fair comparison**: Training matches evaluation (no contexts)

3. **RAG+FT**: Same fine-tuned model + RAG contexts
   - Model: `hybrid_science_v1_ollama.yaml` (uses SAME instruction-only model!)
   - Prompt mode: `rag`
   - ✅ **Tests**: Does fine-tuning help even when RAG is available?

## Key Point

**Only ONE fine-tuned model is needed** (instruction-only), used in two ways:

| Evaluation Mode | Same Model | Prompt Mode | Contexts |
|----------------|------------|-------------|----------|
| FT-only | Instruction-only trained | `instruction` | None |
| RAG+FT | Instruction-only trained | `rag` | Provided |

## Workflow

### 1. Train Model

**In Google Colab:**

Run `lora_science_v1_instruction_only.ipynb` → produces instruction-only model
- This ONE model is used for both FT-only and RAG+FT evaluation

### 2. Export and Quantize

**Locally:**

```bash
# Convert instruction-only model to GGUF
python /path/to/llama.cpp/convert-hf-to-gguf.py \
  --model-dir outputs/lora_science_v1_instruction_only/merged_full_model \
  --outfile outputs/lora_science_v1_instruction_only/merged_full_model/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf \
  --data-type Q4_K_M

# Register with Ollama (used for BOTH FT-only and RAG+FT)
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
```

### 3. Run Comparison

```bash
python scripts/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/compare_0p5b_experiments.yaml \
  --limit 100  # Start with subset
```

## Why This Design?

**Question**: Does fine-tuning help even when RAG is available?

**Answer**: Test the same fine-tuned model:
- Without RAG → FT-only (tests fine-tuning alone)
- With RAG → RAG+FT (tests fine-tuning + RAG combination)

This ensures:
- ✅ Fair comparison: Same model, different evaluation modes
- ✅ Scientifically rigorous: Tests the interaction between FT and RAG
- ✅ Simpler setup: Only one model to train and manage

## Files Created/Updated

- ✅ `notebooks/finetuning/lora_science_v1_instruction_only.ipynb` - Only notebook needed
- ✅ `configs/lora_science_v1_instruction_only_ollama.yaml` - For FT-only
- ✅ Updated `configs/hybrid_science_v1_ollama.yaml` - Points to same model (for RAG+FT)
- ✅ Updated `configs/evaluation/compare_0p5b_experiments.yaml` - Uses same model for FT-only and RAG+FT
- ✅ `ollama/Modelfile.instruction_only` - Single Modelfile for both evaluation modes

## Next Steps

1. Train ONE model in Colab (`lora_science_v1_instruction_only.ipynb`)
2. Export and quantize the model
3. Register with Ollama (one model, used for both evaluation modes)
4. Run comparison evaluation
5. Analyze results

## Simplification

**Before (incorrect understanding)**: Two fine-tuned models needed
**After (correct)**: One fine-tuned model, used two ways

This simplifies the experiment while maintaining scientific rigor!
