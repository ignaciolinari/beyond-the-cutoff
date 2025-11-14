# Final Experiment Setup - 6-Condition 0.5B Model Comparison

## Experimental Design

Compare **six conditions** using the **same evaluation dataset**:

1. **Base Baseline**: Base model WITHOUT RAG contexts
2. **RAG Baseline**: Base model + RAG contexts
3. **FT Only**: Instruction-only trained model WITHOUT RAG contexts
4. **FT+RAG (instruction-only)**: Instruction-only trained model WITH RAG contexts
5. **RAG-trained FT Only**: RAG-trained model WITHOUT RAG contexts
6. **RAG-trained FT+RAG**: RAG-trained model WITH RAG contexts

## Key Point

**We need TWO fine-tuned models** for the complete experiment:
- **Instruction-only model**: Trained without RAG contexts (used for conditions 3-4)
- **RAG-trained model**: Trained with RAG contexts (used for conditions 5-6)

## What This Tests

- **Baseline comparisons**: How much do RAG and fine-tuning help individually?
- **RAG vs FT**: Which approach works better without the other?
- **Transfer learning**: Can models trained without contexts benefit from RAG? (Condition 4)
- **Training alignment**: Does training with contexts improve RAG+FT performance? (Condition 6 vs 4)
- **Training/eval mismatch**: Does training with contexts hurt when contexts aren't available? (Condition 5)

## Training Setup

### Two Fine-Tuning Notebooks Required

1. **`notebooks/finetuning/lora_science_v1_instruction_only.ipynb`**
   - Trains model WITHOUT RAG contexts
   - Input: Instruction only
   - Output: Reference answers
   - Used for conditions 3-4

2. **`notebooks/finetuning/lora_science_v1.ipynb`**
   - Trains model WITH RAG contexts
   - Input: Instruction + RAG contexts
   - Output: Reference answers
   - Used for conditions 5-6

## Evaluation Setup

### 6-Condition Comparison Plan

**`configs/evaluation/compare_0p5b_experiments.yaml`** includes all 6 conditions:

```yaml
runs:
  # 1. Base Baseline: Base model WITHOUT RAG contexts
  - label: base_baseline_0p5b
    model_config: ../rag_baseline_ollama.yaml
    prompt_mode: instruction
    judge_config: ../judges/scientific_default_instruction.yaml

  # 2. RAG Baseline: Base model + RAG contexts
  - label: rag_baseline_0p5b
    model_config: ../rag_baseline_ollama.yaml
    prompt_mode: rag
    judge_config: ../judges/scientific_default_rag.yaml

  # 3. FT Only: Instruction-only trained model WITHOUT RAG contexts
  - label: lora_science_0p5b_ft_only
    model_config: ../lora_science_v1_instruction_only_ollama.yaml
    prompt_mode: instruction
    judge_config: ../judges/scientific_default_instruction.yaml

  # 4. FT+RAG (instruction-only): Instruction-only trained model WITH RAG contexts
  - label: hybrid_science_0p5b_instruction_only
    model_config: ../lora_science_v1_instruction_only_ollama.yaml
    prompt_mode: rag
    judge_config: ../judges/scientific_default_rag.yaml

  # 5. RAG-trained FT Only: RAG-trained model WITHOUT RAG contexts
  - label: lora_science_0p5b_rag_trained_ft_only
    model_config: ../lora_science_v1_rag_trained_ollama.yaml
    prompt_mode: instruction
    judge_config: ../judges/scientific_default_instruction.yaml

  # 6. RAG-trained FT+RAG: RAG-trained model WITH RAG contexts (optimal)
  - label: hybrid_science_0p5b_rag_trained
    model_config: ../lora_science_v1_rag_trained_ollama.yaml
    prompt_mode: rag
    judge_config: ../judges/scientific_default_rag.yaml
```

## Model Configuration

### Config Files

**`configs/rag_baseline_ollama.yaml`**:
- Model: `qwen2.5:0.5b-instruct` (base model)
- Used for: Base baseline (condition 1) and RAG baseline (condition 2)

**`configs/lora_science_v1_instruction_only_ollama.yaml`**:
- Model: `lora_science_0p5_instruction_only` (instruction-only fine-tuned)
- Used for: FT-only (condition 3) and FT+RAG instruction-only (condition 4)

**`configs/lora_science_v1_rag_trained_ollama.yaml`**:
- Model: `lora_science_0p5` (RAG-trained fine-tuned)
- Used for: RAG-trained FT-only (condition 5) and RAG-trained FT+RAG (condition 6)

## Ollama Setup

### Two Modelfiles Needed

**`ollama/Modelfile.instruction_only`**
- Points to: Instruction-only fine-tuned model GGUF
- Used for: Conditions 3-4
- System prompt: Flexible (works with or without contexts)

**`ollama/Modelfile.rag_trained`**
- Points to: RAG-trained fine-tuned model GGUF
- Used for: Conditions 5-6
- System prompt: Mentions citations (optimized for RAG use)

**Register both:**
```bash
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
ollama create lora_science_0p5 -f ollama/Modelfile.rag_trained
```

## Workflow

### 1. Fine-Tune (Google Colab)

Run **both** notebooks:

1. `lora_science_v1_instruction_only.ipynb`:
   - Trains model without RAG contexts
   - Exports adapter and merged checkpoint
   - Used for conditions 3-4

2. `lora_science_v1.ipynb`:
   - Trains model with RAG contexts
   - Exports adapter and merged checkpoint
   - Used for conditions 5-6

### 2. Export & Quantize (Local)

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

### 3. Run Comparison (Local)

```bash
python scripts/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/compare_0p5b_experiments.yaml
```

This will run all **six conditions**:
1. Base baseline: Base model without RAG contexts
2. RAG baseline: Base model + RAG contexts
3. FT-only: Instruction-only trained model without RAG contexts
4. FT+RAG (instruction-only): Instruction-only trained model + RAG contexts
5. RAG-trained FT-only: RAG-trained model without RAG contexts
6. RAG-trained FT+RAG: RAG-trained model + RAG contexts (optimal)

## Expected Results

Based on training/evaluation alignment:

1. **Base Baseline**: Lowest performance - no fine-tuning, no RAG (control)
2. **RAG Baseline**: Should outperform base baseline - RAG helps base model
3. **FT Only**: Should outperform base baseline if fine-tuning memorized knowledge
4. **FT+RAG (instruction-only)**: May underperform - model doesn't know how to use contexts
5. **RAG-trained FT Only**: May underperform - model trained to expect contexts, evaluated without
6. **RAG-trained FT+RAG**: Should perform best - model trained to use contexts, evaluated with contexts

## Files Summary

### Training
- ✅ `notebooks/finetuning/lora_science_v1_instruction_only.ipynb` - Instruction-only training
- ✅ `notebooks/finetuning/lora_science_v1.ipynb` - RAG-trained training

### Configs
- ✅ `configs/rag_baseline_ollama.yaml` - Base model (conditions 1-2)
- ✅ `configs/lora_science_v1_instruction_only_ollama.yaml` - Instruction-only model (conditions 3-4)
- ✅ `configs/lora_science_v1_rag_trained_ollama.yaml` - RAG-trained model (conditions 5-6)

### Evaluation
- ✅ `configs/evaluation/compare_0p5b_experiments.yaml` - 6-condition comparison plan

### Ollama
- ✅ `ollama/Modelfile.instruction_only` - For instruction-only model
- ✅ `ollama/Modelfile.rag_trained` - For RAG-trained model

## Complete Experimental Design

**Before**: Simplified 3-condition experiment
**After**: Complete 6-condition experiment with full 2×2 matrix + baselines

This provides comprehensive insights into the interaction between fine-tuning and RAG, creating a complete experimental design comparable to research literature!
