# Final Experiment Setup - 0.5B Model Comparison

## Experimental Design

Compare three conditions using the **same evaluation dataset**:

1. **RAG Only**: Base model + RAG contexts
2. **FT Only**: Fine-tuned model (no RAG contexts)
3. **RAG+FT**: Same fine-tuned model + RAG contexts

## Key Point

**We only need ONE fine-tuned model** (instruction-only trained), used in two ways:
- Without RAG contexts → FT-only evaluation
- With RAG contexts → RAG+FT evaluation

## What This Tests

- **RAG vs FT**: Which approach works better?
- **RAG vs RAG+FT**: Does fine-tuning improve RAG performance?
- **FT vs RAG+FT**: Does adding RAG help a fine-tuned model?

## Training Setup

### Single Fine-Tuning Notebook

**`notebooks/finetuning/lora_science_v1_instruction_only.ipynb`**
- Trains model WITHOUT RAG contexts
- Input: Instruction only
- Output: Reference answers
- Same model used for both FT-only and RAG+FT evaluations

**Note**: `lora_science_v1.ipynb` (RAG-trained) is NOT needed for this experiment.

## Evaluation Setup

### Comparison Plan

**`configs/evaluation/compare_0p5b_experiments.yaml`**:

```yaml
runs:
  # 1. RAG Only: Base model + RAG contexts
  - label: rag_baseline_0p5b
    model_config: ../rag_baseline_ollama.yaml
    prompt_mode: rag

  # 2. FT Only: Fine-tuned model WITHOUT RAG contexts
  - label: lora_science_0p5b_ft_only
    model_config: ../lora_science_v1_instruction_only_ollama.yaml
    prompt_mode: instruction

  # 3. RAG+FT: Same fine-tuned model WITH RAG contexts
  - label: hybrid_science_0p5b
    model_config: ../lora_science_v1_instruction_only_ollama.yaml  # Same model!
    prompt_mode: rag  # But WITH contexts
```

## Model Configuration

### Config Files

**`configs/rag_baseline_ollama.yaml`**:
- Model: `qwen2.5:0.5b-instruct` (base model)
- Used for: RAG-only evaluation

**`configs/lora_science_v1_instruction_only_ollama.yaml`**:
- Model: `lora_science_0p5_instruction_only` (fine-tuned)
- Used for: FT-only evaluation

**`configs/hybrid_science_v1_ollama.yaml`**:
- Model: `lora_science_0p5_instruction_only` (same fine-tuned model!)
- Used for: RAG+FT evaluation

## Ollama Setup

### Single Modelfile Needed

**`ollama/Modelfile.instruction_only`**
- Points to: Instruction-only fine-tuned model GGUF
- Used for: Both FT-only and RAG+FT
- System prompt: Flexible (works with or without contexts)

**Register once:**
```bash
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
```

## Workflow

### 1. Fine-Tune (Google Colab)

Run **only** `lora_science_v1_instruction_only.ipynb`:
- Trains model without RAG contexts
- Exports adapter and merged checkpoint

### 2. Export & Quantize (Local)

```bash
# Convert to GGUF
python /path/to/llama.cpp/convert-hf-to-gguf.py \
  --model-dir outputs/lora_science_v1_instruction_only/merged_full_model \
  --outfile outputs/lora_science_v1_instruction_only/merged_full_model/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf \
  --data-type Q4_K_M

# Register with Ollama
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
```

### 3. Run Comparison (Local)

```bash
python scripts/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/compare_0p5b_experiments.yaml
```

This will run all three evaluations:
- RAG-only: Base model + RAG contexts
- FT-only: Fine-tuned model (no contexts)
- RAG+FT: Fine-tuned model + RAG contexts

## Expected Results

- **RAG+FT ≥ RAG**: Fine-tuning should help even with RAG
- **RAG+FT ≥ FT-only**: RAG should help even with fine-tuning
- **RAG vs FT-only**: Depends on dataset and model quality

## Files Summary

### Training
- ✅ `notebooks/finetuning/lora_science_v1_instruction_only.ipynb` - Only notebook needed

### Configs
- ✅ `configs/rag_baseline_ollama.yaml` - Base model
- ✅ `configs/lora_science_v1_instruction_only_ollama.yaml` - Fine-tuned model
- ✅ `configs/hybrid_science_v1_ollama.yaml` - Same fine-tuned model (for RAG+FT)

### Evaluation
- ✅ `configs/evaluation/compare_0p5b_experiments.yaml` - Comparison plan

### Ollama
- ✅ `ollama/Modelfile.instruction_only` - Single Modelfile for fine-tuned model

## Simplification

**Before**: Two fine-tuned models (RAG-trained + instruction-only)
**After**: One fine-tuned model (instruction-only), used two ways

This simplifies the experiment while maintaining scientific rigor!
