# Scaling the Pipeline to Larger Models

This guide explains how to adapt the Beyond the Cutoff pipeline for experiments with larger language models.

## Why Scale Up?

The completed experiment used **Qwen 2.5 0.5B** and found:
- RAG provides 4-5x improvement
- Fine-tuning provides marginal gains (+1.3%)
- No significant difference between Base+RAG and FT+RAG (p=0.35)

**Key question**: Do larger models show different patterns? With more capacity, fine-tuning might:
- Better retain domain knowledge
- Show stronger synergy with RAG
- Reduce dependence on retrieval

---

## Quick Start: Changing the Base Model

### Step 1: Choose Your Model

| Model | Ollama Tag | VRAM Needed | Notes |
|-------|------------|-------------|-------|
| Qwen 2.5 1.5B | `qwen2.5:1.5b-instruct` | ~4GB | Minimal upgrade |
| Qwen 2.5 3B | `qwen2.5:3b-instruct` | ~6GB | Good balance |
| Qwen 2.5 7B | `qwen2.5:7b-instruct-q4_K_M` | ~11GB | Significant capacity increase |
| Qwen 2.5 14B | `qwen2.5:14b-instruct-q4_K_M` | ~16GB | May show FT benefits |
| Llama 3.1 8B | `llama3.1:8b-instruct` | ~8GB | Alternative family |

### Step 2: Pull the Model

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### Step 3: Create New Configuration Files

Copy and modify the model configs:

```bash
# Copy base config
cp configs/models/base_ollama.yaml configs/models/base_ollama_7b.yaml
```

Edit `configs/models/base_ollama_7b.yaml`:

```yaml
inference:
  provider: ollama
  model: qwen2.5:7b-instruct-q4_K_M  # Changed from 0.5b
  base_url: http://localhost:11434
  temperature: 0
  max_tokens: 1024
  top_p: 0.9
  repeat_penalty: 1.05
```

### Step 4: Update Experiment Plan

Create a new experiment plan:

```bash
cp configs/evaluation/six_condition_experiment.yaml configs/evaluation/six_condition_7b.yaml
```

Update references to use your new model configs.

### Step 5: Fine-tune the New Model

Use the same notebooks but adjust for the new base model:

1. `notebooks/finetuning/lora_science_v1.ipynb`
   - Change `base_model` to `Qwen/Qwen2.5-7B-Instruct`
   - Adjust LoRA rank if needed (larger models may benefit from higher rank)
   - May need more VRAM (use Kaggle GPU or Colab Pro)

2. Export and convert to GGUF:
   ```bash
   python /path/to/llama.cpp/convert-hf-to-gguf.py \
     --model-dir outputs/lora_science_v1_7b/merged_full_model \
     --outfile outputs/lora_science_v1_7b/model.Q4_K_M.gguf \
     --data-type Q4_K_M
   ```

3. Register with Ollama:
   ```bash
   ollama create lora_science_7b -f ollama/Modelfile.rag_trained_7b
   ```

### Step 6: Run the Experiment

```bash
python scripts/core/compare_models.py --plan configs/evaluation/six_condition_7b.yaml
```

---

## Expected Changes with Larger Models

### Resource Requirements

| Model Size | Fine-tuning (Colab) | Inference (Local) | Dataset Gen |
|------------|---------------------|-------------------|-------------|
| 0.5B | Free Colab | 4GB RAM | ~2 hours |
| 3B | Colab Pro | 8GB RAM | ~4 hours |
| 7B | Kaggle GPU | 16GB RAM | ~8 hours |
| 14B+ | A100 / Cloud | 32GB+ RAM | ~16+ hours |

### Expected Results Patterns

Based on the literature and our 0.5B findings:

| Aspect | 0.5B (Observed) | 3-7B (Expected) | 14B+ (Expected) |
|--------|-----------------|-----------------|-----------------|
| **RAG benefit** | 4-5x | 3-4x | 2-3x |
| **FT benefit** | +1.3% | +5-10% | +10-20% |
| **FT+RAG synergy** | Minimal | Moderate | Strong |
| **FT can replace RAG?** | No | Partially | Possibly |

### Why Larger Models May Differ

1. **Capacity**: More parameters can store more knowledge during fine-tuning
2. **Generalization**: Better ability to transfer knowledge to new questions
3. **Context utilization**: Better at integrating retrieved information
4. **Instruction following**: More reliable structured outputs

---

## Changing the Judge Model

The judge model can also be upgraded for more reliable evaluation:

### Local Judges

```yaml
# configs/judges/qwen3_8b_thinking.yaml
inference:
  model: qwen3:8b  # Current default

# For stronger judging:
inference:
  model: qwen2.5:32b-instruct-q4_K_M  # Larger, more reliable
```

### API-Based Judges

For gold-standard evaluation, use API models:

```yaml
# configs/judges/gpt4_judge.yaml
inference:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  temperature: 0.1
```

```yaml
# configs/judges/claude_judge.yaml
inference:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${ANTHROPIC_API_KEY}
  temperature: 0.1
```

---

## Changing the Dataset Generator

The generator model (Qwen 2.5 7B) can be upgraded:

```yaml
# configs/default.yaml
dataset_generation:
  generator:
    model: qwen2.5:14b-instruct-q4_K_M  # Upgrade from 7B
```

Benefits of larger generators:
- Better question quality
- More diverse QA pairs
- Fewer parsing failures
- Better coverage of paper content

---

## Full Experiment Checklist

### Before Starting

- [ ] Sufficient VRAM/RAM for chosen model
- [ ] Enough Colab/Kaggle credits for fine-tuning
- [ ] Storage for checkpoints (~2-10GB per model)
- [ ] Time budget (larger models = longer training/inference)

### Configuration Files to Create/Modify

- [ ] `configs/models/base_ollama_<size>.yaml`
- [ ] `configs/models/lora_instruction_only_<size>.yaml`
- [ ] `configs/models/lora_rag_trained_<size>.yaml`
- [ ] `configs/evaluation/six_condition_<size>.yaml`
- [ ] `ollama/Modelfile.instruction_only_<size>`
- [ ] `ollama/Modelfile.rag_trained_<size>`

### Training Checklist

- [ ] Modify notebook for new base model
- [ ] Adjust LoRA hyperparameters if needed
- [ ] Train instruction-only variant
- [ ] Train RAG-trained variant
- [ ] Convert to GGUF
- [ ] Register with Ollama
- [ ] Verify models load correctly

### Evaluation Checklist

- [ ] Run 6-condition experiment
- [ ] (Optional) Run pairwise comparison on top 2
- [ ] (Optional) Run quantization comparison
- [ ] Compare results to 0.5B baseline

---

## Comparing Results Across Model Sizes

When you have results from multiple model sizes, create a comparison table:

```markdown
| Metric | 0.5B | 3B | 7B | 14B |
|--------|------|----|----|-----|
| Base pass rate | 4.2% | ? | ? | ? |
| RAG pass rate | 22.8% | ? | ? | ? |
| FT+RAG pass rate | 24.1% | ? | ? | ? |
| RAG benefit | 4.4x | ? | ? | ? |
| FT benefit | +1.3% | ? | ? | ? |
| FT+RAG synergy | ~0% | ? | ? | ? |
```

This will reveal:
- How RAG benefit scales with model size
- At what size fine-tuning becomes valuable
- Whether FT+RAG synergy emerges at larger scales

---

## Troubleshooting

### Out of Memory

- Use smaller batch sizes in training
- Use Q4_K_M quantization for inference
- Split evaluation into smaller batches
- Use `--limit` flag for testing

### Slow Training

- Reduce LoRA rank
- Use gradient checkpointing
- Consider fewer training epochs

### Inconsistent Results

- Increase evaluation examples
- Use multiple judge models
- Add pairwise comparisons for close results

---

## Related Documentation

- [Main README](../README.md) - Project overview and results
- [Pipeline Reference](../reference/pipeline.md) - Full pipeline documentation
- [Experiment Setup](../experiment/setup.md) - 6-condition design
- [Methodology](../experiment/methodology.md) - Evaluation approach
