# Experiment Readiness Checklist

_Last updated: 2025-11-13_

Use this checklist before running the 0.5B model comparison experiments to ensure everything is properly configured.

## Pre-Experiment Setup

### 1. Data & Indexing
- [ ] **Corpus ingested**: Run `python scripts/ingest_and_index.py --config configs/default.yaml`
- [ ] **Index exists**: Verify `data/external/index/` contains FAISS index files
- [ ] **Offline dataset generated**: Run `python scripts/generate_offline_dataset.py --config configs/default.yaml`
- [ ] **Dataset validated**: Check that `evaluation/datasets/offline_dataset.jsonl` exists and has expected structure
- [ ] **Dataset size**: Verify dataset has sufficient examples (recommended: >100 tasks)

### 2. Model Availability

#### Base Model (RAG-only)
- [ ] **Ollama base model**: `ollama pull qwen2.5:0.5b-instruct` (or verify it's already available)
- [ ] **Model accessible**: Test with `ollama list | grep qwen2.5:0.5b-instruct`

#### Fine-Tuned Models (6-condition experiment requires 2 models)

**Instruction-Only Model** (for conditions 3-4):
- [ ] **Model trained**: Fine-tuning completed in Colab using `notebooks/finetuning/lora_science_v1_instruction_only.ipynb`
- [ ] **Checkpoint synced**: Merged model weights synced from Colab to local `outputs/lora_science_v1_instruction_only/merged_full_model/`
- [ ] **GGUF converted**: Model converted to GGUF format (e.g., `Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf`)
- [ ] **Ollama model registered**: `ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only`
- [ ] **Model accessible**: Test with `ollama list | grep lora_science_0p5_instruction_only`

**RAG-Trained Model** (for conditions 5-6):
- [ ] **Model trained**: Fine-tuning completed using `notebooks/finetuning/lora_science_v1.ipynb`
- [ ] **Checkpoint synced**: Merged model weights synced to local (e.g., `outputs/lora_science_v1/merged_full_model/`)
- [ ] **GGUF converted**: Model converted to GGUF format
- [ ] **Ollama model registered**: `ollama create lora_science_0p5 -f ollama/Modelfile.rag_trained`
- [ ] **Model accessible**: Test with `ollama list | grep lora_science_0p5`

#### Judge Model
- [ ] **Judge model available**: `ollama pull qwen2.5:7b-instruct-q4_K_M` (or verify it's already available)
- [ ] **Judge accessible**: Test with `ollama list | grep qwen2.5:7b-instruct`

### 3. Configuration Files

#### Model Configs
- [ ] **`configs/rag_baseline_ollama.yaml`**: Points to `qwen2.5:0.5b-instruct` (used for base baseline and RAG baseline)
- [ ] **`configs/lora_science_v1_instruction_only_ollama.yaml`**: Points to `lora_science_0p5_instruction_only`
- [ ] **`configs/lora_science_v1_rag_trained_ollama.yaml`**: Points to `lora_science_0p5` (RAG-trained model)
- [x] **`configs/hybrid_science_v1_ollama.yaml`**: Legacy config (moved to `vintage/configs/` - not used in current experiment)

#### Comparison Plan
- [ ] **`configs/evaluation/compare_0p5b_experiments.yaml`**:
  - Contains all 6 conditions (base baseline, RAG baseline, FT-only, FT+RAG instruction-only, RAG-trained FT-only, RAG-trained FT+RAG)
  - Uses correct model configs for each condition
  - Has correct `prompt_mode` settings (`rag` for conditions with RAG, `instruction` for conditions without RAG)
  - Points to correct dataset path

#### Judge Config
- [ ] **`configs/judges/scientific_default_rag.yaml`**: Exists and has proper prompt template (for RAG conditions)
- [ ] **`configs/judges/scientific_default_instruction.yaml`**: Exists and has proper prompt template (for instruction-only conditions)
- [ ] **`configs/judges/ollama_qwen7b.yaml`**: Points to judge model

### 4. Environment & Dependencies
- [ ] **Python environment**: Virtual environment activated (`.venv/bin/activate`)
- [ ] **Dependencies installed**: `pip install -e '.[dev]'` completed successfully
- [ ] **Ollama running**: `ollama serve` is running (or Ollama daemon is active)
- [ ] **Ollama accessible**: Test with `curl http://localhost:11434/api/tags`

### 5. Paths & Permissions
- [ ] **Output directory writable**: `evaluation/results/` exists and is writable
- [ ] **Data directories**: `data/processed/`, `data/external/` exist
- [ ] **Sufficient disk space**: Check available space for results and logs

## Running Experiments

### Single Model Evaluation (Optional Pre-Check)
Test a single model before running the full comparison:

```bash
python scripts/evaluate_models.py \
  --config configs/default.yaml \
  --dataset evaluation/datasets/offline_dataset.jsonl \
  --model-config configs/rag_baseline_ollama.yaml \
  --model-label rag_baseline_0p5b_test \
  --judge-config configs/judges/scientific_default_rag.yaml \
  --judge-inference configs/judges/ollama_qwen7b.yaml \
  --output evaluation/results/rag_baseline_0p5b_test/metrics.json \
  --prompt-mode rag \
  --limit 5
```

### Full Comparison Run
Once pre-checks pass, run the full comparison:

```bash
python scripts/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/compare_0p5b_experiments.yaml \
  --output evaluation/results/comparison_report.json
```

### Expected Outputs
After running the comparison, verify:
- [ ] **Metrics files**: `evaluation/results/{label}/metrics.json` for each run
- [ ] **Details files**: `evaluation/results/{label}/details.jsonl` for each run
- [ ] **Metadata files**: `evaluation/results/{label}/metadata.jsonl` for each run
- [ ] **Comparison report**: `evaluation/results/comparison_report.json` (if `--output` specified)

## Post-Experiment Verification

### Data Consistency
- [ ] **Same examples evaluated**: All six runs evaluated the same task IDs (check `evaluated_task_ids` in metrics)
- [ ] **No missing data**: All expected examples have predictions
- [ ] **Error rate acceptable**: Check `examples_with_errors` in summary metrics

### Experimental Correctness
- [ ] **Base baseline**: Used base model (`qwen2.5:0.5b-instruct`) with `prompt_mode: instruction`
- [ ] **RAG baseline**: Used base model (`qwen2.5:0.5b-instruct`) with `prompt_mode: rag`
- [ ] **FT-only**: Used instruction-only model (`lora_science_0p5_instruction_only`) with `prompt_mode: instruction`
- [ ] **FT+RAG (instruction-only)**: Used instruction-only model (`lora_science_0p5_instruction_only`) with `prompt_mode: rag`
- [ ] **RAG-trained FT-only**: Used RAG-trained model (`lora_science_0p5`) with `prompt_mode: instruction`
- [ ] **RAG-trained FT+RAG**: Used RAG-trained model (`lora_science_0p5`) with `prompt_mode: rag`
- [ ] **Same dataset**: All six runs used the same `offline_dataset.jsonl`
- [ ] **Same judge**: All six runs used the same judge model and prompt

### Results Analysis
- [ ] **Metrics computed**: Factuality, citation accuracy, BLEU, BERTScore present in summaries for all 6 conditions
- [ ] **Citation metrics**: Conditions without RAG (base baseline, FT-only, RAG-trained FT-only) have citation metrics marked as N/A (expected)
- [ ] **Comparison valid**: Results show clear differences between the six conditions
- [ ] **Baseline comparisons**: Base baseline vs RAG baseline shows RAG benefit; Base baseline vs FT-only shows fine-tuning benefit

## Troubleshooting

### Common Issues

**Model not found**:
- Verify Ollama model is registered: `ollama list`
- Check config file points to correct model tag
- Ensure Modelfile path is correct in `ollama/Modelfile.instruction_only`

**Dataset not found**:
- Verify `evaluation/datasets/offline_dataset.jsonl` exists
- Check path in comparison plan is relative to plan file location
- Run dataset generation script if missing

**Judge errors**:
- Verify judge model is available: `ollama list | grep qwen2.5:7b`
- Check judge config file exists and is valid YAML
- Ensure judge inference config points to correct model

**Prompt mode errors**:
- Verify `prompt_mode` is either `rag` or `instruction` (case-sensitive)
- Check that RAG mode examples have `rag.prompt` field
- Check that instruction mode examples have `instruction` field

## Next Steps After 0.5B Experiments

Once 0.5B experiments complete successfully:
1. Analyze results and document findings
2. Prepare for 3B experiments (create 3B configs, comparison plan)
3. Update documentation with lessons learned
4. Consider additional experiments or refinements
