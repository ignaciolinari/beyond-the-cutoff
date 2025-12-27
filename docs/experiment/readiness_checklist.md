# Experiment Readiness Checklist

_Last updated: 2025-12-02_

> ✓ **EXPERIMENT COMPLETED** — The 6-condition experiment on Qwen 2.5 0.5B has been completed. This checklist is preserved for reference when running experiments with larger models.

Use this checklist before running the 0.5B model comparison experiments to ensure everything is properly configured.

## Pre-Experiment Setup

### 1. Data & Indexing
- [x] **Corpus ingested**: Run `python scripts/data/ingest_and_index.py --config configs/default.yaml`
- [x] **Index exists**: Verify `data/external/index/` contains FAISS index files
- [x] **Offline dataset generated**: Run `python scripts/data/generate_offline_dataset.py --config configs/default.yaml`
- [x] **Dataset validated**: Check that `evaluation/datasets/offline_dataset.jsonl` exists and has expected structure
- [x] **Dataset size**: Verify dataset has sufficient examples (recommended: >100 tasks)

### 2. Model Availability

#### Base Model (RAG-only)
- [x] **Ollama base model**: `ollama pull qwen2.5:0.5b-instruct` (or verify it's already available)
- [x] **Model accessible**: Test with `ollama list | grep qwen2.5:0.5b-instruct`

#### Fine-Tuned Models (6-condition experiment requires 2 models)

**Instruction-Only Model** (for conditions 3-4):
- [x] **Model trained**: Fine-tuning completed in Colab using `notebooks/finetuning/lora_science_v1_instruction_only.ipynb`
- [x] **Checkpoint synced**: Merged model weights synced from Colab to local `outputs/lora_science_v1_instruction_only/merged_full_model/`
- [x] **GGUF converted**: Model converted to GGUF format (e.g., `Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf`)
- [x] **Ollama model registered**: `ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only`
- [x] **Model accessible**: Test with `ollama list | grep lora_science_0p5_instruction_only`

**RAG-Trained Model** (for conditions 5-6):
- [x] **Model trained**: Fine-tuning completed using `notebooks/finetuning/lora_science_v1.ipynb`
- [x] **Checkpoint synced**: Merged model weights synced to local (e.g., `outputs/lora_science_v1/merged_full_model/`)
- [x] **GGUF converted**: Model converted to GGUF format
- [x] **Ollama model registered**: `ollama create lora_science_0p5 -f ollama/Modelfile.rag_trained`
- [x] **Model accessible**: Test with `ollama list | grep lora_science_0p5`

#### Judge Model
- [x] **Judge model available**: `ollama pull qwen3:8b` (primary judge with thinking mode)
- [x] **Judge accessible**: Test with `ollama list | grep qwen3:8b`

### 3. Configuration Files

#### Model Configs
- [x] **`configs/models/base_ollama.yaml`**: Points to `qwen2.5:0.5b-instruct` (used for base baseline and RAG baseline)
- [x] **`configs/models/lora_instruction_only.yaml`**: Points to `lora_science_0p5_instruction_only`
- [x] **`configs/models/lora_rag_trained.yaml`**: Points to `lora_science_0p5` (RAG-trained model)
- [x] **`configs/hybrid_science_v1_ollama.yaml`**: Legacy config (moved to `vintage/configs/` - not used in current experiment)

#### Comparison Plan
- [x] **`configs/evaluation/six_condition_experiment.yaml`**:
  - Contains all 6 conditions (base baseline, RAG baseline, FT-only, FT+RAG instruction-only, RAG-trained FT-only, RAG-trained FT+RAG)
  - Uses correct model configs for each condition
  - Has correct `prompt_mode` settings (`rag` for conditions with RAG, `instruction` for conditions without RAG)
  - Points to correct dataset path

#### Judge Config
- [x] **`configs/judges/rag.yaml`**: Exists and has proper prompt template (for RAG conditions)
- [x] **`configs/judges/instruction.yaml`**: Exists and has proper prompt template (for instruction-only conditions)
- [x] **`configs/judges/dataset_quality_judge.yaml`**: Points to Qwen3 8B judge model with thinking mode

### 4. Environment & Dependencies
- [x] **Python environment**: Virtual environment activated (`.venv/bin/activate`)
- [x] **Dependencies installed**: `pip install -e '.[dev]'` completed successfully
- [x] **Ollama running**: `ollama serve` is running (or Ollama daemon is active)
- [x] **Ollama accessible**: Test with `curl http://localhost:11434/api/tags`

### 5. Paths & Permissions
- [x] **Output directory writable**: `evaluation/results/` exists and is writable
- [x] **Data directories**: `data/processed/`, `data/external/` exist
- [x] **Sufficient disk space**: Check available space for results and logs

## Running Experiments

### Quick Smoke Test (Optional Pre-Check)
Test the pipeline before running the full comparison using the interleaved evaluation script:

```bash
python scripts/core/interleaved_evaluation.py \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --limit 5
```

This will evaluate 5 examples across all conditions and verify the pipeline works.

### Full Comparison Run
Once pre-checks pass, run the full comparison:

```bash
python scripts/core/compare_models.py \
  --config configs/default.yaml \
  --plan configs/evaluation/six_condition_experiment.yaml \
  --output evaluation/results/comparison_report.json
```

### Expected Outputs
After running the comparison, verify:
- [x] **Metrics files**: `evaluation/results/{label}/metrics.json` for each run
- [x] **Details files**: `evaluation/results/{label}/details.jsonl` for each run
- [x] **Metadata files**: `evaluation/results/{label}/metadata.jsonl` for each run
- [x] **Comparison report**: `evaluation/results/comparison_report.json` (if `--output` specified)

## Post-Experiment Verification

### Data Consistency
- [x] **Same examples evaluated**: All six runs evaluated the same task IDs (check `evaluated_task_ids` in metrics)
- [x] **No missing data**: All expected examples have predictions
- [x] **Error rate acceptable**: Check `examples_with_errors` in summary metrics

### Experimental Correctness
- [x] **Base baseline**: Used base model (`qwen2.5:0.5b-instruct`) with `prompt_mode: instruction`
- [x] **RAG baseline**: Used base model (`qwen2.5:0.5b-instruct`) with `prompt_mode: rag`
- [x] **FT-only**: Used instruction-only model (`lora_science_0p5_instruction_only`) with `prompt_mode: instruction`
- [x] **FT+RAG (instruction-only)**: Used instruction-only model (`lora_science_0p5_instruction_only`) with `prompt_mode: rag`
- [x] **RAG-trained FT-only**: Used RAG-trained model (`lora_science_0p5`) with `prompt_mode: instruction`
- [x] **RAG-trained FT+RAG**: Used RAG-trained model (`lora_science_0p5`) with `prompt_mode: rag`
- [x] **Same dataset**: All six runs used the same `offline_dataset.jsonl`
- [x] **Same judge**: All six runs used the same judge model and prompt

### Results Analysis
- [x] **Metrics computed**: Factuality, citation accuracy, BLEU, BERTScore present in summaries for all 6 conditions
- [x] **Citation metrics**: Conditions without RAG (base baseline, FT-only, RAG-trained FT-only) have citation metrics marked as N/A (expected)
- [x] **Comparison valid**: Results show clear differences between the six conditions
- [x] **Baseline comparisons**: Base baseline vs RAG baseline shows RAG benefit; Base baseline vs FT-only shows fine-tuning benefit

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

✓ **Completed:**
1. Analyzed results and documented findings (see `docs/reports/`)
2. Ran pairwise comparison between top candidates
3. Ran quantization comparison (Q4_K_M vs F16)

**Future work:**
1. Prepare for 3B experiments (create 3B configs, comparison plan)
2. Consider scaling to larger models (7B, 14B) to test if fine-tuning benefits increase with model capacity
