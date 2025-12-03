# Archived Configuration Files

This directory contains deprecated or superseded configuration files from earlier iterations of the experiment.

## Contents

| Config | Original Purpose | Superseded By |
|--------|------------------|---------------|
| `compare_0p5b_llama_judge.yaml` | Used Llama as judge | `six_condition_experiment.yaml` with Qwen3 8B |
| `compare_0p5b_qwen_no_thinking.yaml` | Qwen without thinking mode | `six_condition_experiment.yaml` with thinking mode |
| `compare_0p5b_sample.yaml` | Sample subset for testing | Full dataset runs |
| `compare_0p5b_sample_llama.yaml` | Sample with Llama judge | Full dataset runs |
| `compare_0p5b_sample_qwen.yaml` | Sample with Qwen judge | Full dataset runs |
| `lora_science_v1_rag_trained_f16_ollama.yaml` | F16 model config for quantization study | Moved here after quantization study completed |

## Why Archived

These configs were used during development and testing but are not part of the final reproducible experiment. The canonical configs are:

- `configs/evaluation/six_condition_experiment.yaml` - Main 6-condition experiment
- `configs/evaluation/quantization_comparison.yaml` - Q4_K_M vs F16 study
- `configs/evaluation/six_condition_comparisons.yaml` - Analysis configuration

## Can I Delete These?

These are safe to delete if you want to clean up the repository. They are preserved for:
1. Historical reference
2. Debugging if issues arise
3. Understanding the evolution of the experiment design
