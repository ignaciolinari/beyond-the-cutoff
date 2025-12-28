# Documentation

This folder contains all project documentation organized by purpose.

> **Experiment Status: COMPLETED** — The Qwen 2.5 0.5B experiment is finished. See [Experiment Results](#experiment-results-completed) for findings.

## Quick Links

| Document | Description |
|----------|-------------|
| [Experiment Setup](experiment/setup.md) | 6-condition experiment design and configuration |
| [Methodology](experiment/methodology.md) | Evaluation methodology and metrics |
| [Analysis Guide](experiment/analysis_guide.md) | How to analyze experiment results |
| [**Six-Condition Results**](reports/six_condition_experiment_results.md) | **Main experiment findings** |
| [Pairwise Evaluation](reports/pairwise_evaluation_results.md) | Best candidates comparison |
| [Quantization Analysis](reports/quantization_comparison_results.md) | Q4_K_M vs F16 impact |

---

## Experiment Results (Completed)

### Key Findings

| Finding | Evidence |
|---------|----------|
| **RAG is critical** | 4-5x improvement in pass rate (22% vs 4%) |
| **Fine-tuning provides marginal gains** | +1.3% when training format matches inference |
| **Training format matters** | RAG-trained > instruction-only for RAG inference |
| **Q4_K_M quantization is safe** | No quality degradation, 60% size reduction |
| **Pairwise comparison** | p=0.35, no significant difference between Base+RAG and FT+RAG |

### Scope Limitation

> These results apply to **Qwen 2.5 0.5B** only. Larger models may show different patterns due to increased capacity for knowledge retention. The pipeline is designed to test this — see [Scaling Guide](scaling/README.md).

---

## Folder Structure

### `experiment/` - Main Experiment Documentation
Core documentation for running and understanding the 6-condition RAG vs Fine-tuning experiment.

- **[setup.md](experiment/setup.md)** - Experiment design, conditions, and configuration
- **[methodology.md](experiment/methodology.md)** - Evaluation methodology, metrics, and phases
- **[analysis_guide.md](experiment/analysis_guide.md)** - Guide for analyzing results
- **[readiness_checklist.md](experiment/readiness_checklist.md)** - Pre-flight checklist before running

### `reports/` - Analysis Reports
Results and findings from various analyses.

- **[judge_comparison.md](reports/judge_comparison.md)** - Comparison of different LLM judges
- **[dataset_quality.md](reports/dataset_quality.md)** - Dataset quality assessment
- **[extraction_quality.md](reports/extraction_quality.md)** - PDF extraction quality metrics
- **[six_condition_experiment_results.md](reports/six_condition_experiment_results.md)** - **Main experiment results**
- **[pairwise_evaluation_results.md](reports/pairwise_evaluation_results.md)** - Pairwise comparison (Base+RAG vs FT+RAG)
- **[quantization_comparison_results.md](reports/quantization_comparison_results.md)** - Q4_K_M vs F16 analysis

### `reference/` - Technical Reference
Comprehensive technical documentation.

- **[pipeline.md](reference/pipeline.md)** - Full pipeline design and implementation plan
- **[model_implementation.md](reference/model_implementation.md)** - Model architecture and training details

### `scaling/` - Scaling Guide
How to adapt the pipeline for different models.

- **[README.md](scaling/README.md)** - Guide for running experiments with larger models

### `future/` - Planned Features
Documentation for features not yet implemented.

- **[elo_and_human_eval.md](future/elo_and_human_eval.md)** - ELO ranking and human evaluation plans

---

## Getting Started

1. **New to the project?** Start with [Experiment Setup](experiment/setup.md)
2. **Want results?** See [Six-Condition Results](reports/six_condition_experiment_results.md)
3. **Running evaluation?** Check [Readiness Checklist](experiment/readiness_checklist.md)
4. **Analyzing results?** See [Analysis Guide](experiment/analysis_guide.md)
5. **Scaling up?** See [Scaling Guide](scaling/README.md)

### Six-Condition Pass Rates

The experiment evaluated 6 conditions on **Qwen 2.5 0.5B**:

| # | Condition | Training | RAG at Eval | Pass Rate |
|---|-----------|----------|-------------|-----------|
| 1 | Base Baseline | None | ✗ | 4.2% |
| 2 | RAG Baseline | None | ✓ | **22.8%** |
| 3 | FT Only (instruction) | Instruction-only | ✗ | 4.3% |
| 4 | FT+RAG (instruction) | Instruction-only | ✓ | 19.7% |
| 5 | FT Only (RAG-trained) | RAG-trained | ✗ | 5.6% |
| 6 | FT+RAG (RAG-trained) | RAG-trained | ✓ | **24.1%** |

**To reproduce:**
```bash
python scripts/core/compare_models.py --plan configs/evaluation/six_condition_experiment.yaml
```
