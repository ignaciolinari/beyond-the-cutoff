# Documentation

This folder contains all project documentation organized by purpose.

## 📋 Quick Links

| Document | Description |
|----------|-------------|
| [Experiment Setup](experiment/setup.md) | 6-condition experiment design and configuration |
| [Methodology](experiment/methodology.md) | Evaluation methodology and metrics |
| [Analysis Guide](experiment/analysis_guide.md) | How to analyze experiment results |

---

## 📁 Structure

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

### `reference/` - Technical Reference
Comprehensive technical documentation.

- **[pipeline.md](reference/pipeline.md)** - Full pipeline design and implementation plan
- **[model_implementation.md](reference/model_implementation.md)** - Model architecture and training details
- **[detailed_usage.md](reference/detailed_usage.md)** - Detailed CLI usage and examples

### `future/` - Planned Features
Documentation for features not yet implemented.

- **[elo_and_human_eval.md](future/elo_and_human_eval.md)** - ELO ranking and human evaluation plans

---

## 🚀 Getting Started

1. **New to the project?** Start with [Experiment Setup](experiment/setup.md)
2. **Running evaluation?** Check [Readiness Checklist](experiment/readiness_checklist.md)
3. **Analyzing results?** See [Analysis Guide](experiment/analysis_guide.md)

## 📊 Current Experiment

The main experiment evaluates 6 conditions:

| # | Condition | Training | RAG at Eval |
|---|-----------|----------|-------------|
| 1 | Base Baseline | None | ❌ |
| 2 | RAG Baseline | None | ✅ |
| 3 | FT Only (instruction) | Instruction-only | ❌ |
| 4 | FT+RAG (instruction) | Instruction-only | ✅ |
| 5 | FT Only (RAG-trained) | RAG-trained | ❌ |
| 6 | FT+RAG (RAG-trained) | RAG-trained | ✅ |

**Run command:**
```bash
python scripts/core/compare_models.py --plan configs/evaluation/six_condition_experiment.yaml
```
