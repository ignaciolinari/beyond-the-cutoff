# Comparison Report: Beyond the Cutoff vs. RAG vs Fine-tuning Paper (Microsoft, 2024)

**Paper Reference:** [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://arxiv.org/pdf/2401.08406)
**Date:** January 2024
**Comparison Date:** 2025-01-XX

## Executive Summary

Both projects investigate the tradeoffs between Retrieval-Augmented Generation (RAG) and Fine-tuning for domain-specific LLM applications. The Microsoft paper focuses on **agriculture** with geography-specific knowledge, while Beyond the Cutoff targets **scientific research papers** (2025 arXiv papers) to test temporal generalization beyond training cutoffs. Despite different domains, the methodologies share significant structural similarities with notable differences in evaluation depth, model scale, and domain-specific adaptations.

---

## 1. Overall Objectives & Research Questions

### Microsoft Paper (Agriculture)
- **Primary Goal:** Compare RAG vs Fine-tuning for location-specific agricultural knowledge
- **Research Question:** Can LLMs provide geography-specific insights to farmers?
- **Domain:** Agriculture (location-specific crop/livestock knowledge)
- **Focus:** Industry-specific AI copilot for agriculture professionals

### Beyond the Cutoff
- **Primary Goal:** Evaluate how LLMs update knowledge beyond training cutoff dates
- **Research Question:** How effectively can fine-tuning vs RAG incorporate post-training scientific knowledge?
- **Domain:** Scientific research papers (2025 arXiv papers, post-cutoff)
- **Focus:** Temporal generalization and knowledge updating for research assistants

**Key Difference:** Microsoft focuses on **geographic specificity** (spatial context), while Beyond the Cutoff focuses on **temporal specificity** (knowledge cutoff dates).

---

## 2. Data Collection & Preprocessing

### Microsoft Paper
- **Source:** Agricultural documents (PDFs)
- **Processing:** PDF extraction → text cleaning → structured data
- **Geographic Focus:** State-specific agricultural knowledge (U.S. states)
- **Dataset Size:** Not explicitly stated in abstract, but focuses on agriculture corpus

### Beyond the Cutoff
- **Source:** arXiv API (2025 papers, post-training cutoff)
- **Processing:**
  - PDF extraction via PyMuPDF/pypdf with page-level granularity
  - Section-aware chunking (headings, page boundaries)
  - Metadata catalog generation (CSV/Parquet/JSONL)
  - Manifest-based versioning
- **Target:** ≥100 papers (configurable via `--total`)
- **Categories:** Configurable arXiv categories

**Similarities:**
- Both extract text from PDFs
- Both structure documents for downstream processing
- Both track metadata for provenance

**Differences:**
- **Beyond the Cutoff** has more sophisticated chunking (section-aware, page-aware)
- **Beyond the Cutoff** includes explicit versioning/manifest system
- **Microsoft** focuses on geographic metadata; **Beyond the Cutoff** focuses on temporal/version metadata

---

## 3. Question & Answer Generation

### Microsoft Paper
- **Method:** GPT-based Q&A generation from agricultural documents
- **Focus:** Geography-specific questions (state-level variations)
- **Evaluation:** Quality filtering of generated pairs
- **Goal:** Capture location-specific knowledge (e.g., "What crops grow best in Texas?" vs "What crops grow best in California?")

### Beyond the Cutoff
- **Method:** LLM-based task generation (default: `qwen2.5:7b-instruct-q4_K_M`)
- **Task Types:**
  - QA pairs (6-8 per document)
  - Summary prompts (2 per document)
  - Citation-check tasks (1-2 per document)
  - Contextualization prompts (1-2 per document)
- **Process:**
  - Samples chunks from indexed papers
  - Generates diverse task types via structured prompts
  - Creates "offline dataset" with RAG-ready prompts + contexts
  - Validates citation coverage (min 20% lexical overlap)
- **Output:** `offline_dataset.jsonl` + `offline_tasks.jsonl`

**Similarities:**
- Both use LLMs to generate Q&A pairs from documents
- Both filter/validate generated content
- Both create structured datasets for evaluation

**Differences:**
- **Beyond the Cutoff** generates multiple task types (QA, summaries, citations)
- **Beyond the Cutoff** explicitly includes citation requirements
- **Microsoft** emphasizes geographic variation; **Beyond the Cutoff** emphasizes temporal/post-cutoff knowledge
- **Beyond the Cutoff** has more structured validation (citation coverage, chunk alignment)

---

## 4. Fine-Tuning Approach

### Microsoft Paper
- **Models:** Llama2-13B, GPT-3.5, GPT-4
- **Method:** Fine-tuning on generated Q&A pairs
- **Training:** Incorporates domain knowledge into model parameters
- **Results:** +6 percentage points accuracy improvement

### Beyond the Cutoff
- **Models:** Qwen2.5-0.5B-Instruct (primary), Qwen2.5-3B-Instruct (scaled)
- **Method:** LoRA/PEFT fine-tuning (parameter-efficient)
- **Training Modes:**
  - **Instruction-only:** Trains without RAG contexts (for FT-only evaluation)
  - **RAG-trained:** (optional, not used in primary experiment)
- **Configuration:**
  - LoRA rank: 16
  - Learning rate: 0.0001
  - Batch size: 4, gradient accumulation: 4
  - Max steps: 1000
- **Infrastructure:** Cloud notebooks (Colab/Kaggle) → checkpoint sync → local quantization (GGUF) → Ollama

**Similarities:**
- Both fine-tune on generated Q&A pairs
- Both use instruction-following format
- Both evaluate fine-tuned models independently

**Differences:**
- **Microsoft:** Full fine-tuning on larger models (13B+)
- **Beyond the Cutoff:** Parameter-efficient (LoRA) on smaller models (0.5B-3B)
- **Beyond the Cutoff** explicitly separates instruction-only vs RAG-trained modes
- **Microsoft** focuses on accuracy improvements; **Beyond the Cutoff** focuses on temporal knowledge incorporation

---

## 5. RAG Implementation

### Microsoft Paper
- **Retrieval:** Augments prompts with external agricultural data
- **Method:** Not detailed in abstract, but uses retrieved contexts in prompts
- **Results:** +5 percentage points accuracy improvement (cumulative with fine-tuning)

### Beyond the Cutoff
- **Retrieval Stack:**
  - **Embeddings:** `BAAI/bge-m3` (multilingual, dense retrieval)
  - **Reranker:** `BAAI/bge-reranker-v2-m3` (cross-encoder)
  - **Index:** FAISS (vector store)
  - **Chunking:** Section-aware, page-aware (800 tokens, 120 overlap)
- **Pipeline:**
  - Dense retrieval (top-k: 4)
  - Optional reranking
  - Context formatting with citations
  - Page/section metadata in responses
- **Features:**
  - Inline citation markers `[1]`, `[2]`, etc.
  - Source paths with page numbers
  - Citation verification (lexical overlap checks)
  - Retrieval metrics (Hit@K, MRR)

**Similarities:**
- Both augment prompts with retrieved contexts
- Both show cumulative benefits when combined with fine-tuning

**Differences:**
- **Beyond the Cutoff** has more sophisticated retrieval (reranking, section-aware chunking)
- **Beyond the Cutoff** includes explicit citation tracking and verification
- **Beyond the Cutoff** measures retrieval quality (Hit@K, MRR) separately
- **Microsoft** focuses on domain-specific retrieval; **Beyond the Cutoff** focuses on temporal/post-cutoff retrieval

---

## 6. Evaluation Framework

### Microsoft Paper
- **Metrics:** Accuracy (quantitative), qualitative analysis
- **Evaluation:** GPT-4 as evaluator
- **Focus:** Geography-specific answer quality
- **Key Finding:** Fine-tuned model leverages cross-geography information (47% → 72% answer similarity)

### Beyond the Cutoff
- **Automated Metrics:**
  - **Generation:** BLEU, BERTScore, factuality (judge-based)
  - **Retrieval:** Hit@K (1,3,5,10), Mean Reciprocal Rank (MRR)
  - **Citations:** Citation precision/recall, coverage, groundedness
- **Judge System:**
  - Configurable judge models (default: `qwen2.5:7b-instruct-q4_K_M`)
  - Structured judge prompts (factuality, citation adherence)
  - Judge verdicts + scores logged per example
- **Evaluation Modes:**
  1. **RAG Only:** Base model + RAG contexts
  2. **FT Only:** Fine-tuned model (no RAG contexts)
  3. **RAG+FT:** Fine-tuned model + RAG contexts
- **Infrastructure:**
  - Comparative evaluation harness (`compare_models.py`)
  - Per-example + aggregated metrics
  - JSONL outputs for reproducibility

**Similarities:**
- Both use LLM judges for evaluation
- Both measure accuracy/quality improvements
- Both compare RAG vs FT vs Hybrid

**Differences:**
- **Beyond the Cutoff** has more comprehensive metrics (retrieval, citations, multiple generation metrics)
- **Beyond the Cutoff** includes retrieval-specific evaluation (Hit@K, MRR)
- **Beyond the Cutoff** has structured judge prompts and logging
- **Microsoft** focuses on geographic similarity; **Beyond the Cutoff** focuses on citation accuracy and factual consistency

---

## 7. Experimental Design

### Microsoft Paper
- **Comparison:** RAG vs Fine-tuning vs Combined
- **Models:** Multiple (Llama2-13B, GPT-3.5, GPT-4)
- **Domain:** Agriculture (geography-specific)
- **Results:**
  - FT: +6 p.p. accuracy
  - RAG: +5 p.p. accuracy (cumulative)
  - Cross-geography knowledge transfer demonstrated

### Beyond the Cutoff
- **Comparison:** Six-condition comparison (complete 2x2 matrix + 2 baselines)
- **Models:** Qwen2.5-0.5B (primary), Qwen2.5-3B (scaled)
- **Domain:** Scientific papers (2025, post-cutoff)
- **Design:**
  - Two fine-tuned models: instruction-only and RAG-trained
  - Complete 2x2 matrix: training with/without contexts × evaluation with/without contexts
  - Two baselines: base model without RAG, base model with RAG
  - Same evaluation dataset across all conditions
  - Offline dataset generation ensures reproducibility
- **Infrastructure:**
  - Config-driven experiments (`compare_0p5b_experiments.yaml`)
  - Automated comparative runs
  - Result aggregation and reporting

**Similarities:**
- Both compare RAG, FT, and Hybrid conditions
- Both show cumulative benefits
- Both use structured evaluation datasets

**Differences:**
- **Microsoft:** Multiple model sizes, focuses on model comparison
- **Beyond the Cutoff:** Single model size per experiment, focuses on method comparison
- **Beyond the Cutoff** has more explicit experimental controls (same model, same dataset)
- **Beyond the Cutoff** emphasizes reproducibility (offline datasets, manifests)

---

## 8. Key Technical Differences

### Model Scale & Efficiency
- **Microsoft:** Larger models (13B+), full fine-tuning
- **Beyond the Cutoff:** Smaller models (0.5B-3B), parameter-efficient (LoRA)

### Retrieval Sophistication
- **Microsoft:** Basic retrieval (details not specified)
- **Beyond the Cutoff:** Advanced retrieval (reranking, section-aware chunking, citation tracking)

### Evaluation Depth
- **Microsoft:** Accuracy + qualitative analysis
- **Beyond the Cutoff:** Multi-dimensional metrics (generation, retrieval, citations, factuality)

### Domain Focus
- **Microsoft:** Geographic specificity (spatial context)
- **Beyond the Cutoff:** Temporal specificity (knowledge cutoff dates)

### Infrastructure & Reproducibility
- **Microsoft:** Pipeline description, less emphasis on versioning
- **Beyond the Cutoff:** Strong versioning (manifests, metadata catalogs, offline datasets)

---

## 9. Similarities & Shared Insights

### Shared Findings
1. **Cumulative Benefits:** Both show that RAG + Fine-tuning outperforms either alone
2. **Domain Adaptation:** Both demonstrate successful domain-specific adaptation
3. **LLM-based Generation:** Both use LLMs to generate evaluation datasets
4. **Structured Pipelines:** Both have multi-stage pipelines (data → generation → training → evaluation)

### Shared Methodology Elements
1. PDF extraction and text processing
2. Q&A generation from documents
3. Fine-tuning on generated pairs
4. RAG augmentation of prompts
5. LLM-based evaluation/judging
6. Comparative evaluation framework

---

## 10. Unique Contributions of Each Project

### Microsoft Paper
- **Geographic Knowledge Transfer:** Demonstrates cross-geography learning (47% → 72% similarity)
- **Industry Focus:** Agriculture-specific use case with practical implications
- **Model Scale:** Tests larger models (13B+) with full fine-tuning
- **Qualitative Analysis:** Emphasizes expert comparison and qualitative insights

### Beyond the Cutoff
- **Temporal Generalization:** Tests knowledge updating beyond training cutoff
- **Citation Tracking:** Explicit citation accuracy and verification metrics
- **Retrieval Metrics:** Separate evaluation of retrieval quality (Hit@K, MRR)
- **Reproducibility:** Strong versioning, manifests, offline datasets
- **Parameter Efficiency:** LoRA-based fine-tuning for resource-constrained environments
- **Multi-task Evaluation:** QA, summaries, citations, contextualization

---

## 11. Recommendations for Cross-Pollination

### What Beyond the Cutoff Could Learn from Microsoft Paper
1. **Geographic/Spatial Context:** Consider location-specific knowledge in scientific domains (e.g., regional research trends)
2. **Expert Comparison:** Include human expert evaluation alongside automated metrics
3. **Cross-Domain Transfer:** Test whether fine-tuned models transfer knowledge across scientific subfields
4. **Qualitative Analysis:** Add more qualitative case studies alongside quantitative metrics

### What Microsoft Paper Could Learn from Beyond the Cutoff
1. **Citation Tracking:** Explicit citation accuracy metrics for agricultural recommendations
2. **Retrieval Evaluation:** Separate Hit@K/MRR metrics to understand retrieval quality
3. **Reproducibility:** Versioning and manifest systems for dataset tracking
4. **Parameter Efficiency:** LoRA fine-tuning for resource-constrained agricultural applications
5. **Multi-task Evaluation:** Beyond QA, include summarization and citation tasks

---

## 12. Conclusion

Both projects share a **core research question** (RAG vs Fine-tuning) but differ in **domain focus** (agriculture vs scientific papers), **model scale** (13B+ vs 0.5B-3B), and **evaluation depth** (accuracy vs multi-dimensional metrics).

**Beyond the Cutoff** offers:
- More sophisticated retrieval and citation tracking
- Stronger reproducibility infrastructure
- Parameter-efficient fine-tuning approach
- Temporal knowledge updating focus

**Microsoft Paper** offers:
- Geographic knowledge transfer insights
- Larger model scale results
- Industry-specific practical focus
- Qualitative expert comparison

The projects are **complementary** rather than competitive, with each contributing unique insights to the RAG vs Fine-tuning debate. Beyond the Cutoff's focus on **temporal generalization** and **citation accuracy** addresses gaps in the Microsoft paper, while Microsoft's **geographic transfer** and **industry focus** provide valuable real-world validation.

---

## References

- Microsoft Paper: [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://arxiv.org/pdf/2401.08406) (arXiv:2401.08406, January 2024)
- Beyond the Cutoff: Project repository documentation and codebase (2025)
