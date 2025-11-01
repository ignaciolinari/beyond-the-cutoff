# Retrieval Presets

Each YAML file in this directory defines a named retrieval configuration used by the scientific assistant.

## Suggested Structure
```yaml
id: qa_default
description: Literature QA preset for post-cutoff papers
settings:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  chunk_size: 512
  chunk_overlap: 64
  top_k: 6
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  max_context_chars: 6000
```

## Planned Presets
- `qa_default.yaml`
- `method_compare.yaml`
- `summary_structured.yaml`
- `citation_trace.yaml`

Add `owner` and `last_updated` metadata when the preset is activated.
