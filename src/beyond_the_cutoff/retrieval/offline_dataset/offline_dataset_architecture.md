# Offline Dataset Module Architecture

## Overview

The offline dataset generation system has been refactored from a monolithic 1810-line module into a modular architecture with clear separation of concerns. This document describes the new structure and its components.

## Module Structure

```
src/beyond_the_cutoff/retrieval/offline_dataset/
├── __init__.py              # Public API exports
├── types.py                 # Shared dataclasses and types
├── parser.py                # JSON parsing and response cleaning
├── validator.py             # Payload and output validation
├── citation_enforcer.py     # Citation compliance and rewriting
├── document_metadata.py     # Document metadata management
└── generator.py             # Main generation coordinator
```

## Component Responsibilities

### types.py (126 lines)

Defines shared data structures used across the module:

- `MappingRow`: Represents a chunk from the FAISS mapping TSV
- `DocumentStats`: Tracks document-level statistics (pages, tokens)
- `DocumentMetadata`: Comprehensive document metadata container
- `OfflineExample`: Output record for training/evaluation examples

All types include serialization methods and are fully typed.

### parser.py (81 lines)

Handles JSON parsing and text cleaning for LLM responses:

- `ResponseParser.strip_fences()`: Removes markdown code fences from JSON
- `ResponseParser.coerce_text()`: Type-safe text conversion
- `ResponseParser.parse_generator_response()`: Parses JSON with error recovery for invalid escapes

### validator.py (211 lines)

Validates generator payloads and output examples:

- `PayloadValidator.validate_generator_payload()`: Validates and cleans LLM JSON output
- `PayloadValidator.validate_output_examples()`: Validates final examples with citation checks
- `PayloadValidator.has_tasks()`: Checks if payload contains tasks
- `PayloadValidator.missing_minimum_counts()`: Verifies minimum task requirements

Required fields per task type:
- qa: (question, answer)
- summaries: (instruction, response)
- citations: (instruction, answer)
- contextualizations: (instruction, response)

### citation_enforcer.py (127 lines)

Enforces citation compliance with automatic rewriting:

- `CitationEnforcer.ensure_citation_compliance()`: Verifies and rewrites answers for proper citations
- Configurable minimum citation coverage threshold
- Automatic retry with LLM rewriting on compliance failures
- Detailed enforcement metadata tracking

### document_metadata.py (520 lines)

Manages document metadata and statistics:

- `DocumentMetadataManager.get_document_metadata()`: Retrieves metadata for a document
- `DocumentMetadataManager.get_document_stats()`: Retrieves/computes document statistics
- `DocumentMetadataManager.should_skip_document()`: Applies size-based filtering
- Loads from processed and raw manifests
- Handles multiple path variants for robust lookup
- Computes page/token counts from sidecars or raw text

### generator.py (671 lines)

Main coordinator that orchestrates all components:

- `OfflineDatasetGenerator.generate()`: Main entry point for dataset generation
- Manages file I/O and progress tracking
- Delegates to specialized components:
  - Parser for JSON handling
  - Validator for payload validation
  - CitationEnforcer for compliance
  - DocumentMetadataManager for metadata
- Handles retry logic and error reporting
- Builds prompts and examples

## Backward Compatibility

The original `offline_dataset.py` file has been converted to a compatibility wrapper:

```python
from .offline_dataset import (
    DocumentMetadata,
    DocumentStats,
    MappingRow,
    OfflineDatasetGenerator,
    OfflineExample,
)
```

All existing imports continue to work:

```python
# Still works
from beyond_the_cutoff.retrieval.offline_dataset import OfflineDatasetGenerator

# Also works with new module structure
from beyond_the_cutoff.retrieval.offline_dataset.generator import OfflineDatasetGenerator
```

## Testing

All 80 tests pass with the new structure:

- 3 tests for offline dataset generation (backward compatibility)
- 7 new tests for extraction quality metrics
- 70 existing tests for other components

No test modifications were required, demonstrating complete backward compatibility.

## Benefits

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Maintainability**: Smaller files are easier to understand and modify
4. **Extensibility**: New components can be added without affecting existing code
5. **Type Safety**: Comprehensive type hints throughout all modules
6. **Documentation**: Clear docstrings for all public APIs

## Migration Guide

No migration is required. All existing code continues to work without changes.

If you want to use the new modular structure directly:

```python
# Old style (still works)
from beyond_the_cutoff.retrieval.offline_dataset import OfflineDatasetGenerator

# New style (also works)
from beyond_the_cutoff.retrieval.offline_dataset import OfflineDatasetGenerator
from beyond_the_cutoff.retrieval.offline_dataset.parser import ResponseParser
from beyond_the_cutoff.retrieval.offline_dataset.validator import PayloadValidator
```

## Performance

No performance degradation. The refactoring is purely organizational and does not change the execution logic.

## Future Enhancements

The modular structure enables future improvements:

1. Parallel document processing
2. Pluggable validation strategies
3. Alternative citation enforcement methods
4. Custom metadata extractors
5. Streaming output for large datasets
