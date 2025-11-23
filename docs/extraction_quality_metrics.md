# PDF Extraction Quality Metrics

## Overview

The PDF extraction system now includes automatic quality analysis for every converted document. Quality metrics are computed during ingestion and saved as `.quality.json` sidecar files alongside each `.txt` output.

## Metrics

### Parse Success (4 metrics)

- `pages_attempted`: Total number of pages in the PDF
- `pages_extracted`: Pages successfully converted to text
- `pages_failed`: Pages that failed extraction (empty output)
- `extraction_success_rate`: Ratio of successful pages (0.0-1.0)

Example:
```json
{
  "pages_attempted": 10,
  "pages_extracted": 10,
  "pages_failed": 0,
  "extraction_success_rate": 1.0
}
```

### Content Volume (3 metrics)

- `char_count`: Total character count across all pages
- `word_count`: Total word count (whitespace-delimited)
- `line_count`: Total number of non-empty lines

These metrics help identify documents with insufficient content or extraction failures.

### Structural Integrity (3 metrics)

- `has_paragraphs`: Boolean indicating presence of paragraph breaks (double newlines)
- `has_sentences`: Boolean indicating presence of sentence-ending punctuation
- `avg_line_length`: Average characters per line (indicator of proper text flow)

Well-structured academic papers typically have:
- Paragraphs: true
- Sentences: true
- Avg line length: 40-80 characters

### Text Composition (5 metrics)

Character distribution analysis:

- `alphabetic_ratio`: Proportion of alphabetic characters (0.0-1.0)
- `digit_ratio`: Proportion of digit characters (0.0-1.0)
- `whitespace_ratio`: Proportion of whitespace (0.0-1.0)
- `punctuation_ratio`: Proportion of punctuation marks (0.0-1.0)
- `special_char_ratio`: Proportion of special/control characters (0.0-1.0)

Typical ranges for academic papers:
- Alphabetic: 0.65-0.85
- Digits: 0.01-0.05
- Whitespace: 0.12-0.25
- Punctuation: 0.03-0.08
- Special chars: 0.00-0.03

High special character ratios may indicate extraction artifacts or encoding issues.

### Confidence Score (1 metric)

- `confidence_score`: Overall quality score (0.0-1.0)

Weighted aggregate of all metrics:
- 30% extraction success rate
- 20% content volume (word count normalized)
- 20% structural integrity
- 20% text composition quality
- 10% line length consistency

Interpretation:
- 0.8-1.0: Excellent extraction quality
- 0.6-0.8: Good quality, minor issues
- 0.4-0.6: Acceptable quality, review recommended
- 0.0-0.4: Poor quality, manual inspection required

## Output Format

Quality metrics are saved as JSON files with the `.quality.json` extension:

```
data/processed/
├── paper1.txt
├── paper1.quality.json
├── paper1.pages.jsonl
├── paper2.txt
├── paper2.quality.json
└── paper2.pages.jsonl
```

Example `.quality.json`:

```json
{
  "pages_attempted": 12,
  "pages_extracted": 12,
  "pages_failed": 0,
  "extraction_success_rate": 1.0,
  "char_count": 45234,
  "word_count": 7823,
  "line_count": 567,
  "has_paragraphs": true,
  "has_sentences": true,
  "avg_line_length": 79.8,
  "alphabetic_ratio": 0.752,
  "digit_ratio": 0.023,
  "whitespace_ratio": 0.187,
  "punctuation_ratio": 0.035,
  "special_char_ratio": 0.003,
  "confidence_score": 0.87
}
```

## Usage

### Automatic Generation

Quality metrics are automatically generated during PDF ingestion:

```bash
python scripts/ingest_and_index.py --config configs/default.yaml
```

No additional flags required. Quality analysis is performed for every converted PDF.

### Programmatic Access

```python
import json
from pathlib import Path

# Load quality metrics
quality_path = Path("data/processed/paper.quality.json")
metrics = json.loads(quality_path.read_text())

# Check confidence score
if metrics["confidence_score"] < 0.6:
    print(f"Warning: Low extraction quality ({metrics['confidence_score']:.2f})")
    print(f"Success rate: {metrics['extraction_success_rate']:.2%}")
    print(f"Words extracted: {metrics['word_count']}")
```

### Quality Filtering

Filter documents by quality during dataset preparation:

```python
from beyond_the_cutoff.data.extraction_quality import ExtractionQualityMetrics

def should_include_document(quality_path: Path, min_confidence: float = 0.6) -> bool:
    metrics_dict = json.loads(quality_path.read_text())
    return metrics_dict["confidence_score"] >= min_confidence

# Filter dataset
quality_files = Path("data/processed").glob("*.quality.json")
high_quality_docs = [
    f.with_suffix(".txt")
    for f in quality_files
    if should_include_document(f, min_confidence=0.7)
]

print(f"High quality documents: {len(high_quality_docs)}")
```

## Implementation

The quality analysis is implemented in `src/beyond_the_cutoff/data/extraction_quality.py`:

- `ExtractionQualityMetrics`: Dataclass containing all 17 metrics
- `ExtractionQualityAnalyzer.analyze_pages()`: Main analysis function

Integrated into `PDFIngestor.convert_all()` in `pdf_loader.py`.

## Testing

7 comprehensive tests cover all metrics and edge cases:

```bash
pytest tests/test_extraction_quality.py -v
```

Tests include:
- Empty pages (extraction failures)
- Well-structured content
- Partial failures
- Poor structure (garbled text)
- Long academic content
- Confidence score accuracy

All tests pass with 100% coverage of the quality analysis code.

## Troubleshooting

### Low Confidence Scores

Common causes and solutions:

1. **High page failure rate**: PDF may be scanned images without OCR
   - Solution: Use OCR preprocessing or exclude from dataset

2. **High special character ratio**: Encoding issues or extraction artifacts
   - Solution: Check PDF validity, try alternative extraction library

3. **Low word count**: Short documents or failed extraction
   - Solution: Verify PDF is complete, check for encryption

4. **Missing structure**: Text extracted as continuous stream
   - Solution: May be acceptable for some use cases, review manually

### Validation Workflow

```bash
# Find low-quality extractions
find data/processed -name "*.quality.json" -exec \
  python -c "import json, sys; \
    m = json.load(open(sys.argv[1])); \
    print(sys.argv[1], m['confidence_score']) if m['confidence_score'] < 0.6 else None" \
  {} \;

# Review specific document
cat data/processed/low_quality_paper.quality.json | jq .
head -n 50 data/processed/low_quality_paper.txt
```

## Future Enhancements

Potential improvements to the quality system:

1. Language detection and multilingual support
2. Mathematical notation detection
3. Figure/table extraction quality
4. Reference section identification
5. Citation format validation
6. Abstract/conclusion presence checks
7. Automated quality-based filtering in dataset generation
8. Quality dashboard visualization
