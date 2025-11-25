"""PDF ingestion utilities using pypdf.

Converts PDFs under a source directory into UTF-8 plain text files under a
target directory, preserving relative filenames (with `.txt` extension).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .extraction_quality import ExtractionQualityAnalyzer


@dataclass
class PDFConversionResult:
    """Result of converting a single PDF."""

    output_path: Path
    confidence_score: float
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class PDFIngestor:
    """Extract text from PDFs to a processed text directory."""

    source_dir: Path
    target_dir: Path
    write_sidecars: bool = True
    min_extraction_confidence: float = 0.0  # 0.0 means no filtering

    def _pdf_paths(self) -> Iterable[Path]:
        yield from self.source_dir.rglob("*.pdf")

    # Known section names that don't require numbering (case-insensitive)
    _KNOWN_SECTIONS = frozenset(
        {
            "abstract",
            "introduction",
            "background",
            "related work",
            "related works",
            "literature review",
            "methodology",
            "methods",
            "materials and methods",
            "method",
            "approach",
            "proposed method",
            "proposed approach",
            "model",
            "models",
            "architecture",
            "system",
            "framework",
            "implementation",
            "experiments",
            "experiment",
            "experimental setup",
            "experimental settings",
            "experimental results",
            "results",
            "results and discussion",
            "analysis",
            "evaluation",
            "discussion",
            "limitations",
            "future work",
            "conclusion",
            "conclusions",
            "concluding remarks",
            "summary",
            "acknowledgments",
            "acknowledgements",
            "references",
            "bibliography",
            "appendix",
            "appendices",
            "supplementary material",
            "supplementary materials",
            "supplemental material",
            "data availability",
            "code availability",
            "author contributions",
            "competing interests",
            "conflict of interest",
            "funding",
            "ethics",
            "ethical considerations",
        }
    )

    @staticmethod
    def _guess_section_title(page_text: str) -> str | None:
        """Heuristically extract a section heading from the page text if present."""

        # Pattern for numbered headings like "1 Introduction", "2.1 Methods", "A.1 Details"
        heading_pattern = re.compile(r"^(?:[A-Z]?\d+(?:\.\d+)*\.?)\s+.+$")
        # Pattern for letter-prefixed appendix headings like "A Introduction", "B Methods"
        appendix_pattern = re.compile(r"^[A-Z]\s+[A-Z].+$")

        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if len(line) > 120:
                continue

            # Check for known section names first (case-insensitive exact match)
            line_lower = line.lower()
            if line_lower in PDFIngestor._KNOWN_SECTIONS:
                return line.title() if line.islower() else line

            # Check for numbered headings (e.g., "1 Introduction", "2.1 Methods")
            if heading_pattern.match(line):
                return line

            # Check for appendix-style headings (e.g., "A Proofs", "B Additional Results")
            if appendix_pattern.match(line) and len(line.split()) <= 8:
                return line

            if any(c.isalpha() for c in line):
                # ALL CAPS headings (common in older papers)
                if line.isupper() and len(line.split()) <= 12:
                    # Verify it's not just an acronym or short label
                    if len(line) > 3:
                        return line
                # Title Case headings (less common but valid)
                title_case = line.title()
                if line == title_case and len(line.split()) <= 10:
                    # Additional check: first word should be capitalized,
                    # and it shouldn't be a sentence (no ending punctuation)
                    if not line.endswith((".", "?", "!", ":")):
                        return line
        return None

    def convert_all(self) -> list[Path]:
        """Convert all PDFs under `source_dir` to `.txt` under `target_dir`.

        Returns:
            A list of paths to the generated `.txt` files (only those passing quality threshold).
        """

        if not self.source_dir.exists():
            return []

        self.target_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        skipped_low_quality: list[tuple[Path, float]] = []

        for pdf_path in self._pdf_paths():
            rel = pdf_path.relative_to(self.source_dir)
            out_path = (self.target_dir / rel).with_suffix(".txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pages = extract_pages_from_pdf(pdf_path)

            # Compute quality metrics
            quality_metrics = ExtractionQualityAnalyzer.analyze_pages(pages)

            # Check extraction quality threshold
            if self.min_extraction_confidence > 0.0:
                if quality_metrics.confidence_score < self.min_extraction_confidence:
                    skipped_low_quality.append((pdf_path, quality_metrics.confidence_score))
                    # Still write the quality metrics file for inspection
                    quality_path = out_path.with_suffix(".quality.json")
                    quality_data = quality_metrics.to_dict()
                    quality_data["_skipped"] = True
                    quality_data["_skip_reason"] = "below_confidence_threshold"
                    quality_data["_threshold"] = self.min_extraction_confidence
                    quality_path.write_text(
                        json.dumps(quality_data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    continue

            text = "\n\n".join(p for p in pages if p)
            out_path.write_text(text, encoding="utf-8")

            # Write quality metrics sidecar
            quality_path = out_path.with_suffix(".quality.json")
            quality_path.write_text(
                json.dumps(quality_metrics.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if self.write_sidecars:
                # Write sidecar JSONL with per-page texts for downstream page-aware indexing
                pages_path = out_path.with_suffix(".pages.jsonl")
                with pages_path.open("w", encoding="utf-8") as f:
                    for i, page_text in enumerate(pages):
                        section_title = self._guess_section_title(page_text or "")
                        rec = {"page": i + 1, "text": page_text or ""}
                        if section_title:
                            rec["section_title"] = section_title
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outputs.append(out_path)

        # Log summary of skipped files
        if skipped_low_quality:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Skipped %d PDFs due to low extraction confidence (threshold: %.2f):",
                len(skipped_low_quality),
                self.min_extraction_confidence,
            )
            for pdf_path, score in skipped_low_quality[:10]:  # Show first 10
                logger.warning("  - %s (confidence: %.2f)", pdf_path.name, score)
            if len(skipped_low_quality) > 10:
                logger.warning("  ... and %d more", len(skipped_low_quality) - 10)

        return outputs


def extract_pages_from_pdf(path: Path) -> list[str]:
    """Extract text per page from a PDF using PyMuPDF if available, else pypdf."""
    # Try PyMuPDF for higher fidelity extraction
    try:  # pragma: no cover - optional dependency
        import fitz

        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            try:
                text = page.get_text("text")
                if not isinstance(text, str):
                    text = ""
                pages.append(text.strip())
            except (RuntimeError, ValueError) as exc:
                # PyMuPDF can fail on corrupt pages
                import logging

                logger = logging.getLogger(__name__)
                logger.debug("Failed to extract text from page in %s: %s", path, exc)
                pages.append("")
        doc.close()
        return pages
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        # Fallback to pypdf if PyMuPDF not available or file can't be opened
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("PyMuPDF extraction failed for %s: %s, falling back to pypdf", path, exc)

    # Fallback to pypdf
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            content = page.extract_text() or ""
        except (
            RuntimeError,
            ValueError,
            AttributeError,
        ) as exc:  # pragma: no cover - pypdf can raise on corrupt pages
            import logging

            logger = logging.getLogger(__name__)
            logger.debug("Failed to extract text from page: %s", exc)
            content = ""
        parts.append(content.strip())
    return parts


def extract_text_from_pdf(path: Path) -> str:
    """Backward-compatible wrapper returning entire document text."""
    return "\n\n".join(p for p in extract_pages_from_pdf(path) if p)
