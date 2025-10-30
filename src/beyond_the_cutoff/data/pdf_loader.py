"""PDF ingestion utilities using pypdf.

Converts PDFs under a source directory into UTF-8 plain text files under a
target directory, preserving relative filenames (with `.txt` extension).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDFIngestor:
    """Extract text from PDFs to a processed text directory."""

    source_dir: Path
    target_dir: Path
    write_sidecars: bool = True

    def _pdf_paths(self) -> Iterable[Path]:
        yield from self.source_dir.rglob("*.pdf")

    def convert_all(self) -> list[Path]:
        """Convert all PDFs under `source_dir` to `.txt` under `target_dir`.

        Returns:
            A list of paths to the generated `.txt` files.
        """

        if not self.source_dir.exists():
            return []

        self.target_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for pdf_path in self._pdf_paths():
            rel = pdf_path.relative_to(self.source_dir)
            out_path = (self.target_dir / rel).with_suffix(".txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pages = extract_pages_from_pdf(pdf_path)
            text = "\n\n".join(p for p in pages if p)
            out_path.write_text(text, encoding="utf-8")
            if self.write_sidecars:
                # Write sidecar JSONL with per-page texts for downstream page-aware indexing
                pages_path = out_path.with_suffix(".pages.jsonl")
                with pages_path.open("w", encoding="utf-8") as f:
                    for i, page_text in enumerate(pages):
                        rec = {"page": i + 1, "text": page_text or ""}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outputs.append(out_path)
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
                pages.append(page.get_text("text").strip())
            except Exception:
                pages.append("")
        doc.close()
        return pages
    except Exception:
        pass

    # Fallback to pypdf
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            content = page.extract_text() or ""
        except Exception:  # pragma: no cover - pypdf can raise on corrupt pages
            content = ""
        parts.append(content.strip())
    return parts


def extract_text_from_pdf(path: Path) -> str:
    """Backward-compatible wrapper returning entire document text."""
    return "\n\n".join(p for p in extract_pages_from_pdf(path) if p)
