"""PDF ingestion utilities using pypdf.

Converts PDFs under a source directory into UTF-8 plain text files under a
target directory, preserving relative filenames (with `.txt` extension).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDFIngestor:
    """Extract text from PDFs to a processed text directory."""

    source_dir: Path
    target_dir: Path

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
            text = extract_text_from_pdf(pdf_path)
            out_path.write_text(text, encoding="utf-8")
            outputs.append(out_path)
        return outputs


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf with a simple heuristic."""
    # Import locally to keep module importable without optional dependency
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            content = page.extract_text() or ""
        except Exception:  # pragma: no cover - pypdf can raise on corrupt pages
            content = ""
        parts.append(content.strip())
    return "\n\n".join(p for p in parts if p)
