"""Extraction quality metrics for PDF processing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class ExtractionQualityMetrics:
    """Quality metrics for PDF text extraction."""

    # Basic text metrics
    char_count: int
    word_count: int
    line_count: int

    # Parse success metrics
    pages_attempted: int
    pages_extracted: int
    pages_failed: int
    extraction_success_rate: float

    # Structural integrity metrics
    has_paragraphs: bool
    has_sentences: bool
    avg_line_length: float
    whitespace_ratio: float

    # Content quality indicators
    alphabetic_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    special_char_ratio: float

    # Confidence score (0.0 to 1.0)
    confidence_score: float

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Convert metrics to dictionary."""
        return {
            "char_count": self.char_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "pages_attempted": self.pages_attempted,
            "pages_extracted": self.pages_extracted,
            "pages_failed": self.pages_failed,
            "extraction_success_rate": self.extraction_success_rate,
            "has_paragraphs": self.has_paragraphs,
            "has_sentences": self.has_sentences,
            "avg_line_length": self.avg_line_length,
            "whitespace_ratio": self.whitespace_ratio,
            "alphabetic_ratio": self.alphabetic_ratio,
            "digit_ratio": self.digit_ratio,
            "punctuation_ratio": self.punctuation_ratio,
            "special_char_ratio": self.special_char_ratio,
            "confidence_score": self.confidence_score,
        }


class ExtractionQualityAnalyzer:
    """Analyze quality of extracted PDF text."""

    @staticmethod
    def analyze_pages(pages: Sequence[str]) -> ExtractionQualityMetrics:
        """
        Analyze extraction quality from page texts.

        Args:
            pages: List of extracted page texts

        Returns:
            Quality metrics with confidence score
        """
        pages_attempted = len(pages)
        pages_extracted = sum(1 for p in pages if p and p.strip())
        pages_failed = pages_attempted - pages_extracted

        extraction_success_rate = pages_extracted / pages_attempted if pages_attempted > 0 else 0.0

        # Combine all text
        full_text = "\n\n".join(p for p in pages if p)

        # Basic metrics
        char_count = len(full_text)
        words = full_text.split()
        word_count = len(words)
        lines = [line for line in full_text.split("\n") if line.strip()]
        line_count = len(lines)

        # Structural metrics
        has_paragraphs = "\n\n" in full_text
        has_sentences = bool(re.search(r"[.!?]\s+[A-Z]", full_text))
        avg_line_length = sum(len(line) for line in lines) / line_count if line_count > 0 else 0.0

        # Character type ratios
        if char_count > 0:
            alphabetic_count = sum(1 for c in full_text if c.isalpha())
            digit_count = sum(1 for c in full_text if c.isdigit())
            whitespace_count = sum(1 for c in full_text if c.isspace())
            punctuation_count = sum(1 for c in full_text if c in ".,;:!?\"'-()[]{}/")

            alphabetic_ratio = alphabetic_count / char_count
            digit_ratio = digit_count / char_count
            whitespace_ratio = whitespace_count / char_count
            punctuation_ratio = punctuation_count / char_count
            special_char_ratio = 1.0 - (
                alphabetic_ratio + digit_ratio + whitespace_ratio + punctuation_ratio
            )
        else:
            alphabetic_ratio = 0.0
            digit_ratio = 0.0
            whitespace_ratio = 0.0
            punctuation_ratio = 0.0
            special_char_ratio = 0.0

        # Calculate confidence score (0.0 to 1.0)
        confidence_score = ExtractionQualityAnalyzer._calculate_confidence(
            extraction_success_rate=extraction_success_rate,
            char_count=char_count,
            word_count=word_count,
            has_paragraphs=has_paragraphs,
            has_sentences=has_sentences,
            alphabetic_ratio=alphabetic_ratio,
            whitespace_ratio=whitespace_ratio,
            avg_line_length=avg_line_length,
        )

        return ExtractionQualityMetrics(
            char_count=char_count,
            word_count=word_count,
            line_count=line_count,
            pages_attempted=pages_attempted,
            pages_extracted=pages_extracted,
            pages_failed=pages_failed,
            extraction_success_rate=extraction_success_rate,
            has_paragraphs=has_paragraphs,
            has_sentences=has_sentences,
            avg_line_length=avg_line_length,
            whitespace_ratio=whitespace_ratio,
            alphabetic_ratio=alphabetic_ratio,
            digit_ratio=digit_ratio,
            punctuation_ratio=punctuation_ratio,
            special_char_ratio=special_char_ratio,
            confidence_score=confidence_score,
        )

    @staticmethod
    def _calculate_confidence(
        *,
        extraction_success_rate: float,
        char_count: int,
        word_count: int,
        has_paragraphs: bool,
        has_sentences: bool,
        alphabetic_ratio: float,
        whitespace_ratio: float,
        avg_line_length: float,
    ) -> float:
        """
        Calculate overall confidence score from individual metrics.

        Confidence indicators:
        - High extraction success rate (>= 0.95)
        - Sufficient content (>= 1000 chars, >= 100 words)
        - Proper structure (paragraphs and sentences)
        - Normal text ratios (alphabetic 0.6-0.9, whitespace 0.1-0.3)
        - Reasonable line lengths (30-100 chars)
        """
        score = 0.0

        # Extraction success (30% weight)
        score += extraction_success_rate * 0.3

        # Content volume (20% weight)
        char_score = min(char_count / 1000.0, 1.0)
        word_score = min(word_count / 100.0, 1.0)
        score += (char_score + word_score) / 2 * 0.2

        # Structural integrity (20% weight)
        structure_score = 0.0
        if has_paragraphs:
            structure_score += 0.5
        if has_sentences:
            structure_score += 0.5
        score += structure_score * 0.2

        # Text composition (20% weight)
        alpha_optimal = 0.6 <= alphabetic_ratio <= 0.9
        whitespace_optimal = 0.1 <= whitespace_ratio <= 0.3
        composition_score = (
            (1.0 if alpha_optimal else alphabetic_ratio)
            + (1.0 if whitespace_optimal else max(0.0, 1.0 - abs(whitespace_ratio - 0.2) / 0.2))
        ) / 2
        score += composition_score * 0.2

        # Line length (10% weight)
        line_length_optimal = 30 <= avg_line_length <= 100
        line_score = 1.0 if line_length_optimal else max(0.0, 1.0 - abs(avg_line_length - 65) / 65)
        score += line_score * 0.1

        return min(1.0, max(0.0, score))


__all__ = ["ExtractionQualityMetrics", "ExtractionQualityAnalyzer"]
