"""Tests for extraction quality metrics."""

from __future__ import annotations

from beyond_the_cutoff.data.extraction_quality import (
    ExtractionQualityAnalyzer,
)


def test_extraction_quality_empty_pages() -> None:
    """Test quality analysis on empty pages."""
    pages = ["", "", ""]
    metrics = ExtractionQualityAnalyzer.analyze_pages(pages)

    assert metrics.pages_attempted == 3
    assert metrics.pages_extracted == 0
    assert metrics.pages_failed == 3
    assert metrics.extraction_success_rate == 0.0
    assert metrics.char_count == 0
    assert metrics.word_count == 0
    assert metrics.confidence_score < 0.5


def test_extraction_quality_good_content() -> None:
    """Test quality analysis on well-structured content."""
    pages = [
        "This is a sample page with multiple sentences. It contains proper structure.",
        "Another page follows. This one has paragraphs too.\n\nSecond paragraph here.",
        "Final page with academic content. Results show improvement.",
    ]
    metrics = ExtractionQualityAnalyzer.analyze_pages(pages)

    assert metrics.pages_attempted == 3
    assert metrics.pages_extracted == 3
    assert metrics.pages_failed == 0
    assert metrics.extraction_success_rate == 1.0
    assert metrics.char_count > 100
    assert metrics.word_count > 20
    assert metrics.has_paragraphs is True
    assert metrics.has_sentences is True
    assert 0.6 <= metrics.alphabetic_ratio <= 0.9
    assert 0.1 <= metrics.whitespace_ratio <= 0.3
    assert metrics.confidence_score > 0.5


def test_extraction_quality_partial_failure() -> None:
    """Test quality analysis with some failed pages."""
    pages = [
        "Good page with content.",
        "",
        "Another successful page.",
        "",
        "Final page.",
    ]
    metrics = ExtractionQualityAnalyzer.analyze_pages(pages)

    assert metrics.pages_attempted == 5
    assert metrics.pages_extracted == 3
    assert metrics.pages_failed == 2
    assert metrics.extraction_success_rate == 0.6
    assert metrics.word_count > 0


def test_extraction_quality_poor_structure() -> None:
    """Test quality analysis on poorly structured content."""
    # Simulates garbled extraction
    pages = ["###  *** @@@", "|||   +++", ""]
    metrics = ExtractionQualityAnalyzer.analyze_pages(pages)

    assert metrics.pages_attempted == 3
    assert metrics.extraction_success_rate < 1.0
    assert metrics.has_sentences is False
    assert metrics.special_char_ratio > 0.3
    assert metrics.confidence_score < 0.5


def test_extraction_quality_metrics_to_dict() -> None:
    """Test metrics serialization to dict."""
    pages = ["Sample text for testing."]
    metrics = ExtractionQualityAnalyzer.analyze_pages(pages)

    metrics_dict = metrics.to_dict()

    assert isinstance(metrics_dict, dict)
    assert "char_count" in metrics_dict
    assert "word_count" in metrics_dict
    assert "confidence_score" in metrics_dict
    assert "extraction_success_rate" in metrics_dict
    assert isinstance(metrics_dict["confidence_score"], float)
    assert 0.0 <= metrics_dict["confidence_score"] <= 1.0


def test_extraction_quality_long_content() -> None:
    """Test quality analysis on longer, realistic content."""
    # Simulate a real academic paper excerpt
    pages = [
        """Abstract

This paper presents a novel approach to information retrieval in scientific
literature. We introduce a methodology that combines semantic embeddings with
traditional keyword-based search to improve precision and recall.

Introduction

The volume of scientific literature has grown exponentially over the past
decades. Researchers face challenges in discovering relevant papers efficiently.
Our work addresses this problem through enhanced retrieval mechanisms.""",
        """Methods

We employ a two-stage pipeline: (1) candidate generation using dense retrieval,
and (2) re-ranking with a cross-encoder model. The embedding model was fine-tuned
on domain-specific data to capture scientific terminology accurately.

Dataset

Our evaluation uses 10,000 papers from arXiv in computer science categories.
We split the data into training (70%), validation (15%), and test (15%) sets.""",
        """Results

The proposed method achieves 85% precision at k=10, outperforming baseline
approaches by 15 percentage points. Recall@100 reaches 92%, demonstrating
strong coverage of relevant documents.

Discussion

These results validate our hypothesis that domain-specific fine-tuning
significantly improves retrieval quality in specialized fields.""",
    ]

    metrics = ExtractionQualityAnalyzer.analyze_pages(pages)

    # Should have high quality scores
    assert metrics.pages_attempted == 3
    assert metrics.pages_extracted == 3
    assert metrics.extraction_success_rate == 1.0
    assert metrics.char_count > 1000
    assert metrics.word_count > 150
    assert metrics.has_paragraphs is True
    assert metrics.has_sentences is True
    assert metrics.confidence_score > 0.7

    # Should have good text composition
    assert 0.65 <= metrics.alphabetic_ratio <= 0.85
    assert 0.12 <= metrics.whitespace_ratio <= 0.25
    assert metrics.avg_line_length > 30


def test_extraction_quality_confidence_scoring() -> None:
    """Test confidence score calculation logic."""
    # Perfect extraction
    good_pages = [
        "This is excellent content with proper structure. " * 50,
        "More quality text follows here. " * 50,
    ]
    good_metrics = ExtractionQualityAnalyzer.analyze_pages(good_pages)

    # Poor extraction
    bad_pages = ["", "###", ""]
    bad_metrics = ExtractionQualityAnalyzer.analyze_pages(bad_pages)

    # Good should have higher confidence than bad
    assert good_metrics.confidence_score > bad_metrics.confidence_score
    assert good_metrics.confidence_score > 0.6
    assert bad_metrics.confidence_score < 0.3
