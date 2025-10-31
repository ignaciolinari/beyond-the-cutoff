"""Tests for arXiv helper utilities."""

from __future__ import annotations

from datetime import datetime, timezone

from beyond_the_cutoff.data.arxiv import (
    ArxivPaper,
    _parse_response,
    build_category_query,
)


def test_build_category_query_multiple_categories() -> None:
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime(2025, 10, 30, tzinfo=timezone.utc)
    query = build_category_query(["cs.AI", "cs.CL"], submitted_after=start, submitted_before=end)
    assert query.startswith("(cat:cs.AI+OR+cat:cs.CL)+AND+submittedDate:")
    assert "202507010000" in query and "202510300000" in query


def test_parse_response_single_entry() -> None:
    feed = """<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom' xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
      <opensearch:totalResults>1</opensearch:totalResults>
      <entry>
        <id>http://arxiv.org/abs/2508.12345v2</id>
        <updated>2025-08-11T12:00:00Z</updated>
        <published>2025-08-10T10:00:00Z</published>
        <title>Sample Paper Title</title>
        <summary>Summary of the paper.</summary>
        <author><name>Author One</name></author>
        <author><name>Author Two</name></author>
        <link rel='alternate' href='http://arxiv.org/abs/2508.12345v2' type='text/html'/>
        <link rel='related' href='http://arxiv.org/pdf/2508.12345v2.pdf' type='application/pdf'/>
        <category term='cs.CL'/>
        <category term='cs.AI'/>
        <arxiv:primary_category term='cs.CL'/>
      </entry>
    </feed>
    """

    result = _parse_response(feed)
    assert result.total_results == 1
    assert len(result.papers) == 1
    paper: ArxivPaper = result.papers[0]
    assert paper.arxiv_id == "2508.12345v2"
    assert paper.canonical_id == "2508.12345"
    assert paper.title == "Sample Paper Title"
    assert paper.summary.startswith("Summary")
    assert paper.authors == ["Author One", "Author Two"]
    assert paper.primary_category == "cs.CL"
    assert paper.categories == ["cs.CL", "cs.AI"]
    assert paper.link.endswith("2508.12345v2")
    assert paper.pdf_url.endswith("2508.12345v2.pdf")
    assert paper.pdf_url.startswith("https://")
    assert paper.published == datetime(2025, 8, 10, 10, 0, tzinfo=timezone.utc)
    assert paper.updated == datetime(2025, 8, 11, 12, 0, tzinfo=timezone.utc)
