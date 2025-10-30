from beyond_the_cutoff.utils.chunking import chunk_text


def test_chunking_basic() -> None:
    text = " ".join([f"w{i}" for i in range(100)])
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    # First chunk has 20 tokens
    assert len(chunks[0].split()) == 20
    # Overlap of 5 tokens between adjacent chunks
    first = chunks[0].split()
    second = chunks[1].split()
    assert first[-5:] == second[:5]


def test_chunking_edge_cases() -> None:
    assert chunk_text("", 10, 2) == []
    assert chunk_text("short text", 50, 0) == ["short text"]
