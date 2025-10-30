from pathlib import Path

from beyond_the_cutoff.data.pdf_loader import PDFIngestor


def test_convert_all_missing_source(tmp_path: Path) -> None:
    source = tmp_path / "missing"
    target = tmp_path / "processed"
    ingestor = PDFIngestor(source_dir=source, target_dir=target)

    outputs = ingestor.convert_all()

    assert outputs == []
    assert not target.exists()


def test_convert_all_empty_dir(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    target = tmp_path / "processed"
    ingestor = PDFIngestor(source_dir=source, target_dir=target)

    outputs = ingestor.convert_all()

    assert outputs == []
    assert target.exists()
