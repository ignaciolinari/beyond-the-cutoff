from __future__ import annotations

from pathlib import Path

from beyond_the_cutoff.utils.checkpoint_sync import sync_directories


def _create_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_sync_directories_copies_missing_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dest"
    _create_file(source / "adapter" / "config.json", "{}")
    _create_file(source / "adapter" / "weights.safetensors", "weights")

    result = sync_directories(source, destination)

    assert destination.joinpath("adapter", "config.json").exists()
    assert destination.joinpath("adapter", "weights.safetensors").exists()
    assert result.as_dict()["copied"] == 2
    assert not result.errors


def test_sync_directories_skips_up_to_date_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dest"
    _create_file(source / "adapter" / "config.json", "{}")
    _create_file(destination / "adapter" / "config.json", "{}")

    result = sync_directories(source, destination)

    assert result.as_dict()["copied"] == 0
    assert result.as_dict()["skipped"] == 1


def test_sync_directories_respects_include_patterns(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dest"
    _create_file(source / "adapter.bin", "binary")
    _create_file(source / "notes.log", "log")

    result = sync_directories(source, destination, include_patterns=["*.bin"])

    assert destination.joinpath("adapter.bin").exists()
    assert not destination.joinpath("notes.log").exists()
    assert result.as_dict()["copied"] == 1
