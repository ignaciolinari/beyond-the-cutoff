"""Helpers to synchronise fine-tuning checkpoint directories."""

from __future__ import annotations

import fnmatch
import hashlib
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INCLUDE_PATTERNS = (
    "*.bin",
    "*.safetensors",
    "*.pt",
    "*.json",
    "*.txt",
    "*.md",
    "*.yaml",
    "*.ckpt",
)


@dataclass(slots=True)
class SyncResult:
    """Summary of a sync run."""

    copied: list[tuple[Path, Path]]
    skipped: list[tuple[Path, Path]]
    errors: list[tuple[Path, str]]

    def as_dict(self) -> dict[str, int]:
        return {
            "copied": len(self.copied),
            "skipped": len(self.skipped),
            "errors": len(self.errors),
        }


def sync_directories(
    source: Path,
    destination: Path,
    *,
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
    checksum: bool = False,
    dry_run: bool = False,
) -> SyncResult:
    """Copy checkpoint files from *source* into *destination* if they differ."""

    source = source.resolve()
    destination = destination.resolve()

    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    destination.mkdir(parents=True, exist_ok=True)

    include = tuple(include_patterns) if include_patterns else DEFAULT_INCLUDE_PATTERNS
    exclude = tuple(exclude_patterns) if exclude_patterns else ()

    copied: list[tuple[Path, Path]] = []
    skipped: list[tuple[Path, Path]] = []
    errors: list[tuple[Path, str]] = []

    for src_file in _iter_files(source, include, exclude):
        rel_path = src_file.relative_to(source)
        dest_file = destination / rel_path

        try:
            if dest_file.exists():
                if not _should_copy(src_file, dest_file, checksum=checksum):
                    skipped.append((src_file, dest_file))
                    continue
            if not dry_run:
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                _copy_file(src_file, dest_file)
            copied.append((src_file, dest_file))
        except Exception as exc:  # noqa: BLE001 - propagate info to caller summary
            errors.append((src_file, str(exc)))

    return SyncResult(copied=copied, skipped=skipped, errors=errors)


def _iter_files(
    root: Path,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> Iterator[Path]:
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root)
        if exclude_patterns and any(rel.match(pattern) for pattern in exclude_patterns):
            continue
        if include_patterns and not _matches_any(rel, include_patterns):
            continue
        yield file_path


def _matches_any(path: Path, patterns: Sequence[str]) -> bool:
    relative = path.as_posix()
    return any(fnmatch.fnmatch(relative, pattern) or path.match(pattern) for pattern in patterns)


def _should_copy(source: Path, destination: Path, *, checksum: bool) -> bool:
    src_stat = source.stat()
    dest_stat = destination.stat()

    if src_stat.st_size != dest_stat.st_size:
        return True
    if src_stat.st_mtime_ns > dest_stat.st_mtime_ns:
        return True
    if checksum:
        return _hash_file(source) != _hash_file(destination)
    return False


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


__all__ = [
    "DEFAULT_INCLUDE_PATTERNS",
    "SyncResult",
    "sync_directories",
]
