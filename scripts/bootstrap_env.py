#!/usr/bin/env python3
"""Bootstrap a virtual environment and install project dependencies.

This helper creates (or reuses) a virtual environment, installs the project in
editable mode together with optional development extras, and wires up
pre-commit hooks so linting runs automatically. Designed to be idempotent and to
keep the local workflow consistent across contributors."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = PROJECT_ROOT / ".venv"


def run_command(command: list[str], *, description: str) -> None:
    """Run a subprocess command, surfacing context in case of failure."""
    print(f"[bootstrap] {description}: {' '.join(command)}")
    subprocess.run(command, check=True)


def create_virtualenv(python_executable: str, venv_path: Path) -> None:
    """Create the virtual environment if it does not already exist."""
    if venv_path.exists():
        print(f"[bootstrap] Re-using existing virtual environment at {venv_path}")
        return

    run_command(
        [python_executable, "-m", "venv", str(venv_path)],
        description="Creating virtual environment",
    )


def venv_python(venv_path: Path) -> Path:
    """Return the Python executable inside the virtual environment."""
    if os.name == "nt":  # pragma: no cover - Windows path handling
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(
            f"Expected virtual environment Python at {python_path} but it was not found."
        )
    return python_path


def install_project(python_path: Path, *, include_dev: bool) -> None:
    """Install the project (and optional dev extras) into the virtual environment."""
    run_command(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
        description="Upgrading pip",
    )

    target = ".[dev]" if include_dev else "."
    run_command(
        [str(python_path), "-m", "pip", "install", "-e", target],
        description=f"Installing project dependencies ({target})",
    )


def install_pre_commit(python_path: Path) -> None:
    """Install the pre-commit git hook if the executable is available."""
    pre_commit = python_path.parent / ("pre-commit.exe" if os.name == "nt" else "pre-commit")
    if not pre_commit.exists():
        print("[bootstrap] pre-commit is not installed in the environment; skipping hook setup")
        return

    run_command([str(pre_commit), "install"], description="Installing pre-commit hooks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap local development environment")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to create the virtual environment (default: current interpreter)",
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV),
        help="Target virtual environment directory (default: .venv)",
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="Skip installation of optional development dependencies.",
    )
    parser.add_argument(
        "--no-pre-commit",
        action="store_true",
        help="Skip installation of pre-commit git hooks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    venv_path = Path(args.venv).resolve()
    create_virtualenv(args.python, venv_path)

    python_in_venv = venv_python(venv_path)
    install_project(python_in_venv, include_dev=not args.no_dev)

    if not args.no_pre_commit and not args.no_dev:
        install_pre_commit(python_in_venv)
    elif args.no_pre_commit:
        print("[bootstrap] Skipping pre-commit installation as requested")

    print(
        "[bootstrap] Done. Activate the environment with 'source .venv/bin/activate' on Unix shells."
    )


if __name__ == "__main__":
    main()
