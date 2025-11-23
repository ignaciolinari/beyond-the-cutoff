#!/usr/bin/env bash
# Generate requirements lock files using pip-tools
# This ensures reproducible dependency installations

set -e

echo "Generating dependency lock files with pip-tools..."
echo ""

# Ensure pip-tools is installed
if ! python -m pip show pip-tools > /dev/null 2>&1; then
    echo "Installing pip-tools..."
    python -m pip install pip-tools
fi

# Generate lock file for main dependencies
echo "Generating requirements.txt.lock from pyproject.toml..."
pip-compile \
    --output-file=requirements.txt.lock \
    --resolver=backtracking \
    --strip-extras \
    pyproject.toml

# Generate lock file for dev dependencies
echo "Generating requirements-dev.txt.lock from pyproject.toml..."
pip-compile \
    --output-file=requirements-dev.txt.lock \
    --resolver=backtracking \
    --strip-extras \
    --extra=dev \
    pyproject.toml

echo ""
echo "Lock files generated successfully!"
echo ""
echo "To install locked dependencies:"
echo "  pip-sync requirements.txt.lock              # Production only"
echo "  pip-sync requirements-dev.txt.lock          # Development (includes production)"
echo ""
echo "To update a specific package:"
echo "  pip-compile --upgrade-package PACKAGE pyproject.toml"
echo ""
echo "To update all dependencies:"
echo "  pip-compile --upgrade pyproject.toml"
