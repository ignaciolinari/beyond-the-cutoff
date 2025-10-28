PYTHON = .venv/bin/python
PIP = .venv/bin/pip

.PHONY: init install lint format test clean

init:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e '.[dev]'

install:
	$(PIP) install -e '.[dev]'

lint:
	.venv/bin/ruff check src tests
	.venv/bin/mypy src

format:
	.venv/bin/ruff format src tests

test:
	$(PYTHON) -m pytest

clean:
	rm -rf .venv
	rm -rf dist build *.egg-info
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '.pytest_cache' -type d -exec rm -rf {} +
