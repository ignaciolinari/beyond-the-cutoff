PYTHON = .venv/bin/python
PIP = .venv/bin/pip

SCORE_DATASET ?= evaluation/datasets/offline_dataset.jsonl
SCORE_PREDICTIONS ?= evaluation/results/rag_baseline_v1/details.jsonl
SCORE_OUTPUT ?= evaluation/results/rag_baseline_v1/automatic_metrics.json
SCORE_DETAILS ?= evaluation/results/rag_baseline_v1/automatic_metrics_details.jsonl

.PHONY: init install lint format test clean score

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

score:
	BTC_USE_FAISS_STUB=1 $(PYTHON) scripts/evaluation_harness.py \
		--dataset $(SCORE_DATASET) \
		--predictions $(SCORE_PREDICTIONS) \
		--output $(SCORE_OUTPUT) \
		--details-output $(SCORE_DETAILS)
