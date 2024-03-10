.DEFAULT_GOAL=all
PYTHONFILES := $(wildcard *.py)

.PHONY: install
install:
	pip install -e .

.PHONY: install-dev
install-dev:
	pip install -e .[dev]
	pre-commit install
	
.PHONY: format
format:
	ruff check --fix-only .
	ruff format .

.PHONY: check
check:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

.PHONY: test
test:
	python -m pytest

.PHONY: test-all
test-all:
	python -m pytest --runslow