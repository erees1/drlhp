.DEFAULT_GOAL=all
PYTHONFILES := $(wildcard *.py)

all: env check

.PHONY: env
env:
	pip install --upgrade pip
	pip install -r requirements.txt
	[ -d .git ] && pre-commit install || echo "no git repo to install hooks"

.PHONY: check
check:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit