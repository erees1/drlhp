.DEFAULT_GOAL=all
PYTHONFILES := $(wildcard *.py)

all: deps check

deps:
	pip install --upgrade pip
	pip install -r requirements.txt
	[ -d .git ] && pre-commit install || echo "no git repo to install hooks"
check:
	black --check .
	flake8 --max-line-length=120 .
	pylint drlhp

format:
	black .
	isort --profile black **/*.py