PYTHON ?= python3

.PHONY: install dev api mcp eval test

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

api:
	creative-ops-api

mcp:
	creative-ops-mcp

eval:
	creative-ops-eval --dataset data/golden_set.json

test:
	pytest
