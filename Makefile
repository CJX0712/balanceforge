# BalanceForge developer shortcuts.
PY ?= python

.PHONY: help install lint test demo bench profile coverage clean

help:
	@echo "install  - install the package plus test extras"
	@echo "lint     - ruff check + format check"
	@echo "test     - run the unit tests"
	@echo "coverage - run tests with a coverage report"
	@echo "demo     - run the end-to-end demo (writes benchmark.json)"
	@echo "bench    - alias for demo"
	@echo "profile  - print imbalance geometry profiles for every dataset"
	@echo "clean    - remove caches and local artefacts"

install:
	$(PY) -m pip install -e ".[test]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -q

coverage:
	$(PY) -m pytest --cov=balanceforge --cov-report=term-missing

demo:
	$(PY) examples/run_demo.py

bench: demo

profile:
	$(PY) -m balanceforge.cli profile

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
