.PHONY: help install install-models data pipeline lab test lint format

PYTHON ?= python

help:
	@echo "install   Install the project and development tools"
	@echo "install-models  Install dependencies for the baseline notebooks"
	@echo "data      Ensure the official train/test CSV files exist"
	@echo "pipeline  Run validation, audits, cleaning, EDA, and fold generation"
	@echo "lab       Start Jupyter Lab"
	@echo "test      Run the test suite"
	@echo "lint      Check Python style"
	@echo "format    Format Python files"

install:
	$(PYTHON) -m pip install -e ".[dev,eda]"
	$(PYTHON) -m nltk.downloader stopwords wordnet omw-1.4

install-models:
	$(PYTHON) -m pip install -e ".[models]"

data:
	PYTHONPATH=src $(PYTHON) -c "from issue_classifier import ensure_dataset; print(ensure_dataset())"

pipeline:
	PYTHONPATH=src $(PYTHON) examples/data_pipeline.py

lab:
	$(PYTHON) -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m ruff check src tests examples

format:
	$(PYTHON) -m ruff format src tests examples
