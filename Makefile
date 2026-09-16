.PHONY: help install data eda pipeline lab test check lint clean report results setfit setfit-cv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in editable mode with dev extras
	pip install -e ".[dev]"
	python -m nltk.downloader stopwords wordnet omw-1.4

data:  ## Download and cache the NLBSE'24 dataset
	python -c "from ai4se.loader import load_dataset; \
		[print(s, len(r)) for s, r in load_dataset().items()]"

eda:  ## Execute the Track A notebook end to end
	jupyter nbconvert --to notebook --execute --inplace \
		notebooks/01_data_and_eda.ipynb \
		--ExecutePreprocessor.timeout=900

pipeline:  ## Run validation, leakage audits, cleaning and fold generation
	python examples/data_pipeline.py

results:  ## Build report-ready baseline result tables and comparison plot
	python examples/build_results_table.py

setfit:  ## Reproduce SetFit on the official train/test split
	python examples/setfit_reproduction.py --mode official

setfit-cv:  ## Evaluate SetFit with duplicate-safe grouped folds
	python examples/setfit_reproduction.py --mode cross-validation \
		--output results/evaluations/setfit-cross-validation.json

lab:  ## Start Jupyter Lab on port 8888
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=''

test:  ## Run the test suite
	python -m pytest tests/ -v

check:  ## Print the persistence-layer equivalence table for the report
	python tests/test_persistence_equivalence.py

lint:  ## Lint and format with ruff
	ruff check src tests
	ruff format src tests

report:  ## Compile the LaTeX report
	cd report && latexmk -pdf -interaction=nonstopmode report.tex

clean:  ## Remove caches and generated data (raw data re-downloads on demand)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache notebooks/.ipynb_checkpoints
	rm -f data/processed/*.csv data/processed/*.json
