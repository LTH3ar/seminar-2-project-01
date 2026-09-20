.PHONY: help install data eda evaluate models results notebooks experiments lab test check lint clean tree report

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

evaluate:  ## Execute the Track D evaluation notebook end to end
	jupyter nbconvert --to notebook --execute --inplace \
		notebooks/02_evaluation_protocol.ipynb \
		--ExecutePreprocessor.timeout=900

models:  ## Execute the Track B and Track C notebooks
	jupyter nbconvert --to notebook --execute --inplace \
		notebooks/03_classical_models.ipynb notebooks/04_neural_models.ipynb \
		--ExecutePreprocessor.timeout=1800

results:  ## Execute the final comparison and error-analysis notebook
	jupyter nbconvert --to notebook --execute --inplace \
		notebooks/05_final_comparison.ipynb \
		--ExecutePreprocessor.timeout=900

notebooks: eda evaluate models results  ## Execute every notebook in order

experiments:  ## Run every experiment and regenerate results/tables
	python scripts/run_experiments.py --all

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

tree:  ## Verify every expected directory exists (recreates missing ones)
	@for d in data/raw data/processed results/figures results/tables \
	          results/models results/predictions report/figures; do \
		mkdir -p $$d && touch $$d/.gitkeep && echo "  ok $$d"; \
	done

clean:  ## Remove caches and generated data (raw data re-downloads on demand)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache notebooks/.ipynb_checkpoints
	rm -f data/processed/*.csv data/processed/*.json
	@$(MAKE) --no-print-directory tree
