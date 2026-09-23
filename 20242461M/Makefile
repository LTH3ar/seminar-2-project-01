.PHONY: help install install-core install-gpu venv data eda evaluate models results notebooks experiments lab test check lint clean tree report

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

venv:  ## Create a virtual environment (alternative to the dev container)
	python3 -m venv .venv
	@echo
	@echo "  Now activate it and install:"
	@echo "    source .venv/bin/activate      # Windows: .venv\\Scripts\\activate"
	@echo "    make install"

install:  ## Install everything: all tracks + dev tools + NLTK corpora
	pip install -e ".[all]"
	python -m nltk.downloader stopwords wordnet omw-1.4
	@echo "\n  Installed. Verify with: make test"

install-core:  ## Install without torch or sentence-transformers (Tracks A, B, D)
	pip install -e ".[dev,ml]"
	python -m nltk.downloader stopwords wordnet omw-1.4
	@echo "\n  Tracks C and D2 will skip. Add them later with: make install"

install-gpu:  ## Install with the CUDA build of torch and the SetFit extra
	pip install -e ".[all,setfit]"
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
	rm -rf checkpoints runs wandb
	rm -f data/processed/*.csv data/processed/*.json
	@$(MAKE) --no-print-directory tree
