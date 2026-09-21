# Shared evaluation workflow

Track D defines the evaluation contract used by the classical and deep-learning
tracks. A model only needs `fit(texts, labels)` and `predict(texts)`. The
factory passed to the evaluator receives the repository name and fold number,
which ensures every repository and fold receives a fresh model.

## Evaluation modes

- `evaluate_cross_validation` uses the duplicate-safe stratified group folds
  from `ai4se.folds`. It reports out-of-fold predictions for every training
  issue and never places normalized duplicate text in both sides of a fold.
- `evaluate_holdout_by_repository` follows the competition protocol: train one
  model for each of the five repositories and evaluate it on that repository's
  official test records. The test split must not be used for model selection.

Both modes record per-label precision, recall and F1, accuracy, macro and
weighted averages, confusion matrices, timing, stable issue IDs, experiment
metadata, per-repository scores, pooled scores, and the competition's
cross-repository arithmetic mean. Cross-validation results also include fold
macro-F1 mean and population standard deviation. JSON output uses schema
version 1.

## SetFit reproduction

Install the optional dependencies and run the original official protocol:

```bash
python -m pip install -e ".[setfit]"
python examples/setfit_reproduction.py --mode official
```

The defaults reproduce `notebooks/02_setfit_baseline.ipynb`: raw title plus
body, `sentence-transformers/all-mpnet-base-v2`, seed 42, batch sizes 16 and 2,
one epoch, 20 contrastive iterations, and prediction batch size 8. Run grouped
cross-validation with:

```bash
python examples/setfit_reproduction.py \
  --mode cross-validation \
  --output results/evaluations/setfit-cross-validation.json
```

SetFit downloads a pretrained model and is intentionally not run by the unit
tests. The adapter imports SetFit lazily, so data processing, result reporting,
and classical models remain usable without the deep-learning environment.

## Tables and plots

Generate report-ready tables from the supplied baseline JSON files:

```bash
python examples/build_results_table.py
```

The command accepts any number of schema-version-1 result files as positional
arguments, so results from Tracks B and C can be compared without custom
reporting code. It writes CSV and Markdown tables to `results/tables/` and a
macro-F1 chart to `results/figures/`. Each completed evaluation also supports
per-repository and pooled confusion-matrix plots through
`plot_confusion_matrices`.

When no paths are supplied, the table builder includes the supplied baselines
and every Track B `*-official-holdout.json` result under `results/classical/`.

The supplied `results/baselines/setfit.json` records a reproduced
cross-repository F1 of 0.8240. This is close to, but distinct from, the 0.8270
score published with the competition baseline; the table generator preserves
the locally reproduced value rather than substituting the published number.
