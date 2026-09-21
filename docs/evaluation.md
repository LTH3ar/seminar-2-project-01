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
cross-repository arithmetic mean. When a classifier exposes `predict_proba`,
the saved prediction rows also contain aligned class scores and confidence.
Cross-validation results include fold macro-F1 mean and population standard
deviation. Models may additionally expose `training_summary()`; Track C uses
this extension to store internal split sizes, device, parameter count,
per-epoch losses and scores, best epoch, and early-stopping status. JSON output
uses schema version 1.

Track B repeats grouped cross-validation over five seeds and selects by the
mean of the competition metric: per-repository weighted F1, averaged over the
repositories. The official holdout result adds a project-stratified bootstrap
confidence interval. `ai4se.statistical_analysis` also provides exact McNemar
comparisons, Holm correction, and an approximate paired-accuracy power check.
The power result is contextual and is not presented as a formal F1 threshold.

## SetFit reproduction

Install the optional dependencies and run the original official protocol:

```bash
python -m pip install -e ".[setfit]"
python examples/setfit_reproduction.py --mode official
```

The defaults reproduce `notebooks/02_setfit_baseline.ipynb`: raw title plus
body, `sentence-transformers/all-mpnet-base-v2`, seed 42, batch sizes 16 and 2,
one epoch, 20 contrastive iterations, prediction batch size 8, and a maximum
encoder sequence length of 384 tokens. The explicit sequence cap prevents GPU
out-of-memory failures on common Colab T4 runtimes. Run grouped cross-validation
with:

```bash
python examples/setfit_reproduction.py \
  --mode cross-validation \
  --output results/evaluations/setfit-cross-validation.json
```

SetFit downloads a pretrained model and is intentionally not run by the unit
tests. The adapter imports SetFit lazily, so data processing, result reporting,
and classical models remain usable without the deep-learning environment.

## Track C integration

The FFNN, TextCNN, and DistilBERT adapters in `ai4se.deep_learning` satisfy the
same evaluator protocol. Run CPU-friendly neural models with `make deep-neural`
and transformer fine-tuning with `make deep-transformer`. DistilBERT downloads
its checkpoint on first use and should normally run on a GPU. Grouped
cross-validation is available through `make deep-cv`, but evaluates one fresh
model per repository and fold and is therefore especially expensive for the
transformer.

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
and every completed Track B or C `*-official-holdout.json` result.

The supplied `results/baselines/setfit.json` records a reproduced
cross-repository F1 of 0.8240. This is close to, but distinct from, the 0.8270
score published with the competition baseline; the table generator preserves
the locally reproduced value rather than substituting the published number.

## Error and temporal analysis

`examples/classical_ml.py --mode all` writes two additional diagnostics for
the selected model. The error-analysis JSON groups mistakes by confusion
direction and text-length quintile and lists high-confidence errors when class
probabilities are available. The temporal analysis deduplicates the combined
dataset, trains on the older half of every repository/class cell, and compares
that score with a matched random split. These artifacts are robustness checks;
the official holdout remains the primary evaluation.
