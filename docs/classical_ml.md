# Track B: classical machine learning

Track B compares four conventional text classifiers under the exact evaluation
contract defined by Track D:

- Complement Naive Bayes;
- logistic regression;
- linear support-vector machine;
- random forest.

Each classifier is wrapped in one scikit-learn `Pipeline`. The first pipeline
stage is a `FeatureUnion` containing word TF-IDF (unigrams and bigrams) and,
unless disabled for an ablation, character TF-IDF (3-5 character n-grams).
Because vectorisation is part of the estimator, vocabulary and IDF values are
learned from the current training fold only.

## Experiment protocol

The default command performs the safe end-to-end protocol:

```bash
python -m pip install -e ".[ml]"
python examples/classical_ml.py --mode all
```

It performs these steps:

1. load Track A's official training split through `IssueDataService`;
2. use raw normalized text, retain at most 400 words, and repeat the title three
   times, as selected by the training-only preprocessing ablation;
3. evaluate all four classifiers with five duplicate-safe folds over five seeds;
4. rank models by the competition's cross-repository weighted-F1;
5. select the best model without reading official test labels;
6. retrain that model on each repository's full training subset;
7. evaluate it once on the corresponding official test subset;
8. save schema-version-1 JSON and report-ready CSV/Markdown tables.

Results are written to `results/classical/`, while the combined table is
written to `results/tables/classical_ml.{csv,md}`. The selection decision is
recorded in `results/classical/model-selection.json`.

To run only grouped cross-validation:

```bash
python examples/classical_ml.py --mode cross-validation
```

To evaluate explicitly chosen models on the official split after a previous
selection run:

```bash
python examples/classical_ml.py --mode official --models svm lr
```

The `--word-only` option removes character n-grams for an ablation. Other
important reproducibility options are `--cleaning-level`, `--max-words`,
`--title-weight`, `--n-splits`, and `--seeds`. Use `--plots` to create
per-repository and pooled confusion matrices. The official result also records
a project-stratified bootstrap confidence interval, error analysis, and a
matched random-versus-time-aware robustness check.

Recompute the preprocessing decision without touching the test set:

```bash
python examples/classical_ablation.py
```

## Default hyperparameters

| Component | Default |
|---|---|
| Word TF-IDF | 1-2 grams, `min_df=2`, `max_df=0.98`, 30,000 features |
| Character TF-IDF | `char_wb` 3-5 grams, `min_df=2`, 20,000 features |
| Naive Bayes | Complement NB, `alpha=0.25` |
| Logistic regression | `C=4`, LBFGS, 2,000 iterations |
| Linear SVM | `C=1.5` |
| Random forest | 400 trees, balanced subsampling |

These settings are fixed before official holdout evaluation. Model-family
selection is based only on grouped cross-validation over the training split.

The preprocessing ablation compared 36 configurations under the same grouped
folds. Its best configuration was raw normalized text, title weight 3,
400-word truncation, and word+character TF-IDF with a CV F1 of 0.7543. The old
full-cleaning configuration removed useful modal and software-specific terms.

## Current reproducible result

The five-seed grouped-CV result is:

| Model | Mean cross-repository weighted F1 | Standard deviation |
|---|---:|---:|
| Logistic regression | 0.7506 | 0.0061 |
| Linear SVM | 0.7472 | 0.0057 |
| Random forest | 0.7149 | 0.0026 |
| Complement Naive Bayes | 0.6929 | 0.0074 |

Logistic regression leads linear SVM by only 0.0034, so it is the operational
winner rather than evidence of a statistically meaningful superiority. The
selected model obtains **0.7548** cross-repository weighted F1 on the official
holdout, with a project-stratified 95% bootstrap interval of **[0.7329,
0.7763]**. It remains below the locally reproduced SetFit baseline (0.8240).

The matched random robustness split scores 0.7757, while the chronological
split scores 0.6733, a decline of 0.1024. The error analysis records 365
mistakes; the most frequent direction is bug to question, and accuracy is
lowest in the longest-text quintile. These findings support reporting the
temporal check and inspecting long reports rather than relying on one aggregate
score alone.
