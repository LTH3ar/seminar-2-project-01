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
2. apply full cleaning, retain at most 1,000 words, and repeat the title twice;
3. evaluate all four classifiers using five duplicate-safe stratified folds;
4. rank models by cross-repository macro-F1;
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
`--title-weight`, `--n-splits`, and `--random-state`. Use `--plots` to create
per-repository and pooled confusion matrices.

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

## Current reproducible result

With the defaults above, grouped cross-validation selected logistic regression:

| Model | Cross-repository CV macro-F1 |
|---|---:|
| Logistic regression | 0.7226 |
| Linear SVM | 0.7145 |
| Random forest | 0.7095 |
| Complement Naive Bayes | 0.6855 |

The selected logistic-regression model obtained **0.7502** cross-repository
macro-F1 on the official holdout split. This is below the locally reproduced
SetFit baseline (0.8240), which is expected evidence that pretrained semantic
representations outperform these bag-of-words methods on the small dataset.
