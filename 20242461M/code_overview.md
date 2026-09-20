# Code overview

**Project 1 — Issue Report Classification**
Artificial Intelligence for Software Engineering, academic year 2026–2027.

This document describes what has been built, why it is structured the way it
is, and where the remaining work plugs in. It is written for two audiences:
the group members picking up tracks B and C, and the lecturer reviewing the
repository before the oral.

---

## 1. Status

| Track | Scope | Status |
|---|---|---|
| A — Data & persistence | Loading, EDA, cleaning pipeline, repository abstraction | complete |
| D — Evaluation harness | Metrics, ROC/AUC, k-fold, competition protocol, floors | complete |
| B — Classical ML | TF-IDF + NB, Complement NB, LogReg, SVM, Random Forest | complete |
| C — Deep learning | FFNN, CNN, learning curves, early stopping | complete |
| D2 — Sentence transformers | Frozen-embedding baseline run; full SetFit written, needs a GPU | partial |
| — | Error analysis | complete |
| — | LaTeX report | not started |

Current size: 3,908 lines across 13 modules in `src/ai4se/`, 1,043 lines across
3 test files (65 tests, all passing), a 336-line experiment runner, and 5
notebooks that all execute end to end with zero errors. `ruff check` is clean
under a configuration that includes docstring linting.

**Blocked on hardware.** `SetFitClassifier` in `ai4se.embeddings` implements the
published baseline's contrastive fine-tuning. Running it needs a GPU;
fine-tuning an encoder five times on one CPU core is not practical. The code
path is written and its import-error path is tested.

---

## 2. The task, as the dataset actually defines it

Three corrections to the assignment brief were established by reading the
competition's own repository rather than the brief alone. All three change
how the work must be done.

**The task is multi-class, not multi-label.** The brief describes multi-label
classification. The NLBSE'24 organisers excluded every issue carrying more
than one label, so each issue has exactly one class from `{bug, feature,
question}`. A multi-label setup would be measuring something the data does not
contain. *(Open question for the lecturer: the brief may have meant Project 3,
Jigsaw toxic comments, which genuinely is multi-label.)*

**Five classifiers are required, not one.** The competition protocol trains a
separate model for each of the five repositories, scores each on its own test
issues, and reports the arithmetic mean of the five. The brief does not
mention this. It changes the training loop for tracks B and C: each model is
fitted five times on 300 issues, not once on 1,500.

**The number to beat is 0.8270.** That is the cross-repository F1 of the
official SetFit baseline, ranging from 0.7555 on `bitcoin/bitcoin` to 0.8718
on `facebook/react`. The per-repository figures are encoded in
`ai4se.evaluation.SETFIT_BASELINE` and a test asserts they average to the
published overall, so a transcription error cannot go unnoticed.

Dataset: 3,000 issues from `facebook/react`, `tensorflow/tensorflow`,
`microsoft/vscode`, `bitcoin/bitcoin` and `opencv/opencv`, collected January
2022 to September 2023, split 50/50 into train and test, balanced at exactly
100 issues per (project, class) cell in each split.

---

## 3. Architecture

```
                    ┌──────────────────────────────┐
                    │   notebooks/ (01, 02)        │   presentation
                    └──────────────┬───────────────┘
                                   │
   ┌──────────┬──────────┬─────────┼─────────┬──────────┬──────────┐
   │          │          │         │         │          │          │
 eda.py  preprocessing  classical  neural  embeddings  baselines  error_
   │          .py         .py      .py       .py         .py    analysis.py
   │          │          └────┬────┴─────┬────┘          │          │
   │          │        evaluation.py  metrics.py         │          │
   └──────────┴───────────────┴───────────┴──────────────┴──────────┘
                                   │                 business logic
                    ┌──────────────▼───────────────┐
                    │      IssueRepository         │   abstract interface
                    │  (the only dependency        │
                    │   business logic may have)   │
                    └──────────────┬───────────────┘
                     ┌─────────────┴─────────────┐
          InMemoryIssueRepository        FileIssueRepository
                  (RAM)                    (CSV / JSON)         persistence
```

The dependency arrow points one way only: persistence knows nothing about
analysis, and analysis knows nothing about storage. `loader.py` sits beside
this as the acquisition step, producing a repository of whichever kind is
requested.

`model.py` defines the single entity everything passes around and depends on
nothing at all.

### External dependencies, by layer

The heavy dependencies are confined to the model modules. The data and
evaluation layers stay usable without any of them, which is why
`import ai4se` never pulls in torch.

| Layer | Modules | Requires |
|---|---|---|
| Domain, persistence | `model`, `repository`, `loader` | standard library only |
| Preprocessing, EDA | `preprocessing`, `eda` | pandas, matplotlib, nltk (lazily) |
| Metrics, protocol | `metrics`, `evaluation` | pandas |
| Reference floors | `baselines` | standard library only |
| Track B | `classical` | scikit-learn |
| Track C | `neural` | torch |
| Track D2 | `embeddings` | sentence-transformers (+ `setfit` for the full run) |

`metrics.py` and `evaluation.py` deliberately do **not** import scikit-learn:
precision, recall, F1, ROC and AUC, and the stratified fold splitter are all
implemented from scratch, so the evaluation layer has no dependency on any
modelling library. scikit-learn appears in `classical.py` as a model provider,
and in the test suite as the oracle those implementations are checked against.
`ai4se/__init__.py` re-exports only the light layers; Tracks B, C and D2 are
imported from their submodules on demand. `import ai4se` therefore works in an
environment with none of scikit-learn, torch or sentence-transformers
installed, and `test_core_package_has_no_heavy_dependencies` parses the source
to enforce it. With only `.[dev]` installed the suite reports 55 passed,
8 skipped.

---

## 4. Modules

### `model.py` (103 lines)

`IssueReport`, a frozen-ish dataclass with `slots`, plus the `LABELS` and
`REPOSITORIES` constants. Carries `repo`, `created_at`, `label`, `title`,
`body` and `clean_text`, with `raw_text`/`text`/`word_count` properties and an
`is_valid()` check. `__post_init__` normalises the NaN values the raw CSV
yields for missing bodies, which would otherwise break every downstream string
operation.

No I/O, no persistence awareness. The same object is produced by both
repository implementations.

### `repository.py` (321 lines)

The persistence layer, and the module that satisfies the project's explicit
non-functional requirement.

`IssueRepository` is an abstract base class with **five primitives** —
`load`, `save`, `all`, `add`, `clear` — and roughly a dozen query helpers
(`by_repo`, `by_label`, `label_distribution`, `texts_and_labels`, `filter`,
`apply`, `to_dataframe`, `__len__`, `__iter__`, `__getitem__`) implemented
*once* on the base class in terms of those primitives. Concrete
implementations therefore cannot drift apart: adding a query helper adds it to
both at the same time.

`InMemoryIssueRepository` wraps a list; `load` and `save` are deliberate
no-ops. `FileIssueRepository` reads and writes CSV or JSON, inferring the
format from the suffix, and caches after first load.

`make_repository(kind, ...)` is the factory that hides the concrete class.
Switching the entire application's persistence layer is one argument:

```python
train = load_split("train", kind="memory")   # held in RAM
train = load_split("train", kind="file")     # backed by the CSV on disk
```

One implementation detail worth knowing: `csv.field_size_limit` is raised at
import, because one issue body in the training set exceeds 21,000 words and
Python's default limit rejects it.

### `loader.py` (112 lines)

Downloads the two official CSVs from the competition's GitHub repository,
caches them under `data/raw/`, and hands them to the persistence layer.
`_project_root()` resolves the cache directory relative to the package rather
than the working directory — without it, running a notebook from
`notebooks/` silently created a second copy of the dataset.

### `preprocessing.py` (301 lines)

Markdown-aware cleaning. Issue bodies are not plain prose: they contain fenced
code blocks, stack traces, URLs, images, HTML comments and GitHub issue
template headings, all of which dilute a bag-of-words vocabulary.

Each step is a pure `str -> str` function, so the pipeline can be reordered or
partially disabled for the ablation study. Three levels, switchable by one
string:

| Level | Operations | Intended consumer |
|---|---|---|
| `raw` | whitespace normalisation only | control condition |
| `light` | strips code, stack traces, markup, URLs, paths, SHAs, mentions | transformer models |
| `full` | `light` + lowercase, punctuation, stop words, lemmatisation | TF-IDF models |

Average retained length: 100% → 63% → 33% of the original word count.
`make_cleaner(**kwargs)` returns a one-argument function for
`IssueRepository.apply`.

The stop word list is NLTK's English list plus a domain list of terms that
appear in nearly every issue regardless of class (`issue`, `repro`,
`expected`, `screenshot`, …) and therefore carry no discriminative signal.

### `eda.py` (266 lines)

Dataset characterisation. Every function takes an `IssueRepository` and is
indifferent to where the data lives — the same analysis code runs unchanged
against both layers, which is the practical demonstration that the
abstraction works.

`overview`, `label_distribution`, `length_statistics`, `structural_noise`,
`cleaning_impact`, `top_terms`, `distinctive_terms`, plus two figure
functions.

`distinctive_terms` ranks by **document frequency** (how many issues contain a
term) rather than raw token count. The first implementation used raw counts
and produced nonsense: a handful of enormous log dumps flooded the table with
vocabulary from single issues.

### `metrics.py` (412 lines)

Precision, recall, F1, confusion matrix, and macro/micro/weighted averaging,
implemented from first principles.

Three reasons for not importing scikit-learn here: the evaluation layer then
has no modelling-library dependency; the definitions match exactly what the
course covered; and deriving these from a confusion matrix is what the oral
exam asks about. `tests/test_evaluation.py` verifies every function against
scikit-learn to twelve decimal places, which is what makes the choice
defensible rather than merely stubborn.

ROC and one-vs-rest AUC are included, as the course's metric list requires.
A model that cannot produce probabilities — `LinearSVC`, the keyword rules —
reports `None` rather than an AUC faked from hard labels.

`Scores` carries per-class results, all three averages, optional AUC, the
confusion matrix and the sample count, with `to_frame()`, `confusion_frame()` and `as_dict()`
for reporting and serialisation. Undefined precision (a class never predicted)
returns 0.0, matching scikit-learn's `zero_division=0` convention.

### `evaluation.py` (725 lines)

The protocol. Two measurements that are **not interchangeable**:

**`cross_validate` / `cross_validate_per_project`** — stratified k-fold on the
*training* split, for model selection and hyperparameter tuning.
`stratified_folds` shuffles each class independently and deals it round-robin
into the k folds, so every fold holds the class proportions of the whole to
within one item. Seeded at 42, so every member gets identical folds and
identical numbers.

**`evaluate_competition`** — the official protocol on the *test* split, used
exactly once. One classifier per repository, each scored as the average F1
over the three classes, reported as the arithmetic mean of the five. This is
the only number comparable with the published baseline.

Models are supplied as a **factory** — a zero-argument callable returning a
fresh untrained model — because each fold and each project needs its own
instance. Any object with `fit(X, y)` and `predict(X)` satisfies the `Model`
protocol.

`grid_search()` cross-validates a hyperparameter grid on the training split,
and `class_probabilities()` normalises scikit-learn's `predict_proba` array and
the project's own dictionary form into one shape so AUC works for both.

`leaderboard_from_disk()` rebuilds the master table from saved JSON — and
raises on duplicate model names, which caught a stale result file that would
otherwise have appeared twice.

`leaderboard()` builds the master comparison table with the SetFit baseline
included as a row. `to_latex()` renders it as a LaTeX `table` environment, and
`save_result()` writes JSON, so the report can be rebuilt without re-training
and the numbers in the report cannot drift from the numbers the code produced.

### `classical.py` (240 lines) — Track B

Five scikit-learn `Pipeline` factories over a shared TF-IDF representation, so
differences in score come from the classifier rather than the features.

The vectoriser is fitted **inside** each pipeline, which means inside each
cross-validation fold. Fitting it once over the whole dataset and then
cross-validating would leak test-fold vocabulary and document frequencies into
training and inflate every score; it is the easiest way to accidentally cheat
at this task, and `test_vectorizer_is_fitted_inside_each_fold` guards against
it.

`TUNED_PREPROCESSING` and the `tuned_*` factories record the configuration the
ablation and grid search selected.

### `neural.py` (474 lines) — Track C

`NeuralClassifier` holds the training loop, early stopping and history
tracking; `FeedForwardClassifier` (TF-IDF input) and `CNNTextClassifier`
(learned embeddings, parallel filter widths, global max-pooling) supply the
features and the architecture.

`TrainingHistory` records training loss, validation loss and validation
accuracy per epoch, which the course explicitly requires, and
`plot_learning_curves` renders the validation curve used to diagnose
overfitting. Early stopping restores the weights from the best epoch rather
than the last.

### `embeddings.py` (263 lines) — Track D2

`FrozenEmbeddingClassifier` encodes with a frozen Sentence Transformer and
fits a logistic-regression head — SetFit with the contrastive fine-tuning
removed, which is what makes the ablation interpretable.
`SetFitClassifier` is the full reproduction, needing the `setfit` extra and a
GPU. Embeddings are cached per (model, text), because cross-validation would
otherwise re-encode the same 1,500 issues once per fold for no benefit.

### `error_analysis.py` (299 lines)

`collect_predictions` mirrors the competition protocol but keeps individual
predictions instead of collapsing them to scores, and a test asserts the two
agree. On top of it: `confusion_summary`, `errors_by_length`,
`misleading_terms` and `worst_mistakes`, which ranks errors by model
confidence — confident errors are the informative ones.

### `baselines.py` (273 lines)

Four reference classifiers with no third-party dependencies:
`MajorityClassifier`, `StratifiedRandomClassifier`, `KeywordClassifier` (cue
words drawn from the Track A term analysis) and `MultinomialNaiveBayes`
(implemented from scratch, log-space, Laplace smoothing).

They exist so the harness was testable before any real model existed, and so
the report has a floor to quote. "Our SVM reaches 0.79" means nothing until
the reader knows that always predicting one class reaches 0.17.

---

## 5. Design decisions and their rationale

These are the points likely to come up at the oral.

**Repository pattern for persistence.** The specification requires data to be
manageable both in memory and through files, with few changes to the
application when switching. Implementing the query helpers once on the
abstract base class — rather than twice, once per concrete class — is what
makes "few changes" structurally guaranteed rather than a promise.

**Factories rather than model instances.** Cross-validation fits k models and
the competition protocol fits five; passing a trained instance would leak
state between folds and silently inflate every score.

**Metrics from scratch, verified against scikit-learn.** Independence from a
modelling library, plus the ability to explain the arithmetic. The
verification test is the part that matters.

**Seeded everything.** `RANDOM_SEED = 42` in one place, used by the splitter
and the stochastic baseline. Two members running the same code on different
machines get identical numbers, which matters when four people are comparing
results.

**Test split touched once.** Cross-validation on the training half drives
every decision; `evaluate_competition` runs when a model is final. Selecting
on the test split would make the baseline comparison meaningless.

**Package installed editable, not `sys.path`-hacked.** `pyproject.toml` plus
`pip install -e ".[dev]"` means `import ai4se` works from any directory —
notebooks, tests, the shell — without path manipulation.

**`zip(..., strict=True)` where lengths must match.** In the metrics and the
naive Bayes fit, a length mismatch between texts and labels would silently
truncate and corrupt every score. `strict=True` turns that into an exception.

---

## 6. Tests

65 tests, all passing, in three files (55 pass and 8 skip in an environment
without torch or sentence-transformers).

`test_persistence_equivalence.py` (7) is the evidence for the graded
non-functional requirement: it runs the *same* analysis pipeline over both
persistence layers and asserts identical results, verifies CSV and JSON
round-trips are lossless, and confirms both implementations satisfy the full
abstract interface. Run standalone it prints a comparison table for the report.

It also guards the public API: `test_public_api_resolves` checks every name in
`__all__` actually exists, added after the list drifted from the import block
and advertised eighteen names the package did not expose — counting the entries
is not the same as checking they resolve. `test_core_package_has_no_heavy_
dependencies` parses the source to keep the optional extras optional.

`test_evaluation.py` (31) covers hand-computed metric values, fold
partitioning, stratification and determinism, the competition protocol's
arithmetic, and the self-consistency of the published baseline constants. Three
tests **cross-check against scikit-learn** — every classification metric, the
fold splitter's proportions, and one-vs-rest AUC — and the AUC comparison
agrees to 1.1e-16.

`test_models.py` (27) covers the model tracks and error analysis. Torch and
sentence-transformers tests skip cleanly when those packages are absent. The
notable ones:

- `test_vectorizer_is_fitted_inside_each_fold` — guards the leakage that would
  inflate every score. It compares using the vectoriser's *own* analyzer,
  because a naive whitespace split tokenises differently and comparing across
  the two compares noise.
- `test_parameterised_factory_does_not_capture_by_reference` — the loop-variable
  bug that would evaluate every grid configuration with the last one's values.
- `test_early_stopping_triggers_and_restores_best_weights`.
- `test_collected_predictions_agree_with_the_harness` — the error analysis must
  describe the same run the leaderboard reports, not a separate one.
- `test_tuned_configuration_is_reproducible` — which caught a real drift when
  the ablation changed the preprocessing and invalidated the recorded grid
  result.

## 7. Results

### Final leaderboard

Cross-repository F1 on the official test split, under the competition protocol
(one classifier per repository, mean of the five).

| Model | react | tensorflow | vscode | bitcoin | opencv | overall | AUC | vs SetFit |
|---|---|---|---|---|---|---|---|---|
| **SetFit (NLBSE'24 baseline)** | 0.8718 | 0.8644 | 0.8262 | 0.7555 | 0.8173 | **0.8270** | — | — |
| TF-IDF + Logistic Regression (tuned) | 0.8334 | 0.8155 | 0.7234 | 0.6793 | 0.7496 | **0.7603** | 0.9018 | −0.0667 |
| TF-IDF + Linear SVM | 0.8247 | 0.8112 | 0.7229 | 0.6757 | 0.7569 | 0.7583 | — | −0.0687 |
| CNN (embeddings) | 0.8187 | 0.8484 | 0.6862 | 0.7050 | 0.7308 | 0.7578 | 0.8904 | −0.0692 |
| TF-IDF + Logistic Regression | 0.8340 | 0.8089 | 0.7163 | 0.6793 | 0.7497 | 0.7576 | 0.9016 | −0.0694 |
| TF-IDF + Linear SVM (tuned) | 0.8002 | 0.8021 | 0.7332 | 0.6854 | 0.7448 | 0.7531 | — | −0.0739 |
| FFNN (TF-IDF) | 0.8175 | 0.7625 | 0.7286 | 0.6791 | 0.7457 | 0.7467 | 0.8926 | −0.0803 |
| TF-IDF + Naive Bayes | 0.7944 | 0.7646 | 0.6833 | 0.6691 | 0.6587 | 0.7140 | 0.8713 | −0.1130 |
| Frozen MiniLM + LogReg | 0.7177 | 0.7024 | 0.6908 | 0.6942 | 0.7202 | 0.7050 | 0.8837 | −0.1220 |
| TF-IDF + Complement NB | 0.7660 | 0.7602 | 0.6515 | 0.6765 | 0.6516 | 0.7012 | 0.8673 | −0.1258 |
| TF-IDF + Random Forest | 0.7270 | 0.8489 | 0.6695 | 0.6312 | 0.6046 | 0.6962 | 0.8630 | −0.1308 |
| Naive Bayes (from scratch) | 0.7914 | 0.6466 | 0.6609 | 0.6030 | 0.6375 | 0.6679 | 0.8246 | −0.1591 |
| Keyword rules | 0.5712 | 0.6239 | 0.4582 | 0.4945 | 0.5498 | 0.5395 | — | −0.2875 |
| Random (stratified) | 0.3428 | 0.3194 | 0.3428 | 0.3365 | 0.3326 | 0.3348 | — | −0.4922 |
| Majority class | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | — | −0.6603 |

**Best result: 0.7603**, about 79% of the distance from a majority-class
classifier to the published baseline. The remaining 0.067 is real and is not
closed by anything implemented here.

### Findings

**Aggressive cleaning hurts.** `full` cleaning — stop words plus lemmatisation
— scores roughly 0.015 macro F1 *below* `light`, consistently across every
title weight and truncation setting. Stop-word removal deletes *would*,
*could*, *should* and *please*, which are precisely the modal words Track A's
own term analysis identified as the signature of a feature request. Track A was
built on the opposite assumption; only the ablation exposed it.

**Frozen sentence embeddings underperform TF-IDF.** A pretrained MiniLM encoder
with a logistic-regression head reaches 0.7050, below a plain TF-IDF pipeline
at 0.7603. Since that is exactly SetFit minus the contrastive fine-tuning step,
the fine-tuning accounts for roughly 0.12 F1 on its own. **The pretrained
representation is not what makes the baseline strong — the adaptation is.**
This is the most useful single result in the project and it is only visible
because the ablated variant was run.

**Neither neural model beats TF-IDF.** The CNN reaches 0.7578 and the FFNN
0.7467, against 0.7603 for tuned logistic regression. With 300 training issues
per project there is not enough data to learn a better representation than
TF-IDF already supplies, and the CNN must learn its embeddings from scratch.

**Model family matters less than data size.** Every reasonable model lands
between 0.70 and 0.76. The spread within Track B is comparable to the spread
between Tracks B, C and D2.

**Per-project training beats a global classifier** by 0.068 on all five
repositories despite using a fifth of the data, confirming the low
cross-project vocabulary overlap measured in the EDA.

**`bug` misclassified as `question` is the dominant error**, over a quarter of
all mistakes. Fourteen of the twenty-five most confident errors have `question`
as their true label.

**About a third of confident errors look like label noise.** Nine of the
twenty-five have a title explicitly declaring a different type than the
ground-truth label — an issue titled `[Feature Request] Add workbench action to
split editor terminal below` is labelled `question` — and the model usually
agrees with the title rather than the label. Labels come from maintainers
applying project conventions, not an adjudicated annotation protocol, so a
ceiling below 1.0 is built into the dataset. The published baseline faces the
same ceiling.

**Accuracy falls at both length extremes**, from 0.80 in the middle quintile to
0.71 on the longest. Very short issues carry too little text; very long ones
are diluted and partly truncated.

**Tuning gained almost nothing.** Across the grid, the best and fifth-best
configurations differ by about 0.002 macro F1 against a fold-to-fold standard
deviation of roughly 0.029. Only the bigram setting clearly exceeds the noise.
Reporting this honestly is better than implying the search mattered.

### A process note worth reporting

The grid search was first run under `full` cleaning. When the ablation moved
the pipeline to `light`, the winning `min_df` changed — so the recorded "tuned"
configuration was briefly stale, and the untuned defaults outscored it.
Hyperparameters and preprocessing are not independent, and re-tuning after
changing the pipeline is not optional. `test_tuned_configuration_is_reproducible`
now fails loudly if the recorded configuration drifts from what the factories
build.

## 8. Reproducing everything

One entry point regenerates every number:

```bash
python scripts/run_experiments.py --list     # the seven stages
python scripts/run_experiments.py --all      # roughly 6 minutes on one CPU core
make notebooks                               # re-execute all five notebooks
```

Stages are independent and skip work already on disk unless `--force` is given,
so an interrupted run resumes cheaply. Results land in `results/tables/*.json`
and figures in `results/figures/`; the notebooks read those rather than
re-training, so the report can be rebuilt without a GPU and its numbers cannot
drift from what the code produced.

### Adding a model

Anything with `fit(X, y)` and `predict(X)`, supplied as a zero-argument
factory. `predict_proba` is optional and adds ROC/AUC automatically.

```python
from ai4se.classical import TUNED_PREPROCESSING
from ai4se.evaluation import cross_validate, evaluate_competition, save_result
from ai4se.loader import load_split
from ai4se.preprocessing import make_cleaner

train = load_split("train"); train.apply(make_cleaner(**TUNED_PREPROCESSING))
test  = load_split("test");  test.apply(make_cleaner(**TUNED_PREPROCESSING))

cv = cross_validate(my_factory, train, k=10, model_name="My model")   # selection
final = evaluate_competition(my_factory, train, test, model_name="My model")
save_result(final, "results/tables/competition_my_model.json")
```

### Rules

1. Select on cross-validation; touch the test split once, when the model is final.
2. Keep `RANDOM_SEED = 42`.
3. Fit vectorisers **inside** the pipeline, never on the full dataset beforehand.
4. `save_result(...)` every run, so the leaderboard rebuilds from disk.
5. Report the spread (`std_macro_f1`), not only the mean.
6. Do not write your own metrics — use `ai4se.metrics`, so every leaderboard row
   is computed identically.

## 9. Environment and reproducibility

`.devcontainer/` builds from the official Docker Hub `python:3.11-slim-trixie`
(Debian 13, multi-arch amd64 + arm64) — not a vendor devcontainer image, and
with no `features` block. The non-root user, git, sudo and the VS Code server
prerequisites are installed explicitly, so the only upstream layer is the
official Python image. NLTK corpora are baked into the image so the cleaning
pipeline works offline; a LaTeX toolchain is included (toggleable via the
`INSTALL_LATEX` build arg, since it adds ~1 GB).

`post-create.sh` installs the package editable, verifies the corpora, caches
the dataset and runs the test suite as a smoke check.

`make help` lists every task: `data`, `eda`, `evaluate`, `notebooks`, `lab`,
`test`, `check`, `lint`, `tree`, `clean`, `report`.

Raw and processed data are git-ignored and regenerate on demand, so the
repository stays small. Every directory that can legitimately be empty carries
a `.gitkeep`, and the final rule in `.gitignore` re-includes those
placeholders even inside ignored directories, so a fresh clone reproduces the
whole tree.

---

## 10. Remaining work

**Needs hardware.** The full SetFit reproduction (`SetFitClassifier`) and any
transformer fine-tuning need a GPU. Given that frozen embeddings score 0.705
and the published baseline reaches 0.827, the contrastive step is the missing
0.12 and reproducing it is the obvious next experiment. `pip install -e
".[setfit]"`, then add it to the `embeddings` stage of the runner.

**Cheap and worthwhile without hardware.**

- Strip issue-template boilerplate more aggressively. The error analysis shows
  the model learning OpenCV's template rather than its content.
- An ensemble of the tuned linear models and the CNN; their per-repository
  profiles differ noticeably, so they are making different mistakes.
- Language detection, to quantify rather than anecdote the non-English issues.

**Not code.**

- The LaTeX report in `report/`, drawing on `results/tables/*.tex` and
  `results/figures/`.
- A per-member contribution statement.
- Enough commit history from each member for the lecturer to attribute work at
  the oral. This is graded and currently reflects one person.
- The multi-class/multi-label clarification from the lecturer.