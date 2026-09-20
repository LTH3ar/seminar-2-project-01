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
| D — Evaluation harness | Metrics, k-fold, competition protocol, reference floors | complete |
| B — Classical ML | TF-IDF + Naive Bayes, Logistic Regression, SVM, Random Forest | not started |
| C — Deep learning | FFNN, CNN, fine-tuned transformer | not started |
| D2 — SetFit reproduction | Reproduce the published baseline end to end | not started |
| — | LaTeX report | not started |

Current size: 2,117 lines across 9 modules in `src/ai4se/`, 483 lines across
2 test files (28 tests, all passing), 2 executed notebooks, and a dev
container. `ruff check` is clean under a configuration that includes docstring
linting.

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
      ┌───────────────┬────────────┼────────────┬──────────────┐
      │               │            │            │              │
   eda.py      preprocessing.py  evaluation.py  metrics.py  baselines.py
      │               │            │                            │
      └───────────────┴────────────┼────────────────────────────┘
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

### External dependencies

`src/` imports only **pandas**, **matplotlib** and **nltk** — and nltk only
lazily, inside the functions that need it, with a built-in fallback word list
so the pipeline still runs offline. Everything else is the standard library.

Notably, `src/` does **not** import scikit-learn. The metrics and the fold
splitter are implemented from scratch, so the evaluation layer has no
dependency on any modelling library. scikit-learn appears only in the test
suite, where it is used to verify those implementations.

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

### `metrics.py` (253 lines)

Precision, recall, F1, confusion matrix, and macro/micro/weighted averaging,
implemented from first principles.

Three reasons for not importing scikit-learn here: the evaluation layer then
has no modelling-library dependency; the definitions match exactly what the
course covered; and deriving these from a confusion matrix is what the oral
exam asks about. `tests/test_evaluation.py` verifies every function against
scikit-learn to twelve decimal places, which is what makes the choice
defensible rather than merely stubborn.

`Scores` carries per-class results, all three averages, the confusion matrix
and the sample count, with `to_frame()`, `confusion_frame()` and `as_dict()`
for reporting and serialisation. Undefined precision (a class never predicted)
returns 0.0, matching scikit-learn's `zero_division=0` convention.

### `evaluation.py` (468 lines)

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

`leaderboard()` builds the master comparison table with the SetFit baseline
included as a row. `to_latex()` renders it as a LaTeX `table` environment, and
`save_result()` writes JSON, so the report can be rebuilt without re-training
and the numbers in the report cannot drift from the numbers the code produced.

### `baselines.py` (248 lines)

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

28 tests, all passing, in two files.

`test_persistence_equivalence.py` (5 tests) is the evidence for the graded
non-functional requirement. It runs the *same* analysis pipeline over both
persistence layers and asserts the results are identical, verifies CSV and
JSON round-trips are lossless, confirms both implementations satisfy the full
abstract interface, and checks the dataset's expected shape. Run standalone,
it prints a comparison table intended for the report.

`test_evaluation.py` (23 tests) covers:

- hand-computed confusion matrix and metric values;
- **cross-checks against scikit-learn** for every metric and for the fold
  splitter's proportions;
- `micro F1 == accuracy` for single-label problems;
- fold partitioning, stratification, determinism, and rejection of invalid `k`;
- the `fit`/`predict` contract of all four reference models;
- the competition protocol's shape (5 repositories × 300 test issues, the
  mean-of-means arithmetic);
- self-consistency of the published baseline constants;
- that the evaluation harness, like the EDA, is indifferent to the
  persistence layer.

---

## 7. Results so far

### Reference floors

| Model | CV macro F1 | Test F1 (competition) | vs SetFit |
|---|---|---|---|
| Majority class | 0.1667 ± 0.0000 | 0.1667 | −0.660 |
| Random (stratified) | 0.3626 ± 0.0145 | 0.3348 | −0.492 |
| Keyword rules | 0.5107 ± 0.0340 | 0.5395 | −0.288 |
| Naive Bayes (from scratch) | 0.6009 ± 0.0461 | 0.6679 | −0.159 |
| **SetFit (NLBSE'24 baseline)** | — | **0.8270** | — |

A from-scratch naive Bayes on cleaned text closes roughly three-quarters of
the distance from zero to the baseline. **0.159 remains** for tracks B and C.
Anything below 0.54 is not beating hand-written keyword rules and should be
treated as a bug rather than a result.

### Findings worth reporting

**Per-project training beats a global classifier.** Trained per project on 300
issues each, naive Bayes reaches 0.6679; trained once on all 1,500 and scored
per project, it reaches 0.6003. Per-project wins on **all five** repositories,
by 0.068 on average and by 0.196 on `facebook/react`, despite using a fifth of
the data. This independently confirms the low cross-project vocabulary overlap
measured in the EDA, and shows the competition's protocol is the right
modelling choice rather than an arbitrary constraint.

**Structural noise is class-correlated.** 47% of issues contain fenced code
blocks, but 58% of bugs against 35% of features. Stripping code removes noise
and signal at the same time, so the ablation must test both conditions rather
than assume cleaning helps.

**Length is extremely skewed.** Median 150 words, maximum 21,595. Truncation
is not optional, and the threshold should be reported as a hyperparameter.

**Feature requests are lexically distinctive.** Their characteristic terms are
modal and evaluative (*allow*, *nice*, *considered*, *easily*, *approach*)
rather than technical, which is exactly the signal a lexical model can
exploit. Bugs and questions are dominated by project-specific technical
vocabulary — hence the per-project result above.

**Macro and micro F1 coincide on this dataset** because the splits are exactly
balanced. Both are reported anyway: if any resampling or filtering is
introduced during development, they will diverge, and that divergence is the
signal that something changed.

---

## 8. How tracks B and C plug in

Any object with `fit(X, y)` and `predict(X)`, passed as a factory:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from ai4se.evaluation import cross_validate, evaluate_competition, save_result
from ai4se.loader import load_split
from ai4se.preprocessing import make_cleaner

train = load_split("train"); train.apply(make_cleaner(level="full", max_words=400))
test  = load_split("test");  test.apply(make_cleaner(level="full", max_words=400))

def svm_factory():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LinearSVC())

# model selection -- run as often as needed
cv = cross_validate(svm_factory, train, k=10, model_name="TF-IDF + LinearSVC")
print(cv.mean_macro_f1, cv.std_macro_f1)

# final number -- run once, when the model is final
final = evaluate_competition(svm_factory, train, test, model_name="TF-IDF + LinearSVC")
save_result(final, "results/tables/competition_svm.json")
```

For a transformer, wrap fine-tuning in the same two methods and it drops into
exactly these calls.

### Rules for everyone

1. Select on cross-validation; report on the test split, once.
2. Keep `RANDOM_SEED = 42`.
3. `save_result(...)` into `results/tables/` for every run, so the report
   rebuilds without re-training.
4. Report the spread (`std_macro_f1`), not only the mean.
5. Do not write your own metrics — use `ai4se.metrics` so every row of the
   leaderboard is computed identically.

---

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

## 10. Not done yet

- **Track B**: TF-IDF pipelines for Naive Bayes, Logistic Regression, SVM and
  Random Forest, with hyperparameter tuning over the cross-validation harness.
- **Track C**: FFNN and CNN over embeddings, then transformer fine-tuning.
  Training/validation loss curves per epoch are expected in the report.
- **Track D2**: reproducing the SetFit baseline end to end, to confirm the
  published numbers are reachable in this environment.
- **Ablations**: `raw` vs `light` vs `full` cleaning, title-only vs
  title + body, truncation threshold.
- **Error analysis**: a sample of misclassified issues, categorised.
- **Report**: LaTeX sources in `report/`, drawing on the generated tables in
  `results/tables/` and figures in `results/figures/`.
- **Per-member contribution statement**, and enough commit history from each
  member for the lecturer to attribute work at the oral.
