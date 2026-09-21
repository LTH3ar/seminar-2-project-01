# Project 1 — Issue Report Classification

Course project based on the [NLBSE'24 tool competition on issue report classification](https://nlbse2024.github.io/tools/).

## Task

Classify GitHub issue reports into one of three types: `bug`, `feature`, `question`.

The dataset contains 3,000 labelled issues extracted from five open-source projects — `facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`, `opencv/opencv` — collected between January 2022 and September 2023. It is split 50/50 into training and test sets, balanced at 100 issues per (project, class) cell in each split.

**The task is multi-class, not multi-label.** Issues carrying more than one label were excluded by the organisers, so every issue has exactly one class. The course brief describes the task as multi-label; this discrepancy has been raised with the lecturer.

The competition requires **one classifier per project** (five per submission). Per-repository performance is the average F1 over the three classes; the reported score is the arithmetic mean of the five per-repository F1 scores.

### Baseline to beat

| Repository | SetFit F1 |
|---|---|
| facebook/react | 0.8718 |
| tensorflow/tensorflow | 0.8644 |
| microsoft/vscode | 0.8262 |
| bitcoin/bitcoin | 0.7555 |
| opencv/opencv | 0.8173 |
| **cross-repository** | **0.8270** |

These are the published competition values. The supplied local SetFit result
reproduces a cross-repository F1 of 0.8240; Track D keeps both values explicit
instead of presenting them as the same run.

## Team and tracks

| Track | Owner | Scope |
|---|---|---|
| A — Data & persistence | *(name)* | Loading, EDA, cleaning pipeline, repository abstraction |
| B — Classical ML | *(name)* | TF-IDF + Naive Bayes, Logistic Regression, SVM, Random Forest |
| C — Deep learning | *(name)* | FFNN, CNN, fine-tuned transformer |
| D — Baseline & evaluation | *(name)* | SetFit reproduction, k-fold harness, metrics, results tables |

Tracks A, B, and D are implemented. Track B consumes the same cleaned
`IssueReport` objects produced by A and uses D's shared evaluator.

## Layout

```
.devcontainer/         Reproducible environment (see below)
src/ai4se/
    model.py           IssueReport entity, label and repository constants
    repository.py      IssueRepository (abstract) + InMemory / File implementations
    loader.py          Downloads and caches the official NLBSE'24 splits
    preprocessing.py   Markdown-aware cleaning and structural features
    eda.py             Dataset characterisation and figures
    validation.py      Strict record and split validation
    audit.py           Duplicate and train/test leakage analysis
    folds.py           Per-project stratified group folds
    classical.py       TF-IDF + NB, logistic regression, SVM, random forest
    evaluation.py      Shared metrics, grouped CV, holdout evaluation, plots
    reporting.py       CSV/Markdown result tables and model comparison plots
    setfit_baseline.py SetFit adapter matching the supplied notebook
    service.py         Unified application workflow
notebooks/
    00_dataset_extraction_reference.ipynb
    01_data_and_eda.ipynb
    02_setfit_baseline.ipynb
    03_roberta_baseline.ipynb
    04_fasttext_baseline.ipynb
    05_classical_ml.ipynb
tests/
    test_persistence_equivalence.py
    test_data_quality.py
    test_evaluation.py
data/raw/              Cached competition CSVs (git-ignored, downloaded on demand)
data/processed/        Cleaned datasets handed to tracks B, C, and D
results/baselines/     Reproduced SetFit, RoBERTa, and fastText metrics
results/classical/     Track B CV, selection, and official-holdout metrics
results/figures/       Figures referenced by the LaTeX report
results/tables/        Shared model comparison tables
docs/                  Competition reference and dataset notes
report/                LaTeX sources
```

## Persistence design

The project specification requires that data be managed both in memory and through files, and that switching between the two involve few changes to the application. This is implemented with the Repository pattern:

```
IssueRepository                 abstract interface -- the only thing the
  |                             business logic is allowed to depend on
  |-- InMemoryIssueRepository   a Python list; nothing touches the disk
  `-- FileIssueRepository       CSV or JSON on the file system
```

Generic query methods (`by_repo`, `by_label`, `label_distribution`, `texts_and_labels`, `apply`, `to_dataframe`) are implemented once on the abstract base class in terms of five primitives (`load`, `save`, `all`, `add`, `clear`), so both implementations inherit them and cannot diverge.

Switching layer is one argument:

```python
train = load_split("train", kind="memory")   # held in RAM
train = load_split("train", kind="file")     # backed by the CSV on disk
```

Everything downstream — EDA, preprocessing, vectorisation, training, evaluation — is written against `IssueRepository` and is unaffected by that choice. `tests/test_persistence_equivalence.py` runs the same analysis pipeline over both layers and asserts the results are identical; it also checks CSV and JSON round-trips are lossless.

## Cleaning pipeline

Issue bodies are Markdown documents. In the training set, 47% contain fenced
code blocks (58% of bug reports), 59% contain URLs, and 43% use a GitHub issue
template whose headings are class-independent boilerplate. Four cleaning levels
are provided so the ablation study can be run by changing one string:

| Level | Operations | Intended consumer |
|---|---|---|
| <code>conservative</code> | replaces URLs and code blocks while preserving versions, numbers, and punctuation | software-aware default |
| `raw` | whitespace normalisation only | control condition |
| `light` | removes code, stack traces, Markdown markup, URLs, paths, SHAs, mentions | transformer models |
| `full` | `light` + lowercasing, punctuation removal, stop words, lemmatisation | TF-IDF models |

Run the EDA or integrated pipeline to compare retained text across all four
levels on the current dataset.

## Getting started

### With the dev container (recommended)

The repository ships a `.devcontainer/` so every group member — and the
lecturer at the exam — runs an identical environment. Open the folder in
VS Code and accept **Reopen in Container**, or run
`devcontainer up --workspace-folder .` with the CLI. GitHub Codespaces picks
the same definition up automatically.

On first build the container:

- pins Python 3.11 and installs `requirements.txt`;
- bakes the NLTK corpora into the image, so the cleaning pipeline works offline;
- installs a LaTeX toolchain (`latexmk` + TeX Live), so `make report` compiles
  the report without a second environment;
- installs the project with `pip install -e ".[dev]"`, so `import ai4se` works
  from any directory with no `sys.path` manipulation;
- downloads and caches the competition dataset;
- runs the test suite as a smoke check.

Pip and HuggingFace caches live in named volumes and survive rebuilds, which
matters once track C starts downloading transformer checkpoints. To give the
container a GPU for that track, uncomment the `runArgs` block in
`devcontainer.json`.

### Without the dev container

```bash
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]" + NLTK corpora
```

### Common commands

```bash
make help     # list every target
make data     # download and cache the NLBSE'24 dataset
make eda      # execute the Track A notebook end to end
make pipeline # validate, audit, prepare and generate grouped folds
make classical # select Track B model by CV, then evaluate the winner once
make classical-cv # compare all classical models without touching test labels
make classical-holdout # explicit official evaluation of Track B models
make results  # build shared baseline tables and comparison plot
make setfit   # reproduce SetFit on the official test split
make setfit-cv # run duplicate-safe grouped cross-validation for SetFit
make lab      # start Jupyter Lab on port 8888
make test     # run the test suite
make check    # print the persistence-equivalence table for the report
make lint     # ruff check + format
make report   # compile report/report.tex
```

Optional dependency groups, installed per track:
`pip install -e ".[ml]"` for track B, `pip install -e ".[dl]"` for track C,
`pip install -e ".[evaluation]"` for D's reporting tools, and
`pip install -e ".[setfit]"` for the SetFit reproduction.

Minimal example:

```python
from ai4se.loader import load_split
from ai4se.preprocessing import make_cleaner

train = load_split("train", kind="memory")
train.apply(make_cleaner(level="full", max_words=400))

X, y = train.texts_and_labels()                 # ready for scikit-learn
react = train.by_repo("facebook/react")         # per-project classifier input
```

## Key EDA findings

| Finding | Value | Consequence |
|---|---|---|
| Dataset size | 1,500 train / 1,500 test | Small; favours pretrained and few-shot approaches |
| Balance | 100 per (project, class) cell | Micro- and macro-F1 coincide; resampling forbidden on the test set |
| Label cardinality | Exactly one per issue | Multi-class, not multi-label |
| Median length | ~150 words, max >21,000 | Truncation required; report the threshold as a hyperparameter |
| Code blocks | 47% overall, 58% of bugs | Cleaning may remove genuine signal — verify in the ablation |
| Cross-project vocabulary overlap | Low | Per-project classifiers justified |

## References

Kallis, Colavito, Al-Kaswan, Pascarella, Chaparro, Rani. *The NLBSE'24 Tool Competition.* NLBSE'24.

Kallis, Di Sorbo, Canfora, Panichella. *Predicting issue types on GitHub.* Science of Computer Programming 205, 2021.

Colavito, Lanubile, Novielli. *Few-Shot Learning for Issue Report Classification.* NLBSE'23.


## Shared evaluation

Track D provides one evaluation contract for Tracks B, C, and D. Models expose
`fit(texts, labels)` and `predict(texts)` while the shared evaluator handles
duplicate-safe folds, fixed label ordering, per-class metrics, macro and
weighted averages, confusion matrices, out-of-fold predictions, timing, and
the competition's mean across repositories.

The SetFit command reproduces the supplied notebook on the official split by
default. Its output uses the same versioned JSON schema as future classical and
deep-learning runs. Generate the current cross-model tables with:

    make results

See `docs/evaluation.md` for the schema, SetFit settings, evaluation protocol,
and integration instructions for other model tracks.

## Classical machine learning

Track B is implemented in `ai4se.classical`. Every model is a complete
scikit-learn pipeline containing word/character TF-IDF and one classifier, so
the vectorizer is fitted independently inside every fold. The default workflow
compares Complement Naive Bayes, logistic regression, linear SVM, and random
forest with five-fold grouped cross-validation, selects by cross-repository
macro-F1, and evaluates only the winner on the official test split.

Run it with:

    python -m pip install -e ".[ml]"
    make classical

See `docs/classical_ml.md` for hyperparameters, ablations, outputs, and the
model-selection protocol.
