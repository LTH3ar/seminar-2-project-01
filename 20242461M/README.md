# Project 1 — Issue Report Classification

Artificial Intelligence for Software Engineering, academic year 2026–2027.
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

For a full technical walkthrough of the codebase — architecture, module
reference, design decisions, results and open work — see
[`code_overview.md`](code_overview.md).

## Team and tracks

| Track | Owner | Scope | Status |
|---|---|---|---|
| A — Data & persistence | *(name)* | Loading, EDA, cleaning pipeline, repository abstraction | done |
| B — Classical ML | *(name)* | TF-IDF + NB, LogReg, SVM, Random Forest; ablation, grid search | done |
| C — Deep learning | *(name)* | FFNN, CNN, learning curves, early stopping | done |
| D — Evaluation harness | *(name)* | Metrics, ROC/AUC, k-fold, competition protocol, floors | done |
| D2 — SetFit reproduction | *(name)* | Frozen-embedding baseline done; full SetFit needs a GPU | partial |

Track A's outputs under `data/processed/` are the agreed input for every other
track. Track D's runners in `ai4se.evaluation` are how every model is scored —
B and C plug into them rather than writing their own metrics.

## Layout

```
.devcontainer/           Reproducible environment (see .devcontainer/README.md)
.dockerignore            Keeps the Docker build context small
pyproject.toml           Packaging; `pip install -e ".[dev]"`
Makefile                 Task shortcuts, run `make help`
src/ai4se/
    model.py             IssueReport entity, label and repository constants
    repository.py        IssueRepository (abstract) + InMemory / File implementations
    loader.py            Downloads and caches the official NLBSE'24 splits
    preprocessing.py     Markdown-aware cleaning pipeline (raw / light / full)
    eda.py               Dataset characterisation and figures
    metrics.py           Precision, recall, F1, confusion matrix (from scratch)
    evaluation.py        Stratified k-fold + the competition protocol
    baselines.py         Reference floors, no third-party dependencies
notebooks/
    01_data_and_eda.ipynb        Track A
    02_evaluation_protocol.ipynb Track D
tests/
    test_persistence_equivalence.py
    test_evaluation.py
data/
    raw/                 Cached competition CSVs (ignored, downloaded on demand)
    processed/           Cleaned datasets handed to tracks B, C, D (ignored)
results/
    figures/             Committed -- referenced by the LaTeX report
    tables/              Committed -- results tables for the report
    models/              Ignored -- checkpoints are large and reproducible
    predictions/         Ignored -- raw prediction dumps
report/
    figures/             Figures copied in for the LaTeX build
```

Every directory that can legitimately be empty carries a `.gitkeep`, and the
final rule in `.gitignore` re-includes those placeholders even inside ignored
directories. A fresh `git clone` therefore reproduces the whole tree, so no
member has to guess where their outputs belong.

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

Issue bodies are Markdown documents. In the training set, 47% contain fenced code blocks (58% of bug reports), 59% contain URLs, and 43% use a GitHub issue template whose headings are class-independent boilerplate. Three cleaning levels are provided so the ablation study can be run by changing one string:

| Level | Operations | Intended consumer |
|---|---|---|
| `raw` | whitespace normalisation only | control condition |
| `light` | removes code, stack traces, Markdown markup, URLs, paths, SHAs, mentions | transformer models |
| `full` | `light` + lowercasing, punctuation removal, stop words, lemmatisation | TF-IDF models |

Average retained length: 100% → 63% → 33% of the original word count.

## Getting started

### With the dev container (recommended)

The repository ships a `.devcontainer/` so every group member — and the
lecturer at the exam — runs an identical environment. Open the folder in
VS Code and accept **Reopen in Container**, or run
`devcontainer up --workspace-folder .` with the CLI. GitHub Codespaces picks
the same definition up automatically.

The image is built from the **official Docker Hub `python:3.11-slim-trixie`**
(Debian 13, slim variant, multi-arch amd64 + arm64), not from a vendor
devcontainer image. Everything else — the non-root user, git, sudo, the VS Code
server prerequisites — is installed explicitly in the Dockerfile, so there are
no `features` and no third-party layers to trust.

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

Two build arguments in `devcontainer.json`:

| Argument | Default | Effect |
|---|---|---|
| `PYTHON_VERSION` | `3.11` | Newest version with prebuilt wheels for torch, transformers and setfit on both architectures. Change it and rebuild to move to 3.12 or 3.13. |
| `INSTALL_LATEX` | `true` | Installs `latexmk` + TeX Live, roughly 1 GB. Set to `false` if you are not the one compiling the report. |

### Without the dev container

```bash
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]" + NLTK corpora
```

### Common commands

```bash
make experiments  # regenerate every result (~6 min on one CPU core)
make notebooks    # re-execute all five notebooks
make help         # list every target
make data     # download and cache the NLBSE'24 dataset
make eda      # execute the Track A notebook end to end
make lab      # start Jupyter Lab on port 8888
make test     # run the test suite
make check    # print the persistence-equivalence table for the report
make lint     # ruff check + format
make report   # compile report/report.tex
```

Optional dependency groups, installed per track:
`pip install -e ".[ml]"` for track B, `pip install -e ".[dl]"` for tracks C and D.

Minimal example:

```python
from ai4se.loader import load_split
from ai4se.preprocessing import make_cleaner

train = load_split("train", kind="memory")
train.apply(make_cleaner(level="full", max_words=400))

X, y = train.texts_and_labels()                 # ready for scikit-learn
react = train.by_repo("facebook/react")         # per-project classifier input
```

## Evaluation protocol

Defined in `ai4se.evaluation` and documented in notebook 02. Two measurements,
not interchangeable:

**Model selection** — stratified 10-fold cross-validation on the *training*
split only, seeded at 42 so every member gets identical folds.

**Final result** — the competition protocol on the *test* split, used once: a
separate classifier per repository, each scored as the average F1 over the
three classes, reported as the arithmetic mean of the five.

Any object with `fit(X, y)` and `predict(X)` plugs in, supplied as a
zero-argument factory:

```python
from ai4se.evaluation import cross_validate, evaluate_competition

def svm_factory():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC())

cross_validate(svm_factory, train, k=10, model_name="TF-IDF + LinearSVC")
evaluate_competition(svm_factory, train, test, model_name="TF-IDF + LinearSVC")
```

Metrics are implemented from scratch in `ai4se.metrics` and verified against
scikit-learn to twelve decimal places in `tests/test_evaluation.py`.

### Results

Cross-repository F1 on the official test split, under the competition protocol.

| Model | overall | AUC | vs SetFit |
|---|---|---|---|
| **SetFit (NLBSE'24 baseline)** | **0.8270** | — | — |
| TF-IDF + Logistic Regression (tuned) | **0.7603** | 0.9018 | −0.0667 |
| TF-IDF + Linear SVM | 0.7583 | — | −0.0687 |
| CNN (embeddings) | 0.7578 | 0.8904 | −0.0692 |
| FFNN (TF-IDF) | 0.7467 | 0.8926 | −0.0803 |
| Frozen MiniLM + LogReg | 0.7050 | 0.8837 | −0.1220 |
| TF-IDF + Random Forest | 0.6962 | 0.8630 | −0.1308 |
| Naive Bayes (from scratch) | 0.6679 | 0.8246 | −0.1591 |
| Keyword rules | 0.5395 | — | −0.2875 |
| Majority class | 0.1667 | — | −0.6603 |

Full per-repository table: `results/tables/final_leaderboard.tex`, or
`leaderboard_from_disk()`.

### Findings

- **Aggressive cleaning hurts.** `full` (stop words + lemmatisation) scores
  0.015 below `light` — stop-word removal deletes *would* and *could*, the
  modal words that mark a feature request.
- **Frozen sentence embeddings underperform TF-IDF** (0.7050 vs 0.7603). Since
  that is SetFit minus the contrastive fine-tuning, the fine-tuning is worth
  roughly 0.12 F1 on its own: the adaptation, not the pretrained encoder, is
  what makes the baseline strong.
- **Neither neural model beats TF-IDF.** With 300 training issues per project
  there is not enough data to learn a better representation.
- **Per-project training beats a global classifier** by 0.068 on all five
  repositories, despite using a fifth of the data.
- **`bug` → `question` is the dominant error**, over a quarter of all mistakes.
- **About a third of confident errors look like label noise** — the title
  declares a different type than the label, and the model usually agrees with
  the title.

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
