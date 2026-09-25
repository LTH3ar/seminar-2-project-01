# Project 1 — Issue Report Classification

Artificial Intelligence for Software Engineering, academic year 2026–2027.
Course project based on the [NLBSE'24 tool competition on issue report classification](https://nlbse2024.github.io/tools/).

## Task

Classify GitHub issue reports into one of three types: `bug`, `feature`, `question`.

The dataset contains 3,000 labelled issues extracted from five open-source projects — `facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`, `opencv/opencv` — created between March 2016 and September 2023 (about 77% from 2022–2023). It is split 50/50 into training and test sets, balanced at 100 issues per (project, class) cell in each split.

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
README.md                This file
RUNBOOK.md               Step-by-step: fresh clone to every result in the report
code_overview.md         Architecture, design decisions, findings, corrections
pyproject.toml           Packaging and optional extras (dev, ml, dl, embeddings, setfit)
requirements.txt         Flat dependency list, for tools that do not read pyproject
Makefile                 Task shortcuts -- run `make help`
.devcontainer/           Optional container setup; the virtualenv is the supported path

src/ai4se/
    __init__.py          Public API -- the dependency-light layers only
    model.py             IssueReport entity, label and repository constants
    repository.py        IssueRepository (abstract) + InMemory / File implementations
    loader.py            Downloads and caches the official NLBSE'24 splits
    preprocessing.py     Markdown-aware cleaning pipeline (raw / light / full)
    eda.py               Dataset characterisation and figures
    metrics.py           Precision, recall, F1, confusion matrix, ROC/AUC (from scratch)
    evaluation.py        Stratified k-fold, competition protocol, leaderboard, grid search
    baselines.py         Reference floors -- no third-party dependencies
    classical.py         Track B: TF-IDF + NB, LogReg, SVM, Random Forest   [scikit-learn]
    neural.py            Track C: FFNN and CNN, learning curves             [torch]
    embeddings.py        Track D2: frozen encoders and SetFit       [sentence-transformers]
    ensemble.py          Soft-voting ensembles over any of the above
    error_analysis.py    Confusions, length effects, misleading terms, confident errors
    validity.py          Temporal confound, statistical power, per-project vs pooled

scripts/
    run_experiments.py   Regenerates every result -- `--list` shows the stages

notebooks/
    01_data_and_eda.ipynb          Track A -- dataset, persistence, EDA       (standalone)
    02_evaluation_protocol.ipynb   Track D -- metrics, k-fold, floors         (standalone)
    03_classical_models.ipynb      Track B -- ablation, grid search, results  (needs results)
    04_neural_models.ipynb         Track C -- learning curves, early stopping (needs results)
    05_final_comparison.ipynb      Leaderboard, analysis, error analysis      (needs results)

tests/
    test_persistence_equivalence.py   Memory/file equivalence, public API, dependency layering
    test_evaluation.py                Metrics and splitter, cross-checked against scikit-learn
    test_models.py                    Tracks B, C, D2, ensembles, error analysis

data/
    raw/                 Competition CSVs -- downloaded on demand, never committed
    processed/           Cleaned datasets written by notebook 01 -- never committed

results/
    tables/              JSON results and generated .tex -- safe to commit once generated
    figures/             PNG figures -- safe to commit once generated
    models/              Checkpoints -- git-ignored
    predictions/         Prediction dumps -- git-ignored

report/
    report.md            The full report, in Markdown -- source for the LaTeX version
    figures/             Figures copied in for the LaTeX build
```

Notebooks 03–05 read results written by `scripts/run_experiments.py`, so run
that first; 01 and 02 are standalone. See `RUNBOOK.md` for the order.

**What gets committed.** Source code always. Generated tables and figures may be
committed once produced — the report references them — but every one of them
regenerates from `RUNBOOK.md`, so a source-only checkout loses nothing. Data,
model checkpoints and prediction dumps are git-ignored. `make strip` returns the
tree to source-only (clears notebook outputs and generated results) if you want
a clean commit.

Every directory that can legitimately be empty carries a `.gitkeep`, and the
final rule in `.gitignore` re-includes those placeholders even inside ignored
directories. A fresh clone therefore reproduces the whole tree.

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

### Without the dev container (plain virtualenv)

The dev container is a convenience, not a requirement — a virtualenv works
exactly as well, and this path is verified:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

make install                       # everything, all four tracks
# or:
make install-core                  # no torch, no sentence-transformers
make install-gpu                   # CUDA torch + the full SetFit extra
make test
```

Requires Python 3.10 or newer (3.11 recommended, matching the container).

**The optional extras really are optional.** `import ai4se` needs only pandas,
matplotlib and NLTK; the data, preprocessing, metrics and evaluation layers all
work with nothing else installed. Tracks B, C and D2 live in submodules that
are imported on demand, and their tests skip cleanly when the dependency is
absent:

```
72 passed, 9 skipped        # make install-core   (no torch, no sentence-transformers)
81 passed                   # make install
80 passed, 1 skipped        # make install-gpu    (the SetFit missing-extra test cannot run)
```

So a member who cannot install torch is not blocked from any other part of the
project. The only cost is that `ai4se.neural` is unavailable to them.

| Command | Installs | Tracks available |
|---|---|---|
| `make install-core` | scikit-learn | A, B, D |
| `make install` | + torch, sentence-transformers | A, B, C, D, D2 (frozen) |
| `make install-gpu` | + setfit, accelerate | all, including the SetFit reproduction |

If torch is the problem specifically, install the CPU wheel explicitly — it is
a few hundred MB rather than the ~2 GB CUDA default:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### If SetFit runs out of GPU memory

Contrastive training embeds **both** sentences of every pair, so peak memory is
roughly twice plain fine-tuning — and MPNet is twelve layers of width 768
against MiniLM's six of 384. MPNet at MiniLM's settings will not fit on a
12 GB card.

The knobs, in order of effect:

| Knob | Default | Effect |
|---|---|---|
| `max_seq_length` | 128 | **Dominant.** Attention cost grows with its square, so halving it cuts peak memory ~4×. Encoder defaults are 256 (MiniLM) and 384 (MPNet). |
| `batch_size` | 16 (8 for MPNet) | Peak memory is roughly linear in it. |
| `num_iterations` | 20 | Affects runtime, not peak memory. |

```python
from ai4se.embeddings import SetFitClassifier, STRONG_ENCODER

SetFitClassifier(STRONG_ENCODER, batch_size=4, max_seq_length=96)
```

128 tokens is ample here: the median issue is around 150 words and the text is
already truncated by preprocessing. `free_gpu_memory()` is called between the
five per-project fits, so a later project cannot OOM on memory the previous one
left cached.

If it still will not fit, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
reduces fragmentation.

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
| **Ensemble (SetFit + TF-IDF + MPNet)** | **0.8168** | **0.9319** | −0.0102 |
| SetFit (MPNet) | 0.8108 | 0.9195 | −0.0162 |
| Ensemble (SetFit + TF-IDF) | 0.8053 | 0.9244 | −0.0217 |
| SetFit (reproduction, MiniLM) | 0.7982 | 0.9181 | −0.0288 |
| Ensemble (TF-IDF + MPNet + CNN) | 0.7909 | 0.9245 | −0.0361 |
| SetFit (MiniLM, matched settings) | 0.7816 | 0.9144 | −0.0454 |
| TF-IDF + Logistic Regression (tuned) | 0.7603 | 0.9018 | −0.0667 |
| Frozen MPNet + LogReg | 0.7557 | 0.8970 | −0.0713 |
| CNN (embeddings) | 0.7438 | 0.8885 | −0.0832 |
| Frozen MiniLM + LogReg | 0.7057 | 0.8837 | −0.1213 |
| Keyword rules | 0.5395 | — | −0.2875 |
| Majority class | 0.1667 | — | −0.6603 |

Full per-repository table: `results/tables/final_leaderboard.tex`, or
`leaderboard_from_disk()`.

### Findings

- **The benchmark has a temporal confound.** No bug report predates 2021;
  every 2016–2020 issue is a feature or question. A shallow decision tree reading
  *only the creation timestamp* scores **0.7071**, and on `tensorflow` it matches
  our tuned TF-IDF model. The meaningful floor for this benchmark is ~0.70, not
  the majority class's 0.17.
- **How small a difference the test set can detect depends on the pair of
  models.** McNemar's test at 80% power detects 1.8–3.1 points overall
  (3.9–6.9 within one project) for real model pairs — less for models that
  rarely disagree. Against the published baseline, whose per-issue predictions
  are not available, the conservative threshold is 3.9 points overall and
  7.6–9.8 per project.
- **Best result 0.8168**, a soft-voting ensemble — **statistically
  indistinguishable from the published baseline** (1.02 points below it).
- **The published method reproduces to 0.8108** on MPNet. The organisers' own
  supplied re-run gives 0.8240 against their published 0.8270, so the
  reference point itself moves by 0.3 points.
- **Contrastive fine-tuning is worth +0.0759** on a fixed encoder, and a larger
  encoder **+0.0500** frozen — both detectable under any assumption. At matched
  training settings the encoder is worth +0.0292 after fine-tuning, which is
  not established, so the apparent sub-additivity is suggestive only.
- **Aggressive cleaning hurts.** `full` (stop words + lemmatisation) scores
  0.011 below `light` on average, and lower in all six paired settings —
  stop-word removal deletes *would* and *could*, the modal words that mark a
  feature request.
- **Ensembling helps when members are complementary.** TF-IDF + frozen MPNet +
  CNN gains +0.031 over its best member at no GPU cost. Adding SetFit gives the
  best AUC measured (0.9319), but its F1 gain over SetFit alone (+0.0060) is
  not detectable.
- **No neural model is detectably better or worse than TF-IDF** with 300
  training issues per project.
- **Whether per-project training helps depends on the model.** It gains +0.068
  for unregularised naive Bayes, winning on all five projects; for the tuned
  linear models there is no detectable difference from one pooled classifier.
- **21 of 22 models reproduce bit-identically** across three clean runs. The
  exception, SetFit on MPNet, varies by 0.0006 on the cross-repository mean
  while its per-project scores move by up to 0.0125 — on `tensorflow` it landed
  above the published baseline in two runs and below it in the third. Neural models differ by up
  to 0.014 between CPU and GPU.
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
