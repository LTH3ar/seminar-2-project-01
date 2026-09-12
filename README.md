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

## Team and tracks

| Track | Owner | Scope |
|---|---|---|
| A — Data & persistence | *(name)* | Loading, EDA, cleaning pipeline, repository abstraction |
| B — Classical ML | *(name)* | TF-IDF + Naive Bayes, Logistic Regression, SVM, Random Forest |
| C — Deep learning | *(name)* | FFNN, CNN, fine-tuned transformer |
| D — Baseline & evaluation | *(name)* | SetFit reproduction, k-fold harness, metrics, results tables |

Track A is complete and its outputs under `data/processed/` are the agreed input for B, C and D.

## Layout

```
src/ai4se/
    model.py           IssueReport entity, label and repository constants
    repository.py      IssueRepository (abstract) + InMemory / File implementations
    loader.py          Downloads and caches the official NLBSE'24 splits
    preprocessing.py   Markdown-aware cleaning pipeline (raw / light / full)
    eda.py             Dataset characterisation and figures
notebooks/
    01_data_and_eda.ipynb
tests/
    test_persistence_equivalence.py
data/raw/              Cached competition CSVs (git-ignored, downloaded on demand)
data/processed/        Cleaned datasets handed to tracks B, C, D
results/figures/       Figures referenced by the LaTeX report
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

Issue bodies are Markdown documents. In the training set, 47% contain fenced code blocks (58% of bug reports), 59% contain URLs, and 43% use a GitHub issue template whose headings are class-independent boilerplate. Three cleaning levels are provided so the ablation study can be run by changing one string:

| Level | Operations | Intended consumer |
|---|---|---|
| `raw` | whitespace normalisation only | control condition |
| `light` | removes code, stack traces, Markdown markup, URLs, paths, SHAs, mentions | transformer models |
| `full` | `light` + lowercasing, punctuation removal, stop words, lemmatisation | TF-IDF models |

Average retained length: 100% → 63% → 33% of the original word count.

## Usage

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Reproduce Track A end to end
jupyter lab notebooks/01_data_and_eda.ipynb

# Verify the persistence requirement
python -m pytest tests/ -v
python tests/test_persistence_equivalence.py    # prints the comparison table
```

Minimal example:

```python
import sys; sys.path.insert(0, "src")
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