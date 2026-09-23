# Issue Report Classification on the NLBSE'24 Benchmark

**Artificial Intelligence for Software Engineering — Project 1**
Academic year 2026–2027

---

## Abstract

We evaluate twenty-two models on the NLBSE'24 issue report classification task —
assigning each GitHub issue to `bug`, `feature` or `question` — from trivial
floors through classical machine learning, neural networks and frozen sentence
encoders to a reproduction of the published SetFit baseline and ensembles of
these. Our best result, a soft-voting ensemble, scores 0.8168 cross-repository
F1 against the published 0.8270, a gap that is not statistically detectable on
this test set. Decomposing the baseline, contrastive fine-tuning contributes
+0.076 F1 and a larger pretrained encoder +0.050, and these are the only two
effects in the study large enough to be established beyond doubt. We also report
that the benchmark carries a temporal confound — a classifier that reads only an
issue's creation date scores 0.707 — that conventional text cleaning is harmful
here, and that most of the finer ordering on our leaderboard is statistical
noise. Three of our own initial explanations did not survive testing, and we
document each correction.

---

## 1. Introduction

### 1.1 Problem

Issue trackers are the primary interface between a software project and its
users. Before any report can be triaged, someone must decide what kind of report
it is — a defect, a request for new functionality, or a question about existing
behaviour. That decision determines who sees the issue and how urgently, and it
is repetitive, high-volume work that maintainers perform manually.

Automating it is a classification problem over short, noisy, semi-structured
text. Issue reports are Markdown documents in which prose is interleaved with
stack traces, code blocks, logs, screenshots and template boilerplate. They vary
enormously in length — in our data, from a single sentence to over twenty-one
thousand words. And the class boundaries are genuinely soft: a user who is
unsure whether behaviour is broken writes something that reads as both a bug
report and a question.

### 1.2 Task definition

We address the **NLBSE'24 tool competition on issue report classification**. The
dataset comprises 3,000 labelled issue reports from five open-source projects —
`facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`
and `opencv/opencv` — created between March 2016 and September 2023, with about
77% from 2022–2023. Each report carries a title, a body, a creation timestamp,
and exactly one label from `{bug, feature, question}`.

Two properties of the competition shape everything that follows.

**The task is multi-class, not multi-label.** The organisers excluded every
issue carrying more than one label, so each report has exactly one class.[^1]

**Five classifiers are required, not one.** The protocol trains a separate model
for each repository, evaluates it on that repository's own test issues, scores
it as the mean F1 over the three classes, and reports the arithmetic mean of the
five per-repository scores. Each classifier therefore sees only 300 training
examples. This is a few-shot problem by construction, and it is the single most
consequential constraint on which approaches can work.

The data is split 50/50 into training and test sets, balanced at exactly 100
issues per (project, class) cell in each split. The reference point is the
competition's published baseline, **SetFit at 0.8270** cross-repository F1.

[^1]: The project brief describes the task as multi-label classification. We
raised this with the course instructor and proceeded with the multi-class
formulation the data supports.

### 1.3 Approach

We evaluate twenty-two models spanning the range of available approaches, all
measured through a single evaluation harness so that every leaderboard entry is
computed identically:

- **Reference floors** — majority class, stratified random, hand-written keyword
  rules, and naive Bayes implemented from scratch.
- **Classical machine learning** — TF-IDF features with naive Bayes, logistic
  regression, linear SVM and random forest, with a preprocessing ablation and
  hyperparameter search.
- **Neural networks** — a feed-forward network over TF-IDF features and a 1-D
  convolutional network over learned embeddings.
- **Sentence transformers** — frozen pretrained encoders with a classification
  head, and a full reproduction of the published SetFit baseline.
- **Ensembles** — soft voting over models with complementary per-project error
  profiles.

### 1.4 Contributions

1. **A decomposition of the published baseline.** Contrastive fine-tuning
   contributes +0.0759 F1 on a fixed encoder and a larger pretrained encoder
   +0.0500 — the only two effects in this study large enough to be detectable
   under any assumption. An additive projection of the two predicted 0.8482 and
   was wrong by 0.037; we report the failed prediction alongside the
   measurement.

2. **A reproduction and a result at the level of the baseline.** Our SetFit
   reproduction reaches 0.8108 and our best ensemble 0.8168, against the
   published 0.8270. The gap is not statistically detectable, and the
   organisers' own supplied re-run of their baseline gives 0.8240 — the
   reference point itself moves by 0.3 points.

3. **Evidence of a temporal confound in the benchmark.** No bug report predates
   2021. A classifier that reads only the creation timestamp scores 0.7071, which
   makes ~0.70 — not the majority class's 0.17 — the meaningful floor.

4. **Evidence that conventional text cleaning is harmful here.** Stop-word
   removal and lemmatisation lower macro F1 by 0.011 on average and in every one
   of six paired settings, because the modal words they discard are what
   distinguishes a feature request.

5. **An account of how much of the leaderboard can be trusted.** We measure the
   minimum detectable difference for real model pairs (1.8–3.1 points), show
   that most of the leaderboard's finer ordering falls below it, and document
   three of our own explanations that did not survive testing.

### 1.5 Structure

Section 2 characterises the dataset, including the temporal confound. Section 3
describes our approach, including the persistence architecture and the
preprocessing pipeline. Section 4 sets out the evaluation methodology and the
test set's statistical power. Section 5 presents results, Section 6 analyses
where the models fail, Section 7 discusses threats to validity, and Section 8
concludes.

---

## 2. Dataset

### 2.1 Composition

We use the dataset published by the NLBSE'24 tool competition: 3,000 GitHub
issue reports from five open-source projects. Each record has five fields —
repository, creation timestamp, label, title and body — and carries exactly one
class from `{bug, feature, question}`.

The organisers balanced the dataset deliberately. It is split 50/50 into
training and test sets, with **exactly 100 issues per (project, class) cell in
each split**:

| Repository | bug | feature | question | total |
|---|---|---|---|---|
| bitcoin/bitcoin | 100 | 100 | 100 | 300 |
| facebook/react | 100 | 100 | 100 | 300 |
| microsoft/vscode | 100 | 100 | 100 | 300 |
| opencv/opencv | 100 | 100 | 100 | 300 |
| tensorflow/tensorflow | 100 | 100 | 100 | 300 |
| **Total (per split)** | **500** | **500** | **500** | **1,500** |

*(Figure: `label_distribution.png`)*

This balance has two consequences we return to in Section 4. Macro- and
micro-averaged F1 coincide, so the choice of averaging does not affect the
ranking; and accuracy is a meaningful metric here, though F1 remains the
competition's reported figure.

**Integrity checks.** No empty titles, no unexpected labels, and one duplicate
title-and-body pair in the training split. The test split contains two issues
with empty bodies. None of these is material at this scale.

### 2.2 Temporal distribution and the confound it creates

The issues span March 2016 to September 2023, but not evenly, and — more
importantly — not evenly *by class*:

| Year | issues | bug | feature | question |
|---|---|---|---|---|
| 2016 | 16 | 0.00 | 1.00 | 0.00 |
| 2017 | 28 | 0.00 | 0.89 | 0.11 |
| 2018 | 58 | 0.00 | 0.55 | 0.45 |
| 2019 | 86 | 0.00 | 0.23 | 0.77 |
| 2020 | 63 | 0.00 | 0.51 | 0.49 |
| 2021 | 98 | 0.26 | 0.47 | 0.28 |
| 2022 | 406 | 0.32 | 0.38 | 0.30 |
| 2023 | 745 | 0.46 | 0.23 | 0.30 |

*(Training split; class shares per year.)*

**No bug report in the dataset predates 2021.** Every issue from 2016 to 2020 is
a feature request or a question. The organisers evidently filled the feature and
question quotas by reaching further back in time than the bug quota. The effect
is strongest in `facebook/react`, where the median bug report dates from 2022 and
the median feature request from 2018.

This matters because text carries temporal proxies — library versions, API
names, deprecated features — so any model can learn *when* an issue was filed as
a stand-in for *what kind* of issue it is. Section 5.6 measures how far that
shortcut alone goes.

### 2.3 Length distribution

Issue length is extremely skewed. The median report is 150 words; the longest
is 21,595.

| Class | mean | median | 75th pct | max |
|---|---|---|---|---|
| bug | 253.0 | 157.5 | 254.0 | 3,653 |
| feature | 163.2 | 129.0 | 196.5 | 2,373 |
| question | 293.7 | 160.5 | 290.5 | 21,595 |
| **overall** | **236.6** | **150.0** | **244.0** | **21,595** |

*(Figure: `length_distribution.png`)*

The standard deviation for `question` is more than six times its median, driven
by a small number of reports containing pasted log output. Truncation is
therefore a hyperparameter rather than an implementation detail, and
term-frequency statistics computed over raw counts are dominated by a handful of
documents (Section 2.5).

### 2.4 Issue reports are not plain prose

Issue bodies are Markdown documents in which natural language is interleaved
with material that is not natural language at all:

| | code block | stack trace | URL | image | HTML | template heading |
|---|---|---|---|---|---|---|
| bug | **57.6%** | **17.2%** | 67.0% | 17.2% | 27.8% | **59.0%** |
| feature | 34.8% | 1.6% | 63.4% | 6.8% | 35.2% | 36.2% |
| question | 49.8% | 3.0% | 47.2% | 12.2% | 45.6% | 32.8% |
| **overall** | **47.4%** | **7.3%** | **59.2%** | **12.1%** | **36.2%** | **42.7%** |

**The noise is class-correlated.** Code blocks appear in 57.6% of bug reports
but 34.8% of feature requests; stack traces in 17.2% of bugs against 1.6% of
features. Stripping them therefore removes signal along with noise — a reason to
*measure* whether cleaning helps rather than assume it. Section 5.1 reports that
the most aggressive cleaning we implemented is harmful.

Our pipeline offers three levels, and their effect on document length is:

| Level | operations | mean words | retained |
|---|---|---|---|
| `raw` | whitespace normalisation only | 189.6 | 100% |
| `light` | strips code, stack traces, markup, URLs, paths, mentions | 114.4 | 60.3% |
| `full` | `light` + lowercasing, punctuation, stop words, lemmatisation | 60.1 | 31.7% |

### 2.5 Are the classes lexically separable?

Terms are ranked by **document frequency** — the number of issues containing
them — rather than raw token count, because the length skew above means a few
enormous log dumps would otherwise dominate.

| Class | characteristic terms |
|---|---|
| bug | `cudatoolkit`, `onednn`, `avx`, `geforce`, `rebuild`, `critical` |
| feature | `willing`, `allow`, `nice`, `considered`, `easily`, `approach`, `great` |
| question | `liveserver`, `runner`, `canvas`, `argv`, `oop`, `reporter` |

**Feature requests are marked by modal and evaluative language** — *would be
nice*, *allow*, *willing*, *considered* — which is generic across projects.
`allow` appears in 9.6% of feature requests at seven times the rate of other
classes.

**Bug reports and questions are marked by project-specific technical
vocabulary.** `cudatoolkit`, `onednn` and `geforce` identify TensorFlow;
`liveserver` identifies VS Code; `canvas`, `argv` and `oop` identify OpenCV.
These terms identify the project, and within one project's training data they
happen to correlate with a class.

### 2.6 Cross-project vocabulary overlap

Mean pairwise Jaccard overlap of the five projects' cleaned vocabularies is
**0.266** — roughly a quarter of terms shared between any two projects.

| Repository | vocabulary size |
|---|---|
| bitcoin/bitcoin | 3,467 |
| facebook/react | 2,824 |
| microsoft/vscode | 2,668 |
| tensorflow/tensorflow | 2,551 |
| opencv/opencv | 2,535 |

This predicts that a classifier trained on one project will transfer poorly to
another, and therefore that the competition's per-project protocol should help.
Section 5.3 tests the prediction and finds it holds only for unregularised
models: per-project training gains +0.068 for naive Bayes, but makes no
detectable difference for the tuned linear models.

### 2.7 Summary

| Property | Value | Consequence for modelling |
|---|---|---|
| Size | 1,500 train / 1,500 test | Few-shot; favours pretrained representations |
| Training data per classifier | 300 issues | The binding constraint on neural models |
| Balance | 100 per (project, class) cell | Macro- and micro-F1 coincide |
| Label cardinality | Exactly one per issue | Multi-class, not multi-label |
| Time span | March 2016 – Sept 2023, no bugs before 2021 | A temporal shortcut exists |
| Median length | 150 words, max 21,595 | Truncation is a hyperparameter |
| Code blocks | 47.4%, class-correlated | Cleaning removes signal as well as noise |
| Cross-project vocabulary overlap | 0.266 | Per-project training should help weak models |

---

## 3. Approach

### 3.1 Architecture

The implementation is organised in layers, with dependencies pointing in one
direction only:

```
                    notebooks (01–05)                     presentation
                           │
   ┌─────────┬─────────────┼─────────────┬──────────┐
   │         │             │             │          │
  eda   preprocessing  classical     neural    embeddings        business
   │         │         evaluation ── metrics    ensemble           logic
   │         │          validity    error_analysis
   └─────────┴─────────────┼─────────────┴──────────┘
                           │
                  ┌────────▼────────┐
                  │ IssueRepository │                     abstract interface
                  └────────┬────────┘
             ┌─────────────┴─────────────┐
   InMemoryIssueRepository      FileIssueRepository        persistence
           (RAM)                   (CSV / JSON)
```

Persistence knows nothing about analysis, and analysis knows nothing about
storage. `IssueReport`, the single domain entity, depends on nothing.

The layering is also enforced by dependency weight. The domain, persistence,
preprocessing, metrics and evaluation layers require only pandas, matplotlib and
NLTK; `import ai4se` works with neither scikit-learn nor PyTorch installed, and
the model tracks are imported from submodules on demand. A test parses the
package source to keep it that way, so a team member unable to install PyTorch
is not blocked from any other part of the project.

### 3.2 Persistence layer

The project specification requires that data be manageable both in memory and
through files, and that switching between the two involve few changes to the
application. We implement this with the Repository pattern.

`IssueRepository` is an abstract base class exposing **five primitives** —
`load`, `save`, `all`, `add`, `clear` — and roughly a dozen query helpers
(`by_repo`, `by_label`, `label_distribution`, `texts_and_labels`, `filter`,
`apply`, `to_dataframe`, and the standard container protocol). The helpers are
implemented **once, on the base class**, in terms of the five primitives.

This makes behavioural equivalence structural rather than a promise: adding a
query adds it to both implementations simultaneously, and they cannot drift
apart. `InMemoryIssueRepository` wraps a Python list and its `load`/`save` are
deliberate no-ops; `FileIssueRepository` reads and writes CSV or JSON, inferring
the format from the file suffix.

Switching the entire application's persistence layer is one argument:

```python
train = load_split("train", kind="memory")   # held in RAM
train = load_split("train", kind="file")     # backed by the CSV on disk
```

**Evidence.** `tests/test_persistence_equivalence.py` runs the identical analysis
pipeline — load, clean, extract features and labels — over both layers and
asserts that the results match:

```
property          in-memory               file                    equal
------------------------------------------------------------------------
n                 1500                    1500                    yes
distribution      (('bug', 500), ('featu  (('bug', 500), ('featu  yes
checksum          658455                  658455                  yes
first_label       bug                     bug                     yes
------------------------------------------------------------------------
all properties identical: True
```

The same test verifies that CSV and JSON round-trips are lossless and that both
implementations satisfy the full abstract interface. Every function in the EDA,
evaluation and validity modules takes an `IssueRepository`, so the entire
analysis runs unchanged against either.

One implementation detail: `csv.field_size_limit` must be raised at import,
because the 21,595-word issue in Section 2.3 exceeds Python's default field limit.

### 3.3 Preprocessing

Each cleaning step is a pure `str → str` function, so the pipeline can be
reordered or partially disabled for ablation. Three levels are selectable by one
string:

| Level | Operations | Intended consumer |
|---|---|---|
| `raw` | whitespace normalisation only | control condition |
| `light` | strips fenced code, stack traces, Markdown markup, URLs, file paths, commit hashes, issue references, mentions | transformer models |
| `full` | `light` + lowercasing, punctuation removal, stop words, lemmatisation | TF-IDF models |

Stop words are NLTK's English list plus a domain list of terms appearing in
nearly every issue regardless of class (`issue`, `reproduce`, `expected`,
`screenshot`, and similar).

**The configuration selected by ablation** (Section 5.1) is `level="light"`, with
the title repeated three times and the text truncated to 200 words. Titles are
short and dense with class-characteristic vocabulary, so repeating them
up-weights them in a bag-of-words representation at no cost. We emphasise that
`full` — the default in most text-classification pipelines — is **not** what we
use, for reasons Section 5.1 reports.

### 3.4 Models

All models expose the same `fit(X, y)` / `predict(X)` interface and are supplied
to the evaluation harness as zero-argument factories, so that each
cross-validation fold and each repository receives a fresh untrained instance.
Models that also expose `predict_proba` additionally get ROC and AUC reported.

#### 3.4.1 Reference floors

**Majority class**, **stratified random**, **keyword rules** over cue words drawn
from Section 2.5, and **multinomial naive Bayes** implemented from scratch in log
space with Laplace smoothing. Section 5.6 adds a fifth, measured separately: a
classifier that reads only the creation timestamp.

#### 3.4.2 Classical machine learning

Five scikit-learn pipelines over a shared TF-IDF representation, so that score
differences come from the classifier rather than the features: multinomial naive
Bayes, complement naive Bayes, logistic regression, linear SVM, and random
forest.

**Vectorisation happens inside each pipeline, and therefore inside each
cross-validation fold.** Fitting the vectoriser once over the whole dataset
before cross-validating would leak test-fold vocabulary and document frequencies
into training and inflate every reported score. We guard against this with a
dedicated test.

Selected settings after grid search: bigrams for both linear models; logistic
regression at `C=10, min_df=2`; linear SVM at `C=1, min_df=1`. `LinearSVC` has no
probability head, so its AUC is reported as absent rather than approximated.

#### 3.4.3 Neural networks

**Feed-forward network** over TF-IDF features (20,000 max features, hidden layers
of 256 and 64, ReLU, dropout 0.5) — directly comparable with the classical
pipelines.

**1-D convolutional network** over learned 100-dimensional embeddings, with
parallel filters of widths 2, 3 and 4 acting as learned n-gram detectors, 64
filters each, and global max-pooling.

Both hold out 20% of the training data for validation and record training loss,
validation loss and validation accuracy at every epoch, with early stopping after
8 epochs without improvement, restoring the weights from the best epoch.

#### 3.4.4 Sentence transformers

**Frozen encoder baselines.** A pretrained Sentence Transformer encodes each
issue and a logistic-regression head is fitted on the embeddings; the encoder is
not updated. We evaluate `all-MiniLM-L6-v2` (6 layers, 384 dimensions) and
`all-mpnet-base-v2` (12 layers, 768 dimensions). This configuration is precisely
**SetFit with the contrastive fine-tuning step removed**, which is what makes the
decomposition in Section 5.4 possible.

**SetFit reproduction.** Generate pairs from the labelled examples, fine-tune the
encoder contrastively, then fit a logistic-regression head on the adapted
embeddings: 20 pair-generation iterations, 1 epoch.

Contrastive training embeds both sentences of every pair, so MPNet at MiniLM's
settings exceeds 12 GB of GPU memory. We therefore run MPNet at `batch_size=8,
max_seq_length=128`, and — critically — also run MiniLM at those same settings,
so that the encoder comparison is controlled rather than confounded.

#### 3.4.5 Ensembles

Soft voting over the averaged class probabilities of two or three members. Soft
rather than majority voting because with three classes and two members, hard
votes tie often, and the tie discards exactly the confidence information that
would resolve it. Members were chosen because they fail on different projects
(Section 5.5).

### 3.5 Reproducibility infrastructure

A single entry point regenerates every number in this report:

```bash
python scripts/run_experiments.py --all
```

Nine stages write JSON results to `results/tables/` and figures to
`results/figures/`. Each skips work already on disk unless forced. The notebooks
read those saved results rather than re-training, and the LaTeX tables are
generated from the same JSON, so the numbers here cannot drift from what the code
produced. A fixed seed of 42 is used throughout.

---

## 4. Evaluation methodology

### 4.1 Two measurements, not one

**Model selection** uses stratified *k*-fold cross-validation over the **training
split only**. Every decision in this report — preprocessing, truncation,
hyperparameters, architecture — was made against cross-validation scores.

**The final result** uses the competition protocol on the **test split**, run
once per model when that model was final. This is the only number comparable
with the published baseline.

### 4.2 Metrics

The course specifies accuracy, precision, recall, F1 and the area under the ROC
curve. All derive from the confusion matrix. For a class *c*:

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 · precision · recall / (precision + recall)
```

We implement these from first principles rather than importing them, so the
evaluation layer carries no dependency on a modelling library and the
definitions match the course exactly. That choice is only defensible if
verified: `tests/test_evaluation.py` cross-checks every metric against
scikit-learn on 500 synthetic predictions — per-class precision, recall, F1 and
support; macro, micro and weighted averaging; the fold splitter's proportions;
and one-vs-rest AUC. **Agreement is exact to machine precision**, with a maximum
deviation of 1.1 × 10⁻¹⁶.

#### 4.2.1 Averaging

Macro averages the per-class scores unweighted, so a class the model ignores
cannot be hidden. Micro pools the counts first, which for a single-label
multi-class problem makes it identical to accuracy — a property we assert in a
test. A model that always predicts `bug` on our balanced data scores:

| | value |
|---|---|
| accuracy | 0.3333 |
| micro F1 | 0.3333 |
| **macro F1** | **0.1667** |
| weighted F1 | 0.1667 |

**Macro F1 is the figure used throughout**, matching the competition.

#### 4.2.2 ROC and AUC

ROC is computed one-vs-rest, and the macro AUC is the unweighted mean of the
three curves. AUC is reported only for models that expose class probabilities;
models returning hard labels report it as absent rather than approximated.

### 4.3 Cross-validation

Stratified *k*-fold with *k* = 10 over the 1,500 training issues, seeded at 42.
Each fold contains exactly 150 issues — 50 per class — with 1,350 for training.
Grid searches use *k* = 5 to rank configurations; the selected configuration is
then re-evaluated at *k* = 10. The neural models additionally hold out 20% of the
training data to record loss curves per epoch, as the course prescribes.

### 4.4 The competition protocol

1. Train a **separate classifier for each of the five repositories**.
2. Evaluate each on that repository's 300 test issues.
3. Score each repository as the **mean F1 over the three classes**.
4. Report the **arithmetic mean of the five per-repository scores**.

### 4.5 Baselines

**Floors** establish what costs no effort. **The published baseline** is the
competition's SetFit result:

| Repository | SetFit F1 |
|---|---|
| facebook/react | 0.8718 |
| tensorflow/tensorflow | 0.8644 |
| microsoft/vscode | 0.8262 |
| opencv/opencv | 0.8173 |
| bitcoin/bitcoin | 0.7555 |
| **cross-repository** | **0.8270** |

A test asserts that the five per-repository figures average to the published
overall. The organisers also supply a re-run of their own baseline code, which
gives **0.8240** — so the published reference point itself moves by 0.3 points
between runs of the organisers' code.

### 4.6 Statistical power

How small a difference can this test set detect? McNemar's exact test compares
two models on the same items and uses only the items on which they disagree, so
its power depends on that **disagreement rate**, not on the test-set size alone.
We estimate the minimum detectable difference (MDD) at 80% power by Monte Carlo
simulation of McNemar's test, using disagreement rates measured between real
model pairs:

| Compared with tuned logistic regression | disagreement | MDD overall | MDD per project |
|---|---|---|---|
| Linear SVM (tuned) | 5.5% | 1.8 points | 3.9 points |
| Naive Bayes | 13.1% | 2.7 points | 6.0 points |
| Random forest | 17.4% | 3.1 points | 6.9 points |

Two similar models (logistic regression and SVM on the same features) rarely
disagree, so even a small difference between them is detectable; two dissimilar
models need a larger gap. There is therefore no single "detectable difference"
for this benchmark — only one per pair of models.

Comparisons against the **published baseline** are harder: the organisers publish
scores, not per-issue predictions, so no paired test is possible. Treating the
two scores as independent gives a conservative threshold of **3.9 points
overall** and **7.6–9.8 points within one project**.

On balanced data macro F1 and accuracy move together closely, so these
thresholds are read in F1 points. Section 5.7 applies them to every comparison
in this report.

### 4.7 Guarding against leakage

**Vectorisers are fitted inside the pipeline**, and a test asserts that no term
occurring only in held-out data enters the learned vocabulary. **Models are
supplied as factories**, so no fitted state can leak between folds or
repositories. **The error analysis re-derives its predictions** through the same
protocol as the leaderboard, and a test asserts the two agree.

---

## 5. Results

### 5.1 Feature engineering and classical models

#### 5.1.1 Does cleaning help?

We cross-validated the same logistic-regression pipeline across all eighteen
combinations of cleaning level, title weight and truncation:

| Cleaning level | mean macro F1 | best macro F1 |
|---|---|---|
| `raw` | 0.7446 | 0.7472 |
| `light` | 0.7434 | 0.7472 |
| **`full`** | **0.7326** | **0.7364** |

**`full` cleaning is the worst of the three levels.** It scores 0.011 below
`light` on the mean and on the best configuration, and it is lower in **all six**
paired settings of title weight and truncation.

The magnitude is small, within the fold-to-fold spread of any single
configuration, but the direction is perfectly consistent. The mechanism is in
what `full` removes: stop-word removal deletes *would*, *could*, *should*,
*please* and *nice*; lemmatisation collapses tense. Those are precisely the modal
and evaluative words that Section 2.5 identified as the signature of a feature
request.

`full` — lowercase, strip punctuation, remove stop words, lemmatise — is the
default pipeline in most text-classification work, and it was the configuration
this project was originally built around. **An obviously sensible preprocessing
step was consistently harmful, and only an ablation revealed it.** We select
`light`, which is indistinguishable from `raw` but discards code blocks and stack
traces that inflate the vocabulary.

#### 5.1.2 Hyperparameter search

| Model | best configuration | CV macro F1 |
|---|---|---|
| Logistic regression | `C=10`, bigrams, `min_df=2` | 0.7481 |
| Linear SVM | `C=1`, bigrams, `min_df=1` | 0.7484 |

**Tuning gained almost nothing.** The best and fifth-best configurations differ
by about 0.002 against a fold-to-fold standard deviation of roughly 0.03. Only
the bigram setting clearly exceeds the noise.

The grid was first run under `full` cleaning, and the winning `min_df` changed
once the ablation moved the pipeline to `light`. Hyperparameters and
preprocessing are not independent, and re-tuning after altering the pipeline is
not optional.

#### 5.1.3 Classical results

| Model | CV macro F1 | test F1 | AUC |
|---|---|---|---|
| **TF-IDF + Logistic Regression (tuned)** | 0.7446 ± 0.0398 | **0.7603** | 0.9018 |
| TF-IDF + Linear SVM | 0.7426 ± 0.0367 | 0.7583 | — |
| TF-IDF + Logistic Regression | 0.7478 ± 0.0409 | 0.7576 | 0.9016 |
| TF-IDF + Linear SVM (tuned) | 0.7499 ± 0.0430 | 0.7531 | — |
| TF-IDF + Naive Bayes | 0.6919 ± 0.0334 | 0.7140 | 0.8713 |
| TF-IDF + Complement NB | 0.6902 ± 0.0311 | 0.7012 | 0.8673 |
| TF-IDF + Random Forest | 0.7252 ± 0.0316 | 0.6978 | 0.8630 |

The three linear models cluster within 0.007 of each other; naive Bayes trails
by about 0.045. Random forest is the only model whose test score falls
materially below its cross-validation score — trees cope poorly with
high-dimensional sparse TF-IDF features. **Every difference among the top four
rows is inside one standard deviation**, and should not be read as a ranking.

### 5.2 Validation curves and overfitting control

With 300 training issues per project, overfitting appears quickly.

| Model | epochs run | best epoch | stopped early | train loss | validation loss |
|---|---|---|---|---|---|
| FFNN (TF-IDF) | 23 | 15 | yes | 1.099 → 0.007 | 1.093 → 0.579 (min 0.572) |
| CNN (embeddings) | 38 | 30 | yes | 1.267 → 0.053 | 0.989 → 0.574 (min 0.513) |

*(Figures: `learning_curve_ffnn_tf_idf.png`, `learning_curve_cnn_embeddings.png`)*

The FFNN drives training loss essentially to zero while validation loss bottoms
out at 0.572 and then rises — a model that has stopped generalising and started
memorising, within fifteen epochs. The CNN overfits more slowly, reaching its
best epoch at 30, because it must learn its embedding table from scratch.
Both use early stopping with a patience of 8 and restore the weights from the
best epoch.

| Model | test F1 | AUC |
|---|---|---|
| CNN (embeddings) | 0.7438 | 0.8885 |
| FFNN (TF-IDF) | 0.7424 | 0.8909 |

Both score about 0.017 below tuned logistic regression (0.7603) — **a difference
too small to detect** (Section 5.7). With 300 training issues per project, a
learned representation does no better than an engineered one. That is the
argument for Section 5.4: a *pretrained* representation is the only way to bring
outside knowledge into a problem this small.

### 5.3 Per-project versus pooled training

Section 2.6 predicted that per-project training would help, because
cross-project vocabulary overlap is low. We trained each model both ways and
scored both per project:

| Model | per-project | pooled | difference | per-project wins |
|---|---|---|---|---|
| Naive Bayes (from scratch, `full`) | 0.6679 | 0.6003 | **+0.0676** | 5 / 5 |
| Naive Bayes (from scratch, `light`) | 0.7046 | 0.6401 | +0.0645 | 4 / 5 |
| TF-IDF + Naive Bayes | 0.7140 | 0.6822 | +0.0318 | 4 / 5 |
| TF-IDF + Logistic Regression (tuned) | 0.7603 | 0.7621 | −0.0019 | 2 / 5 |
| TF-IDF + Linear SVM (tuned) | 0.7531 | 0.7639 | −0.0108 | 2 / 5 |

**The prediction holds for weak models and not for strong ones.** The
per-project advantage is large for naive Bayes, halves for TF-IDF naive Bayes,
and disappears for the regularised linear models, where neither configuration
is detectably better.

The mechanism is regularisation. Naive Bayes weights every term by its
frequency, so the project-specific vocabulary of Section 2.5 — `cudatoolkit`,
`liveserver`, `geforce` — actively misleads a pooled model, because those terms
signal a class *within one project* and nothing across projects. A regularised
linear model can down-weight exactly those terms, at which point five times the
training data compensates for the lost specialisation.

### 5.4 Decomposing the published baseline

#### 5.4.1 The reproduction

Our implementation of the published method reaches **0.8108** on MPNet against
the published **0.8270** — and against the organisers' own re-run of 0.8240.

#### 5.4.2 Where the performance comes from

The frozen-encoder configuration is SetFit with the contrastive step removed.
Running both encoders in both configurations decomposes the baseline:

| Configuration | MiniLM | MPNet | encoder effect |
|---|---|---|---|
| Frozen encoder + LogReg | 0.7057 | 0.7557 | **+0.0500** |
| SetFit @ `batch=8, seq=128` | 0.7816 | 0.8108 | +0.0292 |
| **fine-tuning effect** | **+0.0759** | **+0.0551** | |

**Contrastive fine-tuning is worth +0.0759 on a fixed encoder**, with everything
else held constant. It is the largest single effect in this study, and it and
the frozen encoder effect (+0.0500) are the only two large enough to be
detectable under any assumption about the models' disagreement rate.

#### 5.4.3 A prediction that failed, and a correction that was itself wrong

We report all three stages of this measurement, because the process is itself a
finding.

**Stage 1 — projection.** From the frozen models and a first fine-tuned run, we
measured an encoder effect of +0.0500 and a fine-tuning effect of +0.0925, and
projected SetFit-on-MPNet at **0.8482** — above the published baseline.

**Stage 2 — measurement.** The actual value was **0.8108**, wrong by 0.037.
Taken at face value the encoder appeared worth only +0.0126 after fine-tuning,
implying severe sub-additivity.

**Stage 3 — the confound.** That comparison was not controlled. MiniLM had run at
`batch_size=16` with a sequence length of 256; MPNet required 8 and 128 to fit in
memory. Re-running MiniLM at MPNet's settings gives **0.7816** — the reduced
settings alone cost 0.0166 — and the controlled encoder effect is **+0.0292**,
not +0.0126.

Whether the two effects are sub-additive therefore remains open: the controlled
encoder effect after fine-tuning (+0.0292) is below what this comparison can
reliably detect (Section 5.7). What is established is narrower and still useful:

1. **Effects measured separately cannot be assumed to compose.** The additive
   projection was wrong about both the number and the conclusion.
2. **The first correction was itself overstated**, because the comparison behind
   it was confounded by training settings.
3. **Training settings are not a detail.** Halving batch size and sequence length
   cost 0.0166 — comparable to the effect of the encoder itself. A result quoted
   without them is not comparable with another.

### 5.5 Ensembles

Tuned TF-IDF and a frozen MPNet encoder reach similar averages by opposite
routes:

| | react | tensorflow | vscode | bitcoin | opencv | spread |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg (tuned) | **0.8334** | **0.8155** | 0.7234 | 0.6793 | 0.7496 | 0.154 |
| Frozen MPNet + LogReg | 0.8017 | 0.7228 | **0.7566** | **0.7279** | **0.7696** | **0.079** |

TF-IDF is stronger on the two easiest projects and MPNet on the three hardest,
with half the spread. Individually, per-project gaps of this size are within
noise; the value of the observation is the *pattern* — two models of similar
strength failing in different places, which is the precondition for ensembling.

| Ensemble | test F1 | AUC | gain over best member |
|---|---|---|---|
| **SetFit + TF-IDF + MPNet** | **0.8168** | **0.9319** | +0.0060 |
| SetFit + TF-IDF | 0.8053 | 0.9244 | +0.0071 |
| TF-IDF + MPNet + CNN | 0.7909 | 0.9245 | +0.0306 |
| TF-IDF + MPNet | 0.7838 | 0.9198 | +0.0235 |

Every ensemble scores above its best member. **The GPU-free combination of
TF-IDF, frozen MPNet and the CNN gains +0.031**, the one ensemble gain large
enough to be plausibly real (Section 5.7), and reaches within 0.007 of the SetFit
reproduction at no fine-tuning cost.

The best ensemble has the **highest AUC of any model measured, 0.9319**, above
SetFit-on-MPNet's 0.9195: its ranking of the classes is better than its argmax
decisions suggest. Its F1 gain over SetFit alone, however, is 0.0060 — far below
what can be detected.

### 5.6 Final results

The leaderboard below adds one row not produced by the harness: a depth-3
decision tree per project that reads **only each issue's creation timestamp**
(Section 2.2).

| Model | overall F1 | AUC | vs baseline |
|---|---|---|---|
| **SetFit (published baseline)** | **0.8270** | — | — |
| **Ensemble (SetFit + TF-IDF + MPNet)** | **0.8168** | **0.9319** | −0.0102 |
| SetFit (MPNet) | 0.8108 | 0.9195 | −0.0162 |
| Ensemble (SetFit + TF-IDF) | 0.8053 | 0.9244 | −0.0217 |
| SetFit (reproduction, MiniLM) | 0.7982 | 0.9181 | −0.0288 |
| Ensemble (TF-IDF + MPNet + CNN) | 0.7909 | 0.9245 | −0.0361 |
| Ensemble (TF-IDF + MPNet) | 0.7838 | 0.9198 | −0.0432 |
| SetFit (MiniLM, matched settings) | 0.7816 | 0.9144 | −0.0454 |
| TF-IDF + Logistic Regression (tuned) | 0.7603 | 0.9018 | −0.0667 |
| TF-IDF + Linear SVM | 0.7583 | — | −0.0687 |
| TF-IDF + Logistic Regression | 0.7576 | 0.9016 | −0.0694 |
| Frozen MPNet + LogReg | 0.7557 | 0.8970 | −0.0713 |
| TF-IDF + Linear SVM (tuned) | 0.7531 | — | −0.0739 |
| CNN (embeddings) | 0.7438 | 0.8885 | −0.0832 |
| FFNN (TF-IDF) | 0.7424 | 0.8909 | −0.0846 |
| TF-IDF + Naive Bayes | 0.7140 | 0.8713 | −0.1130 |
| ***Timestamp only (depth-3 tree)*** | ***0.7071*** | — | *−0.1199* |
| Frozen MiniLM + LogReg | 0.7057 | 0.8837 | −0.1213 |
| TF-IDF + Complement NB | 0.7012 | 0.8673 | −0.1258 |
| TF-IDF + Random Forest | 0.6978 | 0.8630 | −0.1292 |
| Naive Bayes (from scratch) | 0.6679 | 0.8246 | −0.1591 |
| Keyword rules | 0.5395 | — | −0.2875 |
| Random (stratified) | 0.3348 | — | −0.4922 |
| Majority class | 0.1667 | — | −0.6603 |

*(Full per-repository table: `final_leaderboard.tex`)*

Four observations:

**A model that reads no text at all outscores five that do.** The timestamp-only
classifier (0.7071) sits above frozen MiniLM, complement naive Bayes, random
forest and both floors built from text. On `tensorflow` it scores 0.8375 against
0.8155 for tuned TF-IDF. The meaningful floor for this benchmark is ~0.70, and
the useful headroom above trivial is roughly 0.70 → 0.83 — not 0.17 → 0.83.

**Our best result is statistically indistinguishable from the published
baseline.** The 1.02-point gap is well under the 3.9-point threshold for this
comparison (Section 4.6). We neither beat the baseline nor fall detectably short
of it.

**Every purely lexical model lands between 0.70 and 0.76**, whatever the
classifier. What moved the number was a better representation (+0.050 for the
larger encoder) and adapting it to the task (+0.076 for fine-tuning).

**AUC and F1 do not rank models identically.** The best ensemble leads on both,
but SetFit-on-MPNet is second on F1 and fifth on AUC. Reporting both, as the
course requires, surfaces information either alone would hide.

### 5.7 Which differences are real

Applying the thresholds of Section 4.6 to every comparison drawn above, using
for each claim the threshold most favourable to it:

| Claim | difference | most favourable threshold | verdict |
|---|---|---|---|
| Contrastive fine-tuning (MiniLM) | 7.59 | 1.8 | **detectable** |
| Larger encoder, frozen | 5.00 | 1.8 | **detectable** |
| TF-IDF + MPNet + CNN vs its TF-IDF member | 3.06 | 1.8 | likely detectable |
| Larger encoder, fine-tuned (matched) | 2.92 | 1.8 | not established |
| Neural models vs TF-IDF | 1.65 | 1.8 | not detectable |
| Our best vs published baseline | 1.02 | 3.9 (unpaired) | not detectable |
| Best ensemble vs SetFit on MPNet | 0.60 | 1.8 | not detectable |
| Per-project differences from the baseline | ≤ 1.4 | 7.6–9.8 (unpaired) | not detectable |

"Likely detectable" and "not established" mark the two claims whose disagreement
rate could not be measured, because one side needs a GPU to regenerate. The
ensemble contains the TF-IDF model it is compared with, so the two rarely
disagree and the favourable threshold probably applies.

**The two largest effects in the study survive under any assumption. Most of the
leaderboard's finer ordering does not, and should be read as ties.**

---

## 6. Error analysis

We analyse the best classical model, TF-IDF with tuned logistic regression
(0.7603). It is fully interpretable — every decision traces to term weights — and
its confusion structure matches that of the stronger models, which make the same
errors less often. A test asserts that the predictions analysed here agree with
the leaderboard's to four decimal places.

### 6.1 Where the model fails

Of 1,500 test issues, 359 are misclassified:

| True | Predicted | count | share of errors | share of all |
|---|---|---|---|---|
| **bug** | **question** | **96** | **26.7%** | 6.4% |
| question | feature | 70 | 19.5% | 4.7% |
| feature | question | 57 | 15.9% | 3.8% |
| bug | feature | 53 | 14.8% | 3.5% |
| question | bug | 49 | 13.6% | 3.3% |
| feature | bug | 34 | 9.5% | 2.3% |

**`bug` misclassified as `question` is the single largest error type.** More
broadly, `question` is involved in 272 of 359 errors (76%) as either the true or
the predicted class. Any class appears in four of the six possible confusion
types, so an even spread would give 67%; `question` is over-represented, though
by less than the raw figure suggests. The sharper evidence that it is the problem
class comes from the confident errors in Section 6.5.

| Repository | F1 bug | F1 feature | F1 question | F1 avg | errors |
|---|---|---|---|---|---|
| facebook/react | 0.9175 | 0.8302 | 0.7526 | 0.8334 | 50 |
| tensorflow/tensorflow | 0.8290 | 0.8526 | 0.7650 | 0.8155 | 56 |
| opencv/opencv | 0.6630 | 0.8195 | 0.7664 | 0.7496 | 74 |
| microsoft/vscode | 0.6957 | 0.7117 | 0.7629 | 0.7234 | 83 |
| bitcoin/bitcoin | 0.6374 | 0.7586 | 0.6419 | 0.6793 | 96 |

`bug` is the *best* class on the two easy projects and the *worst* on the two
hardest. Bug reports in a JavaScript UI library are lexically distinctive —
crash, render, hook, component — while bug reports in a cryptocurrency node or a
computer-vision library read much like questions about the same subsystems.

### 6.2 The class boundary is genuinely soft

A user who is unsure whether observed behaviour is a defect writes a report that
reads as both a bug and a question; the label reflects a maintainer's later
judgement, not anything determinable from the text. `[Bug] Forders not opening`
— a four-word title with a typo — carries no information that distinguishes the
two. The `question → feature` confusion has the same character in reverse:
titles such as `please allow for multiple tunnels on same machine` are labelled
`question` but are, on any reading, feature requests.

### 6.3 What misleads the model

Ranking terms by their lift in a specific confusion shows *why* the model errs.

**`question → bug`** is dominated by fragments of OpenCV's issue template:

| Term | in errors | in correct | lift |
|---|---|---|---|
| `forum.opencv.org,` | 32.7% | 1.8% | 15.6× |
| `overflow,` | 32.7% | 1.8% | 15.6× |
| `(videos,` | 28.6% | 1.6% | 15.6× |
| `solution` | 32.7% | 3.1% | 9.6× |

These are pieces of the boilerplate OpenCV inserts into new issues — a line
directing users to its forum, Stack Overflow and documentation. The model has
learned the template rather than the issue. **`bug → question`** shows the same
pathology with different boilerplate — `reproduce**`, `**actual`, `behavior**` —
the residue of issue-template headings that our Markdown stripping left behind.

This is the clearest actionable finding in the analysis: **template boilerplate
correlates with class within a project, and the per-project protocol lets each
classifier overfit to its own project's conventions.** More aggressive template
stripping is the cheapest available improvement we did not implement.

### 6.4 Document length

| Length quintile | issues | median words | accuracy |
|---|---|---|---|
| 1–70 words | 300 | 41 | 0.7467 |
| 70–127 | 307 | 103 | 0.7655 |
| **127–178** | **297** | **152** | **0.8148** |
| 178–280 | 296 | 219 | 0.7601 |
| 280–5,050 | 300 | 394 | **0.7167** |

Accuracy peaks in the middle quintile, whose median of 152 words almost exactly
matches the dataset median, and falls at both ends. Very short issues carry too
little text; very long ones are diluted by log output and partly discarded by
truncation. **The model performs best on the typical issue and degrades on both
tails.**

### 6.5 Confident errors and label quality

Among the 25 errors the model was most confident about, the true label is
`question` 14 times, `feature` 6 and `bug` 5. More striking: **nine of the
twenty-five have a title that explicitly declares a different type than their
label**, and in eight of those nine the model agreed with the title rather than
the label.

| True label | Title declares | Model said | Title |
|---|---|---|---|
| question | feature | feature | `Allow webview context menus triggered by primary click` |
| question | feature | feature | `please allow for multiple tunnels on same machine` |
| feature | question | question | `How to use opencv to erase text from images...` |
| question | bug | bug | `Error when runnning tensorflow.python.ops.sparse_ops...` |
| bug | question | question | `Forders not opening` |

These labels come from maintainers applying project conventions, not from an
annotation protocol with adjudication. **A ceiling below 1.0 is built into the
dataset**, and part of the remaining gap between any model and perfect
classification is unreachable. The published baseline faces the same ceiling,
which is why the comparison against it remains fair.

At least one confident error is an issue written in Portuguese
(`[Bug] Tela preta no terminal do VS Code`). The dataset is not language-filtered,
but this is one case in twenty-five, not a major error source.

### 6.6 Summary

| Finding | Evidence | Actionable? |
|---|---|---|
| `question` is the problem class | 76% of errors vs 67% expected; 14/25 confident errors | Needs better features, not tuning |
| Template boilerplate is learned | OpenCV forum line at 15.6× lift | **Yes** — strip templates harder |
| Accuracy falls on both length tails | 0.81 mid-quintile vs 0.72 longest | Partly — revisit truncation |
| ~⅓ of confident errors look like label noise | 9/25 titles contradict their label | No — dataset ceiling |
| Project difficulty varies widely | 50 errors on React vs 96 on Bitcoin | No — data property |

---

## 7. Threats to validity

### 7.1 Construct validity

**The benchmark rewards knowing when an issue was filed.** No bug report
predates 2021, and a classifier reading only the creation date scores 0.7071
(Section 5.6). Text carries temporal proxies, so every model in this study may
have learned some of its score from *when* rather than *what*. We cannot
separate the two within this dataset. A benchmark resampled so that each class
spans the same period would be needed; our figures are best read as an upper
bound on how well these models identify issue *type*.

**The labels are not ground truth.** Nine of the twenty-five most confident
errors have a title declaring a different type than their label (Section 6.5).
The labels come from maintainers applying project conventions, without
adjudication or reported inter-annotator agreement, so part of every model's
residual error is unreachable.

**The class boundary between `bug` and `question` is genuinely soft.** A user
unsure whether behaviour is defective writes a report that is legitimately both.

**The published baseline was not re-executed in our environment.** Our figures
are compared with published scores and with the organisers' supplied re-run
(0.8240), not with a run of their code on our hardware. Since the organisers'
own two figures differ by 0.3 points, the reference point is itself approximate.

### 7.2 Internal validity

**Most of the leaderboard's finer ordering is not statistically resolved.** The
minimum detectable difference is 1.8–3.1 points overall for real model pairs,
and 3.9 against the published baseline (Section 4.6). Section 5.7 applies these
thresholds: only the fine-tuning and frozen-encoder effects are detectable under
any assumption. Rankings among models within about three points of each other
should be read as ties.

**The test split is evaluated once per model.** Confidence intervals come from
the power analysis rather than from repeated sampling, and we did not bootstrap
the test set.

**Selection used cross-validation, but the choice of models to report was
informed by test results.** The ensembles of Section 5.5 exist because the
per-project test profiles looked complementary — a mild form of test-informed
design that we flag rather than claim complete isolation from.

**One confound was found and removed; one may remain.** The encoder comparison
was initially confounded with training settings and corrected by a matched
re-run. MPNet still runs at a smaller batch than the library default because of
memory limits, so its score is a lower bound.

**SetFit's hyperparameters were not searched.** Twenty iterations and one epoch
were used throughout; MPNet's contrastive loss collapsed to about 10⁻⁴, so fewer
iterations might generalise better. Our reproduction is a lower bound on the
method.

### 7.3 External validity

**Five projects is a small sample**, all large, mature, English-language
repositories, four of five using issue templates. The per-project spread —
0.6793 to 0.8334 for the same model — suggests transfer should not be assumed.

**Three classes is a simplification.** Real triage involves duplicates,
documentation, security and support, and the organisers removed multi-labelled
issues.

**300 training examples per classifier is a property of this competition.** A
project with years of labelled history would support different approaches, and
the finding that neural models do no better than TF-IDF would likely not hold.

### 7.4 Reproducibility

Every result was regenerated three times, from clean checkouts of the final code
on one machine (an NVIDIA RTX 3060, Python 3.11, PyTorch 2.14, scikit-learn 1.9,
sentence-transformers 6.1, SetFit 1.2). **Twenty-one of twenty-two models
returned bit-identical scores in all three runs**, including every scikit-learn
pipeline, both neural models, three of four SetFit variants and all four
ensembles.

The exception is SetFit on MPNet, and it is informative:

| Repository | run 1 | run 2 | run 3 | range |
|---|---|---|---|---|
| bitcoin/bitcoin | 0.7520 | 0.7488 | 0.7553 | 0.0066 |
| facebook/react | 0.8401 | 0.8438 | 0.8438 | 0.0037 |
| microsoft/vscode | 0.8203 | 0.8104 | 0.8104 | 0.0099 |
| opencv/opencv | 0.7676 | 0.7771 | 0.7801 | 0.0125 |
| tensorflow/tensorflow | 0.8710 | 0.8710 | 0.8642 | 0.0068 |
| **cross-repository mean** | **0.8102** | **0.8102** | **0.8108** | **0.0006** |

**Per-project scores move about twenty times as much as the cross-repository
mean**, because the protocol averages five independent classifiers whose
deviations partly cancel. The reported metric is therefore far more robust than
any single project's figure.

The `tensorflow` row makes the point concretely. In the first two runs this
model scored 0.8710 there, above the published baseline's 0.8644; in the third it
scored 0.8642, below it. Had we reported "our reproduction beats the baseline on
`tensorflow`" — as an earlier draft of this report did — the claim would have
been true or false depending on which run was used. Per-project figures in this
report show *where* models differ, not whether one beats another on one project.

**Across devices the picture differs.** The CNN scores 0.7578 on a CPU and 0.7438
on a GPU from identical code and seed. Each device reproduces its own number
exactly; the two backends compute the same operations with different kernels
and accumulation orders. Neural results must be reported with the device that
produced them — ours are from a GPU.

### 7.5 Three explanations that did not survive testing

We record these because each correction carries more information than the claim
it replaced.

**The cross-device gap was not nondeterminism.** We first attributed the CNN's
CPU/GPU difference to cuDNN's autotuned kernels and changed the seeding to request
deterministic ones. Re-running returned *exactly* the same numbers: training was
already deterministic per device. The flags remain as insurance, not as a fix.

**SetFit's run-to-run variation was overstated.** We first estimated it at about
0.011, from an ensemble score that moved between two runs. Those runs spanned a
change to SetFit's defaults, confounding a code change with run-to-run variation.
At fixed code, the cross-repository score varies by 0.0006 across three runs.

**The detectable difference is not a single number.** We first adopted a threshold
of "about 3 points overall, 7 per project" for every comparison. That figure
assumed a 16% disagreement rate between the compared models. Measured on real
pairs, the threshold ranges from 1.8 to 3.1 points, depending on how similar the
two models are.

Each had the same shape: an explanation was adopted without being tested against
the prediction it implied.

### 7.6 Summary of limitations

| Threat | Severity | Mitigated? |
|---|---|---|
| Temporal confound in the benchmark | High | No — measured and reported, not removable here |
| Label noise / soft class boundary | High | No — dataset property, affects baseline equally |
| Finer leaderboard ordering unresolved | Medium | Partly — thresholds quantified per comparison |
| Published baseline not re-executed locally | Medium | Partly — organisers' own re-run available |
| Five English projects only | Medium | No |
| SetFit hyperparameters unsearched | Medium | No — reproduction is a lower bound |
| MPNet limited by memory | Low | Partly — matched comparison run |
| Per-project score instability | Low | Yes — quantified; the averaged metric is stable |
| Cross-device neural variation | Low | Yes — quantified; device reported |

---

## 8. Conclusion

### 8.1 What we did

We evaluated twenty-two models on the NLBSE'24 issue report classification task
through a single harness, so that every entry on the leaderboard is computed
identically. Our best result, a soft-voting ensemble of SetFit, TF-IDF and a
frozen MPNet encoder, scores **0.8168** against the published baseline's 0.8270 —
a gap the test set cannot resolve. Our reproduction of the published method
reaches 0.8108, against the organisers' own re-run of 0.8240.

### 8.2 What we learned

**The benchmark rewards knowing when an issue was filed.** A classifier that
never reads text scores 0.7071, because no bug report predates 2021. The
meaningful floor is ~0.70, and every result above it should be read against that.

**Representation matters more than classifier choice.** Every purely lexical
model lands between 0.70 and 0.76. What moved the number was a better pretrained
representation (+0.050) and adapting it to the task through contrastive
fine-tuning (+0.076) — the only two effects in the study large enough to be
established beyond doubt.

**With 300 training examples per classifier, learned representations do no better
than engineered ones.** Neither neural model outscored TF-IDF.

**Conventional preprocessing was harmful.** Stop-word removal and lemmatisation
lowered macro F1 in every configuration tested, because the modal words they
discard are what marks a feature request.

**Most small differences are not differences.** The test set can detect gaps of
1.8–3.1 points between real model pairs, and much of our leaderboard sits inside
that. Models that score alike can still fail differently, which is what made
ensembling worthwhile.

### 8.3 What we got wrong

Four expectations did not survive measurement. We assumed aggressive text
cleaning would help; the ablation showed it hurt. We assumed the encoder and
fine-tuning effects would compose additively; the projection missed by 0.037, and
our first correction was itself confounded by training settings. We attributed a
cross-device discrepancy to nondeterminism; the fix changed nothing. And we
adopted a single detectability threshold for every comparison, which turned out
to depend on the pair of models compared.

Each had the same shape: an explanation was adopted without being tested against
the prediction it implied. If this project has one methodological lesson, it is
that an explanation never tested against a prediction is not yet a finding.

### 8.4 Future work

In order of expected value:

**Remove the temporal confound.** Resampling the benchmark so that each class
spans the same period — or reporting results within matched time windows — would
separate what these models learn about issue type from what they learn about
issue age. This is the largest open question about the benchmark itself.

**Strip issue-template boilerplate more aggressively.** OpenCV's forum-referral
line appears at 15.6× lift in one confusion. This is the clearest defect we
identified and did not fix.

**Re-execute the published baseline locally**, to attribute the remaining gap
between method, encoder choice and library versions.

**Search SetFit's hyperparameters.** The contrastive loss collapsed to about 10⁻⁴,
suggesting the objective was over-solved; our reproduction is a lower bound.

**Report bootstrap confidence intervals** alongside the power analysis, so that
every row of the leaderboard carries its own uncertainty.

---

## References

- Kallis, R., Colavito, G., Al-Kaswan, A., Pascarella, L., Chaparro, O., Rani, P.
  *The NLBSE'24 Tool Competition.* Proceedings of the 3rd International Workshop
  on Natural Language-based Software Engineering (NLBSE), 2024.
- Kallis, R., Di Sorbo, A., Canfora, G., Panichella, S. *Predicting issue types on
  GitHub.* Science of Computer Programming 205, 2021.
- Colavito, G., Lanubile, F., Novielli, N. *Few-Shot Learning for Issue Report
  Classification.* NLBSE, 2023.
- Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M.,
  Pereg, O. *Efficient Few-Shot Learning Without Prompts.* arXiv:2209.11055, 2022.
- McNemar, Q. *Note on the sampling error of the difference between correlated
  proportions or percentages.* Psychometrika 12(2), 1947.

---

## Appendix: reproducing this report

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-gpu                             # or make install without a GPU
make test                                    # expect 80 passed, 1 skipped
make data
python scripts/run_experiments.py --all      # ~20 minutes
python scripts/run_experiments.py --embeddings --setfit   # GPU, ~50 minutes
python scripts/run_experiments.py --ensemble  --setfit    # GPU, ~25 minutes
make notebooks                               # ~10 minutes
```

Every number in this report regenerates from these commands. Full instructions
are in `RUNBOOK.md`.
