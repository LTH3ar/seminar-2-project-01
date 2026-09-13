# Benchmark audit — what the NLBSE'24 test set can and cannot tell us

**Author:** Minh Khanh · **Branch:** `minh-khanh` · **Track:** D (baseline & evaluation)

Reproduce everything here with:

```bash
pip install -e ".[ml]"
python experiments/benchmark_audit.py
```

Runtime is about two minutes on a laptop CPU. Numbers are written to
`results/benchmark_audit.json`. Every figure below is output of that script, not
an estimate.

---

## Summary

Two properties of this benchmark change how the rest of the project should be run.

**1. The test set cannot resolve small differences.** The minimum detectable
difference is **3.0 F1 points** on the full 1,500-item test set and **7.0 points**
within a single project. Most gaps reported between competition entries are
smaller than that.

**2. Class and creation date are confounded.** A model reading *only* the
timestamp — no text at all — scores **0.684** where chance is 0.333. Evaluating on
a chronological split instead of the official random one costs **8.1 F1 points**.

Neither finding stops us competing. Both change what we are allowed to claim.

---

## 1. Baseline under the official protocol

TF-IDF (1–2 grams) + LinearSVC, no cleaning, raw title + body.

| Configuration | Cross-repo F1 |
|---|---|
| One model for all five projects | 0.7653 |
| One model per project *(what the competition requires)* | 0.7520 |
| **Official SetFit baseline** | **0.8270** |

95% bootstrap CI on the first row, stratified by project: **[0.7440, 0.7867]** —
a width of 4.3 points. That interval is wider than most improvements claimed in
this competition, which is the first hint of the problem in section 4.

The single number hides a wide spread. Per project and class:

| Project | bug | feature | question |
|---|---|---|---|
| facebook/react | **0.922** | 0.764 | **0.632** |
| tensorflow/tensorflow | 0.819 | 0.838 | 0.734 |
| bitcoin/bitcoin | 0.705 | 0.821 | 0.698 |
| microsoft/vscode | 0.727 | 0.748 | 0.762 |
| opencv/opencv | 0.693 | 0.802 | 0.783 |

Range 0.632 to 0.922 — a 29-point spread inside one 0.765 average. **The report
needs all fifteen cells, not the mean.** `ai4se.evaluation.classification_rows`
produces this table directly.

`question` is the weakest class almost everywhere. That is coherent rather than
accidental: *question* describes the speech act of the person writing, while
*bug* and *feature* describe technical content. The three classes do not sit on a
single axis, which is a construct-validity limit of the benchmark, not a modelling
failure.

---

## 2. The temporal confound

A `GradientBoostingClassifier` given exactly one feature, `created_at`, and no
text whatsoever:

| Project | F1 from timestamp alone |
|---|---|
| facebook/react | 0.802 |
| tensorflow/tensorflow | 0.792 |
| bitcoin/bitcoin | 0.622 |
| opencv/opencv | 0.620 |
| microsoft/vscode | 0.587 |
| **mean** | **0.684** |

Chance is 0.333. For `facebook/react` the clock alone beats our TF-IDF model
reading the full text of the issue.

**Mechanism.** The dataset was built by sampling exactly 100 issues per class per
project. Class prevalence drifts over a project's lifetime, so fixing the count
per class makes the sampling date carry class information. Measured: in **5 of 5
projects** the median creation date of `bug` is the latest of the three classes.
The gap between the earliest and latest class median ranges from 50 days
(vscode) to **1,522 days** (react).

**It cannot be fixed by dropping the column.** Adding `created_at` as an explicit
feature to a text model changes nothing (0.7627 → 0.7633): the signal is already
in the vocabulary of each era. Jensen–Shannon divergence between the vocabulary
of the older and newer half of the data is 0.1475, with era-specific tokens
(`useeffect`, `setstate`, `jsfiddle` in the old half; `webgl`, `insiders`,
`rasterization` in the new).

---

## 3. What the random split costs

`ai4se.evaluation.splits.time_aware_split` puts the oldest 50% of **each
(project, class) cell** into training and the newest 50% into test. Splitting
inside the cell is what keeps class balance identical; splitting globally instead
collapses the training set to 37 bugs against 814 features and confounds the
temporal effect with an imbalance effect.

`random_split_control` is the matched comparison: same cell sizes, same balance,
random cut instead of chronological.

| Split protocol | Cross-repo F1 |
|---|---|
| Random (matched control) | 0.7713 |
| Time-aware | 0.6907 |
| **Difference** | **−0.0807** |

Eight points is well above the 3.0-point detection threshold, so unlike most
results in this project, this one is solid.

What it means in practice: the official protocol reports the score of a model
that has been allowed to see the future. A tool deployed on a real issue tracker
only ever sees the past. The gap between those two numbers is the honest estimate
of what deployment would look like.

---

## 4. Significance, and why it matters here

`ai4se.evaluation.significance` runs exact McNemar per project and applies the
Holm–Bonferroni correction across the family of five tests.

Comparing one-model-per-project against one-model-for-all — a 1.3-point gap:

| Project | n01 | n10 | discordant | p | p (Holm) | |
|---|---|---|---|---|---|---|
| bitcoin/bitcoin | 17 | 9 | 26 | 0.169 | 0.843 | ns |
| facebook/react | 14 | 17 | 31 | 0.720 | 1.000 | ns |
| microsoft/vscode | 17 | 10 | 27 | 0.248 | 0.991 | ns |
| opencv/opencv | 16 | 14 | 30 | 0.856 | 1.000 | ns |
| tensorflow/tensorflow | 21 | 15 | 36 | 0.405 | 1.000 | ns |
| POOLED | 85 | 65 | 150 | 0.121 | — | ns |

**0 of 5 projects reach significance.** The two configurations are not
distinguishable on this test set. Anyone reporting the 1.3-point gap as a result
would be reporting noise.

---

## 5. The detection threshold

Monte Carlo simulation of McNemar's test, using the discordant rate actually
observed between two reasonable models (0.161 of items):

| Test set | Minimum detectable difference, 80% power, α = 0.05 |
|---|---|
| Full 1,500 items | **3.0 F1 points** |
| One project, 300 items | **7.0 F1 points** |

For context, on top of that: repeating the same configuration with a different
random seed moves the score by up to **2.4 points** on its own (12 seeds, SGD
classifier, identical settings).

### What this means for tracks B, C and D

- Report a gain below 3.0 points as **"no detectable difference"**, never as an
  improvement. `ai4se.evaluation.power.is_conclusive` is a one-line guard for this.
- Never claim a per-project win from a gap under 7 points.
- Run **5 seeds minimum** and report mean ± sd, plus the raw values in an appendix.
- Beating SetFit's 0.8270 by a *detectable* margin means reaching roughly 0.857.
  That is the real target.

None of this makes the project harder. It makes the report defensible, and it
turns a weakness of the benchmark into something we can write about.

---

## Two notes on the current README

Both are small and offered as corrections, not criticism — section 2 depends on
the first one being right.

**Date range.** The README says the issues were "collected between January 2022
and September 2023". Measured from the CSVs, three of the five projects start
well before that:

| Project | Earliest issue | Latest issue |
|---|---|---|
| facebook/react | 2016-03-12 | 2023-08-26 |
| bitcoin/bitcoin | 2018-10-30 | 2023-09-13 |
| tensorflow/tensorflow | 2021-12-03 | 2023-09-29 |
| opencv/opencv | 2022-01-05 | 2023-09-29 |
| microsoft/vscode | 2023-06-14 | 2023-09-29 |

react spans 7.5 years, not 20 months. The span matters because it is what makes
the temporal confound large in exactly those two projects.

**`data/raw/` is not actually ignored.** The README says the cached CSVs are
git-ignored, but the root `.gitignore` has no rule for them — `make data`
followed by `git add .` commits 7 MB. This branch adds `data/raw/.gitignore` to
close the gap without touching the root file. Drop that one file if the team
would rather fix it centrally.

---

## What is in this branch

```
src/ai4se/evaluation/       metrics, significance, power, alternative splits
  metrics.py                cross_repo_f1, classification_rows, bootstrap_ci
  significance.py           mcnemar_exact, holm, compare_per_repo, cliffs_delta
  power.py                  minimum_detectable_difference, is_conclusive
  splits.py                 time_aware_split, random_split_control, stratified_folds
tests/test_evaluation.py    24 tests, all with hand-checkable expectations
experiments/                benchmark_audit.py reproduces this document
docs/benchmark-audit.md     this file
preregistration.md          hypotheses committed before the modelling starts
data/raw/.gitignore         the gap noted above
```

Nothing outside these paths is modified, so the branch merges cleanly and can be
reverted with a single `git revert`.

`ai4se.evaluation` imports with the **base** requirements — numpy only, no
scikit-learn, no scipy — so tracks B and C can use it before installing their
extras. Only `experiments/benchmark_audit.py` needs `.[ml]`.

### Suggested use by the other tracks

```python
from ai4se.evaluation import bootstrap_ci, compare_per_repo, is_conclusive

point, low, high = bootstrap_ci(y_true, y_pred, repos)
print(f"{point:.4f}  95% CI [{low:.4f}, {high:.4f}]")

if not is_conclusive(point - baseline_score):
    print("No detectable difference — do not call this an improvement.")

for row in compare_per_repo(baseline_pred, y_pred, y_true, repos):
    print(row["repo"], row["p_holm"], row["significant"])
```
