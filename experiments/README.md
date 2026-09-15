# Experiments

Run in this order. Each script writes a JSON file to `results/`, and
`04_make_tables.py` turns those into every table in the report. No number in
the report is typed by hand.

| Script | What it does | Needs | Time |
|---|---|---|---|
| `benchmark_audit.py` | Baseline, temporal confound, split protocol, significance, power | `.[ml]` | ~2 min CPU |
| `02_baselines.py` | k-fold selection then test evaluation of six models, five seeds each | `.[ml,dl]` | ~10 min CPU |
| `03_setfit.py` | Reproduces the organisers' published baseline | `.[dl]` | 15 min GPU / 1–3 h CPU |

> **Run `03_setfit.py` on a GPU.** `all-mpnet-base-v2` has 110M parameters and
> each project generates 12,000 contrastive pairs, so a laptop CPU needs hours
> for what a T4 does in about fifteen minutes.
> [`notebooks/02_setfit_colab.ipynb`](../notebooks/02_setfit_colab.ipynb) does
> the whole thing on Colab — open it there, run every cell, and download the two
> result files into `results/`. The repository is private, so the notebook asks
> for a read-only token (or takes a zip upload instead).
| `04_make_tables.py` | Regenerates `report/tables/*.tex` from `results/*.json` | base | seconds |

```bash
pip install -e ".[ml,dl]"
make data
python experiments/benchmark_audit.py
python experiments/02_baselines.py
python experiments/03_setfit.py          # --preset fast to iterate
python experiments/04_make_tables.py
make report
```

`04_make_tables.py` writes a visible placeholder for any table whose experiment
has not run, so the report always compiles and never shows stale numbers.

## Two rules these scripts enforce

**The test set is read once, at the end.** All hyper-parameter selection
happens in `02_baselines.py` stage 1, by 5-fold cross-validation stratified on
(project, class) over the training split only. Inside each fold one model is
trained per project, mirroring the competition protocol rather than a
simplification of it.

**A difference below 3.0 F1 points is not an improvement.** That is the
measured detection threshold of this test set (`docs/benchmark-audit.md`).
`02_baselines.py` prints `detectable` or `NOT detectable` next to every
comparison, and `ai4se.evaluation.is_conclusive` is the one-line guard to use
elsewhere.

## `exploratory/` — the raw working scripts

The unpolished scripts the audit grew out of, kept for provenance: each figure
in the audit can be traced to the script that first produced it. They are
standalone `pandas` + `scikit-learn` programs that read the CSVs directly and
print to stdout. They do **not** use the `ai4se` package and are not part of
the supported surface.

| Script | What it produced |
|---|---|
| `eda.py` | Class and length distributions, structural noise rates, duplicates, temporal structure, distinctive terms, first TF-IDF baselines, confusion matrix |
| `exp2.py` | Markdown cleaning strategies, structure-only classifier, first time-aware split, leave-one-project-out transfer, learning curve |
| `exp3.py` | Isolation of the temporal confound: label from timestamp alone, balance-preserving chronological split, vocabulary drift |
| `rigor.py` | Bootstrap intervals, exact McNemar, power simulation, seed variance, permutation negative control, accuracy-vs-coverage, leakage audit |

They expect the CSVs in `data/raw/`, which `make data` fills. `rigor.py` also
needs `scipy`.

### Findings not yet folded into the main pipeline

Worth picking up by whoever extends tracks B and C:

- **Structural cues alone reach 0.556** — fifteen boolean flags, no words at
  all. Deleting code blocks and Markdown discards real signal: stack traces
  appear in 32% of bugs against 8% of features, leftover template HTML
  comments in 30% of questions against 10% of bugs. A permutation control
  confirms signal rather than capacity: shuffled labels give 0.340 ± 0.039,
  z = 5.5. Suggests replacing structure with typed tokens (`xxcode`,
  `xxtrace`) instead of stripping it.
- **The body carries roughly twice the signal of the title** — title only
  0.639, body only 0.749, both 0.765. Truncation should protect the body.
- **The learning curve is flat after ~100 labels per project**: 25 → 0.618,
  100 → 0.701, 300 → 0.743. A new project needs about 100 hand-labelled issues
  to reach most of what n-grams can do.
- **Near-duplicate leakage**: 61 test issues (4.1%) have a training issue at
  cosine ≥ 0.90; three are exact duplicates. Small, but the same order of
  magnitude as the differences competitors argue over.
- **Two models disagree on 17% of items**, and accuracy on that subset falls
  from 79% to 63%. That subset is where the label noise concentrates, and
  where manual re-annotation should be sampled from.
