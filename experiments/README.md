# Experiments

## `benchmark_audit.py` — start here

Reproduces every number in [`docs/benchmark-audit.md`](../docs/benchmark-audit.md)
using only the `ai4se` package, so anyone who can run `make data` can verify it.

```bash
pip install -e ".[ml]"
python experiments/benchmark_audit.py
```

About two minutes on a laptop CPU. Writes `results/benchmark_audit.json`.

It runs five steps: the classical baseline with a bootstrap confidence interval;
the label-from-timestamp confound; time-aware split against a matched random
control; McNemar with Holm correction; and the minimum detectable difference of
the test set.

## `exploratory/` — the raw working scripts

The unpolished scripts the audit grew out of. They are kept for provenance: each
figure in the audit can be traced back to the script that first produced it. They
are standalone `pandas` + `scikit-learn` programs that read the CSVs directly and
print to stdout — they do **not** use the `ai4se` package and are not part of the
supported surface. Use `benchmark_audit.py` for anything that goes in the report.

| Script | What it produced |
|---|---|
| `eda.py` | Class and length distributions, structural noise rates, duplicates, temporal structure, distinctive terms per class, first TF-IDF baselines, confusion matrix |
| `exp2.py` | Markdown cleaning strategies, structure-only classifier, first time-aware split, leave-one-project-out transfer, learning curve |
| `exp3.py` | Isolation of the temporal confound: label from timestamp alone, balance-preserving chronological split, vocabulary drift |
| `rigor.py` | Bootstrap intervals, exact McNemar, power simulation, seed variance, permutation negative control, accuracy-vs-coverage, leakage audit |

They expect the competition CSVs in `data/raw/`, which `make data` fills:

```bash
make data
python experiments/exploratory/rigor.py
```

`rigor.py` additionally needs `scipy` (included in the `.[ml]` extra).

### Findings from `exploratory/` not yet folded into the audit

Worth picking up by whoever takes tracks B and C:

- **Structural cues alone reach 0.556** (15 boolean flags, no words at all).
  Deleting code blocks and Markdown discards real signal; stack traces appear in
  32% of bugs against 8% of features, and leftover HTML comments from the issue
  template in 30% of questions against 10% of bugs. A permutation negative
  control confirms it is signal and not capacity: shuffled labels give
  0.340 ± 0.039, z = 5.5.
- **The body carries roughly twice the signal of the title** — title only 0.639,
  body only 0.749, both 0.765. Truncation should protect the body.
- **Learning curve is flat after ~100 labels per project**: 25 → 0.618,
  100 → 0.701, 300 → 0.743. A new project needs about 100 hand-labelled issues to
  reach most of what n-grams can do.
- **Near-duplicate leakage**: 61 test issues (4.1%) have a train issue at cosine
  ≥ 0.90, and 3 are exact duplicates. Small, but the same order of magnitude as
  the differences competitors argue over — worth a sensitivity analysis.
- **Two models disagree on 17% of items**, and accuracy on that subset collapses
  from 79% to 63%. That subset is where the label noise lives, and where manual
  re-annotation should be sampled from.
