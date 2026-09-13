# Pre-registration — modelling experiments

**Author:** Minh Khanh · **Branch:** `minh-khanh`
**Committed before the first modelling experiment. Not to be edited afterwards.**

The point of this file is its Git timestamp. A hypothesis written down before the
result is evidence; the same sentence written afterwards is not. If a prediction
here turns out wrong, it stays wrong and gets reported as wrong — see the
*Predictions already falsified* section, which is not empty.

Everything below follows from the measurements in `docs/benchmark-audit.md`,
which were made on the baselines only, before any transformer was trained.

---

## Primary metric — fixed, not to be changed

Cross-repository F1: the arithmetic mean of five per-project **weighted-average**
F1 scores on `issues_test.csv`, computed by `ai4se.evaluation.cross_repo_f1`.
This is the metric the organisers' own notebook computes.

Other metrics may be reported. They are secondary and may not be used to claim
success.

> **Amendment, 2026-09-13, before any model was trained.** This line originally
> said *micro-F1*, which was a mistake on my part rather than a choice: micro-F1
> is accuracy, and it is not what the competition ranks on. Correcting the
> definition of the metric is not the same as changing the hypothesis, and the
> amendment is recorded here rather than made silently. The decision threshold,
> the hypotheses and the seeds are untouched. The commit that made this edit is
> still earlier than the commit of any modelling experiment, which is the
> property the file exists to guarantee.

## Decision threshold

**3.0 F1 points**, the measured minimum detectable difference of this test set at
80% power (`docs/benchmark-audit.md`, section 5).

A difference smaller than that is reported as *"no detectable difference"*. It is
not described as better, higher, improved, or an advantage. If the proposed model
does not clear the threshold, that is the finding, and we do not go looking for a
subgroup where it does.

---

## Hypotheses

### H1 — primary

A fine-tuned DeBERTa-v3 with structure-aware preprocessing beats the SetFit
baseline of 0.8240 by at least 3.0 points, i.e. reaches **≥ 0.854**.

*Prior:* plausible but not likely. Stated at roughly 40%.

### H2 — secondary

Typed masking of code and Markdown (`xxcode`, `xxtrace`, `xximg`, `xxurl`) beats
deleting them outright.

*Predicted effect: +0.5 to +1.5 points — below the threshold.* Recorded as a
direction, and it will not be claimed as a result whatever happens.

### H3 — secondary

Prepending a `[REPO=facebook/react]` token to a single pooled model helps.

*Predicted effect: +1.0 to +2.0 points — below the threshold.*

### H4 — secondary

A chronological split lowers every model's score relative to the official random
split.

*Predicted effect: −8 to −10 points.* Already measured at −8.8 for the TF-IDF
baseline; the prediction is that transformers behave the same way.

### H5 — secondary

With Platt scaling and an abstention option, accuracy at 90% coverage is
approximately **0.85**.

*This replaces an earlier prediction of "above 0.90", which was wrong — measured
0.792. The revised figure is deliberately anchored to that measurement.*

---

## Operational commitments

1. **The test set is touched once per configuration**, at the end. Access goes
   through a logged wrapper; the log ships in the appendix. If a configuration
   shows six test reads, the reader will see it.
2. **Seeds are fixed in advance: 41, 42, 43, 44, 45.** All five raw scores are
   published, not only the mean. Measured spread from seed alone is 2.4 points,
   so reporting a single run would be meaningless.
3. **Tuning budget is equal.** However many configurations the proposed model is
   given, SetFit gets the same number. Both counts go in a table. An
   under-tuned baseline is the most common way an honest paper reaches a false
   conclusion.
4. **Every configuration that is run appears in the results tables**, including
   the ones that failed.
5. **Every claimed improvement ships a negative control** — the same architecture
   with its signal destroyed but its capacity intact (shuffled labels, randomised
   repo tokens, Gaussian noise in place of the structural feature vector). If the
   destroyed version gains as much, the gain came from capacity, not from the
   idea.
6. **Multiple comparisons are corrected** with Holm–Bonferroni across the family
   of five per-project tests. Subgroup analyses are labelled exploratory and
   cannot support a claim.

---

## Predictions already falsified

Kept deliberately. The mechanism only has value if it is allowed to say no.

| Prediction | Outcome |
|---|---|
| "A pooled model beats per-project models" | 1.3-point gap, 0 of 5 projects significant after Holm. **Not distinguishable.** |
| "The title adds about 1 point over the body alone" | 1.3 points, not significant. Only the reverse holds: removing the *body* costs 12.4 points, p = 4e−16. |
| "Accuracy at 90% coverage will exceed 0.90" | Measured **0.792**. Off by 11 points. |

---

## Analysis plan, fixed in advance

| Step | Method |
|---|---|
| Point estimate | `cross_repo_f1`, mean over 5 seeds ± sd |
| Uncertainty | Stratified bootstrap, B = 10,000, 95% percentile interval |
| Paired comparison | Exact McNemar per project + Holm across the five |
| Effect size | Cliff's delta with the conventional interpretation bands |
| Power | Reported alongside every comparison, never omitted |
| Hyper-parameters | `StratifiedKFold(5)` on `(project, class)`, training split only |
| Secondary protocol | `time_aware_split`, all models re-run, rankings compared |
