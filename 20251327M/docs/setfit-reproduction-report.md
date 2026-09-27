# Reproducing the NLBSE'24 SetFit baseline, and failing to beat it

**Author:** Minh Khanh · **Branch:** `minh-khanh` · **Track:** D (baseline & evaluation)

Every figure below is read out of `results/setfit-repro/`. Nothing here is an
estimate, and nothing is carried over from an earlier draft.

```bash
# on a GPU with >= 24 GB, about 4.6 hours end to end
python setfit-repro/run_phase1.py --seeds 41 42 43 44 45
python setfit-repro/run_phase2.py select
python setfit-repro/run_phase2.py final --config baseline
```

---

## Summary

**1. The baseline reproduces.** Averaged over the five preregistered seeds the
reproduction scores **0.8224**, against a published 0.8240. The gap is
**0.16 points** — a fifth of the noise floor.

**2. Reporting one seed would have been misleading, and the organisers' own
seed is the unlucky one.** Seed 42 scores 0.8138, the *worst* of the five.
Reporting it alone makes the reproduction look 1.02 points short of a figure it
actually matches.

**3. Nine candidate improvements were tried and none beat the control.** The
best was the unmodified official configuration. This is reported as the result,
not as a preliminary before a better one.

**4. The one change large enough to clear the decision threshold made things
worse.** Stripping issue-template boilerplate — the fix the group report
recommends most confidently — cost **3.03 points** on validation.

**5. Per-project scores on `bitcoin/bitcoin` swing 4.47 points from the seed
alone**, which is wider than any gap in the group's leaderboard. Single-seed
per-project comparisons on this benchmark carry almost no information.

---

## 1. What was run

| | |
|---|---|
| GPU | NVIDIA RTX 4090, 47.5 GiB |
| Configuration | `all-mpnet-base-v2`, batch (16, 2), 1 epoch, 20 contrastive iterations |
| Sequence length | 384 tokens, pinned explicitly (see §7) |
| Text | `title + " " + body`, as the organisers assemble it |
| Protocol | one classifier per project, five per run |
| Seeds | 41, 42, 43, 44, 45 — fixed in `preregistration.md` before any run |
| Total GPU time | ~4.6 h (1.7 h reproduction, 2.6 h selection, 0.3 h final) |

Library versions are recorded in every results file. They are **not** the
generation the organisers used, and that is a threat to validity rather than a
detail — see §7.

---

## 2. The reproduction

| Seed | Cross-repository F1 | 95% CI |
|---|---|---|
| 41 | 0.8286 | [0.8094, 0.8476] |
| 42 | 0.8138 | [0.7944, 0.8342] |
| 43 | 0.8279 | [0.8090, 0.8476] |
| 44 | 0.8192 | [0.8004, 0.8384] |
| 45 | 0.8224 | [0.8029, 0.8422] |
| **mean ± sd** | **0.8224 ± 0.0062** | |

Against the two published figures:

| Published figure | Source | Our mean − published |
|---|---|---|
| 0.8240 | organisers' committed results file | **−0.16 points** |
| 0.8270 | competition README | −0.46 points |

Both sit inside the confidence interval of every individual seed. The
reproduction succeeds.

The two published numbers differ from each other by 0.3 points and neither the
README nor the results file explains why. Reproductions of this baseline should
say which one they are comparing against; this report uses 0.8240 because it is
the figure the organisers' own output file carries.

### Per project

| Project | 5-seed mean | sd | Spread | Published | Delta |
|---|---|---|---|---|---|
| tensorflow/tensorflow | 0.8650 | 0.0035 | 0.94p | 0.8644 | +0.06p |
| facebook/react | 0.8576 | 0.0054 | 1.31p | 0.8718 | −1.42p |
| opencv/opencv | 0.8203 | 0.0158 | 3.83p | 0.8173 | +0.30p |
| microsoft/vscode | 0.8144 | 0.0093 | 2.58p | 0.8262 | −1.18p |
| bitcoin/bitcoin | 0.7547 | 0.0203 | 4.47p | 0.7555 | −0.08p |

Three of the five reproduce to within a third of a point. The shortfall is
concentrated in `react` and `vscode`, not — as a single seed suggested — in
`bitcoin`.

---

## 3. Seed variance, and a correction

Measured spread across the five seeds: **1.48 points**. Standard deviation
0.0062. Reporting only the best seed would have inflated the headline by 0.62
points.

`preregistration.md` states the spread as 2.4 points, measured on the classical
baselines in `docs/benchmark-audit.md`. **That figure does not transfer.** SetFit
at the official configuration is more stable than the classical baselines were,
and the honest number for this model is 1.48. The preregistered figure is left
where it stands; this is the correction, recorded rather than quietly swapped.

Seed stability is strongly project-dependent, and the variation is the finding:

| Project | Spread from seed alone |
|---|---|
| bitcoin/bitcoin | **4.47 points** |
| opencv/opencv | 3.83 points |
| microsoft/vscode | 2.58 points |
| facebook/react | 1.31 points |
| tensorflow/tensorflow | 0.94 points |

`bitcoin` is both the hardest project and the least stable one. Any claim about
a single project, from any model, needs several seeds behind it before it means
anything. A per-project win of one or two points on `bitcoin` is indistinguishable
from a lucky draw.

---

## 4. Selection — nine candidates, no winner

Selection ran on a validation set carved out of the *training* split
(n = 375, stratified within each project × class cell). The test split was not
read; §6 gives the evidence.

| Candidate | Validation F1 | vs control | 95% CI | GPU min |
|---|---|---|---|---|
| **baseline** (control) | **0.7922** | +0.00p | [0.750, 0.835] | 15.5 |
| iters_10 | 0.7883 | −0.40p | [0.746, 0.829] | 8.0 |
| epochs_2 | 0.7870 | −0.52p | [0.746, 0.830] | 31.0 |
| iters_40 | 0.7838 | −0.84p | [0.741, 0.826] | 31.1 |
| strip_and_mask | 0.7798 | −1.25p | [0.736, 0.822] | 14.6 |
| mask_structure | 0.7775 | −1.48p | [0.736, 0.818] | 14.5 |
| encoder_e5 | 0.7755 | −1.67p | [0.733, 0.818] | 12.6 |
| encoder_bge | 0.7702 | −2.20p | [0.727, 0.812] | 12.6 |
| strip_templates | 0.7619 | **−3.03p** | [0.717, 0.807] | 15.4 |

Eight of the nine differences fall inside the decision threshold and are
reported as *no detectable difference*. One clears it, and it clears it in the
wrong direction.

Validation scores are lower than test scores throughout because each classifier
trains on 225 issues instead of 300. They rank candidates; they do not estimate
test performance, and they are not comparable with 0.8240.

### 4.1 Contrastive budget

`num_iterations` and `num_epochs` were untouched by the organisers, and the
group report names searching them as expected-value item 3, on the grounds that
the reproduction is therefore a lower bound on the method.

Three configurations tested that: 10 iterations (−0.40p), 40 iterations
(−0.84p), 2 epochs (−0.52p). All three lose to the default of 20 iterations and
1 epoch.

The training log explains why. Embedding loss falls from 0.2052 to 0.0011 within
a single epoch — the contrastive objective is solved early and stays solved.
Giving it more budget does not help because there is nothing left to optimise,
and giving it less does not help because it was already converging fast. **The
reproduction is not a lower bound in this direction.**

### 4.2 Preprocessing

`strip_templates` removes issue-template boilerplate and deliberately leaves
code blocks and stack traces alone. `mask_structure` does the opposite: it
replaces code, traces, images and URLs with typed placeholders (`xxcode`,
`xxtrace`, `xximg`, `xxurl`) and leaves templates in place.

Template stripping is the worst candidate in the study at −3.03 points. Masking
costs 1.48 points, and doing both costs 1.25.

This bears directly on the group report, which identifies template boilerplate
as *"the clearest actionable defect we identified and did not fix"* and ranks
stripping it second in expected value. On this evidence it is not a fix. Two
caveats keep that from being a refutation:

- The group's finding is on **TF-IDF with logistic regression**; this test is on
  **SetFit with MPNet**. A lexical model that matches the forum-referral line
  directly and a sentence encoder that reads structure as meaning need not
  respond the same way.
- My `strip_templates` is one implementation of an idea, not the idea itself. A
  better regex might behave differently.

The defensible statement is that the cheapest recommended fix made things
measurably worse for the strongest model in the study, and that anyone
proposing it should measure before recommending it again.

### 4.3 H2 was not directly tested

`preregistration.md` H2 predicts that typed masking of code and Markdown beats
deleting them outright, by +0.5 to +1.5 points.

**The comparison H2 names was not run.** The variant set has no "delete code
blocks outright" condition — `strip_templates` deletes templates, not code. The
nearest available comparison, masking (0.7775) against template stripping
(0.7619), favours masking by 1.56 points and so points the direction H2
predicted, at roughly the predicted size. It is not a test of H2 and is not
claimed as one. H2 also carried the instruction that it *"will not be claimed as
a result whatever happens"*, which stands.

### 4.4 Newer encoders

`bge-base-en-v1.5` (−2.20p) and `e5-base-v2` (−1.67p) both lose to
`all-mpnet-base-v2` from 2021, despite being trained later on more data.

Both were flagged in advance as contamination risks: the test set has been
public on GitHub since 2024 and these encoders postdate it. The flag turned out
not to matter, because neither won. That is the comfortable version of this
problem, and it was comfortable by luck rather than by design — had one of them
won by four points, the result would have needed a paragraph in Threats to
Validity and an argument about whether it was admissible at all.

---

## 5. The final run, and the noise floor

One configuration was taken to the test split: the control, because nothing beat
it.

| Project | Final run | Seed 42 | Difference |
|---|---|---|---|
| facebook/react | 0.8552 | 0.8552 | +0.00p |
| tensorflow/tensorflow | 0.8604 | 0.8638 | −0.34p |
| microsoft/vscode | 0.8297 | 0.8258 | +0.39p |
| opencv/opencv | 0.8038 | 0.7954 | +0.84p |
| bitcoin/bitcoin | 0.7289 | 0.7289 | +0.00p |
| **cross-repository** | **0.8156** | 0.8138 | **+0.18p** |

These two runs share a configuration, a seed and a machine. Everything
separating them is GPU nondeterminism. They disagree on **21 of 1,500 issues
(1.4%)**, and no project is significant after Holm correction — every adjusted
p-value is 1.000.

**The noise floor of the whole pipeline is 0.18 points.** Seed choice moves the
score roughly eight times further than rerunning does.

### A distinction worth keeping

The final run reports that this comparison resolves **1.0 point**, against a
preregistered threshold of 3.0. Those are different quantities and conflating
them is a mistake:

- **3.0 points** is the preregistered decision threshold, fixed by a power
  analysis before any experiment ran. It does not move.
- **1.0 point** is what *this particular comparison* resolves, and it follows
  from its own discordant rate. That rate is a property of the pair of models,
  not of the test set.

Two runs of one configuration disagree on 1.4% of issues and so resolve a small
difference. Two genuinely different models disagree far more often — at a
discordant rate of 21%, the same test set resolves about 3.5 points.

This matters outside this report. The group's best ensemble sits 1.02 points
below the published baseline, and the group report calls that gap *real*. Two
different model families will not have a 1.4% discordant rate; at a realistic
rate the resolution is roughly 3.5 points, and 1.02 falls well inside it.

---

## 6. Did selection ever read the test split?

`results/setfit-repro/test_access.log`, in full:

```
2026-09-27T05:58:58+00:00	smoke-test
2026-09-27T05:59:46+00:00	phase1-baseline
2026-09-27T06:25:30+00:00	phase1-baseline
2026-09-27T10:55:30+00:00	phase2-final:baseline
```

Four reads across a nine-hour session, each with a declared reason. Two are the
reproduction (seed 42, then the batch of 41/43/44/45); one is the smoke test;
one is the single final evaluation. **Selection appears nowhere**, across nine
configurations and 2.6 hours of GPU time.

This is what operational commitment 1 of the preregistration promised, and it is
the difference between asserting that selection was clean and being able to show
it.

---

## 7. Threats to validity

**Library drift is large and unfixable.** The organisers ran SetFit ~1.0 and
Transformers ~4.39. This run used:

| Package | Ours |
|---|---|
| Python | 3.12.14 |
| torch | 2.11.0+cu128 |
| transformers | 5.17.0 |
| sentence-transformers | 6.1.0 |
| setfit | 1.2.0 |
| datasets | 5.0.1 |
| scikit-learn | 1.9.1 |
| numpy | 2.5.3 |

The organisers' generation does not install on Python 3.12 and is not safe
alongside torch 2.11. Some unknown share of the 0.16-point gap is library drift
rather than method. Recording the versions is the only honest handling
available; it does not remove the threat.

**One behavioural difference is known and was controlled.** Newer
sentence-transformers releases do not apply the encoder's configured
`max_seq_length`, where older ones did. Left unset, a 21,595-word issue goes
through the encoder whole and exhausts a 15 GB GPU — which is how the first
attempt at this run died. `max_seq_length` is therefore pinned to 384, the
checkpoint's own limit. This is a deviation from the organisers' notebook as
written and a restoration of the organisers' notebook as it behaved.

**The dataset contains train/test overlap, which nobody reports.**

| Check | Count | Share of test |
|---|---|---|
| Exact duplicate (title, body) across train and test | 3 | 0.20% |
| Identical titles across train and test | 23 | 1.53% |
| Cosine ≥ 0.99 | 5 | 0.33% |
| Cosine ≥ 0.95 | 28 | 1.87% |
| Cosine ≥ 0.90 | 60 | **4.00%** |
| Cosine ≥ 0.80 | 131 | **8.73%** |

Near-duplicates at 0.90 are unevenly spread: `react` 20, `opencv` 20,
`tensorflow` 12, `vscode` 6, `bitcoin` 2. The two projects with the most overlap
are not the two easiest, so this is not a clean explanation of project
difficulty — but `bitcoin`, the hardest project for every model including the
baseline, has the least.

This is small, and it inflates our score and the baseline's equally, so it does
not affect the comparison. It is the same order of magnitude as the differences
this benchmark is used to report, which is reason enough to state it.

**Selection used one validation split, not cross-validation.** Nine
configurations were ranked on a single 375-item holdout. With intervals that
wide, the ranking below the control is not reliable; the only conclusion drawn
from it is that nothing beat the control, which is robust to the ordering.

**One training-data scope was never tested.** Fine-tuning a single encoder on
all 1,500 training issues and fitting five per-project heads is five classifiers
trained only on provided data, and therefore legal by the letter of the rules.
The rules require five classifiers and say nothing about how much data each may
see. It is gated behind `--allow-global` and was not run. **The organisers or
the course lecturer should rule on that reading before anyone uses it**, and the
result should be reported under both scopes either way. Reading an ambiguity in
your own favour and staying quiet is the failure mode this whole setup exists to
prevent.

---

## 8. What this does not show

The primary hypothesis H1 — a fine-tuned DeBERTa-v3 with structure-aware
preprocessing reaching ≥ 0.854 — **was not tested**. Neither were H3 (repository
tokens on a pooled model), H4 (chronological split) or H5 (Platt scaling with
abstention). This report covers the reproduction and the SetFit improvement
attempts only, and it does not license any claim about the others.

Nothing here shows the benchmark cannot be beaten. It shows that nine specific
attempts did not beat it, on one validation split, with one model family.

---

## 9. Context: how much headroom is there

| | F1 |
|---|---|
| SetFit baseline (2024) | 0.8270 |
| Best published result on this task (fine-tuned GPT-3.5) | **0.828** |
| Fine-tuned GPT-4o | 0.807 |

Two years of published work moved this benchmark by **0.1 points**, and a
fine-tuned GPT-4o scored *below* the baseline. Colavito and colleagues measured
0.8321 on relabelled data against 0.7767 on the challenge labels, which puts a
ceiling near 0.83 that comes from label quality rather than from modelling.

A negative result on a benchmark in that state is not a failed experiment. It is
the expected outcome, and the value of running it is the machinery that makes
the negative statement checkable: five seeds, an access log, per-issue
predictions, and a threshold fixed before the first run.

---

## Files

| Path | Contents |
|---|---|
| `setfit-repro/` | the harness; `run_phase1.py`, `run_phase2.py`, `src/` |
| `results/setfit-repro/phase1_seed4*.json` | five reproduction runs, per-issue predictions |
| `results/setfit-repro/select_*.json` | nine selection candidates |
| `results/setfit-repro/final_baseline.json` | the single test-split run |
| `results/setfit-repro/leakage_audit.json` | the overlap figures in §7 |
| `results/setfit-repro/test_access.log` | every read of the test split |
| `preregistration.md` | hypotheses, threshold and seeds, fixed in advance |
| `reflection.md` | what went wrong and what I would change |
