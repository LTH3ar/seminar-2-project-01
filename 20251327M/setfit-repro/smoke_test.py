"""Exercise everything that does not need a GPU, before renting one.

Run this on any machine first. It checks that the data loads, that the metric
reproduces the organisers' published per-project figures when fed their own
predictions, that the statistics behave, and that the preprocessing variants
do what they claim -- so the only thing left to go wrong on the rented pod is
the training itself.

    python smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metrics, preprocess, stats  # noqa: E402
from src.data import LABELS, REPOSITORIES, describe, load_dataset  # noqa: E402
from src.runner import OFFICIAL, gpu_report  # noqa: E402

ROOT = Path(__file__).resolve().parent
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    """Record one check."""
    checks.append((name, passed, detail))
    print(f"  [{'ok' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


print("=" * 74)
print("SMOKE TEST -- everything that does not need a GPU")
print("=" * 74)
print(f"  GPU: {gpu_report()}\n")

# ---------------------------------------------------------------- data --
print("data")
data = load_dataset(ROOT / "data")
train, test = data["train"], data["test"]
check("both splits load", len(train) == 1500 and len(test) == 1500,
      f"{len(train)} / {len(test)}")

counts = describe(train)
balanced = bool((counts.to_numpy() == 100).all())
check("train is 100 per (project, class)", balanced)
check("test is 100 per (project, class)",
      bool((describe(test).to_numpy() == 100).all()))
check("five projects, three labels",
      set(train.frame.repo) == set(REPOSITORIES)
      and set(train.frame.label) == set(LABELS))
check("text is title + ' ' + body",
      train.frame.text.iloc[0].startswith(str(train.frame.title.iloc[0])))
check("shuffle is deterministic",
      load_dataset(ROOT / "data")["train"].frame.text.iloc[0]
      == train.frame.text.iloc[0])

# ------------------------------------------------------------- metrics --
print("\nmetrics")
truth = test.frame.label.tolist()
repos = test.frame.repo.tolist()

perfect = metrics.cross_repo_f1(truth, truth, repos)
check("a perfect prediction scores 1.0", abs(perfect - 1.0) < 1e-12)

always_bug = ["bug"] * len(truth)
majority = metrics.cross_repo_f1(truth, always_bug, repos)
check("always-bug scores the documented 0.1667", abs(majority - 1 / 6) < 5e-4,
      f"{majority:.4f}")

# The metric must be the weighted average, not accuracy. Symmetric noise makes
# the two agree to four decimals, which would let a wrong implementation pass,
# so the check uses a deliberately asymmetric failure: perfect on two classes,
# never right on the third. Accuracy then reads 0.667 and the weighted average
# 0.556 -- a gap far larger than anything anyone claims as a result.
from sklearn.metrics import f1_score  # noqa: E402

lopsided = [t if t != "question" else "bug" for t in truth]
micro_lop = float(np.mean([
    f1_score(
        np.array(truth)[np.array(repos) == r],
        np.array(lopsided)[np.array(repos) == r], average="micro",
    )
    for r in REPOSITORIES
]))
weighted_lop = metrics.cross_repo_f1(truth, lopsided, repos)
check("metric is weighted-avg F1, not accuracy",
      abs(weighted_lop - micro_lop) > 0.05
      and abs(micro_lop - 2 / 3) < 1e-9
      and abs(weighted_lop - 5 / 9) < 1e-9,
      f"weighted {weighted_lop:.4f} vs accuracy {micro_lop:.4f}")

rng = np.random.default_rng(0)
noisy = [t if rng.random() < 0.75 else rng.choice(LABELS) for t in truth]

report = metrics.full_report(truth, noisy, repos)
check("full report has five projects and all classes",
      len(report["per_repo"]) == 5
      and all(lbl in report["classification_report"][REPOSITORIES[0]]
              for lbl in LABELS))
check("verdict wording is graded",
      metrics.verdict(0.8240) .startswith("reproduced")
      and metrics.verdict(0.700).startswith("FAILED"))

# --------------------------------------------------------------- stats --
print("\nstats")
identical = stats.mcnemar(noisy, noisy, truth)
check("identical models have zero discordant pairs and p = 1",
      identical.n_discordant == 0 and identical.p_value == 1.0)

better = [t if rng.random() < 0.85 else rng.choice(LABELS) for t in truth]
result = stats.mcnemar(noisy, better, truth)
check("a clearly better model is favoured and significant",
      result.favours_challenger and result.p_value < 0.001,
      f"n01={result.n01} n10={result.n10} p={result.p_value:.2e}")

adjusted = stats.holm([0.001, 0.04, 0.03, 0.2, 0.5])
check("Holm never lowers a p-value and stays ordered",
      all(a >= r for a, r in zip(adjusted, [0.001, 0.04, 0.03, 0.2, 0.5]))
      and abs(adjusted[0] - 0.005) < 1e-12)

point, low, high = stats.bootstrap_ci(truth, noisy, repos, n_resamples=300)
check("bootstrap interval brackets the estimate", low <= point <= high,
      f"{point:.4f} in [{low:.4f}, {high:.4f}]")

comparison = stats.compare(noisy, better, truth, repos)
check("comparison reports five projects plus a pooled row",
      len(comparison["per_repo"]) == 5 and "pooled" in comparison)

rate = comparison["pooled"]["discordant"] / len(truth)
mdd = stats.minimum_detectable_difference(rate, n_items=len(truth))
check("detectable difference is a few F1 points", 0.005 <= mdd <= 0.08,
      f"{mdd * 100:.1f} points at a discordant rate of {rate:.3f}")

small = stats.describe_claim(0.010, comparison)
check("a one-point gain is called no detectable difference",
      "no detectable difference" in small)

# ---------------------------------------------------------- preprocess --
print("\npreprocessing")
sample = (
    "Bug: crash on load\n"
    "<!-- please read the guide -->\n"
    "### Steps to reproduce\n"
    "Please ask usage questions on forum.opencv.org or Stack Overflow\n"
    "```python\nprint(1)\n```\n"
    "See https://example.com/x.png for a screenshot."
)
stripped = preprocess.strip_templates(sample)
check("template stripping removes the forum line",
      "forum.opencv.org" not in stripped)
check("template stripping keeps the code block", "print(1)" in stripped)
masked = preprocess.mask_structure(sample)
check("masking replaces code with a typed token",
      "xxcode" in masked and "print(1)" not in masked)
check("all variants are applicable to a frame",
      all(len(preprocess.apply(test.frame.head(5), v)) == 5
          for v in preprocess.VARIANTS))

# ---------------------------------------------------------------- audit --
print("\naudit")
from src import audit  # noqa: E402

for name in ("issues_train.csv", "issues_test.csv"):
    path = ROOT / "data" / name
    check(f"{name} matches the published checksum",
          audit.sha256(path) == audit.EXPECTED_SHA256[name])

tampered = ROOT / "data" / "_tampered.csv"
tampered.write_text("not the dataset", encoding="utf-8")
audit.EXPECTED_SHA256["_tampered.csv"] = "0" * 64
try:
    audit.verify(tampered)
    caught = False
except ValueError:
    caught = True
finally:
    tampered.unlink(missing_ok=True)
    audit.EXPECTED_SHA256.pop("_tampered.csv", None)
check("a modified file is rejected", caught)

log_dir = ROOT / "results"
before = audit.test_access_count(log_dir).get("smoke-test", 0)
audit.log_test_access(log_dir, "smoke-test")
check("test reads are logged",
      audit.test_access_count(log_dir).get("smoke-test", 0) == before + 1)

overlap = audit.leakage(train.frame, test.frame, thresholds=(0.90,))
check("leakage audit finds the known duplicates",
      overlap["exact_duplicates_train_test"] == 3
      and overlap["duplicates_within_train"] == 1,
      f"{overlap['exact_duplicates_train_test']} exact, "
      f"{overlap['near_duplicates']['cosine_ge_0.90']['n']} near at 0.90")

# ----------------------------------------------------------- selection --
print("\nselection split (phase 2)")
import runpy  # noqa: E402

phase2 = runpy.run_path(str(ROOT / "run_phase2.py"), run_name="not_main")
inner, validation = phase2["split_for_selection"](train, 0.25, 42, verbose=False)
check("split sizes are 75/25 of the training split",
      len(inner) == 1125 and len(validation) == 375,
      f"{len(inner)} / {len(validation)}")
check("every (project, class) cell is split in the same proportion",
      set(validation.frame.groupby(["repo", "label"]).size()) == {25}
      and set(inner.frame.groupby(["repo", "label"]).size()) == {75})
check("no text straddles the split",
      not (set(inner.frame.text) & set(validation.frame.text)),
      "the training split's duplicate pair is kept on one side")
check("the test split is untouched by selection",
      not (set(validation.frame.text) & set(test.frame.text.head(0))))
check("gated candidate is marked gated",
      phase2["build"]("global_pairs", 42)[2]
      and not phase2["build"]("baseline", 42)[2])
check("contaminated encoders are flagged",
      phase2["build"]("encoder_bge", 42)[0].encoder
      in phase2["POSSIBLY_CONTAMINATED"])

# -------------------------------------------------------------- config --
print("\nconfiguration")
check("official config matches the organisers' notebook",
      OFFICIAL.encoder.endswith("all-mpnet-base-v2")
      and (OFFICIAL.body_batch, OFFICIAL.head_batch) == (16, 2)
      and OFFICIAL.num_epochs == 1
      and OFFICIAL.num_iterations == 20
      and OFFICIAL.seed == 42
      and OFFICIAL.scope == "per_repo")
check("max_seq_length is pinned to the checkpoint's own 384",
      OFFICIAL.max_seq_length == 384)

# -------------------------------------------------------------- result --
failed = [name for name, ok, _ in checks if not ok]
print("\n" + "=" * 74)
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
    sys.exit(1)
print("Ready for the GPU. Start with:  python run_phase1.py --seeds 42")
