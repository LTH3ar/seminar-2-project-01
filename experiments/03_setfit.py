"""Reproduce the organisers' SetFit baseline.

Everything else in the project is measured against this number, so it has to
be reproduced rather than quoted. The configuration is copied exactly from the
organisers' notebook -- ``all-mpnet-base-v2``, batch ``(16, 2)``, one epoch,
twenty contrastive iterations, seed 42, one model per project.

Two published figures exist for that configuration and they disagree: the
competition README says **0.8270**, the result file committed in the same
repository says **0.8240**. Landing between them is a successful
reproduction; the gap between the organisers and themselves is a useful
reminder that the third decimal place carries no information here.

    pip install -e ".[dl]"
    python experiments/03_setfit.py                 # official, slow on CPU
    python experiments/03_setfit.py --preset fast   # MiniLM, for iterating

On a GPU the official preset takes about fifteen minutes. On CPU expect one
to three hours: ``all-mpnet-base-v2`` has 110M parameters and each project
generates 12,000 contrastive pairs. Predictions are cached under
``results/setfit_predictions_<preset>.json`` so the scoring can be re-run
without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4se.classifiers.base import train_per_repo  # noqa: E402
from ai4se.classifiers.setfit_model import make_setfit  # noqa: E402
from ai4se.evaluation import (  # noqa: E402
    OFFICIAL_BASELINES,
    bootstrap_ci,
    classification_rows,
    cross_repo_f1,
    per_repo_f1,
)
from ai4se.loader import load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("official", "fast"),
        default="official",
        help="'official' reproduces the published baseline; 'fast' does not.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Score cached predictions instead of retraining.",
    )
    return parser.parse_args()


def main() -> None:
    """Train (or load) SetFit, score it, and compare with the published run."""
    args = parse_args()
    cache = ROOT / "results" / f"setfit_predictions_{args.preset}.json"

    train = load_split("train", kind="memory")
    test = load_split("test", kind="memory")
    y_true = [issue.label for issue in test.all()]
    repos = [issue.repo for issue in test.all()]

    if args.reuse_cache and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        predictions = np.asarray(cached, dtype=object)
        elapsed = 0.0
        print(f"Loaded cached predictions from {cache.name}")
    else:
        if args.preset == "fast":
            print("WARNING: the 'fast' preset is NOT the official baseline.")
            print("         Its score must not be reported as a reproduction.\n")
        print(f"Training SetFit ({args.preset} preset), one model per project:")
        started = time.time()
        predictions = train_per_repo(
            make_setfit(preset=args.preset), train, test, verbose=True
        )
        elapsed = time.time() - started
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(list(predictions)), encoding="utf-8")
        print(f"\nTrained in {elapsed / 60:.1f} minutes; predictions cached.")

    score = cross_repo_f1(y_true, predictions, repos)
    point, low, high = bootstrap_ci(y_true, predictions, repos, n_resamples=2000)
    per_repo = per_repo_f1(y_true, predictions, repos)
    published = OFFICIAL_BASELINES["setfit"]

    print(f"\n{'=' * 74}\nREPRODUCTION vs PUBLISHED\n{'=' * 74}")
    print(f"  {'project':<24} {'ours':>8} {'published':>10} {'delta':>8}")
    for repo, value in per_repo.items():
        reference = published.get(repo)
        delta = f"{value - reference:+.4f}" if reference else "n/a"
        print(f"  {repo:<24} {value:>8.4f} {reference:>10.4f} {delta:>8}")
    print(f"  {'-' * 54}")
    print(
        f"  {'cross-repo':<24} {score:>8.4f} "
        f"{published['cross-repo']:>10.4f} {score - published['cross-repo']:>+8.4f}"
    )
    print(f"  95% CI [{low:.4f}, {high:.4f}]")
    print("  README of the competition quotes 0.8270 for the same configuration.")

    gap = abs(score - published["cross-repo"])
    if args.preset != "official":
        verdict = "NOT a reproduction (fast preset)"
    elif gap <= 0.01:
        verdict = "reproduced (within 1.0 point)"
    elif gap <= 0.03:
        verdict = "close, but investigate (within the 3.0-point noise floor)"
    else:
        verdict = "FAILED to reproduce - stop and find the cause before continuing"
    print(f"\n  Verdict: {verdict}")

    payload = {
        "preset": args.preset,
        "cross_repo_f1": score,
        "ci": [low, high],
        "per_repo": per_repo,
        "published": published,
        "gap": score - published["cross-repo"],
        "verdict": verdict,
        "minutes": elapsed / 60,
        "classification_rows": classification_rows(y_true, predictions, repos),
    }
    out = ROOT / "results" / f"setfit_{args.preset}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Written to {out.name}")


if __name__ == "__main__":
    main()
