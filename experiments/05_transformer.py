"""Fine-tuned transformer experiment: selection, then test evaluation.

This script mirrors the structure of ``02_baselines.py`` exactly — two
stages, five seeds, bootstrap intervals, McNemar comparisons — so the
numbers are directly comparable and the pre-registration commitments
(one test read per configuration, all seeds reported, tuning budget
recorded) are enforced mechanically.

Two stages:

1. **Selection** -- 5-fold cross-validation on the *training split only*,
   choosing learning rate and max-length.  The test set is not read.
2. **Evaluation** -- the selected configuration trained five times with
   different seeds and scored once on the test set.

Usage::

    pip install -e ".[dl]"
    python experiments/05_transformer.py                  # full (GPU recommended)
    python experiments/05_transformer.py --preset fast    # quick CPU iteration
    python experiments/05_transformer.py --skip-selection # stage 2 only

Cost, because one model is fine-tuned per project and that multiplies fast:
selection is 4 configurations × 5 folds × 5 projects = **100 fine-tunes**,
evaluation is 5 seeds × 5 projects = **25** more.  At the ~90 s per
fine-tune quoted in ``ai4se.classifiers.transformer`` that is roughly
**3 hours on a T4** for the whole script, and the better part of a day on
CPU.  ``--skip-selection`` runs stage 2 alone, about 40 minutes on a T4.

Writes ``results/transformer.json``.
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
from ai4se.classifiers.selection import cross_validate, summarise_search  # noqa: E402
from ai4se.classifiers.transformer import make_transformer  # noqa: E402
from ai4se.evaluation import (  # noqa: E402
    OFFICIAL_BASELINES,
    bootstrap_ci,
    classification_rows,
    cross_repo_f1,
    is_conclusive,
    per_repo_f1,
)
from ai4se.loader import load_split  # noqa: E402
from ai4se.preprocessing import make_cleaner  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "transformer.json"
SEEDS = (41, 42, 43, 44, 45)


def _banner(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def select_hyperparameters(train, preset: str) -> dict:
    """Choose learning rate and max-length by 5-fold CV on train only."""
    _banner("STAGE 1 - SELECTION BY 5-FOLD CROSS-VALIDATION (TRAINING SPLIT ONLY)")

    # Keep the search space small: the pre-registration commits to giving
    # the baseline the same tuning budget, and each CV fold fine-tunes a
    # transformer five times (one per project).
    configs = []

    if preset == "fast":
        search_space = [
            {"learning_rate": 2e-5, "max_length": 256},
            {"learning_rate": 3e-5, "max_length": 256},
        ]
    else:
        search_space = [
            {"learning_rate": 1e-5, "max_length": 512},
            {"learning_rate": 2e-5, "max_length": 512},
            {"learning_rate": 3e-5, "max_length": 512},
            {"learning_rate": 2e-5, "max_length": 256},
        ]

    for params in search_space:
        setting = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n  Evaluating: {setting}")
        factory = make_transformer(preset=preset, **params)
        result = cross_validate(factory, train, n_folds=5)
        configs.append({"params": params, **result})
        print(
            f"    CV score: {result['mean']:.4f} ± {result['std']:.4f}  "
            f"folds: {' '.join(f'{s:.4f}' for s in result['scores'])}"
        )

    configs.sort(key=lambda r: r["mean"], reverse=True)

    # Same rendering as 02_baselines.py, so the two selection tables read
    # alike and both carry the detection-threshold caveat.
    print("\n" + summarise_search(configs))
    print(f"\n  Best configuration: {configs[0]['params']}")

    return {
        "best_params": configs[0]["params"],
        "cv_mean": configs[0]["mean"],
        "cv_std": configs[0]["std"],
        "n_configurations": len(configs),
        "all_configs": [
            {"params": c["params"], "mean": c["mean"], "std": c["std"]}
            for c in configs
        ],
    }


def evaluate(name: str, factory_for_seed, train, test) -> dict:
    """Train one configuration across five seeds and score on the test set."""
    test_issues = test.all()
    y_true = [issue.label for issue in test_issues]
    repos = [issue.repo for issue in test_issues]

    scores, predictions = [], []
    started = time.time()
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        predicted = train_per_repo(factory_for_seed(seed), train, test, verbose=True)
        predictions.append(predicted)
        score = cross_repo_f1(y_true, predicted, repos)
        scores.append(score)
        print(f"    cross-repo F1: {score:.4f}")

    elapsed = time.time() - started

    best_seed = int(np.argmax(scores))
    median_seed = int(np.argsort(scores)[len(scores) // 2])
    representative = predictions[median_seed]
    point, low, high = bootstrap_ci(y_true, representative, repos, n_resamples=2000)

    setfit = OFFICIAL_BASELINES["setfit"]["cross-repo"]
    delta = float(np.mean(scores)) - setfit

    print(f"\n  {name}")
    print(
        f"    mean ± sd:     {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}"
    )
    print(f"    seeds:         {' '.join(f'{s:.4f}' for s in scores)}")
    print(f"    95% CI:        [{low:.4f}, {high:.4f}]")
    print(
        f"    vs SetFit:     {delta:+.4f} -> "
        f"{'detectable' if is_conclusive(delta) else 'NOT detectable'}"
    )
    print(f"    time:          {elapsed:.0f}s")

    return {
        "name": name,
        "seed_scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
        "best_seed_score": scores[best_seed],
        "cherry_pick_inflation": scores[best_seed] - float(np.mean(scores)),
        "ci": [low, high],
        "per_repo": per_repo_f1(y_true, representative, repos),
        "delta_vs_setfit": delta,
        "detectable_vs_setfit": bool(is_conclusive(delta)),
        "seconds": elapsed,
        "classification_rows": classification_rows(y_true, representative, repos),
    }


def main() -> None:
    """Run selection, then evaluation, then write the results file."""
    parser = argparse.ArgumentParser(description="Fine-tuned transformer experiment")
    parser.add_argument(
        "--preset",
        choices=["default", "fast"],
        default="default",
        help="'default' for DeBERTa-v3-base (GPU), 'fast' for DeBERTa-v3-small (CPU)",
    )
    parser.add_argument(
        "--clean",
        choices=["raw", "conservative", "light", "full"],
        default="light",
        help="Cleaning level applied to both splits. 'light' is the level the "
        "README designates for transformer models; 'raw' is the control.",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Skip CV selection and use the preset's own hyperparameters",
    )
    args = parser.parse_args()

    train = load_split("train", kind="memory")
    test = load_split("test", kind="memory")

    # Without this the entity falls back to raw Markdown, and H1 of the
    # pre-registration is explicitly about DeBERTa-v3 *with structure-aware
    # preprocessing*. No word cap is set: max_length is meant to be the single
    # truncation knob, and stage 1 searches it.
    cleaner = make_cleaner(level=args.clean)
    train.apply(cleaner)
    test.apply(cleaner)

    print(
        f"Loaded {len(train)} training and {len(test)} test issues, "
        f"cleaning level {args.clean!r}."
    )

    # ---- Stage 1: Selection --------------------------------------------------
    if args.skip_selection:
        # Empty params, so the preset's own values apply rather than a second
        # copy of them that can drift out of step with transformer.py.
        selection = {
            "best_params": {},
            "cv_mean": None,
            "cv_std": None,
            "n_configurations": 0,
            "note": "selection skipped via --skip-selection",
        }
        print(f"\nSelection skipped — using the {args.preset!r} preset's defaults.")
    else:
        selection = select_hyperparameters(train, args.preset)

    best_params = selection["best_params"]
    eval_params = {k: v for k, v in best_params.items() if k != "seed"}

    # ---- Stage 2: Test evaluation --------------------------------------------
    _banner("STAGE 2 - TEST EVALUATION, FIVE SEEDS")
    print("  Published baselines for reference:")
    for model, cells in OFFICIAL_BASELINES.items():
        print(f"    {model:<10} {cells['cross-repo']:.4f}")
    print()

    result = evaluate(
        f"transformer ({args.preset})",
        lambda seed: make_transformer(preset=args.preset, seed=seed, **eval_params),
        train,
        test,
    )

    # ---- Seed variance -------------------------------------------------------
    _banner("SEED VARIANCE")
    spread = max(result["seed_scores"]) - min(result["seed_scores"])
    print(
        f"  spread {spread * 100:5.2f} pts   "
        f"cherry-picking the best seed would add "
        f"{result['cherry_pick_inflation'] * 100:.2f} pts"
    )

    # ---- Write results -------------------------------------------------------
    payload = {
        "preset": args.preset,
        "clean": args.clean,
        "selection": selection,
        "seeds": list(SEEDS),
        "official_baselines": OFFICIAL_BASELINES,
        "results": result,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # No default=str: a numpy scalar slipping in should fail here, loudly,
    # rather than reach 04_make_tables.py as the string "0.83".
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
