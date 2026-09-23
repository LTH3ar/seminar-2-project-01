"""Classical and neural baselines: selection, then a single test evaluation.

Two stages, in this order and never the other way round:

1. **Selection** -- 5-fold cross-validation on the *training split only*,
   choosing the regularisation strength of the linear models. The test set is
   not read at this point.
2. **Evaluation** -- each selected configuration trained five times with
   different seeds and scored once on the test set, with a bootstrap interval
   and a McNemar comparison against the published SetFit baseline.

Reporting five seeds rather than one is not ceremony: on this data the same
configuration moves by up to 2.4 F1 points between seeds, so a single number
would be indistinguishable from a lucky draw.

Needs both extras::

    pip install -e ".[ml,dl]"
    python experiments/02_baselines.py

About four minutes on a laptop CPU. Writes ``results/baselines.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4se.classifiers.base import train_per_repo  # noqa: E402
from ai4se.classifiers.classical import make_classical  # noqa: E402
from ai4se.classifiers.neural import make_ffnn, make_text_cnn  # noqa: E402
from ai4se.classifiers.selection import grid_search, summarise_search  # noqa: E402
from ai4se.evaluation import (  # noqa: E402
    OFFICIAL_BASELINES,
    bootstrap_ci,
    classification_rows,
    cross_repo_f1,
    is_conclusive,
    per_repo_f1,
)
from ai4se.loader import load_split  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "baselines.json"
SEEDS = (41, 42, 43, 44, 45)


def _banner(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def select_hyperparameters(train) -> dict:
    """Choose the linear models' regularisation by 5-fold CV on train only."""
    _banner("STAGE 1 - SELECTION BY 5-FOLD CROSS-VALIDATION (TRAINING SPLIT ONLY)")
    chosen: dict[str, dict] = {}

    for estimator in ("logreg", "linear_svm"):
        print(f"\n  {estimator}")
        rows = grid_search(
            lambda c_value, estimator=estimator: make_classical(
                estimator=estimator, c_value=c_value
            ),
            {"c_value": [0.25, 0.5, 1.0, 2.0, 4.0]},
            train,
            n_folds=5,
        )
        print("\n" + summarise_search(rows))
        chosen[estimator] = {
            "params": rows[0]["params"],
            "cv_mean": rows[0]["mean"],
            "cv_std": rows[0]["std"],
            "n_configurations": len(rows),
        }

    # Naive Bayes and the random forest are run at their defaults, and that is
    # recorded rather than hidden: the pre-registration commits to reporting
    # the tuning budget of every model, including the ones that got none.
    for estimator in ("naive_bayes", "random_forest"):
        chosen[estimator] = {"params": {}, "n_configurations": 1}
    return chosen


def evaluate(name: str, factory_for_seed, train, test) -> dict:
    """Train one configuration across five seeds and score it on the test set."""
    y_true = [issue.label for issue in test.all()]
    repos = [issue.repo for issue in test.all()]

    scores, predictions = [], []
    started = time.time()
    for seed in SEEDS:
        predicted = train_per_repo(factory_for_seed(seed), train, test)
        predictions.append(predicted)
        scores.append(cross_repo_f1(y_true, predicted, repos))
    elapsed = time.time() - started

    best_seed = int(np.argmax(scores))
    median_seed = int(np.argsort(scores)[len(scores) // 2])
    representative = predictions[median_seed]
    point, low, high = bootstrap_ci(y_true, representative, repos, n_resamples=2000)

    setfit = OFFICIAL_BASELINES["setfit"]["cross-repo"]
    delta = float(np.mean(scores)) - setfit

    print(
        f"  {name:<16} {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}   "
        f"seeds {' '.join(f'{s:.4f}' for s in scores)}"
    )
    print(
        f"  {'':<16} 95% CI [{low:.4f}, {high:.4f}]   "
        f"vs SetFit {delta:+.4f} -> "
        f"{'detectable' if is_conclusive(delta) else 'NOT detectable'}   "
        f"({elapsed:.0f}s)"
    )
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
    train = load_split("train", kind="memory")
    test = load_split("test", kind="memory")
    print(f"Loaded {len(train)} training and {len(test)} test issues.")

    chosen = select_hyperparameters(train)

    _banner("STAGE 2 - TEST EVALUATION, FIVE SEEDS EACH")
    print("  Published baselines for reference:")
    for model, cells in OFFICIAL_BASELINES.items():
        print(f"    {model:<10} {cells['cross-repo']:.4f}")
    print()

    configurations = {
        "naive_bayes": lambda seed: make_classical("naive_bayes", seed=seed),
        "logreg": lambda seed: make_classical(
            "logreg", seed=seed, **chosen["logreg"]["params"]
        ),
        "linear_svm": lambda seed: make_classical(
            "linear_svm", seed=seed, **chosen["linear_svm"]["params"]
        ),
        "random_forest": lambda seed: make_classical("random_forest", seed=seed),
        "ffnn": lambda seed: make_ffnn(seed=seed),
        "text_cnn": lambda seed: make_text_cnn(seed=seed),
    }

    results = {}
    for name, factory_for_seed in configurations.items():
        results[name] = evaluate(name, factory_for_seed, train, test)

    _banner("SEED VARIANCE - WHY A SINGLE NUMBER IS NOT A RESULT")
    for name, result in results.items():
        spread = max(result["seed_scores"]) - min(result["seed_scores"])
        print(
            f"  {name:<16} spread {spread * 100:5.2f} pts   "
            f"cherry-picking the best seed would add "
            f"{result['cherry_pick_inflation'] * 100:.2f} pts"
        )

    payload = {
        "selection": chosen,
        "seeds": list(SEEDS),
        "official_baselines": OFFICIAL_BASELINES,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
