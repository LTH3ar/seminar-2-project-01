"""Every model in the project: selection, then a single test evaluation.

Seven configurations -- four classical, two neural trained from scratch, and
the fine-tuned transformer the pre-registration names as H1 -- are put through
the identical protocol so the report has one comparable table rather than
several that were produced under different conditions.

Two stages, in this order and never the other way round:

1. **Selection** -- 5-fold cross-validation on the *training split only*,
   choosing the regularisation strength of the linear models and the learning
   rate and max-length of the transformer. The test set is not read at this
   point.
2. **Evaluation** -- each selected configuration trained five times with
   different seeds and scored once on the test set, with a bootstrap interval
   and a McNemar comparison against the published SetFit baseline.

Reporting five seeds rather than one is not ceremony: on this data the same
configuration moves by up to 2.4 F1 points between seeds, so a single number
would be indistinguishable from a lucky draw.

Needs both extras::

    pip install -e ".[ml,dl]"
    python experiments/02_baselines.py                          # everything
    python experiments/02_baselines.py --transformer off        # ~4 min CPU
    python experiments/02_baselines.py --transformer fast       # CPU iteration
    python experiments/02_baselines.py --skip-transformer-selection

Cost is dominated by the transformer, because one model is fine-tuned per
project and that multiplies fast: its selection is 4 configurations x 5 folds
x 5 projects = **100 fine-tunes** and its evaluation another **25**, roughly
**3 hours on a T4** at the ~90 s per fine-tune quoted in
``ai4se.classifiers.transformer``. The other six models together take about
four minutes on a laptop CPU, which is what ``--transformer off`` gives back.

Writes ``results/baselines.json``.
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
from ai4se.classifiers.classical import make_classical  # noqa: E402
from ai4se.classifiers.neural import make_ffnn, make_text_cnn  # noqa: E402
from ai4se.classifiers.selection import (  # noqa: E402
    cross_validate,
    grid_search,
    summarise_search,
)
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

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "baselines.json"
SEEDS = (41, 42, 43, 44, 45)

#: Transformer configurations tried during selection. Deliberately not a full
#: cross-product: every entry costs 25 fine-tunes, and the pre-registration
#: commits to giving the baseline a comparable tuning budget (the linear
#: models get five configurations each).
TRANSFORMER_SEARCH = {
    "default": [
        {"learning_rate": 1e-5, "max_length": 512},
        {"learning_rate": 2e-5, "max_length": 512},
        {"learning_rate": 3e-5, "max_length": 512},
        {"learning_rate": 2e-5, "max_length": 256},
    ],
    "fast": [
        {"learning_rate": 2e-5, "max_length": 256},
        {"learning_rate": 3e-5, "max_length": 256},
    ],
}


def _banner(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="All models, one protocol")
    parser.add_argument(
        "--transformer",
        choices=["default", "fast", "off"],
        default="default",
        help="'default' fine-tunes DeBERTa-v3-base (GPU strongly advised), "
        "'fast' uses DeBERTa-v3-small, 'off' runs the other six models only.",
    )
    parser.add_argument(
        "--skip-transformer-selection",
        action="store_true",
        help="Use the transformer preset's own hyper-parameters instead of "
        "cross-validating them, saving 100 fine-tunes.",
    )
    parser.add_argument(
        "--clean",
        choices=["none", "raw", "conservative", "light", "full"],
        default="none",
        help="Cleaning level applied to both splits before anything is "
        "trained. One level for every model, so the table compares models "
        "rather than model-and-preprocessing pairs. 'none' leaves the raw "
        "Markdown, which is what the committed results were measured on.",
    )
    return parser.parse_args()


def _select_transformer(train, preset: str) -> dict:
    """Choose the transformer's learning rate and max-length by 5-fold CV.

    ``grid_search`` is not used here because it expands a full cross-product
    and each cell costs 25 fine-tunes; :data:`TRANSFORMER_SEARCH` is a
    hand-picked slice of the same space.

    Args:
        train: Training issues. The test split is not read.
        preset: Which entry of :data:`TRANSFORMER_SEARCH` to evaluate.

    Returns:
        The same record shape the other models produce in
        :func:`select_hyperparameters`.
    """
    rows = []
    for params in TRANSFORMER_SEARCH[preset]:
        setting = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"    evaluating {setting} ...")
        result = cross_validate(make_transformer(preset=preset, **params), train, 5)
        rows.append({"params": params, **result})
    rows.sort(key=lambda row: row["mean"], reverse=True)
    print("\n" + summarise_search(rows))
    return {
        "params": rows[0]["params"],
        "cv_mean": rows[0]["mean"],
        "cv_std": rows[0]["std"],
        "n_configurations": len(rows),
    }


def select_hyperparameters(train, args: argparse.Namespace) -> dict:
    """Choose every tunable hyper-parameter by 5-fold CV on train only.

    Args:
        train: Training issues. The test split is not read.
        args: Parsed command line, for the transformer options.

    Returns:
        One record per model, carrying the chosen parameters and the number of
        configurations that were tried to reach them.
    """
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

    if args.transformer != "off":
        print("\n  transformer")
        if args.skip_transformer_selection:
            # Empty params, so the preset's own values apply rather than a
            # second copy of them that can drift out of step.
            chosen["transformer"] = {
                "params": {},
                "n_configurations": 0,
                "note": "selection skipped via --skip-transformer-selection",
            }
            print(f"    skipped; using the {args.transformer!r} preset defaults.")
        else:
            chosen["transformer"] = _select_transformer(train, args.transformer)

    return chosen


def evaluate(name: str, factory_for_seed, train, test, verbose: bool = False) -> dict:
    """Train one configuration across five seeds and score it on the test set.

    Args:
        name: Label used in the printed table and the results file.
        factory_for_seed: Called with a seed, returns a classifier factory.
        train: Training issues.
        test: Test issues. Read exactly once per configuration.
        verbose: Print per-project progress, worth it for the transformer
            where a single seed takes minutes.

    Returns:
        The record consumed by ``04_make_tables.py``.
    """
    test_issues = test.all()
    y_true = [issue.label for issue in test_issues]
    repos = [issue.repo for issue in test_issues]

    scores, predictions = [], []
    started = time.time()
    for seed in SEEDS:
        if verbose:
            print(f"\n  {name} seed {seed}:")
        predicted = train_per_repo(factory_for_seed(seed), train, test, verbose=verbose)
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
    args = parse_args()

    train = load_split("train", kind="memory")
    test = load_split("test", kind="memory")
    if args.clean != "none":
        # No word cap: for the transformer max_length is meant to be the only
        # truncation knob, and stage 1 searches it.
        cleaner = make_cleaner(level=args.clean)
        train.apply(cleaner)
        test.apply(cleaner)
    print(
        f"Loaded {len(train)} training and {len(test)} test issues, "
        f"cleaning level {args.clean!r}."
    )

    chosen = select_hyperparameters(train, args)

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
    # Last, because it is the only one measured in hours rather than seconds:
    # the cheap models have already printed their numbers by the time it runs.
    if args.transformer != "off":
        configurations["transformer"] = lambda seed: make_transformer(
            preset=args.transformer, seed=seed, **chosen["transformer"]["params"]
        )

    results = {}
    for name, factory_for_seed in configurations.items():
        results[name] = evaluate(
            name, factory_for_seed, train, test, verbose=name == "transformer"
        )

    _banner("SEED VARIANCE - WHY A SINGLE NUMBER IS NOT A RESULT")
    for name, result in results.items():
        spread = max(result["seed_scores"]) - min(result["seed_scores"])
        print(
            f"  {name:<16} spread {spread * 100:5.2f} pts   "
            f"cherry-picking the best seed would add "
            f"{result['cherry_pick_inflation'] * 100:.2f} pts"
        )

    payload = {
        "clean": args.clean,
        "transformer_preset": args.transformer,
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
