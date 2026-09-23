"""k-fold cross-validation and hyper-parameter search.

The course brief asks explicitly for k-fold cross-validation, and there is a
second reason to want it here: the test set is small enough that selecting
anything on it would burn the only clean measurement the project has. Every
decision -- which estimator, which regularisation strength, which cleaning
level -- is made inside this module, on the **training split only**.

The folds mirror the final protocol rather than simplifying it: inside each
fold, one model is trained per project and the five scores are averaged
exactly as the competition does. A cheaper pooled cross-validation would
select hyper-parameters for a model that is never the one submitted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import numpy as np

from ..evaluation.metrics import cross_repo_f1
from ..evaluation.splits import stratified_folds
from ..repository import IssueRepository, make_repository
from .base import ClassifierFactory


def cross_validate(
    factory: ClassifierFactory,
    repository: IssueRepository,
    n_folds: int = 5,
    seed: int = 42,
    per_repo: bool = True,
) -> dict:
    """Estimate a configuration's score without touching the test set.

    Args:
        factory: Produces a fresh untrained classifier.
        repository: Training issues. Never pass the test split here.
        n_folds: Number of folds.
        seed: Seed of the fold assignment.
        per_repo: Train one model per project inside each fold, as the
            competition requires. Set to ``False`` only for the pooled
            ablation.

    Returns:
        Mapping with ``"mean"``, ``"std"`` and the per-fold ``"scores"``.
    """
    issues = repository.all()
    scores: list[float] = []

    for train_index, validation_index in stratified_folds(repository, n_folds, seed):
        fold_train = [issues[i] for i in train_index]
        fold_validation = [issues[i] for i in validation_index]
        truth = [issue.label for issue in fold_validation]
        repos = [issue.repo for issue in fold_validation]

        if per_repo:
            predictions = np.empty(len(fold_validation), dtype=object)
            # Projects present in this fold, not the REPOSITORIES constant --
            # see the same note in ai4se.classifiers.base.train_per_repo.
            for repo in sorted({issue.repo for issue in fold_validation}):
                positions = [
                    i for i, issue in enumerate(fold_validation) if issue.repo == repo
                ]
                subset = [issue for issue in fold_train if issue.repo == repo]
                if not positions or not subset:
                    continue
                model = factory().fit(
                    [issue.text for issue in subset],
                    [issue.label for issue in subset],
                )
                predicted = model.predict([fold_validation[i].text for i in positions])
                for position, label in zip(positions, predicted, strict=True):
                    predictions[position] = label
        else:
            model = factory().fit(
                [issue.text for issue in fold_train],
                [issue.label for issue in fold_train],
            )
            predictions = model.predict([issue.text for issue in fold_validation])

        scores.append(cross_repo_f1(truth, predictions, repos))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "scores": scores,
    }


def grid_search(
    build: Callable[..., ClassifierFactory],
    grid: dict[str, Sequence],
    repository: IssueRepository,
    n_folds: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """Evaluate every combination in ``grid`` by cross-validation.

    A note on interpreting the ranking: the differences between neighbouring
    configurations are usually far below the 3.0-point threshold the test set
    can resolve (``docs/benchmark-audit.md``). Treat the winner as *a*
    reasonable configuration rather than *the* best one, and record how many
    configurations were tried -- the pre-registration commits to giving the
    baseline the same budget.

    Args:
        build: Called with one combination of keyword arguments, returns a
            classifier factory.
        grid: Mapping from parameter name to the values to try.
        repository: Training issues.
        n_folds: Number of folds.
        seed: Seed of the fold assignment.
        verbose: Print each configuration as it finishes.

    Returns:
        One row per configuration, sorted by mean score, best first. Each row
        carries ``"params"``, ``"mean"``, ``"std"`` and ``"scores"``.
    """
    names = list(grid)
    rows: list[dict] = []

    for values in product(*(grid[name] for name in names)):
        params = dict(zip(names, values, strict=True))
        result = cross_validate(build(**params), repository, n_folds, seed)
        rows.append({"params": params, **result})
        if verbose:
            setting = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    {setting:<52} {result['mean']:.4f} ± {result['std']:.4f}")

    rows.sort(key=lambda row: row["mean"], reverse=True)
    return rows


def summarise_search(rows: Sequence[dict]) -> str:
    """Render a grid-search result as a short text table for the report.

    Args:
        rows: Output of :func:`grid_search`.

    Returns:
        A formatted multi-line string.
    """
    if not rows:
        return "(no configurations evaluated)"
    lines = [f"{'configuration':<52} {'mean':>8} {'sd':>8}"]
    lines.append("-" * 70)
    for row in rows:
        setting = ", ".join(f"{k}={v}" for k, v in row["params"].items())
        lines.append(f"{setting:<52} {row['mean']:>8.4f} {row['std']:>8.4f}")
    spread = rows[0]["mean"] - rows[-1]["mean"]
    lines.append("-" * 70)
    lines.append(
        f"{len(rows)} configurations, spread {spread * 100:.2f} F1 points "
        f"({'below' if spread < 0.03 else 'above'} the 3.0-point detection threshold)"
    )
    return "\n".join(lines)


def make_subset(repository: IssueRepository, repo: str) -> IssueRepository:
    """Return a new in-memory repository holding one project's issues.

    Args:
        repository: Source issues.
        repo: Full project name, e.g. ``"facebook/react"``.

    Returns:
        An :class:`~ai4se.repository.IssueRepository` with that subset.
    """
    return make_repository("memory", issues=repository.by_repo(repo))
