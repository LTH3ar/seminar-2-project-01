"""Scoring functions for the cross-repository evaluation protocol.

The competition score is the arithmetic mean of five per-project F1 scores,
not a single F1 over the pooled test set. The two differ whenever the projects
are unequally hard, which they are: per-project F1 of the same model ranges
from 0.716 to 0.838 (see ``docs/benchmark-audit.md``). Always report the mean
of five, and always report the five.

Implemented with numpy only, so this module imports with the base
requirements -- tracks B, C and D can use it before installing their extras.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..model import LABELS


def _confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict]:
    """Count TP, FP and FN per class.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Mapping ``{label: {"tp": int, "fp": int, "fn": int}}`` covering every
        label in :data:`~ai4se.model.LABELS`, including unseen ones.
    """
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    counts = {}
    for label in LABELS:
        is_true = true == label
        is_pred = pred == label
        counts[label] = {
            "tp": int(np.sum(is_true & is_pred)),
            "fp": int(np.sum(~is_true & is_pred)),
            "fn": int(np.sum(is_true & ~is_pred)),
        }
    return counts


def _f1(tp: int, fp: int, fn: int) -> float:
    """Return the F1 score of one class from its raw counts."""
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def micro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Return the micro-averaged F1 score.

    For a single-label problem this equals accuracy. It is kept under its own
    name because the competition reports it as F1 and the report must use the
    same vocabulary.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Micro-F1 in ``[0, 1]``.
    """
    counts = _confusion(y_true, y_pred)
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values())
    fn = sum(c["fn"] for c in counts.values())
    return _f1(tp, fp, fn)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Return the unweighted mean of the per-class F1 scores.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Macro-F1 in ``[0, 1]``.
    """
    counts = _confusion(y_true, y_pred)
    return float(np.mean([_f1(**c) for c in counts.values()]))


def per_repo_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    repos: Sequence[str],
    average: str = "micro",
) -> dict[str, float]:
    """Score each project separately.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        repos: Source project of each issue, same length as the labels.
        average: ``"micro"`` or ``"macro"``.

    Returns:
        Mapping ``{project: F1}``, ordered by project name.

    Raises:
        ValueError: If ``average`` is not a supported value.
    """
    if average not in {"micro", "macro"}:
        raise ValueError(f"Unknown average {average!r}; use 'micro' or 'macro'.")
    score: Callable[[Sequence[str], Sequence[str]], float] = (
        micro_f1 if average == "micro" else macro_f1
    )
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    source = np.asarray(repos, dtype=object)
    return {
        repo: score(true[source == repo], pred[source == repo])
        for repo in sorted(set(source.tolist()))
    }


def cross_repo_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    repos: Sequence[str],
    average: str = "micro",
) -> float:
    """Return the official competition score.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        repos: Source project of each issue.
        average: ``"micro"`` or ``"macro"``.

    Returns:
        Arithmetic mean of the per-project F1 scores.
    """
    return float(np.mean(list(per_repo_f1(y_true, y_pred, repos, average).values())))


def classification_rows(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    repos: Sequence[str],
) -> list[dict]:
    """Build the full per-project, per-class results table.

    This is the table the report needs: fifteen cells, not one number. The
    single cross-repository figure hides a range of 0.620 to 0.916 across
    those cells for the TF-IDF baseline.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        repos: Source project of each issue.

    Returns:
        One row per (project, class) pair with precision, recall, F1 and
        support, plus a ``"cross-repo"`` row per class for the averages.
    """
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    source = np.asarray(repos, dtype=object)
    rows: list[dict] = []
    for repo in sorted(set(source.tolist())):
        mask = source == repo
        counts = _confusion(true[mask], pred[mask])
        for label, c in counts.items():
            precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
            recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
            rows.append(
                {
                    "repo": repo,
                    "label": label,
                    "precision": precision,
                    "recall": recall,
                    "f1": _f1(**c),
                    "support": c["tp"] + c["fn"],
                }
            )
    per_repo_rows = list(rows)
    for label in LABELS:
        cells = [r for r in per_repo_rows if r["label"] == label]
        rows.append(
            {
                "repo": "cross-repo",
                "label": label,
                "precision": float(np.mean([r["precision"] for r in cells])),
                "recall": float(np.mean([r["recall"] for r in cells])),
                "f1": float(np.mean([r["f1"] for r in cells])),
                "support": sum(r["support"] for r in cells),
            }
        )
    return rows


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    repos: Sequence[str],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Estimate a confidence interval for the cross-repository F1.

    Resampling is stratified by project so that each bootstrap replicate keeps
    the same five-project structure as the real evaluation; pooling the test
    set and resampling it flat would understate the variance.

    On the TF-IDF baseline this returns an interval roughly 4.3 F1 points
    wide -- wider than most improvements claimed in the competition, which is
    the point.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        repos: Source project of each issue.
        n_resamples: Number of bootstrap replicates.
        confidence: Width of the interval, e.g. ``0.95``.
        seed: Seed of the random generator, for reproducibility.

    Returns:
        Tuple ``(point_estimate, lower, upper)``.
    """
    rng = np.random.default_rng(seed)
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    source = np.asarray(repos, dtype=object)
    index_by_repo = {
        repo: np.flatnonzero(source == repo) for repo in sorted(set(source.tolist()))
    }

    replicates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        scores = []
        for indices in index_by_repo.values():
            draw = rng.choice(indices, size=indices.size, replace=True)
            scores.append(micro_f1(true[draw], pred[draw]))
        replicates[i] = float(np.mean(scores))

    tail = (1.0 - confidence) / 2.0 * 100.0
    lower, upper = np.percentile(replicates, [tail, 100.0 - tail])
    point = cross_repo_f1(true, pred, source)
    return point, float(lower), float(upper)
