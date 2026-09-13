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


def _confusion(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, dict]:
    """Count TP, FP and FN per class.

    By default the label set is the union of those appearing in ``y_true`` and
    ``y_pred``, which is what scikit-learn does. Scoring a fixed three-label
    set instead would award F1 = 0 to a class that is simply absent from the
    sample -- harmless on the full test set, where every project has all three
    classes, but wrong on any subset, on a bootstrap replicate, or in a fold.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Explicit label set, when the caller wants absent classes
            represented anyway (the report's per-class table does).

    Returns:
        Mapping ``{label: {"tp": int, "fp": int, "fn": int}}``.
    """
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    if labels is None:
        labels = sorted(set(true.tolist()) | set(pred.tolist()))
    counts = {}
    for label in labels:
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


def weighted_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Return the support-weighted mean of the per-class F1 scores.

    **This is the competition's metric.** The organisers' notebook reads
    scikit-learn's ``weighted avg`` row out of ``classification_report`` for
    each project and then averages the five. Weighting by the support in
    ``y_true`` means a class with no instances contributes nothing rather than
    contributing a zero.

    On the official test set every project holds exactly 100 issues per class,
    so this coincides with :func:`macro_f1` there. The two diverge on any
    unbalanced subset -- a bootstrap replicate, a cross-validation fold -- which
    is why the distinction is implemented rather than assumed away.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Weighted F1 in ``[0, 1]``.
    """
    counts = _confusion(y_true, y_pred)
    supports = {label: c["tp"] + c["fn"] for label, c in counts.items()}
    total = sum(supports.values())
    if total == 0:
        return 0.0
    return float(
        sum(_f1(**counts[label]) * support for label, support in supports.items())
        / total
    )


def per_repo_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    repos: Sequence[str],
    average: str = "weighted",
) -> dict[str, float]:
    """Score each project separately.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        repos: Source project of each issue, same length as the labels.
        average: ``"weighted"`` reproduces the official per-project figure;
            ``"macro"`` and ``"micro"`` are available for ablations. See
            :func:`cross_repo_f1`.

    Returns:
        Mapping ``{project: F1}``, ordered by project name.

    Raises:
        ValueError: If ``average`` is not a supported value.
    """
    averages: dict[str, Callable[[Sequence[str], Sequence[str]], float]] = {
        "micro": micro_f1,
        "macro": macro_f1,
        "weighted": weighted_f1,
    }
    if average not in averages:
        raise ValueError(
            f"Unknown average {average!r}; use one of {sorted(averages)}."
        )
    score = averages[average]
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
    average: str = "weighted",
) -> float:
    """Return the official competition score.

    The organisers' notebook computes, per project, scikit-learn's *weighted
    average* F1 over the classes and then takes the plain mean of the five.
    ``average="weighted"`` reproduces that exactly, and is the default.

    ``average="micro"`` is *not* the competition metric: for a single-label
    problem it equals accuracy, which differs from the weighted average
    whenever the per-class F1 scores are uneven -- as they are here, where
    ``question`` trails the other two classes in almost every project. Keep it
    for ablation tables, not for headline numbers.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        repos: Source project of each issue.
        average: ``"weighted"`` to reproduce the official score, or
            ``"macro"`` / ``"micro"`` for ablations.

    Returns:
        Arithmetic mean of the per-project F1 scores.
    """
    return float(np.mean(list(per_repo_f1(y_true, y_pred, repos, average).values())))


#: Baseline scores published by the organisers, read from the result files
#: committed in the competition repository. The competition README quotes
#: 0.8270 for SetFit while the committed run of the same configuration gives
#: 0.8240 -- a 0.3-point gap between the organisers and themselves, which is a
#: useful reminder of how little a third decimal place means here.
OFFICIAL_BASELINES: dict[str, dict[str, float]] = {
    "setfit": {
        "facebook/react": 0.8751,
        "tensorflow/tensorflow": 0.8679,
        "microsoft/vscode": 0.8031,
        "bitcoin/bitcoin": 0.7507,
        "opencv/opencv": 0.8231,
        "cross-repo": 0.8240,
    },
    "roberta": {
        "facebook/react": 0.8441,
        "tensorflow/tensorflow": 0.8585,
        "microsoft/vscode": 0.7643,
        "bitcoin/bitcoin": 0.7526,
        "opencv/opencv": 0.7422,
        "cross-repo": 0.7923,
    },
    "fasttext": {
        "facebook/react": 0.7876,
        "tensorflow/tensorflow": 0.7063,
        "microsoft/vscode": 0.7275,
        "bitcoin/bitcoin": 0.6618,
        "opencv/opencv": 0.7088,
        "cross-repo": 0.7184,
    },
}


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
        counts = _confusion(true[mask], pred[mask], labels=LABELS)
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
            # weighted, to match the metric the competition ranks on.
            scores.append(weighted_f1(true[draw], pred[draw]))
        replicates[i] = float(np.mean(scores))

    tail = (1.0 - confidence) / 2.0 * 100.0
    lower, upper = np.percentile(replicates, [tail, 100.0 - tail])
    point = cross_repo_f1(true, pred, source)
    return point, float(lower), float(upper)
