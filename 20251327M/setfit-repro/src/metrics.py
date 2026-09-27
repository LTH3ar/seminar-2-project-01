"""The competition's metric, computed the way the organisers compute it.

Their notebook calls scikit-learn's ``classification_report`` per project and
reads the **weighted average** F1 out of it, then takes the plain mean of the
five. We call the same function rather than reimplementing it, because here
fidelity to their computation matters more than independence from their
library: a metric that is 0.002 different for a defensible reason is still a
metric that cannot be compared against 0.8270.

The distinction is not cosmetic. For a single-label problem micro-F1 equals
accuracy, and on this data it sits about 0.0015 above the weighted average --
enough to turn "just below the baseline" into "just above it".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from .data import LABELS, REPOSITORIES

#: Published per-project figures. The competition README says 0.8270 overall
#: while the result file committed in the same repository says 0.8240; both are
#: kept so a reproduction can be judged against either.
PUBLISHED = {
    "facebook/react": 0.8718,
    "tensorflow/tensorflow": 0.8644,
    "microsoft/vscode": 0.8262,
    "bitcoin/bitcoin": 0.7555,
    "opencv/opencv": 0.8173,
}
PUBLISHED_OVERALL_README = 0.8270
PUBLISHED_OVERALL_FILE = 0.8240

#: Best published result on this benchmark that we are aware of: a fine-tuned
#: GPT-3.5 at 0.828, i.e. one tenth of a point above the baseline. Recorded
#: here so that "beats the baseline" is never mistaken for "beats the field".
PUBLISHED_BEST_KNOWN = 0.828


def repo_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Return one project's score: the weighted-average F1 over the classes."""
    report = classification_report(
        y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0
    )
    return float(report["weighted avg"]["f1-score"])


def per_repo(
    y_true: Sequence[str], y_pred: Sequence[str], repos: Sequence[str]
) -> dict[str, float]:
    """Score each project separately, in the canonical project order."""
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    source = np.asarray(repos, dtype=object)
    present = [r for r in REPOSITORIES if (source == r).any()]
    return {r: repo_f1(true[source == r], pred[source == r]) for r in present}


def cross_repo_f1(
    y_true: Sequence[str], y_pred: Sequence[str], repos: Sequence[str]
) -> float:
    """Return the ranked figure: the mean of the five per-project scores."""
    scores = per_repo(y_true, y_pred, repos)
    return float(np.mean(list(scores.values()))) if scores else 0.0


def full_report(
    y_true: Sequence[str], y_pred: Sequence[str], repos: Sequence[str]
) -> dict:
    """Build the complete results record for one model.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        repos: Source project of each issue.

    Returns:
        A dictionary with the overall score, the per-project scores, and the
        per-project per-class precision/recall/F1/support table the
        competition also asks for.
    """
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    source = np.asarray(repos, dtype=object)

    detail = {}
    for repo in REPOSITORIES:
        mask = source == repo
        if not mask.any():
            continue
        detail[repo] = classification_report(
            true[mask], pred[mask], labels=list(LABELS),
            output_dict=True, zero_division=0,
        )

    scores = per_repo(true, pred, source)
    overall = float(np.mean(list(scores.values()))) if scores else 0.0
    return {
        "cross_repo_f1": overall,
        "per_repo": scores,
        "vs_published_readme": overall - PUBLISHED_OVERALL_README,
        "vs_published_file": overall - PUBLISHED_OVERALL_FILE,
        "vs_published_per_repo": {
            r: scores[r] - PUBLISHED[r] for r in scores if r in PUBLISHED
        },
        "classification_report": detail,
    }


def macro_auc(
    y_true: Sequence[str], probabilities: np.ndarray, classes: Sequence[str]
) -> float | None:
    """Return one-vs-rest macro AUC, or None when probabilities are absent.

    Args:
        y_true: Ground-truth labels.
        probabilities: Array of shape ``(n_items, n_classes)``.
        classes: Column order of ``probabilities``.

    Returns:
        Macro AUC, or None if it cannot be computed.
    """
    if probabilities is None:
        return None
    try:
        return float(
            roc_auc_score(
                y_true, probabilities, multi_class="ovr",
                average="macro", labels=list(classes),
            )
        )
    except ValueError:
        # Raised when a class is absent from y_true; not an error worth
        # failing a run over, but not a number worth reporting either.
        return None


def verdict(score: float, reference: float = PUBLISHED_OVERALL_FILE) -> str:
    """Judge a reproduction against a published figure.

    The thresholds come from the measured detection limit of this test set:
    differences below about three F1 points cannot be resolved here, so a gap
    inside that band is agreement rather than disagreement.

    Args:
        score: Our cross-repository F1.
        reference: The published figure to judge against.

    Returns:
        A short human-readable verdict.
    """
    gap = abs(score - reference)
    if gap <= 0.010:
        return "reproduced (within 1.0 point)"
    if gap <= 0.030:
        return "consistent (inside the 3.0-point noise floor) -- investigate"
    return "FAILED to reproduce -- find the cause before going further"
