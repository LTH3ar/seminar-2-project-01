"""Classification metrics for the issue report task.

The metrics are implemented directly rather than taken from scikit-learn, for
three reasons: the evaluation module then has no hard dependency on a
modelling library, the definitions match exactly what the course covered
(Evaluation of Machine Learning Systems), and being able to derive precision,
recall and F1 from a confusion matrix is the kind of thing the oral asks
about. ``tests/test_metrics.py`` cross-checks every function against
scikit-learn when it is installed.

Definitions, for a class ``c``:

    TP  predicted c, truly c
    FP  predicted c, truly something else
    FN  predicted something else, truly c

    precision = TP / (TP + FP)      of what we predicted, how much was right
    recall    = TP / (TP + FN)      of what was there, how much we found
    F1        = 2PR / (P + R)       harmonic mean of the two

Averaging:

    macro  unweighted mean of the per-class scores -- every class counts the
           same, so a rare class cannot be ignored
    micro  computed from the pooled TP/FP/FN over all classes; for
           single-label multi-class problems it equals accuracy
    weighted  mean of per-class scores weighted by class support

The NLBSE'24 dataset is exactly balanced, so macro and micro coincide here.
They are all reported anyway: the agreement is itself worth stating in the
report, and any resampling during development would break it immediately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassScore:
    """Precision, recall, F1 and support for a single class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict[str, float | str | int]:
        """Serialise to a plain dictionary."""
        return {
            "label": self.label,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True)
class Scores:
    """Full evaluation of one set of predictions."""

    per_class: dict[str, ClassScore]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    confusion: dict[str, dict[str, int]]
    n: int
    labels: tuple[str, ...] = field(default=())
    #: One-vs-rest ROC AUC per class. Populated only when the model exposed
    #: class probabilities; ``None`` for a model that returns hard labels.
    per_class_auc: dict[str, float] | None = None
    macro_auc: float | None = None

    def f1(self, label: str) -> float:
        """F1 of a single class."""
        return self.per_class[label].f1

    def auc(self, label: str) -> float | None:
        """One-vs-rest ROC AUC of a single class, if probabilities were given."""
        return None if self.per_class_auc is None else self.per_class_auc[label]

    def as_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_auc": self.macro_auc,
            "per_class": {k: v.as_dict() for k, v in self.per_class.items()},
            "per_class_auc": self.per_class_auc,
            "confusion": self.confusion,
        }

    def to_frame(self):
        """Return the per-class table as a pandas DataFrame with an average row."""
        import pandas as pd

        rows = [
            {
                "class": score.label,
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "support": score.support,
                **(
                    {"auc": self.per_class_auc[score.label]}
                    if self.per_class_auc
                    else {}
                ),
            }
            for score in self.per_class.values()
        ]
        rows.append(
            {
                "class": "macro avg",
                "precision": self.macro_precision,
                "recall": self.macro_recall,
                "f1": self.macro_f1,
                "support": self.n,
                **({"auc": self.macro_auc} if self.per_class_auc else {}),
            }
        )
        return pd.DataFrame(rows).set_index("class").round(4)

    def confusion_frame(self):
        """Return the confusion matrix as a DataFrame (rows: true, cols: predicted)."""
        import pandas as pd

        order = list(self.labels) or sorted(self.confusion)
        return pd.DataFrame(
            [[self.confusion[t][p] for p in order] for t in order],
            index=pd.Index(order, name="true"),
            columns=pd.Index(order, name="predicted"),
        )

    def __repr__(self) -> str:
        """Compact one-line summary."""
        return (
            f"Scores(n={self.n}, accuracy={self.accuracy:.4f}, "
            f"macro_f1={self.macro_f1:.4f})"
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or 0.0 when the denominator is 0.

    A class that was never predicted has an undefined precision. Reporting 0.0
    is the scikit-learn convention (``zero_division=0``) and is the
    conservative choice: it penalises a model that ignores a class.
    """
    return numerator / denominator if denominator else 0.0


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix as nested dictionaries.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels, same length as ``y_true``.
        labels: Label order. Inferred from the data when omitted.

    Returns:
        ``matrix[true_label][predicted_label] -> count``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(y_true)} true labels, {len(y_pred)} predictions."
        )
    order = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    matrix = {t: dict.fromkeys(order, 0) for t in order}
    for true, predicted in zip(y_true, y_pred, strict=True):
        if true not in matrix:
            raise ValueError(f"Unexpected true label {true!r}; known: {order}")
        if predicted not in matrix[true]:
            raise ValueError(
                f"Unexpected predicted label {predicted!r}; known: {order}"
            )
        matrix[true][predicted] += 1
    return matrix


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    y_score: Sequence[dict[str, float]] | None = None,
) -> Scores:
    """Compute the full set of metrics for one set of predictions.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Label order. Inferred from the data when omitted.
        y_score: Optional per-sample mapping from class to predicted
            probability. When given, one-vs-rest ROC AUC is computed as well;
            when omitted, the AUC fields stay ``None`` rather than being
            approximated from the hard labels.

    Returns:
        A :class:`Scores` object.

    Example:
        >>> scores = evaluate(["bug", "feature"], ["bug", "bug"])
        >>> round(scores.accuracy, 2)
        0.5
    """
    order = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, order)
    total = len(y_true)

    per_class: dict[str, ClassScore] = {}
    for label in order:
        true_positives = matrix[label][label]
        false_negatives = sum(matrix[label][p] for p in order) - true_positives
        false_positives = sum(matrix[t][label] for t in order) - true_positives
        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, true_positives + false_negatives)
        per_class[label] = ClassScore(
            label=label,
            precision=precision,
            recall=recall,
            f1=_safe_divide(2 * precision * recall, precision + recall),
            support=true_positives + false_negatives,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

    n_classes = len(order) or 1
    macro_precision = sum(s.precision for s in per_class.values()) / n_classes
    macro_recall = sum(s.recall for s in per_class.values()) / n_classes
    macro_f1 = sum(s.f1 for s in per_class.values()) / n_classes

    # Micro averaging pools the counts before dividing.
    pooled_tp = sum(s.true_positives for s in per_class.values())
    pooled_fp = sum(s.false_positives for s in per_class.values())
    pooled_fn = sum(s.false_negatives for s in per_class.values())
    micro_precision = _safe_divide(pooled_tp, pooled_tp + pooled_fp)
    micro_recall = _safe_divide(pooled_tp, pooled_tp + pooled_fn)
    micro_f1 = _safe_divide(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )

    weighted_f1 = _safe_divide(sum(s.f1 * s.support for s in per_class.values()), total)

    per_class_auc, macro_auc = (
        roc_auc_ovr(y_true, y_score, order) if y_score is not None else (None, None)
    )

    return Scores(
        per_class=per_class,
        accuracy=_safe_divide(pooled_tp, total),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        weighted_f1=weighted_f1,
        confusion=matrix,
        n=total,
        labels=tuple(order),
        per_class_auc=per_class_auc,
        macro_auc=macro_auc,
    )


# --------------------------------------------------------------------------- #
# ROC and AUC
#
# The course lists AUC and ROC alongside accuracy, precision, recall and F1, so
# they are reported wherever a model can produce class probabilities. A model
# that only returns hard labels (the keyword rules, for instance) simply has no
# AUC, and the corresponding fields stay None rather than being faked from the
# predicted labels.
#
# The task is multi-class, so ROC is computed one-vs-rest: for each class in
# turn, that class is the positive one and the other two are pooled as
# negative. The macro AUC is the unweighted mean over the three curves.
# --------------------------------------------------------------------------- #


def roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    """Compute a receiver operating characteristic curve.

    Samples are sorted by descending score and the decision threshold is swept
    from "predict nothing positive" down through every observed score. At each
    threshold the true positive rate is plotted against the false positive
    rate; a perfect ranking reaches the top-left corner.

    Args:
        y_true: Binary ground truth, 1 for the positive class and 0 otherwise.
        y_score: Score or probability assigned to the positive class.

    Returns:
        ``(fpr, tpr, thresholds)``, each a list of equal length, starting at
        the ``(0, 0)`` origin.

    Raises:
        ValueError: If the inputs differ in length.
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: {len(y_true)} labels, {len(y_score)} scores."
        )

    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        # One class is absent, so the curve is undefined. Return the diagonal.
        return [0.0, 1.0], [0.0, 1.0], [float("inf"), float("-inf")]

    ordered = sorted(zip(y_score, y_true, strict=True), key=lambda p: -p[0])

    fpr: list[float] = [0.0]
    tpr: list[float] = [0.0]
    thresholds: list[float] = [float("inf")]

    true_positives = 0
    false_positives = 0
    previous_score = None

    for score, label in ordered:
        # Samples sharing a score cannot be separated by any threshold, so the
        # curve only gains a point when the score actually changes.
        if previous_score is not None and score != previous_score:
            fpr.append(false_positives / negatives)
            tpr.append(true_positives / positives)
            thresholds.append(previous_score)
        if label:
            true_positives += 1
        else:
            false_positives += 1
        previous_score = score

    fpr.append(false_positives / negatives)
    tpr.append(true_positives / positives)
    thresholds.append(previous_score if previous_score is not None else 0.0)
    return fpr, tpr, thresholds


def auc(x: Sequence[float], y: Sequence[float]) -> float:
    """Area under a curve, by the trapezoidal rule.

    Args:
        x: Monotonically non-decreasing x coordinates.
        y: Corresponding y coordinates.

    Returns:
        The area, 0.0 when fewer than two points are given.
    """
    if len(x) < 2:
        return 0.0
    return sum(
        (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0 for i in range(1, len(x))
    )


def roc_auc_ovr(
    y_true: Sequence[str],
    y_score: Sequence[dict[str, float]],
    labels: Sequence[str] | None = None,
) -> tuple[dict[str, float], float]:
    """One-vs-rest ROC AUC per class, and the macro average.

    Args:
        y_true: Ground-truth labels.
        y_score: Per-sample mapping from class label to predicted probability.
        labels: Label order. Inferred from the data when omitted.

    Returns:
        ``(per_class_auc, macro_auc)``.

    Example:
        >>> scores = [{"bug": 0.9, "feature": 0.1}, {"bug": 0.2, "feature": 0.8}]
        >>> per_class, macro = roc_auc_ovr(["bug", "feature"], scores)
        >>> macro
        1.0
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: {len(y_true)} labels, {len(y_score)} score maps."
        )
    order = list(labels) if labels is not None else sorted(set(y_true))

    per_class: dict[str, float] = {}
    for label in order:
        binary = [1 if true == label else 0 for true in y_true]
        scores = [probabilities.get(label, 0.0) for probabilities in y_score]
        false_positive_rate, true_positive_rate, _ = roc_curve(binary, scores)
        per_class[label] = auc(false_positive_rate, true_positive_rate)

    macro = sum(per_class.values()) / (len(per_class) or 1)
    return per_class, macro
