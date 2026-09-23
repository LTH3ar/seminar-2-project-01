"""Paired significance testing for classifier comparisons.

Two classifiers are evaluated on the *same* test issues, so the comparison is
paired and McNemar's test applies. The exact binomial form is used rather than
the chi-square approximation: with five projects of 300 issues each, the
discordant counts per project are often below thirty, where the approximation
is unreliable.

Because the comparison is repeated once per project, the five p-values form a
family and must be corrected. :func:`holm` does that; without it, one project
out of five reaching ``p < 0.05`` by chance alone is unremarkable.

Implemented with the standard library and numpy only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Outcome of one paired comparison.

    Attributes:
        n01: Items the first model got wrong and the second got right.
        n10: Items the first model got right and the second got wrong.
        n_discordant: ``n01 + n10``; the only items the test looks at.
        p_value: Two-sided exact binomial p-value against ``p = 0.5``.
    """

    n01: int
    n10: int
    n_discordant: int
    p_value: float

    @property
    def favours_second(self) -> bool:
        """True when the second model won more of the discordant items."""
        return self.n01 > self.n10


def _binom_two_sided(successes: int, trials: int) -> float:
    """Return the two-sided exact binomial p-value against ``p = 0.5``.

    Under ``p = 0.5`` the distribution is symmetric, so the two-sided p-value
    is twice the smaller tail, clipped at one.

    Args:
        successes: Observed count in one of the two discordant cells.
        trials: Total discordant count.

    Returns:
        p-value in ``(0, 1]``.
    """
    if trials == 0:
        return 1.0
    k = min(successes, trials - successes)
    tail = sum(comb(trials, i) for i in range(k + 1)) / 2**trials
    return min(1.0, 2.0 * tail)


def mcnemar_exact(
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    y_true: Sequence[str],
) -> McNemarResult:
    """Compare two sets of predictions on the same items.

    Args:
        pred_a: Predictions of the first model.
        pred_b: Predictions of the second model.
        y_true: Ground-truth labels.

    Returns:
        A :class:`McNemarResult`. ``n01`` counts items where ``pred_b`` is
        right and ``pred_a`` is wrong, so a large ``n01`` favours the second
        model.
    """
    correct_a = np.asarray(pred_a, dtype=object) == np.asarray(y_true, dtype=object)
    correct_b = np.asarray(pred_b, dtype=object) == np.asarray(y_true, dtype=object)
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    return McNemarResult(
        n01=n01,
        n10=n10,
        n_discordant=n01 + n10,
        p_value=_binom_two_sided(min(n01, n10), n01 + n10),
    )


def holm(p_values: Sequence[float]) -> list[float]:
    """Apply the Holm-Bonferroni step-down correction.

    Controls the family-wise error rate without assuming the tests are
    independent, which they are not here -- the five per-project comparisons
    share a model.

    Args:
        p_values: Raw p-values of one family of tests.

    Returns:
        Adjusted p-values in the original order, each clipped at one and
        monotonically non-decreasing in rank.
    """
    values = list(p_values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, min(candidate, 1.0))
        adjusted[index] = running
    return adjusted


def compare_per_repo(
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    y_true: Sequence[str],
    repos: Sequence[str],
) -> list[dict]:
    """Run McNemar once per project and correct across the five tests.

    Args:
        pred_a: Predictions of the first model.
        pred_b: Predictions of the second model.
        y_true: Ground-truth labels.
        repos: Source project of each issue.

    Returns:
        One row per project with the discordant counts, the raw p-value and
        the Holm-adjusted p-value, plus a final pooled row whose adjusted
        p-value is left equal to the raw one (it is not part of the family).
    """
    source = np.asarray(repos, dtype=object)
    names = sorted(set(source.tolist()))
    results = []
    for repo in names:
        mask = source == repo
        results.append(
            mcnemar_exact(
                np.asarray(pred_a, dtype=object)[mask],
                np.asarray(pred_b, dtype=object)[mask],
                np.asarray(y_true, dtype=object)[mask],
            )
        )
    adjusted = holm([r.p_value for r in results])

    rows = [
        {
            "repo": repo,
            "n01": r.n01,
            "n10": r.n10,
            "discordant": r.n_discordant,
            "p_value": r.p_value,
            "p_holm": p,
            "significant": p < 0.05,
        }
        for repo, r, p in zip(names, results, adjusted, strict=True)
    ]
    pooled = mcnemar_exact(pred_a, pred_b, y_true)
    rows.append(
        {
            "repo": "POOLED",
            "n01": pooled.n01,
            "n10": pooled.n10,
            "discordant": pooled.n_discordant,
            "p_value": pooled.p_value,
            "p_holm": pooled.p_value,
            "significant": pooled.p_value < 0.05,
        }
    )
    return rows


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> tuple[float, str]:
    """Measure how large a difference is, independently of sample size.

    A p-value says a difference exists; it does not say the difference
    matters. Report both. Thresholds follow the conventional reading of
    Cliff's delta: negligible below 0.147, small below 0.33, medium below
    0.474, large above.

    Args:
        a: Observations of the first group, e.g. per-seed scores.
        b: Observations of the second group.

    Returns:
        Tuple ``(delta, interpretation)`` with delta in ``[-1, 1]``.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0, "undefined"
    greater = int(np.sum(x[:, None] > y[None, :]))
    less = int(np.sum(x[:, None] < y[None, :]))
    delta = (greater - less) / (x.size * y.size)

    magnitude = abs(delta)
    if magnitude < 0.147:
        label = "negligible"
    elif magnitude < 0.33:
        label = "small"
    elif magnitude < 0.474:
        label = "medium"
    else:
        label = "large"
    return float(delta), label
