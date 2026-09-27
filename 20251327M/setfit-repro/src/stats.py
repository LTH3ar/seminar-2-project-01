"""Significance, uncertainty and statistical power.

Three facts about this benchmark shape everything reported here:

* Repeating a configuration with a different seed moves the score by up to
  **2.4 F1 points** on its own.
* The test set cannot resolve a difference below roughly **3.0 points** at
  80% power.
* The best published result on the task, a fine-tuned GPT-3.5 at 0.828, sits
  **one tenth of a point** above the baseline it beat.

Together they mean that a leaderboard delta of one or two points is not a
result. The functions here exist so that a claim can be checked rather than
asserted.

**A paired test against the published baseline is impossible**: the organisers
released aggregate scores, not per-issue predictions. That is the reason the
reproduction in phase 1 is not optional -- once we have run the baseline
ourselves, we hold its predictions, and :func:`mcnemar` becomes available for
every later comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np

from .data import REPOSITORIES
from .metrics import cross_repo_f1

#: Difference below which a claim of improvement is not made, in F1 points.
DECISION_THRESHOLD = 0.030


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of one paired comparison.

    Attributes:
        n01: Items the baseline got wrong and the challenger got right.
        n10: Items the baseline got right and the challenger got wrong.
        n_discordant: ``n01 + n10``; the only items the test can see.
        p_value: Two-sided exact binomial p-value against ``p = 0.5``.
    """

    n01: int
    n10: int
    n_discordant: int
    p_value: float

    @property
    def favours_challenger(self) -> bool:
        """True when the challenger won more of the discordant items."""
        return self.n01 > self.n10


def _binom_two_sided(successes: int, trials: int) -> float:
    """Two-sided exact binomial p-value against ``p = 0.5``."""
    if trials == 0:
        return 1.0
    k = min(successes, trials - successes)
    tail = sum(comb(trials, i) for i in range(k + 1)) / 2**trials
    return min(1.0, 2.0 * tail)


def mcnemar(
    baseline_pred: Sequence[str],
    challenger_pred: Sequence[str],
    y_true: Sequence[str],
) -> McNemarResult:
    """Compare two models that were evaluated on the same items.

    The exact binomial form is used rather than the chi-square approximation:
    per project there are only 300 items and the discordant counts run to a
    few dozen, where the approximation is unreliable.

    Args:
        baseline_pred: Predictions of the reference model.
        challenger_pred: Predictions of the model being proposed.
        y_true: Ground-truth labels.

    Returns:
        A :class:`McNemarResult`.
    """
    truth = np.asarray(y_true, dtype=object)
    base_ok = np.asarray(baseline_pred, dtype=object) == truth
    chal_ok = np.asarray(challenger_pred, dtype=object) == truth
    n01 = int(np.sum(~base_ok & chal_ok))
    n10 = int(np.sum(base_ok & ~chal_ok))
    return McNemarResult(n01, n10, n01 + n10, _binom_two_sided(min(n01, n10), n01 + n10))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down correction for a family of tests.

    Args:
        p_values: Raw p-values, one per project.

    Returns:
        Adjusted p-values in the original order.
    """
    values = list(p_values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min((len(values) - rank) * values[index], 1.0))
        adjusted[index] = running
    return adjusted


def compare(
    baseline_pred: Sequence[str],
    challenger_pred: Sequence[str],
    y_true: Sequence[str],
    repos: Sequence[str],
) -> dict:
    """Run McNemar per project, correct across the five, and pool.

    Args:
        baseline_pred: Predictions of the reference model.
        challenger_pred: Predictions of the model being proposed.
        y_true: Ground-truth labels.
        repos: Source project of each issue.

    Returns:
        A record with one row per project plus a pooled row, and the overall
        score difference.
    """
    source = np.asarray(repos, dtype=object)
    # Canonical project order, not order of appearance: the frames are
    # shuffled, so appearance order changes with the seed and the significance
    # table would not line up with the score table above it.
    present = set(source.tolist())
    names = [r for r in REPOSITORIES if r in present]
    names += [r for r in dict.fromkeys(source.tolist()) if r not in REPOSITORIES]
    base = np.asarray(baseline_pred, dtype=object)
    chal = np.asarray(challenger_pred, dtype=object)
    truth = np.asarray(y_true, dtype=object)

    results = [
        mcnemar(base[source == r], chal[source == r], truth[source == r])
        for r in names
    ]
    adjusted = holm([r.p_value for r in results])
    rows = [
        {
            "repo": name,
            "n01": r.n01,
            "n10": r.n10,
            "discordant": r.n_discordant,
            "p_value": r.p_value,
            "p_holm": p,
            "significant": bool(p < 0.05),
        }
        for name, r, p in zip(names, results, adjusted, strict=True)
    ]
    pooled = mcnemar(base, chal, truth)
    delta = cross_repo_f1(truth, chal, source) - cross_repo_f1(truth, base, source)
    return {
        "delta_cross_repo_f1": delta,
        "clears_threshold": bool(abs(delta) >= DECISION_THRESHOLD),
        "per_repo": rows,
        "pooled": {
            "n01": pooled.n01,
            "n10": pooled.n10,
            "discordant": pooled.n_discordant,
            "n_items": len(truth),
            "p_value": pooled.p_value,
        },
        "survivors": sum(1 for row in rows if row["significant"]),
    }


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    repos: Sequence[str],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Confidence interval for the cross-repository score.

    Resampling is stratified by project, so each replicate keeps the
    five-project structure the metric is built on. Pooling the test set and
    resampling it flat would understate the variance.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        repos: Source project of each issue.
        n_resamples: Number of bootstrap replicates.
        confidence: Interval width.
        seed: Seed of the generator.

    Returns:
        ``(point_estimate, lower, upper)``.
    """
    rng = np.random.default_rng(seed)
    truth = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    source = np.asarray(repos, dtype=object)
    index = {r: np.flatnonzero(source == r) for r in dict.fromkeys(source.tolist())}

    replicates = np.empty(n_resamples)
    for i in range(n_resamples):
        draw = np.concatenate(
            [rng.choice(ix, size=ix.size, replace=True) for ix in index.values()]
        )
        replicates[i] = cross_repo_f1(truth[draw], pred[draw], source[draw])

    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(replicates, [tail, 100.0 - tail])
    return cross_repo_f1(truth, pred, source), float(low), float(high)


def minimum_detectable_difference(
    discordant_rate: float,
    n_items: int = 1500,
    target_power: float = 0.8,
    alpha: float = 0.05,
    n_simulations: int = 2000,
    seed: int = 42,
) -> float:
    """Smallest difference this test set can detect, by simulation.

    Args:
        discordant_rate: Fraction of items on which the two models differ in
            outcome. Measure it with :func:`mcnemar` rather than assuming.
        n_items: Size of the evaluation set.
        target_power: Power to reach.
        alpha: Significance level.
        n_simulations: Simulated experiments per candidate difference.
        seed: Seed of the generator.

    Returns:
        The minimum detectable difference as a fraction.
    """
    rng = np.random.default_rng(seed)
    n_discordant = int(round(discordant_rate * n_items))
    if n_discordant == 0:
        return 1.0
    delta = 0.0025
    while delta <= 0.15:
        p_favourable = 0.5 + (delta * n_items) / (2 * n_discordant)
        if not 0.0 < p_favourable < 1.0:
            return round(delta, 6)
        draws = rng.binomial(n_discordant, p_favourable, n_simulations)
        power = np.mean(
            [
                _binom_two_sided(min(int(k), n_discordant - int(k)), n_discordant)
                < alpha
                for k in draws
            ]
        )
        if power >= target_power:
            return round(delta, 6)
        delta += 0.0025
    return 0.15


def describe_claim(delta: float, comparison: dict) -> str:
    """Turn a measured difference into the sentence we are allowed to write.

    Args:
        delta: Difference in cross-repository F1.
        comparison: Output of :func:`compare`.

    Returns:
        The wording to use in the report.
    """
    points = delta * 100
    if abs(delta) < DECISION_THRESHOLD:
        # Two different numbers, and calling both of them "the resolution of
        # this test set" is how a reader stops trusting the rest of the file.
        # DECISION_THRESHOLD is the preregistered bar, fixed before any
        # experiment ran. What *this* comparison resolves follows from its own
        # discordant rate -- a property of the pair of models, not of the test
        # set. Two runs of one configuration disagree on barely 1% of issues
        # and so resolve a much smaller difference than two genuinely
        # different models would. Report both, and label them.
        pooled = comparison["pooled"]
        resolved = minimum_detectable_difference(
            pooled["discordant"] / pooled["n_items"], n_items=pooled["n_items"]
        )
        return (
            f"no detectable difference ({points:+.2f} points, below the "
            f"preregistered {DECISION_THRESHOLD * 100:.1f}-point decision "
            f"threshold; this comparison resolves {resolved * 100:.1f} points; "
            f"{comparison['survivors']}/5 projects significant after Holm)"
        )
    direction = "higher" if delta > 0 else "lower"
    return (
        f"{points:+.2f} points {direction}, above the detection threshold; "
        f"{comparison['survivors']}/5 projects significant after Holm, "
        f"pooled p = {comparison['pooled']['p_value']:.2e}"
    )
