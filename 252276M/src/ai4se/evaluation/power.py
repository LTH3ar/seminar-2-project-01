"""Statistical power of the NLBSE'24 test set.

Before claiming an improvement, it is worth knowing how small an improvement
this test set can detect at all. The answer, measured on the official split,
is about **3.0 F1 points** at 80% power -- and repeating the same
configuration with a different random seed already moves the score by up to
2.4 points. Differences below that threshold are not evidence of anything.

The estimate is a Monte Carlo simulation of McNemar's test rather than a
closed form, because the test statistic depends on the discordant rate
between the two specific models being compared, which is an empirical
quantity (15.8% for two reasonable baselines here).

Use :func:`observed_discordant_rate` to measure that rate for your own pair of
models before trusting the default.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .significance import _binom_two_sided, mcnemar_exact

#: Discordant rate measured between the word-level and character-level TF-IDF
#: baselines on the official test split. See ``docs/benchmark-audit.md``.
DEFAULT_DISCORDANT_RATE = 0.158


def observed_discordant_rate(
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    y_true: Sequence[str],
) -> float:
    """Return the fraction of items on which two models disagree in outcome.

    Args:
        pred_a: Predictions of the first model.
        pred_b: Predictions of the second model.
        y_true: Ground-truth labels.

    Returns:
        ``n_discordant / n_items`` in ``[0, 1]``.
    """
    result = mcnemar_exact(pred_a, pred_b, y_true)
    return result.n_discordant / max(1, len(y_true))


def power_at(
    delta: float,
    n_items: int = 1500,
    discordant_rate: float = DEFAULT_DISCORDANT_RATE,
    alpha: float = 0.05,
    n_simulations: int = 2000,
    seed: int = 42,
) -> float:
    """Estimate the probability of detecting a true difference of ``delta``.

    Args:
        delta: True difference in accuracy, as a fraction (``0.03`` is three
            F1 points).
        n_items: Size of the test set.
        discordant_rate: Fraction of items the two models resolve differently.
        alpha: Significance level of the two-sided test.
        n_simulations: Number of simulated experiments.
        seed: Seed of the random generator.

    Returns:
        Estimated power in ``[0, 1]``.
    """
    n_discordant = int(round(discordant_rate * n_items))
    if n_discordant == 0:
        return 0.0

    # A true accuracy difference of `delta` means delta * n_items more items
    # fall in one discordant cell than the other.
    p_favourable = 0.5 + (delta * n_items) / (2 * n_discordant)
    if not 0.0 < p_favourable < 1.0:
        return 1.0

    rng = np.random.default_rng(seed)
    draws = rng.binomial(n_discordant, p_favourable, n_simulations)
    detected = sum(
        _binom_two_sided(min(int(k), n_discordant - int(k)), n_discordant) < alpha
        for k in draws
    )
    return detected / n_simulations


def power_curve(
    deltas: Sequence[float] = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05),
    **kwargs,
) -> list[dict]:
    """Tabulate power against effect size.

    Args:
        deltas: True differences to evaluate, as fractions.
        **kwargs: Forwarded to :func:`power_at`.

    Returns:
        One row per delta with ``"delta_points"`` and ``"power"``.
    """
    return [{"delta_points": d * 100, "power": power_at(d, **kwargs)} for d in deltas]


def minimum_detectable_difference(
    target_power: float = 0.8,
    n_items: int = 1500,
    discordant_rate: float = DEFAULT_DISCORDANT_RATE,
    alpha: float = 0.05,
    resolution: float = 0.0025,
    max_delta: float = 0.15,
    **kwargs,
) -> float:
    """Find the smallest difference detectable at ``target_power``.

    This number belongs in the report next to every comparison. On the full
    1,500-item test set it is about 0.030; restricted to a single project
    (300 items) it rises to roughly 0.067, which is why per-project claims
    need far more care than cross-repository ones.

    Args:
        target_power: Power to reach, conventionally ``0.8``.
        n_items: Size of the test set.
        discordant_rate: Fraction of items the two models resolve differently.
        alpha: Significance level of the two-sided test.
        resolution: Step of the search, as a fraction.
        max_delta: Upper bound of the search.
        **kwargs: Forwarded to :func:`power_at`.

    Returns:
        Minimum detectable difference as a fraction, or ``max_delta`` if the
        target power is unreachable within the search range.
    """
    delta = resolution
    while delta <= max_delta:
        power = power_at(
            delta,
            n_items=n_items,
            discordant_rate=discordant_rate,
            alpha=alpha,
            **kwargs,
        )
        if power >= target_power:
            return round(delta, 6)
        delta += resolution
    return max_delta


def is_conclusive(delta: float, **kwargs) -> bool:
    """Return whether an observed difference is large enough to interpret.

    Intended as a guard in experiment scripts::

        if not is_conclusive(new_score - baseline_score):
            print("No detectable difference -- do not call this an improvement.")

    Args:
        delta: Observed difference in score, as a fraction.
        **kwargs: Forwarded to :func:`minimum_detectable_difference`.

    Returns:
        True when ``abs(delta)`` reaches the minimum detectable difference.
    """
    return abs(delta) >= minimum_detectable_difference(**kwargs)


estimate_mdd = minimum_detectable_difference
