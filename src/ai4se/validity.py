"""Analyses of how far the benchmark's results can be trusted.

Three questions, each of which changes how the leaderboard should be read:

**Is there a shortcut in the data?** No bug report in the dataset predates
2021, so the creation date alone predicts the class surprisingly well.
:func:`timestamp_only_scores` measures how well, with a classifier that never
reads a word of text. Its score is the meaningful floor for this benchmark.

**How small a difference can the test set detect?**
:func:`minimum_detectable_difference` simulates McNemar's exact test at a given
discordant rate and returns the smallest accuracy difference detected with 80%
power. On balanced data, macro F1 and accuracy move together closely enough for
the threshold to be read in F1 points.

**Does the per-project protocol help?** :func:`pooling_comparison` trains each
model once per project (the competition rule) and once on all projects pooled,
then scores both per project.

Every number these produce is written to ``results/tables/validity.json`` by
``scripts/run_experiments.py --validity``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from statistics import mean

import numpy as np

from .evaluation import RANDOM_SEED
from .metrics import evaluate
from .model import LABELS
from .repository import IssueRepository

# --------------------------------------------------------------------------- #
# Temporal confound
# --------------------------------------------------------------------------- #


def _years(repository: IssueRepository) -> np.ndarray:
    """Creation time of every issue, as fractional years."""
    import pandas as pd

    stamps = pd.to_datetime([issue.created_at for issue in repository])
    return (stamps.year + (stamps.dayofyear - 1) / 366.0).to_numpy()


def label_share_by_year(repository: IssueRepository):
    """Share of each class among the issues created in each year.

    Returns:
        A pandas DataFrame indexed by year, one column per class plus ``n``.
    """
    import pandas as pd

    frame = pd.DataFrame(
        {
            "year": np.floor(_years(repository)).astype(int),
            "label": [issue.label for issue in repository],
        }
    )
    table = pd.crosstab(frame["year"], frame["label"], normalize="index").round(4)
    table["n"] = frame["year"].value_counts().sort_index()
    return table


def timestamp_only_scores(
    train: IssueRepository,
    test: IssueRepository,
    max_depth: int = 3,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Macro F1 of a classifier that sees only each issue's creation date.

    Follows the competition protocol: one shallow decision tree per project,
    scored on that project's test issues. A depth of 3 allows at most eight
    date intervals, so the classifier can only learn "issues from this period
    are mostly this class" -- nothing about their content.

    Returns:
        ``{repository: macro_f1, ..., "mean": cross_repository_f1}``.
    """
    from sklearn.tree import DecisionTreeClassifier

    from .repository import make_repository

    scores: dict[str, float] = {}
    for repo in sorted(set(train.repos()) & set(test.repos())):
        tr = make_repository("memory", issues=train.by_repo(repo))
        te = make_repository("memory", issues=test.by_repo(repo))
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        model.fit(_years(tr).reshape(-1, 1), [i.label for i in tr])
        predictions = model.predict(_years(te).reshape(-1, 1))
        truth = [i.label for i in te]
        scores[repo] = evaluate(truth, list(predictions), LABELS).macro_f1
    scores["mean"] = mean(v for k, v in scores.items() if k != "mean")
    return scores


# --------------------------------------------------------------------------- #
# Statistical power
# --------------------------------------------------------------------------- #


def discordant_rate(
    pred_a: Sequence[str], pred_b: Sequence[str], y_true: Sequence[str]
) -> float:
    """Fraction of items that one model gets right and the other wrong.

    McNemar's test only looks at these items, so this rate -- not the accuracy
    of either model -- is what determines how much power the test has.
    """
    discordant = sum(
        (a == t) != (b == t) for a, b, t in zip(pred_a, pred_b, y_true, strict=True)
    )
    return discordant / max(len(y_true), 1)


def mcnemar_power(
    delta: float,
    n_items: int,
    rate: float,
    alpha: float = 0.05,
    n_simulations: int = 4000,
    seed: int = RANDOM_SEED,
) -> float:
    """Probability that McNemar's exact test detects a true difference ``delta``.

    Each simulation draws the number of discordant items, then how many of them
    favour the better model, and applies the two-sided exact binomial test.

    Args:
        delta: True accuracy difference between the two models, e.g. 0.03.
        n_items: Test-set size.
        rate: Discordant rate between the two models.
        alpha: Significance level.
        n_simulations: Monte Carlo draws.
        seed: Seed for the simulation.

    Returns:
        Estimated power in ``[0, 1]``.
    """
    from scipy.stats import binom

    rng = np.random.default_rng(seed)
    win_share = min(max(0.5 + delta / (2 * rate), 0.0), 1.0)
    n_discordant = rng.binomial(n_items, rate, size=n_simulations)
    wins = rng.binomial(n_discordant, win_share)
    tails = np.minimum(
        binom.cdf(wins, n_discordant, 0.5), binom.sf(wins - 1, n_discordant, 0.5)
    )
    p_values = np.minimum(1.0, 2.0 * tails)
    return float(np.mean(p_values < alpha))


def minimum_detectable_difference(
    n_items: int,
    rate: float,
    power: float = 0.8,
    alpha: float = 0.05,
    step: float = 0.001,
    seed: int = RANDOM_SEED,
) -> float:
    """Smallest accuracy difference McNemar's test detects with the given power.

    Returns:
        The difference as a fraction, e.g. ``0.030`` for three points.
    """
    delta = step
    while delta < rate:
        if mcnemar_power(delta, n_items, rate, alpha=alpha, seed=seed) >= power:
            return round(delta, 4)
        delta += step
    return float("nan")


# --------------------------------------------------------------------------- #
# Per-project versus pooled training
# --------------------------------------------------------------------------- #


def pooling_comparison(
    factory: Callable[[], object],
    train: IssueRepository,
    test: IssueRepository,
) -> dict:
    """Score a model trained per project against the same model trained pooled.

    Both are scored per project on the same test issues, so the only difference
    is how much -- and which -- training data each classifier saw.

    Returns:
        ``{"per_project": {...}, "pooled": {...}, "per_project_mean": float,
        "pooled_mean": float, "per_project_wins": int}``.
    """
    from .evaluation import evaluate_competition

    per_project = evaluate_competition(factory, train, test).repository_f1

    X, y = train.texts_and_labels()
    pooled_model = factory()
    pooled_model.fit(X, y)
    pooled: dict[str, float] = {}
    for repo in per_project:
        issues = test.by_repo(repo)
        predictions = list(pooled_model.predict([i.text for i in issues]))
        pooled[repo] = evaluate([i.label for i in issues], predictions, LABELS).macro_f1

    return {
        "per_project": per_project,
        "pooled": pooled,
        "per_project_mean": mean(per_project.values()),
        "pooled_mean": mean(pooled.values()),
        "per_project_wins": sum(per_project[r] > pooled[r] for r in per_project),
    }


def unpaired_mdd(
    n_items: int, accuracy: float, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Detectable difference when per-item predictions of one model are unavailable.

    Comparing against the *published* baseline is the case that needs this: the
    organisers publish scores, not per-issue predictions, so McNemar's paired
    test cannot be applied. Treating the two scores as independent proportions
    is conservative -- the true paired threshold is lower -- so a difference
    that fails this test is certainly not established.

    Args:
        n_items: Test-set size.
        accuracy: Accuracy of the reference model.
        power: Desired power.
        alpha: Two-sided significance level.

    Returns:
        The difference as a fraction.
    """
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return round(float(z * np.sqrt(2 * accuracy * (1 - accuracy) / n_items)), 4)
