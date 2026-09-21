"""Uncertainty and paired-comparison utilities for evaluation results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from math import comb
from statistics import mean, stdev
from typing import Any

import numpy as np

# Measured between word-only and word+character TF-IDF baselines on the
# official holdout. Callers comparing another pair can pass their own rate.
DEFAULT_DISCORDANT_RATE = 0.158


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def weighted_f1(
    references: Sequence[str],
    predictions: Sequence[str],
) -> float:
    """Return support-weighted F1 without requiring scikit-learn."""

    if len(references) != len(predictions):
        raise ValueError("References and predictions must have the same length")
    labels = sorted(set(references) | set(predictions))
    total = len(references)
    if total == 0:
        return 0.0

    score = 0.0
    for label in labels:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(references, predictions, strict=True)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(references, predictions, strict=True)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(references, predictions, strict=True)
        )
        support = true_positive + false_negative
        score += support * _f1(true_positive, false_positive, false_negative)
    return score / total


def result_repository_score(result: Mapping[str, Any]) -> float:
    """Read the official weighted-F1 repository average from a result."""

    return float(result["overall"]["repository_average"]["weighted_average"]["f1"])


def summarise_repeated_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarise repeated CV runs without duplicating all predictions."""

    if not results:
        raise ValueError("At least one evaluation result is required")
    scores = [result_repository_score(result) for result in results]
    repositories = sorted(results[0]["repositories"])
    repository_scores = {
        repository: [
            float(
                result["repositories"][repository]["metrics"]["weighted_average"]["f1"]
            )
            for result in results
        ]
        for repository in repositories
    }
    return {
        "model_name": results[0]["model_name"],
        "evaluation": "repeated_stratified_group_k_fold",
        "random_states": [int(result["random_state"]) for result in results],
        "scores": scores,
        "mean": mean(scores),
        "standard_deviation": stdev(scores) if len(scores) > 1 else 0.0,
        "minimum": min(scores),
        "maximum": max(scores),
        "repositories": {
            repository: {
                "scores": values,
                "mean": mean(values),
                "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
            }
            for repository, values in repository_scores.items()
        },
    }


def bootstrap_confidence_interval(
    result: Mapping[str, Any],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Bootstrap the competition score while retaining repository structure."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")

    grouped: dict[str, list[Mapping[str, Any]]] = {
        repository: list(values["predictions"])
        for repository, values in result["repositories"].items()
    }
    if not grouped or any(not rows for rows in grouped.values()):
        raise ValueError("Every repository must contain predictions")

    generator = np.random.default_rng(random_state)
    bootstrap_scores = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        repository_scores = []
        for rows in grouped.values():
            positions = generator.integers(0, len(rows), size=len(rows))
            references = [str(rows[position]["actual"]) for position in positions]
            predictions = [str(rows[position]["predicted"]) for position in positions]
            repository_scores.append(weighted_f1(references, predictions))
        bootstrap_scores[index] = mean(repository_scores)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_scores, [tail, 1.0 - tail])
    return {
        "point_estimate": result_repository_score(result),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "random_state": random_state,
    }


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Exact paired-comparison counts and p-value."""

    first_wrong_second_right: int
    first_right_second_wrong: int
    discordant: int
    p_value: float


@cache
def _exact_binomial_p_value(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    smaller_tail = min(successes, trials - successes)
    tail_probability = (
        sum(comb(trials, count) for count in range(smaller_tail + 1)) / 2**trials
    )
    return min(1.0, 2.0 * tail_probability)


def mcnemar_exact(
    first_predictions: Sequence[str],
    second_predictions: Sequence[str],
    references: Sequence[str],
) -> McNemarResult:
    """Compare two models on the same examples using exact McNemar."""

    if not (len(first_predictions) == len(second_predictions) == len(references)):
        raise ValueError("Predictions and references must have the same length")
    first_correct = np.asarray(first_predictions, dtype=object) == np.asarray(
        references, dtype=object
    )
    second_correct = np.asarray(second_predictions, dtype=object) == np.asarray(
        references, dtype=object
    )
    first_wrong_second_right = int(np.sum(~first_correct & second_correct))
    first_right_second_wrong = int(np.sum(first_correct & ~second_correct))
    discordant = first_wrong_second_right + first_right_second_wrong
    return McNemarResult(
        first_wrong_second_right=first_wrong_second_right,
        first_right_second_wrong=first_right_second_wrong,
        discordant=discordant,
        p_value=_exact_binomial_p_value(
            first_wrong_second_right,
            discordant,
        ),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply Holm-Bonferroni correction in the original input order."""

    values = list(p_values)
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def compare_evaluations(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare two saved evaluations per repository with paired tests."""

    first_rows = _predictions_by_id(first)
    second_rows = _predictions_by_id(second)
    if set(first_rows) != set(second_rows):
        raise ValueError("Evaluations must contain the same issue IDs")

    by_repository: dict[str, list[str]] = defaultdict(list)
    for issue_id, row in first_rows.items():
        other = second_rows[issue_id]
        if row["actual"] != other["actual"] or row["repository"] != other["repository"]:
            raise ValueError(f"Evaluation records disagree for issue {issue_id}")
        by_repository[str(row["repository"])].append(issue_id)

    repository_results = []
    for repository, issue_ids in sorted(by_repository.items()):
        outcome = mcnemar_exact(
            [str(first_rows[value]["predicted"]) for value in issue_ids],
            [str(second_rows[value]["predicted"]) for value in issue_ids],
            [str(first_rows[value]["actual"]) for value in issue_ids],
        )
        repository_results.append((repository, outcome))
    adjusted = holm_adjust([outcome.p_value for _, outcome in repository_results])
    return [
        {
            "repository": repository,
            **asdict(outcome),
            "holm_adjusted_p_value": adjusted_value,
            "significant": adjusted_value < 0.05,
        }
        for (repository, outcome), adjusted_value in zip(
            repository_results,
            adjusted,
            strict=True,
        )
    ]


@cache
def minimum_detectable_difference(
    *,
    target_power: float = 0.8,
    n_items: int = 1_500,
    discordant_rate: float = DEFAULT_DISCORDANT_RATE,
    alpha: float = 0.05,
    resolution: float = 0.0025,
    n_simulations: int = 2_000,
    random_state: int = 42,
) -> float:
    """Estimate the smallest paired accuracy difference detectable by McNemar."""

    for delta in np.arange(resolution, 0.150001, resolution):
        if (
            _power_at(
                float(delta),
                n_items=n_items,
                discordant_rate=discordant_rate,
                alpha=alpha,
                n_simulations=n_simulations,
                random_state=random_state,
            )
            >= target_power
        ):
            return round(float(delta), 6)
    return 0.15


def _power_at(
    delta: float,
    *,
    n_items: int,
    discordant_rate: float,
    alpha: float,
    n_simulations: int,
    random_state: int,
) -> float:
    discordant = int(round(discordant_rate * n_items))
    if discordant == 0:
        return 0.0
    favourable_probability = 0.5 + (delta * n_items) / (2 * discordant)
    if not 0.0 < favourable_probability < 1.0:
        return 1.0
    generator = np.random.default_rng(random_state)
    draws = generator.binomial(discordant, favourable_probability, n_simulations)
    detected = sum(
        _exact_binomial_p_value(int(value), discordant) < alpha for value in draws
    )
    return detected / n_simulations


def _predictions_by_id(
    result: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = {
        str(row["issue_id"]): row
        for values in result["repositories"].values()
        for row in values["predictions"]
    }
    expected = sum(
        len(values["predictions"]) for values in result["repositories"].values()
    )
    if len(rows) != expected:
        raise ValueError("Evaluation contains duplicated issue IDs")
    return rows
