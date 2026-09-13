"""Tests for the evaluation, significance and power utilities.

These are correctness tests with hand-checkable expectations, not smoke tests:
a metric that is silently wrong would corrupt every table in the report, and
unlike a crash it would never announce itself.

Run with::

    python -m pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4se.evaluation import (  # noqa: E402
    bootstrap_ci,
    classification_rows,
    cliffs_delta,
    compare_per_repo,
    cross_repo_f1,
    holm,
    is_conclusive,
    macro_f1,
    mcnemar_exact,
    micro_f1,
    minimum_detectable_difference,
    per_repo_f1,
    power_at,
    random_split_control,
    stratified_folds,
    time_aware_split,
)
from ai4se.model import IssueReport  # noqa: E402
from ai4se.repository import make_repository  # noqa: E402

LABELS3 = ["bug", "feature", "question"]


def _issue(repo: str, label: str, day: int, title: str = "t") -> IssueReport:
    """Build a minimal issue whose creation date is controlled by ``day``."""
    return IssueReport(
        repo=repo,
        created_at=f"2023-01-{day:02d} 00:00:00",
        label=label,
        title=title,
        body="body",
    )


@pytest.fixture
def toy_predictions():
    """Two projects, six issues, three of them misclassified."""
    y_true = ["bug", "bug", "feature", "bug", "feature", "question"]
    y_pred = ["bug", "feature", "feature", "bug", "feature", "bug"]
    repos = ["a/a", "a/a", "a/a", "b/b", "b/b", "b/b"]
    return y_true, y_pred, repos


# --------------------------------------------------------------- metrics --


def test_perfect_prediction_scores_one():
    """A perfect classifier must score exactly 1.0 on every metric."""
    y = ["bug", "feature", "question", "bug"]
    assert micro_f1(y, y) == pytest.approx(1.0)
    assert macro_f1(y, y) == pytest.approx(1.0)


def test_micro_f1_equals_accuracy_for_single_label():
    """With one label per item, micro-F1 and accuracy coincide by definition."""
    y_true = ["bug", "bug", "feature", "question"]
    y_pred = ["bug", "feature", "feature", "question"]
    assert micro_f1(y_true, y_pred) == pytest.approx(3 / 4)


def test_macro_f1_is_hand_checkable():
    """Verify macro-F1 against counts worked out by hand."""
    y_true = ["bug", "bug", "feature", "feature"]
    y_pred = ["bug", "feature", "feature", "feature"]
    # bug:      tp=1 fp=0 fn=1 -> F1 = 2/3
    # feature:  tp=2 fp=1 fn=0 -> F1 = 4/5
    # question: absent entirely -> F1 = 0.0 by convention
    assert macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 4 / 5 + 0.0) / 3)


def test_cross_repo_f1_averages_projects_not_items(toy_predictions):
    """The official score weights each project equally, not each issue."""
    y_true, y_pred, repos = toy_predictions
    per_repo = per_repo_f1(y_true, y_pred, repos)
    assert per_repo == {"a/a": pytest.approx(2 / 3), "b/b": pytest.approx(2 / 3)}
    assert cross_repo_f1(y_true, y_pred, repos) == pytest.approx(2 / 3)


def test_cross_repo_differs_from_pooled_when_projects_are_unequal():
    """Unequal project sizes make the mean-of-five differ from the pooled F1."""
    y_true = ["bug"] * 5 + ["bug"]
    y_pred = ["bug"] * 5 + ["feature"]
    repos = ["a/a"] * 5 + ["b/b"]
    assert cross_repo_f1(y_true, y_pred, repos) == pytest.approx(0.5)
    assert micro_f1(y_true, y_pred) == pytest.approx(5 / 6)


def test_classification_rows_shape_and_averages(toy_predictions):
    """One row per project-class cell, plus one averaged row per class."""
    y_true, y_pred, repos = toy_predictions
    rows = classification_rows(y_true, y_pred, repos)
    assert len(rows) == 2 * 3 + 3
    summary = [r for r in rows if r["repo"] == "cross-repo"]
    assert {r["label"] for r in summary} == set(LABELS3)
    for label in LABELS3:
        cells = [r for r in rows if r["label"] == label and r["repo"] != "cross-repo"]
        expected = sum(c["f1"] for c in cells) / len(cells)
        got = next(r["f1"] for r in summary if r["label"] == label)
        assert got == pytest.approx(expected)


def test_bootstrap_ci_brackets_the_point_estimate(toy_predictions):
    """The interval must contain the observed score and be reproducible."""
    y_true, y_pred, repos = toy_predictions
    point, low, high = bootstrap_ci(y_true, y_pred, repos, n_resamples=200, seed=7)
    assert low <= point <= high
    again = bootstrap_ci(y_true, y_pred, repos, n_resamples=200, seed=7)
    assert (point, low, high) == again


# ---------------------------------------------------------- significance --


def test_mcnemar_counts_discordant_pairs_only():
    """Items both models get right or both get wrong carry no information."""
    y_true = ["bug", "bug", "bug", "bug"]
    pred_a = ["bug", "bug", "feature", "feature"]
    pred_b = ["bug", "feature", "bug", "feature"]
    result = mcnemar_exact(pred_a, pred_b, y_true)
    assert (result.n01, result.n10) == (1, 1)
    assert result.n_discordant == 2
    assert result.p_value == pytest.approx(1.0)


def test_mcnemar_identical_models_are_never_significant():
    """Two identical models have zero discordant pairs and p = 1."""
    y_true = ["bug", "feature", "question"] * 4
    pred = ["bug"] * 12
    result = mcnemar_exact(pred, pred, y_true)
    assert result.n_discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_detects_a_one_sided_difference():
    """A model that wins every discordant pair must reach significance."""
    y_true = ["bug"] * 12
    pred_a = ["feature"] * 12
    pred_b = ["bug"] * 12
    result = mcnemar_exact(pred_a, pred_b, y_true)
    assert result.favours_second
    # Two-sided exact binomial with 12 trials, all on one side: 2 / 2**12.
    assert result.p_value == pytest.approx(2 / 4096)


def test_holm_is_monotone_and_never_lowers_a_p_value():
    """Adjustment may only make p-values larger, and must preserve order."""
    raw = [0.001, 0.04, 0.03, 0.2, 0.5]
    adjusted = holm(raw)
    assert all(a >= r for a, r in zip(adjusted, raw, strict=True))
    by_rank = [adjusted[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
    assert by_rank == sorted(by_rank)
    assert all(a <= 1.0 for a in adjusted)


def test_holm_matches_bonferroni_on_the_smallest_p_value():
    """The most significant test receives the full family-size penalty."""
    raw = [0.005, 0.5, 0.5, 0.5, 0.5]
    assert holm(raw)[0] == pytest.approx(0.025)


def test_compare_per_repo_returns_a_row_per_project_plus_pooled():
    """Five projects give five rows and one pooled summary row."""
    y_true = ["bug", "feature"] * 5
    pred_a = ["bug"] * 10
    pred_b = ["feature"] * 10
    repos = [f"p{i}/p{i}" for i in range(5) for _ in range(2)]
    rows = compare_per_repo(pred_a, pred_b, y_true, repos)
    assert len(rows) == 6
    assert rows[-1]["repo"] == "POOLED"
    assert all(r["p_holm"] >= r["p_value"] for r in rows[:-1])


def test_cliffs_delta_signs_and_magnitude():
    """Fully separated groups give delta = 1; identical groups give 0."""
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == (1.0, "large")
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == (-1.0, "large")
    delta, label = cliffs_delta([1, 2, 3], [1, 2, 3])
    assert delta == pytest.approx(0.0)
    assert label == "negligible"


# ----------------------------------------------------------------- power --


def test_power_increases_with_effect_size():
    """Bigger true differences must be easier to detect."""
    powers = [power_at(d, n_simulations=400, seed=1) for d in (0.005, 0.02, 0.05)]
    assert powers == sorted(powers)
    assert powers[0] < 0.5 < powers[-1]


def test_minimum_detectable_difference_is_around_three_points():
    """Reproduce the headline number of the benchmark audit."""
    mdd = minimum_detectable_difference(n_simulations=600, seed=3)
    assert 0.02 <= mdd <= 0.045


def test_smaller_test_sets_detect_less():
    """Per-project claims need a bigger gap than cross-repository ones."""
    full = minimum_detectable_difference(n_items=1500, n_simulations=400, seed=5)
    single = minimum_detectable_difference(n_items=300, n_simulations=400, seed=5)
    assert single > full


def test_is_conclusive_rejects_small_differences():
    """The guard used by experiment scripts must reject sub-threshold gaps."""
    assert not is_conclusive(0.01, n_simulations=400, seed=2)
    assert is_conclusive(0.08, n_simulations=400, seed=2)


# ---------------------------------------------------------------- splits --


@pytest.fixture
def dated_repository():
    """Two projects, three classes, eight issues per cell with known dates."""
    issues = [
        _issue(repo, label, day)
        for repo in ("a/a", "b/b")
        for label in LABELS3
        for day in range(1, 9)
    ]
    return make_repository("memory", issues=issues)


def test_time_aware_split_preserves_class_balance(dated_repository):
    """Splitting inside each cell is what keeps both sides balanced."""
    train, test = time_aware_split(dated_repository)
    assert len(train) == len(test) == 24
    assert train.label_distribution() == {"bug": 8, "feature": 8, "question": 8}
    assert test.label_distribution() == {"bug": 8, "feature": 8, "question": 8}


def test_time_aware_split_puts_older_issues_in_train(dated_repository):
    """Every training issue must predate every test issue of the same cell."""
    train, test = time_aware_split(dated_repository)
    for repo in ("a/a", "b/b"):
        for label in LABELS3:
            newest_train = max(
                i.created_at for i in train.by_repo(repo) if i.label == label
            )
            oldest_test = min(
                i.created_at for i in test.by_repo(repo) if i.label == label
            )
            assert newest_train < oldest_test


def test_random_control_matches_the_time_split_in_shape(dated_repository):
    """The control must differ from the time split only in which issues move."""
    time_train, time_test = time_aware_split(dated_repository)
    rand_train, rand_test = random_split_control(dated_repository, seed=0)
    assert len(rand_train) == len(time_train)
    assert len(rand_test) == len(time_test)
    assert rand_train.label_distribution() == time_train.label_distribution()


def test_time_aware_split_rejects_a_degenerate_fraction(dated_repository):
    """A fraction of 0 or 1 would produce an empty side; refuse it."""
    with pytest.raises(ValueError):
        time_aware_split(dated_repository, train_fraction=1.0)


def test_folds_partition_the_data_exactly_once(dated_repository):
    """Every issue must appear in exactly one validation fold."""
    folds = list(stratified_folds(dated_repository, n_folds=4, seed=0))
    assert len(folds) == 4
    seen: list[int] = []
    for train_index, validation_index in folds:
        assert set(train_index).isdisjoint(validation_index)
        assert len(train_index) + len(validation_index) == len(dated_repository)
        seen.extend(validation_index.tolist())
    assert sorted(seen) == list(range(len(dated_repository)))


def test_folds_are_stratified_by_project_and_class(dated_repository):
    """Each fold must hold the same number of issues from every cell."""
    issues = dated_repository.all()
    for _, validation_index in stratified_folds(dated_repository, n_folds=4, seed=0):
        cells = [(issues[i].repo, issues[i].label) for i in validation_index]
        assert len(set(cells)) == 6
        assert all(cells.count(cell) == 2 for cell in set(cells))
