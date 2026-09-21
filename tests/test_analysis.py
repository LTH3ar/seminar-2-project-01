"""Tests for uncertainty, error analysis, and robustness splits."""

from __future__ import annotations

from copy import deepcopy

import pytest

from ai4se.error_analysis import analyse_evaluation
from ai4se.evaluation import evaluate_holdout_by_repository
from ai4se.model import LABELS, IssueReport
from ai4se.splits import (
    deduplicate_for_evaluation,
    random_split_control,
    time_aware_split,
)
from ai4se.statistical_analysis import (
    bootstrap_confidence_interval,
    compare_evaluations,
    holm_adjust,
    mcnemar_exact,
    summarise_repeated_results,
    weighted_f1,
)


class ProbabilisticKeywordClassifier:
    """Small classifier exposing labels and aligned probabilities."""

    classes_ = list(LABELS)

    def fit(self, texts, labels):
        """Satisfy the shared classifier interface."""

        return self

    def predict(self, texts):
        """Read the embedded class keyword."""

        return [next(label for label in LABELS if label in text) for text in texts]

    def predict_proba(self, texts):
        """Return a high score for the predicted keyword class."""

        rows = []
        for prediction in self.predict(texts):
            rows.append(
                [0.9 if label == prediction else 0.05 for label in self.classes_]
            )
        return rows


def _issues(suffix: str, per_label: int = 3) -> list[IssueReport]:
    return [
        IssueReport(
            repo="owner/project",
            created_at=f"2024-01-{index + 1:02d}",
            label=label,
            title=f"{label} {suffix} {index}",
            body=f"unique {label} {suffix} body {index}",
        )
        for label in LABELS
        for index in range(per_label)
    ]


def _factory(repository: str, fold: int) -> ProbabilisticKeywordClassifier:
    del repository, fold
    return ProbabilisticKeywordClassifier()


def test_weighted_f1_and_mcnemar_have_hand_checkable_results():
    """Core statistics match simple expected values."""

    references = ["bug", "bug", "feature", "question"]
    first = ["bug", "feature", "feature", "question"]
    second = ["bug", "bug", "question", "question"]

    assert weighted_f1(references, references) == 1
    outcome = mcnemar_exact(first, second, references)
    assert outcome.first_wrong_second_right == 1
    assert outcome.first_right_second_wrong == 1
    assert outcome.p_value == 1
    assert holm_adjust([0.01, 0.04, 0.2]) == [0.03, 0.08, 0.2]


def test_evaluation_records_probabilities_and_supports_analysis():
    """Saved predictions contain confidence for later error inspection."""

    train = _issues("train")
    test = _issues("test", per_label=1)
    result = evaluate_holdout_by_repository(
        train,
        test,
        _factory,
        model_name="probabilistic-keyword",
    )

    prediction = result["repositories"]["owner/project"]["predictions"][0]
    assert prediction["confidence"] == 0.9
    assert set(prediction["scores"]) == set(LABELS)
    analysis = analyse_evaluation(result, test)
    assert analysis["total_predictions"] == 3
    assert analysis["total_errors"] == 0


def test_bootstrap_repeated_summary_and_paired_comparison_are_reproducible():
    """Result-level analysis operates on the shared JSON schema."""

    result = evaluate_holdout_by_repository(
        _issues("train"),
        _issues("test", per_label=1),
        _factory,
        model_name="perfect",
        random_state=10,
    )
    second = deepcopy(result)
    second["random_state"] = 11

    summary = summarise_repeated_results([result, second])
    interval = bootstrap_confidence_interval(
        result,
        n_resamples=100,
        random_state=7,
    )

    assert summary["scores"] == [1, 1]
    assert summary["standard_deviation"] == 0
    assert interval["lower"] == interval["upper"] == 1
    assert not compare_evaluations(result, second)[0]["significant"]


def test_robustness_splits_remove_duplicate_and_conflicting_text():
    """Alternative splits retain no normalized duplicate leakage."""

    issues = _issues("history", per_label=4)
    duplicate = IssueReport(
        repo="owner/project",
        created_at="2024-02-01",
        label="bug",
        title=issues[0].title,
        body=issues[0].body,
    )
    conflict = IssueReport(
        repo="owner/project",
        created_at="2024-02-02",
        label="feature",
        title=issues[0].title,
        body=issues[0].body,
    )

    unique, summary = deduplicate_for_evaluation([*issues, duplicate, conflict])
    temporal_train, temporal_test, _ = time_aware_split(issues)
    random_train, random_test, _ = random_split_control(issues, random_state=3)

    assert len(unique) == len(issues) - 1
    assert summary.conflicting_groups_removed == 1
    assert summary.conflicting_records_removed == 3
    assert not {issue.issue_id for issue in temporal_train} & {
        issue.issue_id for issue in temporal_test
    }
    assert not {issue.issue_id for issue in random_train} & {
        issue.issue_id for issue in random_test
    }
    latest_train_bug = max(
        issue.created_at for issue in temporal_train if issue.label == "bug"
    )
    earliest_test_bug = min(
        issue.created_at for issue in temporal_test if issue.label == "bug"
    )
    assert latest_train_bug < earliest_test_bug


def test_robustness_split_rejects_a_cell_that_cannot_be_divided():
    """A one-record cell must not silently produce an empty evaluation set."""

    with pytest.raises(ValueError, match="at least two unique issues"):
        time_aware_split(_issues("too-small", per_label=1))


def test_invalid_bootstrap_configuration_is_rejected():
    """Bootstrap parameters fail early with useful errors."""

    result = evaluate_holdout_by_repository(
        _issues("train"),
        _issues("test", per_label=1),
        _factory,
        model_name="perfect",
    )
    with pytest.raises(ValueError, match="positive"):
        bootstrap_confidence_interval(result, n_resamples=0)
