"""Leakage-aware stratified cross-validation split generation."""

from __future__ import annotations

from dataclasses import dataclass

from issue_classifier.audit import duplicate_group
from issue_classifier.domain import IssueReport


@dataclass(frozen=True, slots=True)
class DataFold:
    number: int
    train: tuple[IssueReport, ...]
    validation: tuple[IssueReport, ...]


def make_stratified_group_folds(issues: list[IssueReport],
                                *,
                                n_splits: int = 5,
                                random_state: int = 42,
                                ) -> tuple[DataFold, ...]:
    """Keep normalized duplicate texts in one fold while stratifying labels."""

    if len(issues) < n_splits:
        raise ValueError("The number of records must be at least n_splits")

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required to generate cross-validation folds"
        ) from exc

    labels = [issue.label for issue in issues]
    groups = [duplicate_group(issue) for issue in issues]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = []
    for number, (train_indices, validation_indices) in enumerate(
        splitter.split(issues, labels, groups),
        start=1,
    ):
        folds.append(
            DataFold(
                number=number,
                train=tuple(issues[index] for index in train_indices),
                validation=tuple(issues[index] for index in validation_indices),
            )
        )
    return tuple(folds)


def make_folds_by_repository(issues: list[IssueReport],
                             *,
                             n_splits: int = 5,
                             random_state: int = 42,
                             ) -> dict[str, tuple[DataFold, ...]]:
    repositories = sorted({issue.repo for issue in issues})
    return {
        repo: make_stratified_group_folds(
            [issue for issue in issues if issue.repo == repo],
            n_splits=n_splits,
            random_state=random_state,
        )
        for repo in repositories
    }
