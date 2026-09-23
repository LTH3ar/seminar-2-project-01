"""Application service for the consolidated data workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .audit import DatasetAudit, LeakageReport, audit_dataset, audit_train_test_leakage
from .folds import DataFold, make_folds_by_repository
from .loader import DEFAULT_RAW_DIR, load_dataset
from .model import IssueReport
from .preprocessing import clean_issue, structural_features
from .repository import IssueRepository
from .validation import ValidationReport, validate_issues


class IssueDataService:
    """Coordinate split repositories without exposing their persistence kind."""

    def __init__(self, repositories: Mapping[str, IssueRepository]) -> None:
        self.repositories = dict(repositories)

    @classmethod
    def from_loader(
        cls,
        *,
        kind: str = "memory",
        raw_dir: str | Path = DEFAULT_RAW_DIR,
    ) -> IssueDataService:
        """Load the official train and test repositories."""
        return cls(load_dataset(kind=kind, raw_dir=raw_dir))

    def load(
        self,
        split: str,
        *,
        repo: str | None = None,
        validate: bool = True,
    ) -> list[IssueReport]:
        """Load a split, optionally restricted to one repository."""
        try:
            repository = self.repositories[split]
        except KeyError as exc:
            known = ", ".join(sorted(self.repositories))
            raise ValueError(
                f"Unknown split {split!r}; expected one of: {known}"
            ) from exc
        issues = repository.by_repo(repo) if repo else repository.all()
        if validate:
            validate_issues(issues, split=split).raise_for_errors()
        return issues

    def validate(self, split: str) -> ValidationReport:
        """Return validation results without raising."""
        return validate_issues(self.load(split, validate=False), split=split)

    def audit(self, split: str) -> DatasetAudit:
        """Return descriptive statistics for a split."""
        return audit_dataset(self.load(split, validate=False), split=split)

    def audit_leakage(self) -> LeakageReport:
        """Compare official train and test data for normalized duplicates."""
        return audit_train_test_leakage(
            self.load("train", validate=False),
            self.load("test", validate=False),
        )

    def prepare(
        self,
        split: str,
        *,
        repo: str | None = None,
        level: str = "conservative",
        max_words: int | None = None,
        title_weight: int = 1,
    ) -> list[IssueReport]:
        """Return cleaned copies of selected issues."""
        return [
            clean_issue(
                issue,
                level=level,
                max_words=max_words,
                title_weight=title_weight,
            )
            for issue in self.load(split, repo=repo)
        ]

    def prepared_dataframe(
        self,
        split: str,
        *,
        repo: str | None = None,
        level: str = "conservative",
        max_words: int | None = None,
    ):
        """Return cleaned records and structural features as a DataFrame."""
        import pandas as pd

        issues = self.prepare(
            split,
            repo=repo,
            level=level,
            max_words=max_words,
        )
        return pd.DataFrame(
            {**issue.to_dict(), **structural_features(issue)} for issue in issues
        )

    def make_folds(
        self,
        *,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, tuple[DataFold, ...]]:
        """Build folds only from official training records."""
        return make_folds_by_repository(
            self.load("train"),
            n_splits=n_splits,
            random_state=random_state,
        )
