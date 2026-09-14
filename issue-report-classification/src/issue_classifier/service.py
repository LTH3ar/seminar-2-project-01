"""Application service coordinating repositories and data transformations."""

from __future__ import annotations

from pathlib import Path

from issue_classifier.audit import (
    DatasetAudit,
    LeakageReport,
    audit_dataset,
    audit_train_test_leakage,
)
from issue_classifier.dataframe import prepared_issues_to_dataframe
from issue_classifier.domain import IssueReport, PreparedIssue
from issue_classifier.folds import DataFold, make_folds_by_repository
from issue_classifier.preprocessing import TextPreprocessor
from issue_classifier.repositories import (
    CsvIssueRepository,
    InMemoryIssueRepository,
    IssueRepository,
    JsonIssueRepository,
)
from issue_classifier.validation import ValidationReport, validate_issues


def create_issue_repository(
    backend: str,
    data_directory: str | Path,
) -> IssueRepository:
    """Create a CSV, JSON, or in-memory repository."""

    backend = backend.lower()
    file_repository = CsvIssueRepository(data_directory)
    if backend in {"csv", "file"}:
        return file_repository
    if backend == "json":
        return JsonIssueRepository(data_directory)
    if backend == "memory":
        return InMemoryIssueRepository.from_repository(file_repository)
    raise ValueError("backend must be 'csv', 'file', 'json', or 'memory'")


class IssueDataService:
    def __init__(
        self,
        repository: IssueRepository,
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        self.repository = repository
        self.preprocessor = preprocessor or TextPreprocessor()

    def load(
        self,
        split: str,
        *,
        repo: str | None = None,
        validate: bool = True,
    ) -> list[IssueReport]:
        issues = (
            self.repository.find_by_repo(split, repo)
            if repo is not None
            else self.repository.load(split)
        )
        if validate:
            validate_issues(issues, split=split).raise_for_errors()
        return issues

    def validate(self, split: str) -> ValidationReport:
        return validate_issues(self.repository.load(split), split=split)

    def audit(self, split: str) -> DatasetAudit:
        return audit_dataset(self.repository.load(split), split=split)

    def audit_leakage(self) -> LeakageReport:
        return audit_train_test_leakage(
            self.repository.load("train"),
            self.repository.load("test"),
        )

    def prepare(
        self,
        split: str,
        *,
        repo: str | None = None,
    ) -> list[PreparedIssue]:
        return self.preprocessor.prepare_many(self.load(split, repo=repo))

    def prepare_dataframe(self, split: str, *, repo: str | None = None):
        return prepared_issues_to_dataframe(self.prepare(split, repo=repo))

    def make_folds(
        self,
        *,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, tuple[DataFold, ...]]:
        """Build folds exclusively from the official training split."""

        return make_folds_by_repository(
            self.load("train"),
            n_splits=n_splits,
            random_state=random_state,
        )

    def copy_split(self, split: str, target: IssueRepository) -> None:
        target.save(split, self.repository.load(split))
