"""Repository contract used by the data and training services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

from issue_classifier.domain import IssueReport


class IssueRepository(ABC):
    @abstractmethod
    def load(self, split: str) -> list[IssueReport]:
        """Return a detached list of records for a named split."""

    @abstractmethod
    def save(self, split: str, issues: Sequence[IssueReport]) -> None:
        """Replace a named split with the supplied records."""

    def find_by_repo(self, split: str, repo: str) -> list[IssueReport]:
        return [issue for issue in self.load(split) if issue.repo == repo]

    def find_by_label(self, split: str, label: str) -> list[IssueReport]:
        return [issue for issue in self.load(split) if issue.label == label]

    def repositories(self, split: str) -> list[str]:
        return sorted({issue.repo for issue in self.load(split)})

    def labels(self, split: str) -> list[str]:
        return sorted({issue.label for issue in self.load(split)})

    def label_distribution(
        self,
        split: str,
        *,
        repo: str | None = None,
    ) -> dict[str, int]:
        issues = self.find_by_repo(split, repo) if repo else self.load(split)
        return dict(sorted(Counter(issue.label for issue in issues).items()))
