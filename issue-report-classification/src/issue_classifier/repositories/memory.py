"""In-memory implementation of the issue repository contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from issue_classifier.domain import IssueReport
from issue_classifier.repositories.base import IssueRepository


class InMemoryIssueRepository(IssueRepository):
    def __init__(
        self,
        initial_data: Mapping[str, Sequence[IssueReport]] | None = None,
    ) -> None:
        self._data = {
            split: list(issues) for split, issues in (initial_data or {}).items()
        }

    @classmethod
    def from_repository(cls,
                        source: IssueRepository,
                        splits: Sequence[str] = ("train", "test"),
                        ) -> "InMemoryIssueRepository":
        return cls({split: source.load(split) for split in splits})

    def load(self, split: str) -> list[IssueReport]:
        return list(self._data.get(split, ()))

    def save(self, split: str, issues: Sequence[IssueReport]) -> None:
        self._data[split] = list(issues)
