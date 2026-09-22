"""Repository pattern for IssueReport collections (in-memory and disk-backed).
"""

from __future__ import annotations

import csv
import json
import sys
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path


from .model import IssueReport


# Raise CSV field size limit to handle huge issue bodies (>21,000 words)
_max_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_limit)
        break
    except OverflowError:
        _max_limit = int(_max_limit / 10)


class IssueRepository(ABC):
    """Abstract interface declaring repository primitives and generic helpers."""

    # --- Five Primitives ---
    @abstractmethod
    def all(self) -> Sequence[IssueReport]:
        """Return all issue reports in order."""

    @abstractmethod
    def add(self, issue: IssueReport) -> None:
        """Add a single issue report."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all issues from the repository."""

    @abstractmethod
    def load(self) -> IssueRepository:
        """Load records from the persistence store."""

    @abstractmethod
    def save(self) -> None:
        """Persist records to the persistence store."""

    # --- Query Helpers ---
    def __len__(self) -> int:
        return len(self.all())

    def __iter__(self) -> Iterator[IssueReport]:
        return iter(self.all())

    def to_dataframe(self):
        """Convert repository records to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([issue.to_dict() for issue in self.all()])

    def __getitem__(self, index: int) -> IssueReport:
        return self.all()[index]

    def repos(self) -> list[str]:
        """Return distinct repository names preserving first seen order."""
        seen = set()
        result = []
        for issue in self.all():
            if issue.repo not in seen:
                seen.add(issue.repo)
                result.append(issue.repo)
        return result

    def by_repo(self, repo: str) -> list[IssueReport]:
        """Return issues belonging to a specific repository."""
        return [issue for issue in self.all() if issue.repo == repo]

    def by_label(self, label: str) -> list[IssueReport]:
        """Return issues with a specific label."""
        return [issue for issue in self.all() if issue.label == label]

    def label_distribution(self, repo: str | None = None) -> dict[str, int]:
        """Return count per label, optionally filtered by repository."""
        issues = self.by_repo(repo) if repo is not None else self.all()
        counts = Counter(issue.label for issue in issues)
        return dict(counts)

    def texts_and_labels(self) -> tuple[list[str], list[str]]:
        """Return parallel lists of (text, label) ready for vectorisers."""
        issues = self.all()
        return [issue.text for issue in issues], [issue.label for issue in issues]

    def filter(self, predicate: Callable[[IssueReport], bool]) -> list[IssueReport]:
        """Return issues satisfying predicate."""
        return [issue for issue in self.all() if predicate(issue)]

    def labels(self) -> Sequence[str]:
        """Return unique labels present in the repository."""
        return list(dict.fromkeys(issue.label for issue in self.all()))

    def apply(self, transform: Callable[..., str]) -> None:
        """Apply an in-place text transformation to all issue bodies or (title, body)."""
        import inspect
        updated = []
        for issue in self.all():
            try:
                sig = inspect.signature(transform)
                if len(sig.parameters) >= 2:
                    new_text = transform(issue.title, issue.body)
                else:
                    new_text = transform(issue.body)
            except Exception:
                new_text = transform(issue.body)
            updated.append(issue.replace_text(new_text))
        self.clear()
        for issue in updated:
            self.add(issue)

    def to_dataframe(self):
        """Export collection to a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([issue.to_dict() for issue in self.all()])


class InMemoryIssueRepository(IssueRepository):
    """Holds all records in a Python list in RAM."""

    def __init__(self, issues: Sequence[IssueReport] | None = None) -> None:
        self._issues: list[IssueReport] = list(issues) if issues is not None else []

    def all(self) -> Sequence[IssueReport]:
        return list(self._issues)

    def add(self, issue: IssueReport) -> None:
        self._issues.append(issue)

    def clear(self) -> None:
        self._issues.clear()

    def load(self) -> InMemoryIssueRepository:
        return self

    def save(self) -> None:
        pass


class FileIssueRepository(IssueRepository):
    """File-backed repository supporting CSV and JSON formats."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._issues: list[IssueReport] = []

    def all(self) -> Sequence[IssueReport]:
        return list(self._issues)

    def add(self, issue: IssueReport) -> None:
        self._issues.append(issue)

    def clear(self) -> None:
        self._issues.clear()

    def load(self) -> FileIssueRepository:
        self.clear()
        if not self.path.exists():
            return self

        suffix = self.path.suffix.lower()
        if suffix == ".json":
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    self.add(IssueReport.from_dict(item))
        else:
           
            from .loader import _load_csv
            
            for iss in _load_csv(self.path):
                self.add(iss)
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.path.suffix.lower()
        if suffix == ".json":
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([issue.to_dict() for issue in self._issues], f, indent=2)
        else:
            fieldnames = ["repo", "created_at", "label", "title", "body"]
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for issue in self._issues:
                    d = issue.to_dict()
                    d.pop("issue_id", None)
                    writer.writerow(d)


def make_repository(
    kind: str = "memory",
    issues: Sequence[IssueReport] | None = None,
    path: str | Path | None = None,
    **kwargs,
) -> IssueRepository:
    """Factory that hides the concrete persistence class from the caller.

    Args:
        kind: 'memory' or 'file'.
        issues: Initial issues to populate.
        path: File path (required when kind='file').
    """
    if kind == "memory":
        return InMemoryIssueRepository(issues=issues)
    if kind == "file":
        if path is None:
            raise ValueError("path is required when kind='file'")
        p = Path(path)
        repo = FileIssueRepository(path=p)
        if issues is not None:
            repo.clear()
            for issue in issues:
                repo.add(issue)
        elif p.exists():
            repo.load()
        return repo
    raise ValueError(f"Unknown persistence kind: {kind!r}")
