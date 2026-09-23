"""Persistence layer for issue reports.

This module satisfies the non-functional requirement of the course project:

    "The data must be managed both in memory and through files present in the
     file system. The application must be designed in such a way that the use
     of one of the two persistence layers involves few changes to the
     application itself."

The design is a classic Repository pattern:

    IssueRepository            (abstract interface -- the only thing the
      |                         business logic is allowed to depend on)
      |-- InMemoryIssueRepository   (a Python list, nothing touches the disk)
      `-- FileIssueRepository       (CSV or JSON on the file system)

Switching persistence layer is a single argument change::

    repo = make_repository("memory", issues=issues)
    repo = make_repository("file", path="data/processed/train.csv")

Everything downstream -- EDA, preprocessing, vectorisation, training,
evaluation -- calls only the methods declared on ``IssueRepository`` and is
therefore completely unaffected by that choice.
"""

from __future__ import annotations

import csv
import json
import sys
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from .model import IssueReport

# CSV fields of the NLBSE'24 dataset can be very long (one body exceeds
# 21,000 words), so the default csv field size limit must be raised.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class IssueRepository(ABC):
    """Abstract collection of :class:`~ai4se.model.IssueReport` objects.

    Concrete subclasses decide *where* the issues live. Callers only ever see
    this interface, which is what keeps the business logic independent of the
    persistence mechanism.
    """

    # ---------------------------------------------------------------- core --

    @abstractmethod
    def load(self) -> "IssueRepository":
        """Populate the repository from its backing store. Returns self."""

    @abstractmethod
    def save(self) -> None:
        """Persist the current content to the backing store."""

    @abstractmethod
    def all(self) -> list[IssueReport]:
        """Return every issue currently held by the repository."""

    @abstractmethod
    def add(self, issue: IssueReport) -> None:
        """Append a single issue."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every issue from the repository."""

    # ------------------------------------------------- generic query helpers --
    # These are implemented once, in terms of the abstract methods above, so
    # every concrete repository inherits them for free.

    def add_all(self, issues: Iterable[IssueReport]) -> None:
        """Append many issues."""
        for issue in issues:
            self.add(issue)

    def filter(self, predicate: Callable[[IssueReport], bool]) -> list[IssueReport]:
        """Return the issues satisfying ``predicate``."""
        return [issue for issue in self.all() if predicate(issue)]

    def by_repo(self, repo: str) -> list[IssueReport]:
        """Return the issues belonging to one source project.

        Needed because the competition requires one classifier per project:
        five models are trained and evaluated separately, then averaged.
        """
        return self.filter(lambda i: i.repo == repo)

    def by_label(self, label: str) -> list[IssueReport]:
        """Return the issues of one class."""
        return self.filter(lambda i: i.label == label)

    def repos(self) -> list[str]:
        """Sorted list of distinct source projects."""
        return sorted({issue.repo for issue in self.all()})

    def labels(self) -> list[str]:
        """Sorted list of distinct class labels."""
        return sorted({issue.label for issue in self.all()})

    def label_distribution(self, repo: str | None = None) -> dict[str, int]:
        """Count issues per class, optionally restricted to one project."""
        issues = self.by_repo(repo) if repo else self.all()
        return dict(Counter(issue.label for issue in issues))

    def texts_and_labels(self) -> tuple[list[str], list[str]]:
        """Return ``(X, y)`` ready to be handed to a scikit-learn estimator."""
        issues = self.all()
        return [issue.text for issue in issues], [issue.label for issue in issues]

    def apply(self, transform: Callable[[IssueReport], IssueReport]) -> None:
        """Apply ``transform`` in place to every issue.

        Used by the preprocessing pipeline to fill ``clean_text``.
        """
        transformed = [transform(issue) for issue in self.all()]
        self.clear()
        self.add_all(transformed)

    def to_dataframe(self):
        """Return the content as a pandas DataFrame (for EDA and plotting).

        pandas is imported lazily so that the persistence layer itself has no
        hard dependency on it.
        """
        import pandas as pd

        return pd.DataFrame([issue.to_dict() for issue in self.all()])

    # ------------------------------------------------------ dunder niceties --

    def __len__(self) -> int:
        return len(self.all())

    def __iter__(self) -> Iterator[IssueReport]:
        return iter(self.all())

    def __getitem__(self, index: int) -> IssueReport:
        return self.all()[index]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self)})"


class InMemoryIssueRepository(IssueRepository):
    """Issue repository backed by a plain Python list.

    ``load`` and ``save`` are no-ops: the data never leaves RAM. The class
    exists so that the rest of the application can be exercised (and unit
    tested) without touching the file system.
    """

    def __init__(self, issues: Sequence[IssueReport] | None = None) -> None:
        self._issues: list[IssueReport] = list(issues or [])

    def load(self) -> "InMemoryIssueRepository":
        """No backing store to read from; present for interface symmetry."""
        return self

    def save(self) -> None:
        """No backing store to write to; present for interface symmetry."""
        return None

    def all(self) -> list[IssueReport]:
        return list(self._issues)

    def add(self, issue: IssueReport) -> None:
        self._issues.append(issue)

    def clear(self) -> None:
        self._issues.clear()


class FileIssueRepository(IssueRepository):
    """Issue repository backed by a CSV or JSON file on the file system.

    The format is inferred from the file suffix (``.csv`` or ``.json``) unless
    given explicitly. Content is cached after the first :meth:`load` so that
    repeated queries do not re-read the file; call :meth:`load` again to
    refresh from disk.

    Args:
        path: Location of the backing file.
        fmt: ``"csv"``, ``"json"`` or ``None`` to infer from the suffix.
        autoload: Read the file immediately on construction.
    """

    def __init__(
        self,
        path: str | Path,
        fmt: str | None = None,
        autoload: bool = True,
    ) -> None:
        self.path = Path(path)
        self.fmt = (fmt or self.path.suffix.lstrip(".")).lower()
        if self.fmt not in {"csv", "json"}:
            raise ValueError(f"Unsupported persistence format: {self.fmt!r}")
        self._issues: list[IssueReport] = []
        self._loaded = False
        if autoload and self.path.exists():
            self.load()

    # ------------------------------------------------------------- reading --

    def load(self) -> "FileIssueRepository":
        """Read the whole backing file into memory."""
        if not self.path.exists():
            raise FileNotFoundError(f"No such dataset file: {self.path}")
        reader = self._read_csv if self.fmt == "csv" else self._read_json
        self._issues = reader()
        self._loaded = True
        return self

    def _read_csv(self) -> list[IssueReport]:
        with self.path.open(newline="", encoding="utf-8") as handle:
            return [IssueReport.from_dict(row) for row in csv.DictReader(handle)]

    def _read_json(self) -> list[IssueReport]:
        with self.path.open(encoding="utf-8") as handle:
            return [IssueReport.from_dict(row) for row in json.load(handle)]

    # ------------------------------------------------------------- writing --

    def save(self) -> None:
        """Write the current content back to the backing file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = self._write_csv if self.fmt == "csv" else self._write_json
        writer()

    def _write_csv(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            out = csv.DictWriter(handle, fieldnames=list(IssueReport.CSV_FIELDS))
            out.writeheader()
            for issue in self._issues:
                out.writerow(issue.to_dict())

    def _write_json(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(
                [issue.to_dict() for issue in self._issues],
                handle,
                ensure_ascii=False,
                indent=2,
            )

    # -------------------------------------------------------------- access --

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def all(self) -> list[IssueReport]:
        self._ensure_loaded()
        return list(self._issues)

    def add(self, issue: IssueReport) -> None:
        self._ensure_loaded()
        self._issues.append(issue)

    def clear(self) -> None:
        self._issues = []
        self._loaded = True

    def __repr__(self) -> str:
        return f"FileIssueRepository(path={self.path.name!r}, n={len(self)})"


def make_repository(
    kind: str = "memory",
    issues: Sequence[IssueReport] | None = None,
    path: str | Path | None = None,
    **kwargs,
) -> IssueRepository:
    """Factory that hides the concrete persistence class from the caller.

    This is the single switch point required by the project specification:
    changing ``kind`` from ``"memory"`` to ``"file"`` changes the persistence
    layer of the whole application without touching any business logic.

    Args:
        kind: ``"memory"`` or ``"file"``.
        issues: Initial content, used by the in-memory implementation.
        path: Backing file, required when ``kind == "file"``.

    Returns:
        A concrete :class:`IssueRepository`.
    """
    kind = kind.lower()
    if kind == "memory":
        return InMemoryIssueRepository(issues)
    if kind == "file":
        if path is None:
            raise ValueError("A file-backed repository requires a `path`.")
        repository = FileIssueRepository(path, autoload=False, **kwargs)
        if issues is not None:
            repository.clear()
            repository.add_all(issues)
        elif repository.path.exists():
            repository.load()
        return repository
    raise ValueError(f"Unknown repository kind: {kind!r}. Use 'memory' or 'file'.")
