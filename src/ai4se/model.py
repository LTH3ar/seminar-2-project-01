"""Domain model for the NLBSE'24 issue report classification task.

This module defines the single entity the whole application works with.
It is intentionally free of any I/O or persistence concern: the same
``IssueReport`` object is produced by the in-memory repository and by the
file-backed repository, so the business logic never has to know where the
data came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, ClassVar

#: The three issue types used by the NLBSE'24 competition.
#: Issues carrying more than one label were removed by the organisers,
#: so this is a single-label (multi-class) problem, not a multi-label one.
LABELS: tuple[str, ...] = ("bug", "feature", "question")

#: The five open-source projects the 3,000 issues were extracted from.
REPOSITORIES: tuple[str, ...] = (
    "facebook/react",
    "tensorflow/tensorflow",
    "microsoft/vscode",
    "bitcoin/bitcoin",
    "opencv/opencv",
)


@dataclass(slots=True)
class IssueReport:
    """A single labelled GitHub issue report.

    Attributes:
        repo: Full name of the source project, e.g. ``"facebook/react"``.
        created_at: Creation timestamp as provided in the raw CSV.
        label: Ground-truth class, one of :data:`LABELS`.
        title: Issue title.
        body: Issue body in its original Markdown form.
        clean_text: Result of the preprocessing pipeline. Empty until
            :mod:`ai4se.preprocessing` has been applied.
    """

    repo: str
    created_at: str
    label: str
    title: str
    body: str
    clean_text: str = ""

    #: Column order used when the entity is written to / read from CSV.
    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "repo",
        "created_at",
        "label",
        "title",
        "body",
        "clean_text",
    )

    def __post_init__(self) -> None:
        # Defensive normalisation: the raw CSV occasionally yields NaN for a
        # missing body, which would break every downstream string operation.
        self.title = "" if self.title is None else str(self.title)
        self.body = "" if self.body is None else str(self.body)
        self.repo = str(self.repo).strip()
        self.label = str(self.label).strip().lower()

    @property
    def raw_text(self) -> str:
        """Title and body concatenated, the default classifier input."""
        return f"{self.title}\n{self.body}".strip()

    @property
    def text(self) -> str:
        """Preprocessed text if available, otherwise the raw text."""
        return self.clean_text or self.raw_text

    @property
    def word_count(self) -> int:
        """Number of whitespace-separated tokens in :attr:`raw_text`."""
        return len(self.raw_text.split())

    def is_valid(self) -> bool:
        """True when the record can be used for training or evaluation."""
        return bool(self.title or self.body) and self.label in LABELS

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entity to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "IssueReport":
        """Build an entity from a dictionary, ignoring unknown keys."""
        return cls(
            repo=record.get("repo", ""),
            created_at=str(record.get("created_at", "")),
            label=record.get("label", ""),
            title=record.get("title", ""),
            body=record.get("body", ""),
            clean_text=record.get("clean_text", "") or "",
        )
