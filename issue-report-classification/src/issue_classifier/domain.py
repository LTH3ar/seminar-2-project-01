"""Domain objects shared by every persistence and model implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Mapping


ALLOWED_LABELS = frozenset({"bug", "feature", "question"})
EXPECTED_REPOSITORIES = frozenset(
    {
        "bitcoin/bitcoin",
        "facebook/react",
        "microsoft/vscode",
        "opencv/opencv",
        "tensorflow/tensorflow",
    }
)


def make_issue_id(
    repo: str,
    created_at: datetime,
    label: str,
    title: str,
    body: str,
) -> str:
    """Create a stable identifier from all fields available in the source CSV."""

    value = "\x1f".join((repo, created_at.isoformat(), label, title, body))
    return sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IssueReport:
    issue_id: str
    repo: str
    created_at: datetime
    label: str
    title: str
    body: str

    @classmethod
    def create(cls,
               *,
               repo: str,
               created_at: datetime,
               label: str,
               title: str = "",
               body: str = "",
               ) -> "IssueReport":
        title = title or ""
        body = body or ""
        
        return cls(
            issue_id=make_issue_id(repo, created_at, label, title, body),
            repo=repo,
            created_at=created_at,
            label=label,
            title=title,
            body=body,
        )


@dataclass(frozen=True, slots=True)
class PreparedIssue:
    """An issue transformed into model-ready text and structural features."""

    issue: IssueReport
    cleaned_title: str
    cleaned_body: str
    text: str
    features: Mapping[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.issue.label

    @property
    def repo(self) -> str:
        return self.issue.repo

    def to_dict(self) -> dict[str, object]:
        return {
            "issue_id": self.issue.issue_id,
            "repo": self.issue.repo,
            "created_at": self.issue.created_at,
            "label": self.issue.label,
            "title": self.issue.title,
            "body": self.issue.body,
            "cleaned_title": self.cleaned_title,
            "cleaned_body": self.cleaned_body,
            "text": self.text,
            **self.features,
        }
