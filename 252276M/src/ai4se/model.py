"""Domain model and constants for the NLBSE'24 issue report dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

#: Target classes in the NLBSE'24 competition.
LABELS: tuple[str, ...] = ("bug", "feature", "question")

#: The five open-source projects in the benchmark.
REPOSITORIES: tuple[str, ...] = (
    "facebook/react",
    "tensorflow/tensorflow",
    "microsoft/vscode",
    "opencv/opencv",
    "bitcoin/bitcoin",
)

#: Official SetFit baseline F1 scores by repository (published in competition).
OFFICIAL_SETFIT_SCORES: dict[str, float] = {
    "facebook/react": 0.8718,
    "tensorflow/tensorflow": 0.8644,
    "microsoft/vscode": 0.8262,
    "opencv/opencv": 0.8173,
    "bitcoin/bitcoin": 0.7555,
    "cross-repository": 0.8270,
}


@dataclass(frozen=True)
class IssueReport:
    """A single issue report from the competition dataset."""

    repo: str
    issue_id: int
    created_at: datetime
    title: str
    body: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Invalid label {self.label!r}; expected one of {LABELS}.")

    @property
    def text(self) -> str:
        """Combined title and body text."""
        return f"{self.title}\n\n{self.body}".strip()

    def replace_text(self, new_text: str) -> IssueReport:
        """Return a copy with transformed text in the body."""
        return IssueReport(
            repo=self.repo,
            issue_id=self.issue_id,
            created_at=self.created_at,
            title=self.title,
            body=new_text,
            label=self.label,
        )

    def to_dict(self) -> dict:
        """Serialize entity to dictionary."""
        return {
            "repo": self.repo,
            "issue_id": self.issue_id,
            "created_at": self.created_at.isoformat(),
            "title": self.title,
            "body": self.body,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IssueReport:
        """Construct entity from dictionary."""
        created = data["created_at"]
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        return cls(
            repo=data["repo"],
            issue_id=int(data["issue_id"]),
            created_at=created,
            title=data.get("title", "") or "",
            body=data.get("body", "") or "",
            label=data["label"],
        )
