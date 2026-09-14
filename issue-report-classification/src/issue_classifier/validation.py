"""Dataset validation rules that are independent of persistence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from issue_classifier.domain import (
    ALLOWED_LABELS,
    EXPECTED_REPOSITORIES,
    IssueReport,
)


@dataclass(frozen=True, slots=True)
class ValidationMessage:
    severity: str
    code: str
    message: str
    issue_id: str | None = None


@dataclass(slots=True)
class ValidationReport:
    split: str
    record_count: int
    messages: list[ValidationMessage] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationMessage]:
        return [message for message in self.messages if message.severity == "error"]

    @property
    def warnings(self) -> list[ValidationMessage]:
        return [message for message in self.messages if message.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise DatasetValidationError(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "record_count": self.record_count,
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "messages": [asdict(message) for message in self.messages],
        }


class DatasetValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        details = "; ".join(message.message for message in report.errors[:5])
        super().__init__(f"Dataset split {report.split!r} is invalid: {details}")


def validate_issues(issues: list[IssueReport],
                    *,
                    split: str,
                    allowed_labels: frozenset[str] = ALLOWED_LABELS,
                    expected_repositories: frozenset[str] | None = EXPECTED_REPOSITORIES,
                    ) -> ValidationReport:
    report = ValidationReport(split=split, record_count=len(issues))
    seen_ids: Counter[str] = Counter(issue.issue_id for issue in issues)

    if not issues:
        report.messages.append(
            ValidationMessage("error", "empty_dataset", "The dataset is empty")
        )

    for issue in issues:
        if not issue.repo:
            report.messages.append(
                ValidationMessage(
                    "error", "missing_repo", "Repository is empty", issue.issue_id
                )
            )
        elif expected_repositories is not None and issue.repo not in expected_repositories:
            report.messages.append(
                ValidationMessage(
                    "error",
                    "unknown_repo",
                    f"Unknown repository: {issue.repo}",
                    issue.issue_id,
                )
            )

        if issue.label not in allowed_labels:
            report.messages.append(
                ValidationMessage(
                    "error",
                    "unknown_label",
                    f"Unknown label: {issue.label}",
                    issue.issue_id,
                )
            )

        if not issue.title.strip() and not issue.body.strip():
            report.messages.append(
                ValidationMessage(
                    "error",
                    "empty_text",
                    "Both title and body are empty",
                    issue.issue_id,
                )
            )
        elif not issue.title.strip():
            report.messages.append(
                ValidationMessage(
                    "warning", "empty_title", "Title is empty", issue.issue_id
                )
            )
        elif not issue.body.strip():
            report.messages.append(
                ValidationMessage(
                    "warning", "empty_body", "Body is empty", issue.issue_id
                )
            )

    for issue_id, count in seen_ids.items():
        if count > 1:
            report.messages.append(
                ValidationMessage(
                    "warning",
                    "duplicate_record",
                    f"Record occurs {count} times",
                    issue_id,
                )
            )
            
    return report
