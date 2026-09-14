"""CSV-backed issue repository with strict schema and timestamp parsing."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from issue_classifier.domain import IssueReport
from issue_classifier.repositories.base import IssueRepository


CSV_COLUMNS = ("repo", "created_at", "label", "title", "body")
MAX_CSV_FIELD_SIZE = 16 * 1024 * 1024
DEFAULT_FILENAMES = {
    "train": "issues_train.csv",
    "test": "issues_test.csv",
}


class DatasetFormatError(ValueError):
    """Raised when a CSV file cannot be represented as issue reports."""


class CsvIssueRepository(IssueRepository):
    def __init__(self, 
                 data_directory: str | Path, 
                 filenames: Mapping[str, str] | None = None) -> None:
        self.data_directory = Path(data_directory)
        self.filenames = dict(DEFAULT_FILENAMES)
        if filenames:
            self.filenames.update(filenames)

    def _path_for(self, split: str) -> Path:
        try:
            filename = self.filenames[split]
        except KeyError as exc:
            known = ", ".join(sorted(self.filenames))
            raise ValueError(f"Unknown split {split!r}; expected one of: {known}") from exc
        return self.data_directory / filename

    def load(self, split: str) -> list[IssueReport]:
        path = self._path_for(split)
        if not path.exists():
            raise FileNotFoundError(f"Dataset split not found: {path}")

        # Some GitHub issue bodies exceed Python's default 128 KiB CSV limit.
        csv.field_size_limit(max(csv.field_size_limit(), MAX_CSV_FIELD_SIZE))
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            missing = set(CSV_COLUMNS).difference(reader.fieldnames or ())
            if missing:
                columns = ", ".join(sorted(missing))
                raise DatasetFormatError(f"{path} is missing columns: {columns}")

            issues = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    created_at = datetime.fromisoformat((row["created_at"] or "").strip())
                except ValueError as exc:
                    raise DatasetFormatError(
                        f"{path}:{row_number} has an invalid created_at value"
                    ) from exc

                issues.append(
                    IssueReport.create(
                        repo=(row["repo"] or "").strip(),
                        created_at=created_at,
                        label=(row["label"] or "").strip(),
                        title=row["title"] or "",
                        body=row["body"] or "",
                    )
                )
        return issues

    def save(self, split: str, issues: Sequence[IssueReport]) -> None:
        path = self._path_for(split)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8",
                                             newline="",
                                             dir=path.parent,
                                             prefix=f".{path.name}.",
                                             suffix=".tmp",
                                             delete=False) as destination:
                temporary_path = Path(destination.name)
                writer = csv.DictWriter(destination, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for issue in issues:
                    writer.writerow(
                        {
                            "repo": issue.repo,
                            "created_at": issue.created_at.isoformat(sep=" "),
                            "label": issue.label,
                            "title": issue.title,
                            "body": issue.body,
                        }
                    )
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
