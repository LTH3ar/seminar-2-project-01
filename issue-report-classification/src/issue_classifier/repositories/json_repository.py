"""JSON-backed issue repository with the same contract as the CSV backend."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from issue_classifier.domain import IssueReport
from issue_classifier.repositories.base import IssueRepository
from issue_classifier.repositories.csv_repository import DatasetFormatError


JSON_COLUMNS = ("repo", "created_at", "label", "title", "body")
DEFAULT_FILENAMES = {
    "train": "issues_train.json",
    "test": "issues_test.json",
}


class JsonIssueRepository(IssueRepository):
    def __init__(
        self,
        data_directory: str | Path,
        filenames: Mapping[str, str] | None = None,
    ) -> None:
        self.data_directory = Path(data_directory)
        self.filenames = dict(DEFAULT_FILENAMES)
        if filenames:
            self.filenames.update(filenames)

    def _path_for(self, split: str) -> Path:
        try:
            filename = self.filenames[split]
        except KeyError as exc:
            known = ", ".join(sorted(self.filenames))
            raise ValueError(
                f"Unknown split {split!r}; expected one of: {known}"
            ) from exc
        return self.data_directory / filename

    def load(self, split: str) -> list[IssueReport]:
        path = self._path_for(split)
        if not path.exists():
            raise FileNotFoundError(f"Dataset split not found: {path}")

        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, list):
            raise DatasetFormatError(f"{path} must contain a JSON array")

        issues = []
        for record_number, row in enumerate(payload, start=1):
            if not isinstance(row, dict):
                raise DatasetFormatError(
                    f"{path}: record {record_number} must be a JSON object"
                )
            missing = set(JSON_COLUMNS).difference(row)
            if missing:
                columns = ", ".join(sorted(missing))
                raise DatasetFormatError(
                    f"{path}: record {record_number} is missing fields: {columns}"
                )
            try:
                created_at = datetime.fromisoformat(str(row["created_at"]).strip())
            except ValueError as exc:
                raise DatasetFormatError(
                    f"{path}: record {record_number} has an invalid created_at value"
                ) from exc
            issues.append(
                IssueReport.create(
                    repo=str(row["repo"]).strip(),
                    created_at=created_at,
                    label=str(row["label"]).strip(),
                    title=str(row["title"] or ""),
                    body=str(row["body"] or ""),
                )
            )
        return issues

    def save(self, split: str, issues: Sequence[IssueReport]) -> None:
        path = self._path_for(split)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                json.dump(
                    [
                        {
                            "repo": issue.repo,
                            "created_at": issue.created_at.isoformat(),
                            "label": issue.label,
                            "title": issue.title,
                            "body": issue.body,
                        }
                        for issue in issues
                    ],
                    destination,
                    ensure_ascii=False,
                    indent=2,
                )
                destination.write("\n")
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
