"""Dataset loading utilities for NLBSE'24 issue report data.
CSV schema: repo, created_at, label, title, body  (no issue_id column).
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal


from .model import IssueReport
from .repository import InMemoryIssueRepository, IssueRepository

# Increase CSV field size limit for very long issue bodies
_max_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_limit)
        break
    except OverflowError:
        _max_limit = int(_max_limit / 10)

# Search locations for data files
_CANDIDATE_DIRS = [
    Path(__file__).resolve().parents[1] / "data" / "raw",
    Path(__file__).resolve().parents[2] / "data" / "raw",
    Path.cwd() / "data" / "raw",
    Path.cwd() / "252276M" / "data" / "raw",
]


def _find_csv(split: str) -> Path:
    filename = f"issues_{split}.csv"
    for d in _CANDIDATE_DIRS:
        p = d / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not locate {filename}.\nSearched:\n" + "\n".join(f"  {d}/{filename}" for d in _CANDIDATE_DIRS)
    )


def _load_csv(csv_path: Path) -> list[IssueReport]:
    """Parse CSV file into IssueReport list. Handles missing issue_id by using row index."""
    issues = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                # Parse created_at; handle both ISO and space-separated formats
                raw_ts = (row.get("created_at") or "").strip()
                if raw_ts:
                    raw_ts = raw_ts.replace("T", " ")
                    # Remove timezone if present
                    if "+" in raw_ts:
                        raw_ts = raw_ts[:raw_ts.index("+")]
                    created_at = datetime.fromisoformat(raw_ts)
                else:
                    created_at = datetime(2020, 1, 1)

                issue_id = int(row["issue_id"]) if "issue_id" in row and row["issue_id"].strip() else idx

                issues.append(IssueReport(
                    repo=(row.get("repo") or "").strip(),
                    issue_id=issue_id,
                    created_at=created_at,
                    title=(row.get("title") or "").strip(),
                    body=(row.get("body") or "").strip(),
                    label=(row.get("label") or "").strip(),
                ))
            except Exception:
                # Skip malformed rows silently
                continue
    return issues


def load_split(
    split: Literal["train", "test"] = "train",
    kind: Literal["memory", "file"] = "memory",
) -> IssueRepository:
    """Load a dataset split into an in-memory or file-backed repository.

    Args:
        split: 'train' or 'test'.
        kind: 'memory' for InMemoryIssueRepository (default), 'file' for disk-backed.
    """
    csv_path = _find_csv(split)
    if kind == "file":
        from .repository import FileIssueRepository
        return FileIssueRepository(csv_path).load()

    issues = _load_csv(csv_path)
    return InMemoryIssueRepository(issues)


def load_dataset(kind: Literal["memory", "file"] = "memory") -> dict[str, IssueRepository]:
    """Load both train and test splits and return as a dict."""
    return {
        "train": load_split("train"),
        "test": load_split("test"),
    }
