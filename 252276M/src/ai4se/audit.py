"""Descriptive statistics and leakage checks for issue-report datasets."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256


from .model import IssueReport


_WHITESPACE = re.compile(r"\s+")


def normalized_text(issue: IssueReport) -> str:
    """Normalize text to identify exact-content duplicates."""
    combined = f"{issue.title}\n{issue.body}"
    combined = unicodedata.normalize("NFKC", combined).casefold()
    return _WHITESPACE.sub(" ", combined).strip()


def duplicate_group(issue: IssueReport) -> str:
    return sha256(normalized_text(issue).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TextOverlap:
    text_hash: str
    title_preview: str
    train_issue_ids: tuple[int, ...]
    test_issue_ids: tuple[int, ...]
    train_labels: tuple[str, ...]
    test_labels: tuple[str, ...]

    @property
    def has_conflicting_labels(self) -> bool:
        return set(self.train_labels) != set(self.test_labels)


def find_data_leakage(train_issues: list[IssueReport], test_issues: list[IssueReport]) -> list[TextOverlap]:
    """Find text overlaps between train and test splits."""
    train_hashes = defaultdict(list)
    for iss in train_issues:
        h = duplicate_group(iss)
        train_hashes[h].append(iss)

    overlaps = []
    test_hashes = defaultdict(list)
    for iss in test_issues:
        h = duplicate_group(iss)
        test_hashes[h].append(iss)

    common = set(train_hashes.keys()) & set(test_hashes.keys())
    for h in common:
        tr = train_hashes[h]
        te = test_hashes[h]
        overlaps.append(
            TextOverlap(
                text_hash=h,
                title_preview=tr[0].title[:60],
                train_issue_ids=tuple(i.issue_id for i in tr),
                test_issue_ids=tuple(i.issue_id for i in te),
                train_labels=tuple(i.label for i in tr),
                test_labels=tuple(i.label for i in te),
            )
        )
    return overlaps
