"""Descriptive statistics and leakage checks for issue-report datasets."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256

from issue_classifier.domain import IssueReport


_WHITESPACE = re.compile(r"\s+")


def normalized_text(issue: IssueReport) -> str:
    """Normalize only enough text to identify exact-content duplicates."""

    combined = f"{issue.title}\n{issue.body}"
    combined = unicodedata.normalize("NFKC", combined).casefold()
    return _WHITESPACE.sub(" ", combined).strip()


def duplicate_group(issue: IssueReport) -> str:
    return sha256(normalized_text(issue).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LengthStatistics:
    minimum: int
    median: float
    mean: float
    percentile_95: int
    maximum: int


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    split: str
    record_count: int
    label_counts: dict[str, int]
    repository_counts: dict[str, int]
    repository_label_counts: dict[str, dict[str, int]]
    empty_title_count: int
    empty_body_count: int
    duplicate_group_count: int
    duplicate_record_count: int
    title_length: LengthStatistics
    body_length: LengthStatistics

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TextOverlap:
    text_hash: str
    title_preview: str
    train_issue_ids: tuple[str, ...]
    test_issue_ids: tuple[str, ...]
    train_labels: tuple[str, ...]
    test_labels: tuple[str, ...]

    @property
    def has_conflicting_labels(self) -> bool:
        return set(self.train_labels) != set(self.test_labels)


@dataclass(frozen=True, slots=True)
class LeakageReport:
    overlap_group_count: int
    overlapping_train_record_count: int
    overlapping_test_record_count: int
    conflicting_label_group_count: int
    overlaps: tuple[TextOverlap, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for index, overlap in enumerate(self.overlaps):
            result["overlaps"][index]["has_conflicting_labels"] = (
                overlap.has_conflicting_labels
            )

        return result


def _length_statistics(values: list[int]) -> LengthStatistics:
    if not values:
        return LengthStatistics(0, 0.0, 0.0, 0, 0)
    
    ordered = sorted(values)
    size = len(ordered)
    middle = size // 2
    median = (
        float(ordered[middle])
        if size % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    percentile_95 = ordered[max(0, math.ceil(size * 0.95) - 1)]

    return LengthStatistics(
        minimum=ordered[0],
        median=median,
        mean=sum(ordered) / size,
        percentile_95=percentile_95,
        maximum=ordered[-1],
    )


def audit_dataset(issues: list[IssueReport], *, split: str) -> DatasetAudit:
    labels = Counter(issue.label for issue in issues)
    repositories = Counter(issue.repo for issue in issues)
    repository_labels: dict[str, Counter[str]] = defaultdict(Counter)
    duplicate_groups = Counter(duplicate_group(issue) for issue in issues)

    for issue in issues:
        repository_labels[issue.repo][issue.label] += 1

    repeated_groups = [count for count in duplicate_groups.values() if count > 1]

    return DatasetAudit(
        split=split,
        record_count=len(issues),
        label_counts=dict(sorted(labels.items())),
        repository_counts=dict(sorted(repositories.items())),
        repository_label_counts={
            repo: dict(sorted(counts.items()))
            for repo, counts in sorted(repository_labels.items())
        },
        empty_title_count=sum(not issue.title.strip() for issue in issues),
        empty_body_count=sum(not issue.body.strip() for issue in issues),
        duplicate_group_count=len(repeated_groups),
        duplicate_record_count=sum(repeated_groups),
        title_length=_length_statistics([len(issue.title) for issue in issues]),
        body_length=_length_statistics([len(issue.body) for issue in issues]),
    )


def audit_train_test_leakage(
    train_issues: list[IssueReport],
    test_issues: list[IssueReport],
) -> LeakageReport:
    train_groups: dict[str, list[IssueReport]] = defaultdict(list)
    test_groups: dict[str, list[IssueReport]] = defaultdict(list)
    for issue in train_issues:
        train_groups[duplicate_group(issue)].append(issue)
    for issue in test_issues:
        test_groups[duplicate_group(issue)].append(issue)

    overlaps = []
    for text_hash in sorted(train_groups.keys() & test_groups.keys()):
        train_matches = train_groups[text_hash]
        test_matches = test_groups[text_hash]
        title = train_matches[0].title.strip().replace("\n", " ")
        overlaps.append(
            TextOverlap(
                text_hash=text_hash,
                title_preview=title[:120],
                train_issue_ids=tuple(issue.issue_id for issue in train_matches),
                test_issue_ids=tuple(issue.issue_id for issue in test_matches),
                train_labels=tuple(sorted({issue.label for issue in train_matches})),
                test_labels=tuple(sorted({issue.label for issue in test_matches})),
            )
        )

    return LeakageReport(
        overlap_group_count=len(overlaps),
        overlapping_train_record_count=sum(
            len(train_groups[overlap.text_hash]) for overlap in overlaps
        ),
        overlapping_test_record_count=sum(
            len(test_groups[overlap.text_hash]) for overlap in overlaps
        ),
        conflicting_label_group_count=sum(
            overlap.has_conflicting_labels for overlap in overlaps
        ),
        overlaps=tuple(overlaps),
    )
