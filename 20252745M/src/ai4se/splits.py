"""Alternative split protocols for temporal robustness analysis."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .audit import duplicate_group
from .model import IssueReport


@dataclass(frozen=True, slots=True)
class DeduplicationSummary:
    """Counts removed before constructing a robustness split."""

    input_records: int
    output_records: int
    duplicate_records_collapsed: int
    conflicting_records_removed: int
    conflicting_groups_removed: int


def deduplicate_for_evaluation(
    issues: Sequence[IssueReport],
) -> tuple[list[IssueReport], DeduplicationSummary]:
    """Remove repeated text and discard groups with conflicting labels."""

    groups: dict[str, list[IssueReport]] = defaultdict(list)
    for issue in issues:
        groups[duplicate_group(issue)].append(issue)

    retained = []
    conflicting_groups = 0
    conflicting_records = 0
    duplicates_collapsed = 0
    for matches in groups.values():
        if len({issue.label for issue in matches}) > 1:
            conflicting_groups += 1
            conflicting_records += len(matches)
            continue
        duplicates_collapsed += len(matches) - 1
        retained.append(min(matches, key=lambda issue: issue.created_at))
    retained.sort(key=lambda issue: (issue.repo, issue.created_at, issue.issue_id))
    return retained, DeduplicationSummary(
        input_records=len(issues),
        output_records=len(retained),
        duplicate_records_collapsed=duplicates_collapsed,
        conflicting_records_removed=conflicting_records,
        conflicting_groups_removed=conflicting_groups,
    )


def time_aware_split(
    issues: Sequence[IssueReport],
    *,
    train_fraction: float = 0.5,
) -> tuple[list[IssueReport], list[IssueReport], DeduplicationSummary]:
    """Train on older issues and evaluate on newer issues within each cell."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    unique_issues, summary = deduplicate_for_evaluation(issues)
    cells = _group_by_repository_and_label(unique_issues)
    train: list[IssueReport] = []
    test: list[IssueReport] = []
    for (repository, label), cell in cells.items():
        _require_splittable_cell(repository, label, cell)
        ordered = sorted(cell, key=lambda issue: issue.created_at)
        cut = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
        train.extend(ordered[:cut])
        test.extend(ordered[cut:])
    return train, test, summary


def random_split_control(
    issues: Sequence[IssueReport],
    *,
    train_fraction: float = 0.5,
    random_state: int = 42,
) -> tuple[list[IssueReport], list[IssueReport], DeduplicationSummary]:
    """Create a class/repository-matched random control for the temporal split."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    unique_issues, summary = deduplicate_for_evaluation(issues)
    generator = random.Random(random_state)
    cells = _group_by_repository_and_label(unique_issues)
    train: list[IssueReport] = []
    test: list[IssueReport] = []
    for (repository, label), cell in cells.items():
        _require_splittable_cell(repository, label, cell)
        shuffled = list(cell)
        generator.shuffle(shuffled)
        cut = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
        train.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return train, test, summary


def _group_by_repository_and_label(
    issues: Sequence[IssueReport],
) -> dict[tuple[str, str], list[IssueReport]]:
    cells: dict[tuple[str, str], list[IssueReport]] = defaultdict(list)
    for issue in issues:
        cells[(issue.repo, issue.label)].append(issue)
    return dict(cells)


def _require_splittable_cell(
    repository: str,
    label: str,
    issues: Sequence[IssueReport],
) -> None:
    if len(issues) < 2:
        raise ValueError(
            "Robustness splits require at least two unique issues per "
            f"repository/label cell; {repository}/{label} has {len(issues)}"
        )
