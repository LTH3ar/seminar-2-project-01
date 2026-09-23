"""Alternative splitting protocols, including the time-aware one.

The official split is random: training and test issues are drawn from the same
period of each project's history. Because the class balance of a project
shifts over its lifetime and the dataset was built by sampling a fixed 100
issues per class, class and creation date end up confounded -- a model reading
*only* the timestamp scores 0.679 where chance is 0.333.

:func:`time_aware_split` is the control for that. It trains on each project's
older issues and tests on its newer ones while holding the class balance
exactly constant, so the only thing that changes relative to a random split is
the direction of time. On the TF-IDF baseline that costs 9.2 F1 points.

See ``docs/benchmark-audit.md`` for the full measurement.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np

from ..model import IssueReport
from ..repository import IssueRepository, make_repository


def _grouped(issues: Sequence[IssueReport]) -> dict[tuple[str, str], list[IssueReport]]:
    """Group issues by ``(project, label)``."""
    groups: dict[tuple[str, str], list[IssueReport]] = defaultdict(list)
    for issue in issues:
        groups[(issue.repo, issue.label)].append(issue)
    return dict(groups)


def time_aware_split(
    repository: IssueRepository,
    train_fraction: float = 0.5,
    kind: str = "memory",
) -> tuple[IssueRepository, IssueRepository]:
    """Split each ``(project, class)`` cell by creation date.

    The oldest ``train_fraction`` of every cell goes to training and the rest
    to test. Splitting *within* each cell rather than globally is what keeps
    the class balance identical on both sides; splitting globally instead
    collapses the training set to 37 bugs against 814 features, which
    confounds the temporal effect with a class-imbalance effect.

    Args:
        repository: Source issues. Pass the union of both official splits to
            get the full history.
        train_fraction: Share of each cell assigned to training.
        kind: Persistence layer of the two returned repositories.

    Returns:
        Tuple ``(train, test)``.

    Raises:
        ValueError: If ``train_fraction`` is outside ``(0, 1)``.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")

    train: list[IssueReport] = []
    test: list[IssueReport] = []
    for cell in _grouped(repository.all()).values():
        # created_at is a fixed-width "YYYY-MM-DD HH:MM:SS" string, so a plain
        # lexicographic sort is chronological and needs no date parsing.
        ordered = sorted(cell, key=lambda issue: issue.created_at)
        cut = int(len(ordered) * train_fraction)
        train.extend(ordered[:cut])
        test.extend(ordered[cut:])

    return (
        make_repository(kind, issues=train),
        make_repository(kind, issues=test),
    )


def random_split_control(
    repository: IssueRepository,
    train_fraction: float = 0.5,
    seed: int = 42,
    kind: str = "memory",
) -> tuple[IssueRepository, IssueRepository]:
    """Build the matched control for :func:`time_aware_split`.

    Identical cell sizes and identical class balance, but the cut is random
    instead of chronological. Comparing the two isolates the effect of the
    split protocol from every other difference.

    Args:
        repository: Source issues.
        train_fraction: Share of each cell assigned to training.
        seed: Seed of the random generator.
        kind: Persistence layer of the two returned repositories.

    Returns:
        Tuple ``(train, test)``.
    """
    rng = np.random.default_rng(seed)
    train: list[IssueReport] = []
    test: list[IssueReport] = []
    for cell in _grouped(repository.all()).values():
        shuffled = [cell[i] for i in rng.permutation(len(cell))]
        cut = int(len(shuffled) * train_fraction)
        train.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return (
        make_repository(kind, issues=train),
        make_repository(kind, issues=test),
    )


def stratified_folds(
    repository: IssueRepository,
    n_folds: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield cross-validation folds stratified by ``(project, class)``.

    Stratifying on the pair, not on the label alone, keeps every fold balanced
    across both dimensions the competition score depends on.

    A warning on the alternative: grouping by project instead
    (``StratifiedGroupKFold(groups=repo)``) looks like the safer choice but is
    wrong here. It removes all in-domain data and turns every fold into a
    leave-one-project-out problem, which scores 0.617 and is not what the
    competition measures. Keep that design for a dedicated transfer
    experiment, not for hyper-parameter selection.

    Args:
        repository: Issues to split. Use the training split only -- the test
            split must stay untouched until the final table.
        n_folds: Number of folds.
        seed: Seed of the random generator.

    Yields:
        Tuples ``(train_index, validation_index)`` of positions into
        ``repository.all()``.

    Raises:
        ValueError: If ``n_folds`` is smaller than two.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    issues = repository.all()
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(issues), dtype=int)

    positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, issue in enumerate(issues):
        positions[(issue.repo, issue.label)].append(position)

    # Deal each stratum round-robin into folds after shuffling it, so fold
    # sizes stay within one item of each other inside every stratum.
    for cell in positions.values():
        order = rng.permutation(len(cell))
        for rank, index in enumerate(order):
            assignment[cell[index]] = rank % n_folds

    all_positions = np.arange(len(issues))
    for fold in range(n_folds):
        validation = all_positions[assignment == fold]
        train = all_positions[assignment != fold]
        yield train, validation
