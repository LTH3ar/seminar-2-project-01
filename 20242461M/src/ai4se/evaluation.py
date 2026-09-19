"""Evaluation protocol for the NLBSE'24 issue report classification task.

Two things are evaluated, and they are not interchangeable:

**Cross-validation on the training half.** Used for model selection and
hyperparameter tuning. Stratified k-fold, as covered in the course: the data
is split into k equal parts, k-1 are used for training and 1 for testing, and
this is repeated k times. Stratification keeps the class proportions of each
fold equal to those of the whole, which matters as soon as any filtering
unbalances the data.

**The competition protocol on the official test half.** This is the only
number comparable with the published baselines, so the test split is touched
exactly once, at the end. The protocol is per project: a separate classifier
is trained for each of the five repositories, evaluated on that repository's
test issues, scored as the average F1 over the three classes, and the reported
figure is the arithmetic mean of the five per-repository scores.

A model is anything with ``fit(X, y)`` and ``predict(X) -> labels``, so
scikit-learn estimators, a fine-tuned transformer wrapped in two methods, and
the trivial baselines in :mod:`ai4se.baselines` all plug in unchanged. Models
are supplied as a *factory* -- a zero-argument callable returning a fresh,
untrained model -- because each fold and each project needs its own.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Protocol

from .metrics import Scores, evaluate
from .model import LABELS, REPOSITORIES
from .repository import IssueRepository, make_repository

#: Per-repository F1 of the official SetFit baseline, from the NLBSE'24
#: competition. Every result table compares against these numbers.
SETFIT_BASELINE: dict[str, float] = {
    "facebook/react": 0.8718,
    "tensorflow/tensorflow": 0.8644,
    "microsoft/vscode": 0.8262,
    "bitcoin/bitcoin": 0.7555,
    "opencv/opencv": 0.8173,
}

#: Cross-repository score of the SetFit baseline: the mean of the five above.
SETFIT_OVERALL: float = 0.8270

#: Fixed seed so every member reproduces identical folds and identical numbers.
RANDOM_SEED: int = 42


class Model(Protocol):
    """Minimal interface an estimator must satisfy to be evaluated."""

    def fit(self, X: Sequence[str], y: Sequence[str]) -> object:
        """Train on texts ``X`` with labels ``y``."""
        ...

    def predict(self, X: Sequence[str]) -> Sequence[str]:
        """Return one predicted label per element of ``X``."""
        ...


ModelFactory = Callable[[], Model]


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def stratified_folds(
    labels: Sequence[str],
    k: int = 10,
    seed: int = RANDOM_SEED,
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield ``(train_indices, validation_indices)`` for stratified k-fold.

    Each class is shuffled independently and dealt round-robin into the k
    folds, so every fold holds the class proportions of the whole dataset to
    within one item.

    Args:
        labels: Ground-truth label of every sample, in dataset order.
        k: Number of folds.
        seed: Seed for the shuffle, so folds are identical across machines.

    Yields:
        A pair of index lists, one fold at a time.

    Raises:
        ValueError: If ``k`` is smaller than 2 or exceeds the rarest class size.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}.")

    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)

    smallest = min(len(indices) for indices in by_label.values())
    if k > smallest:
        raise ValueError(
            f"k={k} exceeds the size of the rarest class ({smallest} samples)."
        )

    rng = random.Random(seed)
    fold_members: list[list[int]] = [[] for _ in range(k)]
    for label in sorted(by_label):
        indices = by_label[label][:]
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            fold_members[position % k].append(index)

    all_indices = set(range(len(labels)))
    for fold in fold_members:
        validation = sorted(fold)
        training = sorted(all_indices - set(validation))
        yield training, validation


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class CrossValidationResult:
    """Outcome of k-fold cross-validation for one model on one dataset."""

    model_name: str
    k: int
    fold_scores: list[Scores]
    scope: str = "all projects"

    @property
    def macro_f1_per_fold(self) -> list[float]:
        """Macro F1 of each fold."""
        return [s.macro_f1 for s in self.fold_scores]

    @property
    def mean_macro_f1(self) -> float:
        """Mean macro F1 across folds -- the model-selection number."""
        return mean(self.macro_f1_per_fold)

    @property
    def std_macro_f1(self) -> float:
        """Standard deviation across folds; a proxy for stability."""
        scores = self.macro_f1_per_fold
        return stdev(scores) if len(scores) > 1 else 0.0

    def as_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "model": self.model_name,
            "scope": self.scope,
            "k": self.k,
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "folds": [s.as_dict() for s in self.fold_scores],
        }

    def __repr__(self) -> str:
        """Compact one-line summary."""
        return (
            f"CrossValidationResult({self.model_name}, k={self.k}, "
            f"macro_f1={self.mean_macro_f1:.4f} +/- {self.std_macro_f1:.4f})"
        )


@dataclass
class CompetitionResult:
    """Outcome of the per-project competition protocol on the test split."""

    model_name: str
    per_repository: dict[str, Scores]
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def repository_f1(self) -> dict[str, float]:
        """Average F1 over the three classes, per repository."""
        return {repo: scores.macro_f1 for repo, scores in self.per_repository.items()}

    @property
    def overall_f1(self) -> float:
        """The reported figure: arithmetic mean of the per-repository scores."""
        return mean(self.repository_f1.values())

    @property
    def delta_vs_baseline(self) -> float:
        """Difference against the SetFit baseline; positive means better."""
        return self.overall_f1 - SETFIT_OVERALL

    def as_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "model": self.model_name,
            "created_at": self.created_at,
            "overall_f1": self.overall_f1,
            "baseline_f1": SETFIT_OVERALL,
            "delta": self.delta_vs_baseline,
            "per_repository": {
                repo: scores.as_dict() for repo, scores in self.per_repository.items()
            },
        }

    def to_frame(self):
        """Per-repository table with the baseline and the delta alongside."""
        import pandas as pd

        rows = []
        for repo in REPOSITORIES:
            if repo not in self.per_repository:
                continue
            scores = self.per_repository[repo]
            baseline = SETFIT_BASELINE[repo]
            rows.append(
                {
                    "repository": repo,
                    **{f"F1 {label}": scores.f1(label) for label in LABELS},
                    "F1 avg": scores.macro_f1,
                    "SetFit": baseline,
                    "delta": scores.macro_f1 - baseline,
                }
            )
        frame = pd.DataFrame(rows).set_index("repository")
        frame.loc["cross-repository"] = {
            **{f"F1 {label}": float("nan") for label in LABELS},
            "F1 avg": self.overall_f1,
            "SetFit": SETFIT_OVERALL,
            "delta": self.delta_vs_baseline,
        }
        return frame.round(4)

    def __repr__(self) -> str:
        """Compact one-line summary."""
        sign = "+" if self.delta_vs_baseline >= 0 else ""
        return (
            f"CompetitionResult({self.model_name}, F1={self.overall_f1:.4f}, "
            f"{sign}{self.delta_vs_baseline:.4f} vs SetFit)"
        )


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #


def cross_validate(
    model_factory: ModelFactory,
    repository: IssueRepository,
    k: int = 10,
    seed: int = RANDOM_SEED,
    model_name: str | None = None,
    scope: str = "all projects",
    verbose: bool = False,
) -> CrossValidationResult:
    """Run stratified k-fold cross-validation over a repository.

    Args:
        model_factory: Zero-argument callable returning a fresh, untrained model.
        repository: Issues to cross-validate over -- the training split only.
        k: Number of folds.
        seed: Seed controlling the fold assignment.
        model_name: Name for the result; inferred from the factory when omitted.
        scope: Free-text note recorded with the result, e.g. the project name.
        verbose: Print progress per fold.

    Returns:
        A :class:`CrossValidationResult`.
    """
    texts, labels = repository.texts_and_labels()
    name = model_name or getattr(model_factory, "__name__", "model")

    fold_scores: list[Scores] = []
    for fold_number, (train_idx, val_idx) in enumerate(
        stratified_folds(labels, k=k, seed=seed), start=1
    ):
        model = model_factory()
        model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        predictions = list(model.predict([texts[i] for i in val_idx]))
        scores = evaluate([labels[i] for i in val_idx], predictions, labels=LABELS)
        fold_scores.append(scores)
        if verbose:
            print(f"  fold {fold_number}/{k}: macro F1 = {scores.macro_f1:.4f}")

    return CrossValidationResult(
        model_name=name, k=k, fold_scores=fold_scores, scope=scope
    )


def cross_validate_per_project(
    model_factory: ModelFactory,
    repository: IssueRepository,
    k: int = 10,
    seed: int = RANDOM_SEED,
    model_name: str | None = None,
    verbose: bool = False,
) -> dict[str, CrossValidationResult]:
    """Cross-validate separately within each project.

    This mirrors the competition protocol during development, so the model
    selected by cross-validation is the one that will actually be submitted.

    Returns:
        Mapping from repository name to its :class:`CrossValidationResult`.
    """
    results: dict[str, CrossValidationResult] = {}
    for repo in repository.repos():
        if verbose:
            print(f"{repo}:")
        project_repository = make_repository("memory", issues=repository.by_repo(repo))
        results[repo] = cross_validate(
            model_factory,
            project_repository,
            k=k,
            seed=seed,
            model_name=model_name,
            scope=repo,
            verbose=verbose,
        )
        if verbose:
            print(f"  mean macro F1 = {results[repo].mean_macro_f1:.4f}\n")
    return results


def evaluate_competition(
    model_factory: ModelFactory,
    train: IssueRepository,
    test: IssueRepository,
    model_name: str | None = None,
    verbose: bool = False,
) -> CompetitionResult:
    """Run the official protocol: one classifier per project, scored per project.

    Args:
        model_factory: Zero-argument callable returning a fresh, untrained model.
        train: The official training split.
        test: The official test split. Use this once, at the end.
        model_name: Name recorded with the result.
        verbose: Print each repository's score as it is computed.

    Returns:
        A :class:`CompetitionResult` directly comparable with the baselines.
    """
    name = model_name or getattr(model_factory, "__name__", "model")
    per_repository: dict[str, Scores] = {}

    for repo in sorted(set(train.repos()) & set(test.repos())):
        train_subset = make_repository("memory", issues=train.by_repo(repo))
        test_subset = make_repository("memory", issues=test.by_repo(repo))

        X_train, y_train = train_subset.texts_and_labels()
        X_test, y_test = test_subset.texts_and_labels()

        model = model_factory()
        model.fit(X_train, y_train)
        predictions = list(model.predict(X_test))

        scores = evaluate(y_test, predictions, labels=LABELS)
        per_repository[repo] = scores
        if verbose:
            baseline = SETFIT_BASELINE.get(repo)
            note = f"  (SetFit {baseline:.4f})" if baseline else ""
            print(f"  {repo:<24} F1 = {scores.macro_f1:.4f}{note}")

    result = CompetitionResult(model_name=name, per_repository=per_repository)
    if verbose:
        print(
            f"\n  cross-repository F1 = {result.overall_f1:.4f} "
            f"(SetFit {SETFIT_OVERALL:.4f}, delta {result.delta_vs_baseline:+.4f})"
        )
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def leaderboard(results: Sequence[CompetitionResult]):
    """Build the master comparison table: every model against the baseline.

    This is the central table of the report's Results section.
    """
    import pandas as pd

    rows = []
    for result in results:
        row = {"model": result.model_name}
        row.update({repo: result.repository_f1.get(repo) for repo in REPOSITORIES})
        row["overall"] = result.overall_f1
        row["vs SetFit"] = result.delta_vs_baseline
        rows.append(row)

    baseline_row = {"model": "SetFit (NLBSE'24 baseline)"}
    baseline_row.update(SETFIT_BASELINE)
    baseline_row["overall"] = SETFIT_OVERALL
    baseline_row["vs SetFit"] = 0.0
    rows.append(baseline_row)

    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("overall", ascending=False)
        .round(4)
    )


def to_latex(frame, caption: str, label: str, path: str | Path | None = None) -> str:
    """Render a results table as a LaTeX ``table`` environment.

    The report is written in LaTeX, so tables are generated rather than
    retyped: the numbers in the report cannot then drift from the numbers the
    code produced.

    Args:
        frame: A pandas DataFrame, e.g. from :func:`leaderboard`.
        caption: Table caption.
        label: LaTeX label, without the ``tab:`` prefix.
        path: If given, also write the snippet to this file.

    Returns:
        The LaTeX source.
    """
    body = frame.to_latex(
        float_format="%.4f",
        na_rep="--",
        escape=True,
        column_format="l" + "r" * len(frame.columns),
    )
    latex = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{tab:{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(latex, encoding="utf-8")
    return latex


def save_result(
    result: CrossValidationResult | CompetitionResult,
    path: str | Path,
) -> Path:
    """Write a result to JSON so the report can be rebuilt without re-training."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return destination


def load_result(path: str | Path) -> dict:
    """Read back a result written by :func:`save_result`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
