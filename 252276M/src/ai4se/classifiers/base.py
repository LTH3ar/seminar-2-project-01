"""Abstract classifier base and standard evaluation harness across repositories.

Protocol: train_per_repo trains 5 separate models (one per project) on 300 training
issues each, mirrors the official NLBSE'24 competition scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
import numpy as np


from ..model import LABELS, IssueReport
from ..repository import IssueRepository



class Classifier(ABC):
    """Abstract interface that every classifier in the project must implement."""

    name: str = "classifier"

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "Classifier":
        """Train on (texts, labels) and return self."""

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Return array of predicted label strings, same length as texts."""

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return (n_samples, n_classes) probability matrix.

        Column order follows classes_ property.
        Raises NotImplementedError if the model has no probabilistic output.
        """
        raise NotImplementedError(f"{self.name} does not implement predict_proba().")

    @property
    def classes_(self) -> np.ndarray:
        """Label array corresponding to probability columns."""
        return np.asarray(LABELS, dtype=object)


ClassifierFactory = Callable[[], Classifier]


def _extract(issues: list[IssueReport]) -> tuple[list[str], list[str]]:
    """Extract texts and labels from a list of IssueReport objects."""
    return [iss.text for iss in issues], [iss.label for iss in issues]


def train_per_repo(
    factory: ClassifierFactory,
    train: IssueRepository,
    test: IssueRepository,
    return_proba: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train one model per project (competition protocol) and predict test issues.

    NLBSE'24 protocol: five separate classifiers, each trained on 300 issues from
    one project. Per-project F1 scores are averaged for the final leaderboard metric.

    Args:
        factory: Zero-argument callable returning a fresh untrained Classifier.
        train: Training split IssueRepository (1,500 issues, 5 repos × 3 classes × 100).
        test: Test split IssueRepository (1,500 issues, same structure).
        return_proba: If True, also return (n_test, 3) probability matrix in LABELS order.
        verbose: Print per-project progress.

    Returns:
        (predicted_labels, probabilities)  — probabilities is None when return_proba=False.
    """
    test_issues = list(test.all())
    n = len(test_issues)
    predictions = np.empty(n, dtype=object)
    probabilities = np.zeros((n, len(LABELS)), dtype=float) if return_proba else None

    # Use repositories actually present in test (not a hardcoded constant),
    # so the function works on subsets and future dataset editions.
    repos = test.repos()

    for repo in repos:
        train_issues = train.by_repo(repo)
        test_repo_issues = test.by_repo(repo)

        if not train_issues:
            raise ValueError(
                f"No training issues found for repository {repo!r}. "
                "Ensure train and test splits cover the same repositories."
            )

        train_texts, train_labels = _extract(train_issues)
        test_texts = [iss.text for iss in test_repo_issues]

        # Positions of this repo's issues inside the full test list
        positions = [i for i, iss in enumerate(test_issues) if iss.repo == repo]

        if verbose:
            print(f"  {repo:<28} train={len(train_texts):<5} test={len(positions)}")

        model = factory().fit(train_texts, train_labels)
        repo_pred = model.predict(test_texts)

        for pos, lbl in zip(positions, repo_pred, strict=True):
            predictions[pos] = lbl

        if return_proba:
            try:
                raw_prob = model.predict_proba(test_texts)
                # Align probability columns to standard LABELS order
                member_classes = list(model.classes_)
                aligned = np.zeros((len(test_texts), len(LABELS)), dtype=float)
                for j, lbl in enumerate(LABELS):
                    if lbl in member_classes:
                        aligned[:, j] = raw_prob[:, member_classes.index(lbl)]
                probabilities[positions] = aligned
            except NotImplementedError:
                pass

    return predictions, probabilities


def train_pooled(
    factory: ClassifierFactory,
    train: IssueRepository,
    test: IssueRepository,
    return_proba: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train a single global model on all 1,500 training issues (ablation baseline).

    Used for the per-project vs global training comparison (Table 5.6 in report).
    """
    train_texts, train_labels = _extract(list(train.all()))
    test_issues = list(test.all())
    test_texts = [iss.text for iss in test_issues]

    model = factory().fit(train_texts, train_labels)
    predictions = np.asarray(model.predict(test_texts), dtype=object)

    probabilities = None
    if return_proba:
        try:
            raw_prob = model.predict_proba(test_texts)
            member_classes = list(model.classes_)
            probabilities = np.zeros((len(test_texts), len(LABELS)), dtype=float)
            for j, lbl in enumerate(LABELS):
                if lbl in member_classes:
                    probabilities[:, j] = raw_prob[:, member_classes.index(lbl)]
        except NotImplementedError:
            pass

    return predictions, probabilities
