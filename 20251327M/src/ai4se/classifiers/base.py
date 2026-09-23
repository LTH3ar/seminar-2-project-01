"""Common interface for every classifier in the project.

The competition requires **one classifier per project**: five models are
trained and evaluated separately and their scores averaged. Every model in
this package therefore implements the same small interface, and
:func:`train_per_repo` drives that protocol once so no experiment script has
to re-implement it.

The interface deliberately speaks in texts and labels rather than in feature
matrices. Vectorisation is part of a model's own definition -- TF-IDF for the
classical models, a sentence encoder for SetFit, a tokenizer for a
transformer -- and hiding it behind :meth:`Classifier.fit` is what lets the
evaluation code treat all of them identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from ..model import IssueReport
from ..repository import IssueRepository


class Classifier(ABC):
    """A text classifier that maps issue text to one of the three labels.

    Attributes:
        name: Short identifier used in results tables and file names.
    """

    name: str = "classifier"

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Classifier:
        """Train on labelled texts and return self.

        Args:
            texts: Training documents.
            labels: Ground-truth label of each document.

        Returns:
            The fitted classifier, to allow chaining.
        """

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Predict one label per document.

        Args:
            texts: Documents to classify.

        Returns:
            Array of predicted labels, aligned with ``texts``.
        """

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return class probabilities, when the model can produce them.

        Needed by the calibration and abstention analysis. Models without a
        probabilistic output override nothing and this raises.

        Args:
            texts: Documents to score.

        Returns:
            Array of shape ``(len(texts), n_classes)`` whose columns follow
            :attr:`classes_`.

        Raises:
            NotImplementedError: If the model has no probabilistic output.
        """
        raise NotImplementedError(f"{self.name} does not expose probabilities.")

    @property
    def classes_(self) -> np.ndarray:
        """Label order used by :meth:`predict_proba`.

        Raises:
            NotImplementedError: If the model has no probabilistic output.
        """
        raise NotImplementedError(f"{self.name} does not expose class order.")

    def __repr__(self) -> str:
        """Return a short description for logs and results tables."""
        return f"{type(self).__name__}(name={self.name!r})"


#: A zero-argument callable returning a fresh, untrained classifier. Passing
#: factories rather than instances is what keeps the five per-project models
#: (and the five seeds) genuinely independent of one another.
ClassifierFactory = Callable[[], Classifier]


def _texts_and_labels(issues: Sequence[IssueReport]) -> tuple[list[str], list[str]]:
    """Extract the model input and the ground truth from issue entities."""
    return [issue.text for issue in issues], [issue.label for issue in issues]


def train_per_repo(
    factory: ClassifierFactory,
    train: IssueRepository,
    test: IssueRepository,
    verbose: bool = False,
) -> np.ndarray:
    """Train one model per project and predict that project's test issues.

    This is the protocol the competition scores, and it is not the same as
    training a single pooled model: each of the five models sees only its own
    project's 300 training issues.

    Args:
        factory: Produces a fresh untrained classifier for each project.
        train: Training issues of all projects.
        test: Test issues of all projects.
        verbose: Print progress, useful for the slower models.

    Returns:
        Predicted labels aligned with ``test.all()``.
    """
    issues = test.all()
    predictions = np.empty(len(issues), dtype=object)

    # Iterate over the projects actually present in the data rather than the
    # REPOSITORIES constant. Trusting the constant silently drops every issue
    # from a project it does not list -- which is exactly what happens on a
    # subset, on a future edition of the dataset, or in a unit test.
    for repo in test.repos():
        positions = [i for i, issue in enumerate(issues) if issue.repo == repo]
        train_texts, train_labels = _texts_and_labels(train.by_repo(repo))
        if not train_texts:
            raise ValueError(
                f"No training issues for {repo!r}, but {len(positions)} test "
                "issues belong to it. Pass matching splits."
            )
        if verbose:
            print(f"    {repo:<24} train={len(train_texts):<5} test={len(positions)}")

        model = factory().fit(train_texts, train_labels)
        predicted = model.predict([issues[i].text for i in positions])
        for position, label in zip(positions, predicted, strict=True):
            predictions[position] = label

    if any(p is None for p in predictions):
        missing = sum(1 for p in predictions if p is None)
        raise RuntimeError(f"{missing} test issues were left unpredicted.")
    return predictions


def train_pooled(
    factory: ClassifierFactory,
    train: IssueRepository,
    test: IssueRepository,
) -> np.ndarray:
    """Train a single model on all five projects at once.

    Kept as a contrast to :func:`train_per_repo`. The two are statistically
    indistinguishable on this test set (see ``docs/benchmark-audit.md``), so
    the per-project protocol is used for every reported result and this one
    only appears in the ablation.

    Args:
        factory: Produces a fresh untrained classifier.
        train: Training issues of all projects.
        test: Test issues of all projects.

    Returns:
        Predicted labels aligned with ``test.all()``.
    """
    train_texts, train_labels = _texts_and_labels(train.all())
    test_texts, _ = _texts_and_labels(test.all())
    model = factory().fit(train_texts, train_labels)
    return np.asarray(model.predict(test_texts), dtype=object)
