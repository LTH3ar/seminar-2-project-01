"""Ensembles over the project's models.

Motivated by a measurement rather than a hunch. On the test split, a tuned
TF-IDF pipeline and a frozen MPNet encoder reach almost the same
cross-repository F1 (0.7603 and 0.7557), but their per-repository profiles are
close to opposite:

==================  ========  ==========  ========  ========  ========
model                  react  tensorflow    vscode   bitcoin    opencv
==================  ========  ==========  ========  ========  ========
TF-IDF (tuned)        0.8334      0.8155    0.7234    0.6793    0.7496
Frozen MPNet          0.8017      0.7228    0.7566    0.7279    0.7696
==================  ========  ==========  ========  ========  ========

TF-IDF wins on the two easiest projects, MPNet on the three hardest, and
MPNet's spread across projects is half as wide (0.079 against 0.154). Two
models of equal average strength that fail in different places is the textbook
precondition for an ensemble to beat either.

:class:`SoftVotingEnsemble` averages class probabilities;
:class:`WeightedVotingEnsemble` lets one member carry more weight. Both satisfy
the usual ``fit``/``predict``/``predict_proba`` interface, so they run through
the unchanged evaluation harness -- including per project, which is what the
competition protocol requires.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .evaluation import class_probabilities
from .model import LABELS

ModelFactory = Callable[[], object]


class SoftVotingEnsemble:
    """Average the class probabilities of several models.

    Soft voting rather than majority voting: with three classes and two or
    three members, hard votes tie often, and averaging probabilities uses the
    confidence information that the tie-break throws away.

    Every member must expose ``predict_proba``. That excludes ``LinearSVC``,
    which has no probability head -- use :func:`ai4se.classical.logistic_regression`
    as the linear member instead, which scores within 0.003 of it anyway.

    Args:
        factories: Mapping from member name to a zero-argument model factory.
        labels: Class order.

    Raises:
        ValueError: If fewer than two members are given.
    """

    def __init__(
        self,
        factories: dict[str, ModelFactory],
        labels: Sequence[str] = LABELS,
    ) -> None:
        """Store the member factories; nothing is built until :meth:`fit`."""
        if len(factories) < 2:
            raise ValueError("An ensemble needs at least two members.")
        self.factories = dict(factories)
        self.labels = list(labels)
        self.members_: dict[str, object] = {}

    def fit(self, X: Sequence[str], y: Sequence[str]) -> SoftVotingEnsemble:
        """Train every member on the same data."""
        self.members_ = {}
        for name, factory in self.factories.items():
            self.members_[name] = factory().fit(X, y)
        return self

    def _member_probabilities(self, X: Sequence[str]) -> dict[str, list[dict]]:
        """Collect each member's class probabilities."""
        if not self.members_:
            raise RuntimeError("Call fit() before predict().")

        output = {}
        for name, model in self.members_.items():
            probabilities = class_probabilities(model, X, self.labels)
            if probabilities is None:
                raise TypeError(
                    f"Ensemble member {name!r} does not expose predict_proba, "
                    "so its votes cannot be combined."
                )
            output[name] = probabilities
        return output

    def _weights(self) -> dict[str, float]:
        """Equal weight per member."""
        return dict.fromkeys(self.members_, 1.0)

    def predict_proba(self, X: Sequence[str]) -> list[dict[str, float]]:
        """Return the weighted mean of the members' class probabilities."""
        per_member = self._member_probabilities(X)
        weights = self._weights()
        total_weight = sum(weights.values()) or 1.0

        combined = []
        for index in range(len(X)):
            averaged = {
                label: sum(
                    weights[name] * per_member[name][index].get(label, 0.0)
                    for name in per_member
                )
                / total_weight
                for label in self.labels
            }
            combined.append(averaged)
        return combined

    def predict(self, X: Sequence[str]) -> list[str]:
        """Return the class with the highest averaged probability."""
        return [
            max(row.items(), key=lambda kv: kv[1])[0] for row in self.predict_proba(X)
        ]


class WeightedVotingEnsemble(SoftVotingEnsemble):
    """Soft voting with an explicit weight per member.

    Weights should be chosen on cross-validation over the training split, never
    on the test split -- tuning them against the final numbers would turn the
    held-out set into a validation set and invalidate the comparison with the
    published baseline.

    Args:
        factories: Mapping from member name to a zero-argument model factory.
        weights: Mapping from member name to a non-negative weight.
        labels: Class order.

    Raises:
        ValueError: If the weights do not cover exactly the members.
    """

    def __init__(
        self,
        factories: dict[str, ModelFactory],
        weights: dict[str, float],
        labels: Sequence[str] = LABELS,
    ) -> None:
        """Store factories and their weights."""
        super().__init__(factories, labels)
        if set(weights) != set(factories):
            raise ValueError(
                f"Weights {sorted(weights)} do not match members {sorted(factories)}."
            )
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("Weights must be non-negative.")
        self.weights = dict(weights)

    def _weights(self) -> dict[str, float]:
        """The caller-supplied weights."""
        return self.weights


def tfidf_plus_mpnet() -> SoftVotingEnsemble:
    """The complementary pair: a tuned linear model and a frozen MPNet encoder.

    Logistic regression rather than the SVM, because soft voting needs
    probabilities and ``LinearSVC`` has none.
    """
    from .classical import tuned_logistic_regression
    from .embeddings import frozen_mpnet

    return SoftVotingEnsemble(
        {
            "TF-IDF + Logistic Regression": tuned_logistic_regression,
            "Frozen MPNet + LogReg": frozen_mpnet,
        }
    )


def tfidf_plus_mpnet_plus_cnn() -> SoftVotingEnsemble:
    """Add the CNN, whose per-project profile differs again from both."""
    from .classical import tuned_logistic_regression
    from .embeddings import frozen_mpnet
    from .neural import CNNTextClassifier

    return SoftVotingEnsemble(
        {
            "TF-IDF + Logistic Regression": tuned_logistic_regression,
            "Frozen MPNet + LogReg": frozen_mpnet,
            "CNN (embeddings)": CNNTextClassifier,
        }
    )


def setfit_plus_tfidf() -> SoftVotingEnsemble:
    """The strongest single model plus the strongest lexical one.

    SetFit is the best model measured here (0.7982) but is still weakest on
    ``bitcoin`` and ``vscode``, where a lexical model contributes different
    information. Needs the ``setfit`` extra and a GPU.
    """
    from .classical import tuned_logistic_regression
    from .embeddings import setfit_minilm

    return SoftVotingEnsemble(
        {
            "SetFit": setfit_minilm,
            "TF-IDF + Logistic Regression": tuned_logistic_regression,
        }
    )


def setfit_plus_tfidf_plus_mpnet() -> SoftVotingEnsemble:
    """Three members whose per-project profiles all differ."""
    from .classical import tuned_logistic_regression
    from .embeddings import frozen_mpnet, setfit_minilm

    return SoftVotingEnsemble(
        {
            "SetFit": setfit_minilm,
            "TF-IDF + Logistic Regression": tuned_logistic_regression,
            "Frozen MPNet + LogReg": frozen_mpnet,
        }
    )


#: Ensemble factories for the evaluation runners.
ENSEMBLE_MODELS: dict[str, Callable[[], SoftVotingEnsemble]] = {
    "Ensemble (TF-IDF + MPNet)": tfidf_plus_mpnet,
    "Ensemble (TF-IDF + MPNet + CNN)": tfidf_plus_mpnet_plus_cnn,
}

#: Ensembles containing a SetFit member. Need the ``setfit`` extra and a GPU,
#: so they are kept separate from :data:`ENSEMBLE_MODELS`.
SETFIT_ENSEMBLES: dict[str, Callable[[], SoftVotingEnsemble]] = {
    "Ensemble (SetFit + TF-IDF)": setfit_plus_tfidf,
    "Ensemble (SetFit + TF-IDF + MPNet)": setfit_plus_tfidf_plus_mpnet,
}
