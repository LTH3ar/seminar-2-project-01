"""Soft-Voting Ensemble over multiple per-repo classifiers (Chapter 3.4.5 & Table 5.9).

Best result: Ensemble(SetFit-MPNet + TF-IDF-LogReg + Frozen-MPNet)
  → Cross-repo F1 = 0.8168, AUC = 0.9319

Combines predict_proba outputs from constituent models by averaging.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List
import numpy as np


from ..model import LABELS
from .base import Classifier



class SoftVotingEnsemble(Classifier):
    """Combine predictions from multiple classifiers via class-probability averaging.

    Each constituent classifier must implement predict_proba().
    Ensemble prediction = argmax of mean probabilities across all members.

    Args:
        classifiers: List of fitted Classifier instances.
        weights: Optional per-model weights (normalised internally).
    """

    def __init__(
        self,
        classifiers: List[Classifier] | None = None,
        weights: List[float] | None = None,
    ) -> None:
        self.classifiers: List[Classifier] = classifiers or []
        self.weights: List[float] = weights or [1.0] * len(self.classifiers)
        self.name = "ensemble_" + "_".join(
            clf.name.replace("tfidf_", "").replace("frozen_", "").replace("setfit_", "")
            for clf in self.classifiers
        )

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> SoftVotingEnsemble:
        """Fit all constituent classifiers on the same training data."""
        for clf in self.classifiers:
            clf.fit(texts, labels)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Weighted average of per-member class probabilities."""
        if not self.classifiers:
            raise RuntimeError("No classifiers in ensemble.")

        total_weight = sum(self.weights)
        proba_sum = np.zeros((len(texts), len(LABELS)), dtype=float)

        for clf, w in zip(self.classifiers, self.weights, strict=True):
            member_proba = clf.predict_proba(list(texts))

            # Align columns to standard LABELS order
            member_classes = list(clf.classes_)
            aligned = np.zeros((len(texts), len(LABELS)), dtype=float)
            for j, lbl in enumerate(LABELS):
                if lbl in member_classes:
                    col_idx = member_classes.index(lbl)
                    aligned[:, j] = member_proba[:, col_idx]

            proba_sum += (w / total_weight) * aligned

        return proba_sum

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        proba = self.predict_proba(texts)
        indices = proba.argmax(axis=1)
        return np.asarray([LABELS[i] for i in indices], dtype=object)

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(LABELS, dtype=object)


def make_ensemble(classifiers: List[Classifier], weights: List[float] | None = None):
    """Return a factory that creates a fresh SoftVotingEnsemble."""
    def _factory():
        import copy
        return SoftVotingEnsemble(
            classifiers=[copy.deepcopy(clf) for clf in classifiers],
            weights=weights,
        )
    return _factory
