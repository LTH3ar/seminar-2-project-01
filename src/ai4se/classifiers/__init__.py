"""Classifiers for the issue report classification task.

Every model implements the same :class:`~ai4se.classifiers.base.Classifier`
interface, so the evaluation code treats them identically and the competition
protocol -- one model per project -- is driven once by
:func:`~ai4se.classifiers.base.train_per_repo`.

``classical``
    TF-IDF with Naive Bayes, logistic regression, a linear SVM or a random
    forest (track B). Base requirements plus ``.[ml]``.
``neural``
    A feed-forward network over TF-IDF and a text CNN trained from scratch
    (track C). Needs ``.[dl]``.
``setfit_model``
    Reproduction of the organisers' published baseline (track D). Needs
    ``.[dl]``.
``selection``
    k-fold cross-validation and grid search, run on the training split only.

Model modules import their heavy dependencies lazily, so this package can be
imported without scikit-learn or torch installed.
"""

from .base import Classifier, ClassifierFactory, train_per_repo, train_pooled
from .selection import cross_validate, grid_search, make_subset, summarise_search

__all__ = [
    "Classifier",
    "ClassifierFactory",
    "cross_validate",
    "grid_search",
    "make_subset",
    "summarise_search",
    "train_per_repo",
    "train_pooled",
]
