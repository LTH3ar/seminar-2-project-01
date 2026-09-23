"""Classical machine learning baselines (track B).

Four estimators over a shared TF-IDF representation: multinomial Naive Bayes,
logistic regression, a linear support vector machine and a random forest.
They are cheap enough that all four can be run for five seeds across five
projects in under a minute, which makes them the right place to establish the
evaluation habits the expensive models will inherit.

Requires the track B extra::

    pip install -e ".[ml]"
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Classifier

#: Estimators exposed by :func:`make_classical`, in the order they appear in
#: the results tables.
ESTIMATORS: tuple[str, ...] = ("naive_bayes", "logreg", "linear_svm", "random_forest")


class TfidfClassifier(Classifier):
    """TF-IDF features followed by a scikit-learn estimator.

    Args:
        estimator: One of :data:`ESTIMATORS`.
        ngram_range: Word n-gram range passed to the vectoriser.
        min_df: Minimum document frequency of a term.
        sublinear_tf: Apply ``1 + log(tf)`` term-frequency scaling.
        c_value: Regularisation strength for the linear models.
        n_estimators: Number of trees, for the random forest only.
        seed: Seed of any stochastic component.

    Raises:
        ValueError: If ``estimator`` is not a supported name.
    """

    def __init__(  # noqa: D107 -- arguments documented on the class
        self,
        estimator: str = "linear_svm",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True,
        c_value: float = 1.0,
        n_estimators: int = 300,
        seed: int = 42,
    ) -> None:
        if estimator not in ESTIMATORS:
            raise ValueError(
                f"Unknown estimator {estimator!r}; use one of {ESTIMATORS}."
            )
        self.name = estimator
        self.estimator = estimator
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.c_value = c_value
        self.n_estimators = n_estimators
        self.seed = seed
        self._pipeline = None

    def _build(self):
        """Assemble the vectoriser and estimator into a pipeline."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC

        vectoriser = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
        )
        # Naive Bayes has no class_weight parameter; the dataset is balanced
        # at 100 issues per class per project, so nothing is lost by that.
        if self.estimator == "naive_bayes":
            estimator = MultinomialNB()
        elif self.estimator == "logreg":
            estimator = LogisticRegression(
                C=self.c_value,
                max_iter=3000,
                class_weight="balanced",
                random_state=self.seed,
            )
        elif self.estimator == "linear_svm":
            estimator = LinearSVC(
                C=self.c_value,
                class_weight="balanced",
                random_state=self.seed,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            )
        return make_pipeline(vectoriser, estimator)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> TfidfClassifier:
        """Fit the vectoriser and the estimator on the training texts."""
        self._pipeline = self._build()
        self._pipeline.fit(list(texts), list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Predict a label for each text.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting.")
        return np.asarray(self._pipeline.predict(list(texts)), dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return class probabilities.

        ``LinearSVC`` has no probabilistic output, so for that estimator the
        decision function is turned into one with a softmax. That is a
        monotone transform: it leaves the predicted labels untouched and is
        only used for the confidence ranking in the abstention analysis, never
        as a calibrated probability. Use :class:`CalibratedTfidfClassifier`
        when genuine calibration is required.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting.")
        final = self._pipeline[-1]
        if hasattr(final, "predict_proba"):
            return np.asarray(self._pipeline.predict_proba(list(texts)))
        scores = self._pipeline.decision_function(list(texts))
        exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    @property
    def classes_(self) -> np.ndarray:
        """Label order of the probability columns.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting.")
        return self._pipeline[-1].classes_


class CalibratedTfidfClassifier(TfidfClassifier):
    """A :class:`TfidfClassifier` wrapped in Platt scaling.

    Used by the abstention analysis, where the confidence score has to mean
    something rather than merely rank correctly. Calibration is fitted by
    internal cross-validation on the training data only.

    Args:
        cv: Number of internal folds used to fit the calibrator.
        **kwargs: Forwarded to :class:`TfidfClassifier`.
    """

    def __init__(self, cv: int = 5, **kwargs) -> None:  # noqa: D107
        super().__init__(**kwargs)
        self.cv = cv
        self.name = f"{self.estimator}_calibrated"

    def _build(self):
        """Wrap the estimator half of the pipeline in a calibrator."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import make_pipeline

        base = super()._build()
        vectoriser, estimator = base[0], base[-1]
        return make_pipeline(
            vectoriser,
            CalibratedClassifierCV(estimator, cv=self.cv, method="sigmoid"),
        )


def make_classical(estimator: str = "linear_svm", **kwargs):
    """Return a factory producing fresh :class:`TfidfClassifier` instances.

    Args:
        estimator: One of :data:`ESTIMATORS`.
        **kwargs: Forwarded to the constructor.

    Returns:
        A zero-argument callable suitable for
        :func:`~ai4se.classifiers.base.train_per_repo`.
    """
    return lambda: TfidfClassifier(estimator=estimator, **kwargs)
