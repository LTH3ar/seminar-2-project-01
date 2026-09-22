"""Classical machine learning estimators over TF-IDF features (Chapter 3.4.2).
Five scikit-learn models:
- Multinomial Naive Bayes
- Complement Naive Bayes
- Logistic Regression (tuned and default)
- Linear Support Vector Machine (tuned and default)
- Random Forest
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


from ..model import LABELS
from .base import Classifier



ESTIMATORS = ("naive_bayes", "complement_nb", "logreg", "linear_svm", "random_forest")


class TfidfClassifier(Classifier):
    """Pipeline with TF-IDF vectorizer and scikit-learn classifier."""

    def __init__(
        self,
        estimator: str = "logreg",
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 1,
        sublinear_tf: bool = True,
        c_value: float = 1.0,
        n_estimators: int = 300,
        seed: int = 42,
    ) -> None:
        if estimator not in ESTIMATORS:
            raise ValueError(f"Unknown estimator {estimator!r}; choose from {ESTIMATORS}")

        self.name = f"tfidf_{estimator}"
        self.estimator = estimator
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.c_value = c_value
        self.n_estimators = n_estimators
        self.seed = seed
        self._pipeline = None

    def _build(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import ComplementNB, MultinomialNB
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
        )

        if self.estimator == "naive_bayes":
            clf = MultinomialNB()
        elif self.estimator == "complement_nb":
            clf = ComplementNB()
        elif self.estimator == "logreg":
            clf = LogisticRegression(
                C=self.c_value,
                max_iter=3000,
                class_weight="balanced",
                random_state=self.seed,
            )
        elif self.estimator == "linear_svm":
            clf = LinearSVC(
                C=self.c_value,
                class_weight="balanced",
                random_state=self.seed,
            )
        elif self.estimator == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            )

        return make_pipeline(vectorizer, clf)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> TfidfClassifier:
        self._pipeline = self._build()
        self._pipeline.fit(list(texts), list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError(f"{self.name} is not fitted.")
        return np.asarray(self._pipeline.predict(list(texts)), dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError(f"{self.name} is not fitted.")

        clf = self._pipeline[-1]
        if hasattr(clf, "predict_proba"):
            return np.asarray(self._pipeline.predict_proba(list(texts)), dtype=float)

        # For LinearSVC which lacks predict_proba, compute softmax over decision function
        df = self._pipeline.decision_function(list(texts))
        if df.ndim == 1:
            # Binary fallback
            df = np.column_stack([-df, df])
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        return exp_df / np.sum(exp_df, axis=1, keepdims=True)

    @property
    def classes_(self) -> np.ndarray:
        if self._pipeline is None:
            return np.asarray(LABELS, dtype=object)
        return self._pipeline[-1].classes_


# Factories matching report Table 5.2 and 5.3 configurations
def make_classical(estimator: str = "logreg", **kwargs):
    return lambda: TfidfClassifier(estimator=estimator, **kwargs)


def make_tuned_logreg():
    """Tuned Logistic Regression from Section 5.1.2: C=10, bigrams, min_df=2."""
    return lambda: TfidfClassifier(estimator="logreg", c_value=10.0, ngram_range=(1, 2), min_df=2)


def make_tuned_linear_svm():
    """Tuned Linear SVM from Section 5.1.2: C=1, bigrams, min_df=1."""
    return lambda: TfidfClassifier(estimator="linear_svm", c_value=1.0, ngram_range=(1, 2), min_df=1)
