"""Track B -- classical machine learning pipelines.

Four families, all over the same TF-IDF representation so that differences in
score come from the classifier rather than from the features:

============================  ==================================================
Naive Bayes                   probabilistic, based on Bayes' theorem; the
                              standard first choice for text classification
Logistic regression           linear, probabilistic, strong on sparse
                              high-dimensional text
Support vector machine        finds the hyperplane that best separates the
                              classes; historically the strongest linear text
                              classifier
Random forest                 an ensemble of decision trees, included as the
                              non-linear counterpoint
============================  ==================================================

Every factory returns a scikit-learn ``Pipeline``, so vectorisation is fitted
**inside** each cross-validation fold. Fitting the vectoriser once over the
whole dataset and then cross-validating would leak test-fold vocabulary and
document frequencies into training, and would inflate every reported score.
This is the single most common way of accidentally cheating at this task.

All factories are zero-argument callables, as
:mod:`ai4se.evaluation` requires, and produce a fresh untrained model each
time.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .evaluation import RANDOM_SEED, parameterised_factory

#: Default vectoriser settings, shared by every pipeline.
#:
#: ``ngram_range=(1, 2)`` captures short phrases such as "does not" or "would
#: be", which the unigram model splits apart and which matter for separating
#: feature requests from bug reports. ``min_df=2`` drops terms occurring in a
#: single document: with only 300 training issues per project those terms are
#: noise, and dropping them roughly halves the feature count.
VECTORIZER_DEFAULTS: dict = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}


def make_vectorizer(**overrides) -> TfidfVectorizer:
    """Build the shared TF-IDF vectoriser, with optional overrides."""
    settings = {**VECTORIZER_DEFAULTS, **overrides}
    return TfidfVectorizer(**settings)


def naive_bayes(alpha: float = 0.3, **vectorizer_kwargs) -> Pipeline:
    """TF-IDF + multinomial naive Bayes.

    Args:
        alpha: Laplace/Lidstone smoothing. Values below 1 work better on TF-IDF
            than on raw counts, because the features are already damped.
        **vectorizer_kwargs: Overrides forwarded to :func:`make_vectorizer`.
    """
    return Pipeline(
        [
            ("tfidf", make_vectorizer(**vectorizer_kwargs)),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )


def complement_naive_bayes(alpha: float = 0.3, **vectorizer_kwargs) -> Pipeline:
    """TF-IDF + complement naive Bayes, a variant designed for text.

    Args:
        alpha: Laplace/Lidstone smoothing.
        **vectorizer_kwargs: Overrides forwarded to :func:`make_vectorizer`.
    """
    return Pipeline(
        [
            ("tfidf", make_vectorizer(**vectorizer_kwargs)),
            ("clf", ComplementNB(alpha=alpha)),
        ]
    )


def logistic_regression(C: float = 5.0, **vectorizer_kwargs) -> Pipeline:  # noqa: N803
    """TF-IDF + multinomial logistic regression.

    Args:
        C: Inverse regularisation strength; larger means weaker regularisation.
        **vectorizer_kwargs: Overrides forwarded to :func:`make_vectorizer`.
    """
    return Pipeline(
        [
            ("tfidf", make_vectorizer(**vectorizer_kwargs)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def linear_svm(C: float = 1.0, **vectorizer_kwargs) -> Pipeline:  # noqa: N803
    """TF-IDF + linear support vector machine.

    ``LinearSVC`` has no probability head, so this model is evaluated on its
    hard labels and its AUC fields stay ``None``. That is deliberate: wrapping
    it in probability calibration would change the model being measured.

    Args:
        C: Regularisation parameter.
        **vectorizer_kwargs: Overrides forwarded to :func:`make_vectorizer`.
    """
    return Pipeline(
        [
            ("tfidf", make_vectorizer(**vectorizer_kwargs)),
            (
                "clf",
                LinearSVC(
                    C=C,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def random_forest(
    n_estimators: int = 300,
    max_features: str = "sqrt",
    **vectorizer_kwargs,
) -> Pipeline:
    """TF-IDF + random forest, an ensemble of decision trees.

    Included as the non-linear counterpoint to the three linear models. Trees
    cope poorly with the very high-dimensional sparse features TF-IDF
    produces, so a weaker result here is a finding rather than a bug.

    Args:
        n_estimators: Number of trees.
        max_features: Features considered at each split.
        **vectorizer_kwargs: Overrides forwarded to :func:`make_vectorizer`.
    """
    return Pipeline(
        [
            ("tfidf", make_vectorizer(**vectorizer_kwargs)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


#: Zero-argument factories for every Track B model, ready for the evaluation
#: runners. Each call returns a fresh, untrained pipeline.
CLASSICAL_MODELS: dict[str, Callable[[], Pipeline]] = {
    "TF-IDF + Naive Bayes": naive_bayes,
    "TF-IDF + Complement NB": complement_naive_bayes,
    "TF-IDF + Logistic Regression": logistic_regression,
    "TF-IDF + Linear SVM": linear_svm,
    "TF-IDF + Random Forest": random_forest,
}


#: Configuration chosen by the ablation in notebook 03, used for the final
#: Track B numbers. Two results drove it:
#:
#: * ``level="light"`` beats ``level="full"`` by roughly 0.015 macro F1. Stop
#:   word removal and lemmatisation destroy signal that TF-IDF exploits --
#:   modal words such as "would" and "could" are stop words, and they are
#:   exactly what separates a feature request from a bug report.
#: * ``min_df=1`` beats ``min_df=2``. With 300 training issues per project,
#:   a term occurring once is still informative more often than it is noise.
TUNED_PREPROCESSING: dict = {
    "level": "light",
    "title_weight": 3,
    "max_words": 200,
}

#: Vectoriser settings shared by the tuned models. Bigrams win consistently:
#: they are the only setting whose advantage exceeds the fold-to-fold spread.
TUNED_VECTORIZER: dict = {
    "ngram_range": (1, 2),
}


def tuned_logistic_regression() -> Pipeline:
    """Logistic regression at the settings chosen by the grid search.

    Selected under ``TUNED_PREPROCESSING`` -- which matters, because the grid
    was first run under ``level="full"`` and the winning ``min_df`` changed
    once the cleaning level changed. Hyperparameters and preprocessing are not
    independent, and re-tuning after altering the pipeline is not optional.
    """
    return logistic_regression(C=10.0, min_df=2, **TUNED_VECTORIZER)


def tuned_linear_svm() -> Pipeline:
    """Linear SVM at the settings chosen by the grid search.

    Note it prefers ``min_df=1`` where logistic regression prefers ``min_df=2``.
    The gap between the two configurations is about 0.002 macro F1 against a
    fold-to-fold standard deviation of roughly 0.029, so this is a coin-flip
    rather than a finding, and the report should say so.
    """
    return linear_svm(C=1.0, min_df=1, **TUNED_VECTORIZER)


#: The Track B models at their tuned settings. Use with ``TUNED_PREPROCESSING``.
TUNED_MODELS: dict[str, Callable[[], Pipeline]] = {
    "TF-IDF + Logistic Regression (tuned)": tuned_logistic_regression,
    "TF-IDF + Linear SVM (tuned)": tuned_linear_svm,
}


#: Re-exported for convenience: freezes keyword arguments into a zero-argument
#: factory, which is what the evaluation runners and :func:`grid_search` expect.
parameterised = parameterised_factory
