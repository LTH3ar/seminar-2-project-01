"""Classical TF-IDF classifiers for Track B.

Every returned estimator is a complete scikit-learn pipeline. TF-IDF is fitted
inside :meth:`fit`, so the shared cross-validation evaluator cannot leak
vocabulary or inverse-document-frequency statistics across folds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

AVAILABLE_MODELS: tuple[str, ...] = (
    "naive_bayes",
    "logistic_regression",
    "linear_svm",
    "random_forest",
)

MODEL_DISPLAY_NAMES = {
    "naive_bayes": "TF-IDF + Complement Naive Bayes",
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "random_forest": "TF-IDF + Random Forest",
}

_MODEL_ALIASES = {
    "nb": "naive_bayes",
    "naive-bayes": "naive_bayes",
    "naive_bayes": "naive_bayes",
    "lr": "logistic_regression",
    "logistic-regression": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "svm": "linear_svm",
    "linear-svm": "linear_svm",
    "linear_svm": "linear_svm",
    "rf": "random_forest",
    "random-forest": "random_forest",
    "random_forest": "random_forest",
}


@dataclass(frozen=True, slots=True)
class TfidfConfig:
    """Configuration shared by all Track B text representations."""

    word_ngram_range: tuple[int, int] = (1, 2)
    character_ngram_range: tuple[int, int] = (3, 5)
    min_document_frequency: int = 2
    max_document_frequency: float = 0.98
    word_max_features: int | None = 30_000
    character_max_features: int | None = 20_000
    use_character_ngrams: bool = True
    character_weight: float = 0.5
    sublinear_term_frequency: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable experiment metadata."""

        return asdict(self)


def normalise_model_name(model_name: str) -> str:
    """Resolve a short alias to one canonical Track B model name."""

    key = model_name.strip().lower()
    try:
        return _MODEL_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(AVAILABLE_MODELS)
        raise ValueError(
            f"Unknown classical model {model_name!r}; choose one of: {choices}"
        ) from exc


def model_display_name(model_name: str) -> str:
    """Return the report-friendly name of a classical model."""

    return MODEL_DISPLAY_NAMES[normalise_model_name(model_name)]


def classifier_parameters(
    model_name: str,
    *,
    random_state: int = 42,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the reproducible classifier parameters for one model."""

    canonical_name = normalise_model_name(model_name)
    defaults: dict[str, dict[str, Any]] = {
        "naive_bayes": {"alpha": 0.25},
        "logistic_regression": {
            "C": 4.0,
            "max_iter": 2_000,
            "solver": "lbfgs",
            "random_state": random_state,
        },
        "linear_svm": {
            "C": 1.5,
            "dual": True,
            "random_state": random_state,
        },
        "random_forest": {
            "n_estimators": 400,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": random_state,
        },
    }
    parameters = dict(defaults[canonical_name])
    parameters.update(overrides or {})
    return parameters


def build_classical_pipeline(
    model_name: str,
    *,
    tfidf: TfidfConfig | None = None,
    random_state: int = 42,
    classifier_overrides: dict[str, Any] | None = None,
):
    """Build a fresh word/character TF-IDF classification pipeline.

    Imports are intentionally local so Track A remains usable when the
    optional ``ml`` dependency group has not been installed.
    """

    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import ComplementNB
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise ImportError(
            "Track B requires scikit-learn; install it with "
            "`python -m pip install -e \".[ml]\"`"
        ) from exc

    canonical_name = normalise_model_name(model_name)
    config = tfidf or TfidfConfig()
    vectorizers: list[tuple[str, Any]] = [
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=config.word_ngram_range,
                min_df=config.min_document_frequency,
                max_df=config.max_document_frequency,
                max_features=config.word_max_features,
                sublinear_tf=config.sublinear_term_frequency,
                strip_accents="unicode",
                dtype=np.float32,
            ),
        )
    ]
    transformer_weights = {"word_tfidf": 1.0}
    if config.use_character_ngrams:
        vectorizers.append(
            (
                "character_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=config.character_ngram_range,
                    min_df=config.min_document_frequency,
                    max_features=config.character_max_features,
                    sublinear_tf=config.sublinear_term_frequency,
                    dtype=np.float32,
                ),
            )
        )
        transformer_weights["character_tfidf"] = config.character_weight

    parameters = classifier_parameters(
        canonical_name,
        random_state=random_state,
        overrides=classifier_overrides,
    )
    classifiers = {
        "naive_bayes": ComplementNB,
        "logistic_regression": LogisticRegression,
        "linear_svm": LinearSVC,
        "random_forest": RandomForestClassifier,
    }
    return Pipeline(
        steps=(
            (
                "tfidf",
                FeatureUnion(
                    transformer_list=vectorizers,
                    transformer_weights=transformer_weights,
                ),
            ),
            ("classifier", classifiers[canonical_name](**parameters)),
        )
    )


def make_classical_estimator_factory(
    model_name: str,
    *,
    tfidf: TfidfConfig | None = None,
    random_state: int = 42,
    classifier_overrides: dict[str, Any] | None = None,
):
    """Return a factory compatible with :mod:`ai4se.evaluation`.

    The fold number is added to the base seed. This keeps runs reproducible
    while ensuring stochastic estimators do not reuse an identical random
    sequence in every fold.
    """

    canonical_name = normalise_model_name(model_name)

    def estimator_factory(repository: str, fold: int):
        """Create one independent pipeline for a repository/fold pair."""

        del repository
        return build_classical_pipeline(
            canonical_name,
            tfidf=tfidf,
            random_state=random_state + fold,
            classifier_overrides=classifier_overrides,
        )

    return estimator_factory
