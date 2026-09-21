"""Tests for the Track B classical machine-learning pipelines."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from ai4se.classical import (  # noqa: E402
    AVAILABLE_MODELS,
    TfidfConfig,
    build_classical_pipeline,
    classifier_parameters,
    make_classical_estimator_factory,
    normalise_model_name,
)
from ai4se.evaluation import evaluate_cross_validation  # noqa: E402
from ai4se.model import LABELS, IssueReport  # noqa: E402


@pytest.fixture
def small_tfidf() -> TfidfConfig:
    """Use a compact representation suitable for fast unit tests."""

    return TfidfConfig(
        min_document_frequency=1,
        word_max_features=500,
        character_max_features=500,
    )


def training_examples() -> tuple[list[str], list[str]]:
    """Create an easily separable balanced text dataset."""

    labels = [label for label in LABELS for _ in range(6)]
    texts = [
        f"{label} {label} classification signal report number {index}"
        for label in LABELS
        for index in range(6)
    ]
    return texts, labels


@pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
def test_every_classical_pipeline_fits_and_predicts(model_name, small_tfidf):
    """All four required model families satisfy the shared contract."""

    overrides = (
        {"n_estimators": 20, "n_jobs": 1}
        if model_name == "random_forest"
        else None
    )
    model = build_classical_pipeline(
        model_name,
        tfidf=small_tfidf,
        classifier_overrides=overrides,
    )
    texts, labels = training_examples()

    predictions = model.fit(texts, labels).predict(texts)

    assert len(predictions) == len(labels)
    assert set(predictions) <= set(LABELS)
    assert list(model.named_steps) == ["tfidf", "classifier"]


def test_estimator_factory_returns_fresh_models(small_tfidf):
    """Every repository and fold must receive an unfitted estimator."""

    factory = make_classical_estimator_factory(
        "svm",
        tfidf=small_tfidf,
        random_state=10,
    )

    first = factory("owner/one", 1)
    second = factory("owner/one", 2)

    assert first is not second
    assert first.named_steps["classifier"].random_state == 11
    assert second.named_steps["classifier"].random_state == 12


def test_naive_bayes_integrates_with_grouped_cross_validation(small_tfidf):
    """Track B models run unchanged through Track D's evaluator."""

    issues = [
        IssueReport(
            repo=repository,
            created_at=f"2024-02-{index + 1:02d}",
            label=label,
            title=f"{label} {label} title {index}",
            body=f"{label} classification signal {repository} unique {index}",
            clean_text=f"{label} {label} classification signal unique {index}",
        )
        for repository in ("owner/one", "owner/two")
        for label in LABELS
        for index in range(6)
    ]

    result = evaluate_cross_validation(
        issues,
        make_classical_estimator_factory("nb", tfidf=small_tfidf),
        model_name="test-naive-bayes",
        n_splits=3,
    )

    assert result["overall"]["sample_count"] == len(issues)
    assert result["overall"]["repository_average"]["macro_average"]["f1"] == 1


def test_aliases_and_parameters_are_reproducible():
    """Short CLI aliases resolve to explicit versioned configurations."""

    assert normalise_model_name("LR") == "logistic_regression"
    assert normalise_model_name("rf") == "random_forest"
    assert classifier_parameters("svm", random_state=7)["random_state"] == 7
    with pytest.raises(ValueError, match="Unknown classical model"):
        normalise_model_name("transformer")
