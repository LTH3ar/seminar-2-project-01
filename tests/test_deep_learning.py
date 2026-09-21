"""Tests for Track C configurations, splitting, and optional models."""

from __future__ import annotations

import importlib.util

import pytest

from ai4se.deep_learning import (
    AVAILABLE_DEEP_MODELS,
    CNNConfig,
    DistilBERTConfig,
    FFNNConfig,
    TrainingConfig,
    build_deep_classifier,
    make_deep_estimator_factory,
    make_validation_split,
    normalise_deep_model_name,
)
from ai4se.model import LABELS


def training_examples(per_label: int = 8) -> tuple[list[str], list[str]]:
    """Create a small balanced corpus with obvious class signals."""

    labels = [label for label in LABELS for _ in range(per_label)]
    texts = [
        f"{label} {label} issue classification signal unique {index}"
        for label in LABELS
        for index in range(per_label)
    ]
    return texts, labels


def test_deep_model_aliases_and_defaults():
    assert AVAILABLE_DEEP_MODELS == ("ffnn", "cnn", "distilbert")
    assert normalise_deep_model_name("MLP") == "ffnn"
    assert normalise_deep_model_name("text-cnn") == "cnn"
    assert normalise_deep_model_name("transformer") == "distilbert"
    assert DistilBERTConfig().model_id == "distilbert-base-uncased"
    with pytest.raises(ValueError, match="Unknown deep-learning model"):
        normalise_deep_model_name("roberta")


def test_validation_split_is_stratified_and_duplicate_safe():
    texts, labels = training_examples()
    texts.extend((texts[0], texts[8], texts[16]))
    labels.extend(LABELS)

    train_indices, validation_indices = make_validation_split(
        texts,
        labels,
        validation_fraction=0.25,
        random_state=7,
    )

    train_texts = {texts[index] for index in train_indices}
    validation_texts = {texts[index] for index in validation_indices}
    assert train_texts.isdisjoint(validation_texts)
    assert {labels[index] for index in train_indices} == set(LABELS)
    assert {labels[index] for index in validation_indices} == set(LABELS)


def test_factory_returns_independently_seeded_models():
    config = FFNNConfig(training=TrainingConfig(random_state=10))
    factory = make_deep_estimator_factory("ffnn", config=config, random_state=10)

    first = factory("owner/repository", 1)
    second = factory("owner/repository", 2)

    assert first is not second
    assert first.training.random_state == 11
    assert second.training.random_state == 12


def test_configuration_type_must_match_model():
    with pytest.raises(TypeError, match="expects CNNConfig"):
        build_deep_classifier("cnn", config=FFNNConfig())
    with pytest.raises(TypeError, match="expects DistilBERTConfig"):
        build_deep_classifier("distilbert", config=CNNConfig())


def test_missing_torch_has_actionable_error():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("PyTorch is installed")
    classifier = build_deep_classifier("cnn")
    texts, labels = training_examples()
    with pytest.raises(ImportError, match=r"pip install -e '\.\[dl\]'"):
        classifier.fit(texts, labels)


@pytest.mark.parametrize("model_name", ("ffnn", "cnn"))
def test_small_neural_models_fit_and_predict(model_name):
    pytest.importorskip("torch")
    texts, labels = training_examples()
    training = TrainingConfig(
        epochs=2,
        batch_size=8,
        validation_fraction=0.25,
        patience=None,
        random_state=3,
        device="cpu",
    )
    if model_name == "ffnn":
        config = FFNNConfig(
            training=training,
            hidden_sizes=(16,),
            max_features=100,
            min_document_frequency=1,
        )
    else:
        config = CNNConfig(
            training=training,
            embedding_dimension=8,
            filter_sizes=(2, 3),
            filters_per_size=4,
            max_sequence_length=12,
            maximum_vocabulary_size=100,
            minimum_token_frequency=1,
        )

    classifier = build_deep_classifier(model_name, config=config)
    classifier.fit(texts, labels)
    predictions = classifier.predict(texts[:4])
    probabilities = classifier.predict_proba(texts[:4])

    assert len(predictions) == len(probabilities) == 4
    assert set(predictions) <= set(LABELS)
    assert all(sum(row.values()) == pytest.approx(1.0) for row in probabilities)
    assert classifier.training_summary()["history"]["epochs_run"] == 2
