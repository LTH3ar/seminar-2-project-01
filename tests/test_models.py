"""Correctness of the model tracks and the error-analysis utilities.

Track B (scikit-learn pipelines) is tested directly. Track C (torch) and
Track D2 (sentence-transformers) pull in heavy optional dependencies, so their
tests skip cleanly when those are absent -- the data and evaluation layers must
stay usable in an environment without them.

The most important test here is
:func:`test_vectorizer_is_fitted_inside_each_fold`. Fitting the vectoriser once
over the whole dataset before cross-validating leaks test-fold vocabulary and
document frequencies into training and inflates every score. It is the easiest
way to accidentally cheat at this task, and the pipeline structure is what
prevents it.

Run with::

    make test
"""

from __future__ import annotations

import pytest

from ai4se.classical import (
    CLASSICAL_MODELS,
    TUNED_PREPROCESSING,
    linear_svm,
    logistic_regression,
    naive_bayes,
    tuned_linear_svm,
    tuned_logistic_regression,
)
from ai4se.evaluation import (
    cross_validate,
    evaluate_competition,
    grid_search,
    parameterised_factory,
)
from ai4se.loader import load_split
from ai4se.model import LABELS
from ai4se.preprocessing import make_cleaner

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional extra
    HAS_TORCH = False

try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - optional extra
    HAS_SENTENCE_TRANSFORMERS = False


@pytest.fixture(scope="module")
def train():
    repository = load_split("train", kind="memory")
    repository.apply(make_cleaner(**TUNED_PREPROCESSING))
    return repository


@pytest.fixture(scope="module")
def test_split():
    repository = load_split("test", kind="memory")
    repository.apply(make_cleaner(**TUNED_PREPROCESSING))
    return repository


@pytest.fixture(scope="module")
def one_project(train):
    """A single project's issues: 300 samples, enough to train on quickly.

    Shuffled deterministically, because the dataset is stored grouped by label
    and any test that slices the list would otherwise get a single class.
    """
    import random

    from ai4se.repository import make_repository

    issues = train.by_repo("facebook/react")
    random.Random(0).shuffle(issues)
    return make_repository("memory", issues=issues)


# --------------------------------------------------------------------------- #
# Track B
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(CLASSICAL_MODELS))
def test_classical_models_satisfy_the_interface(name, one_project):
    model = CLASSICAL_MODELS[name]()
    X, y = one_project.texts_and_labels()
    model.fit(X[:150], y[:150])
    predictions = model.predict(X[150:170])
    assert len(predictions) == 20
    assert set(predictions) <= set(LABELS)


def test_every_factory_returns_a_fresh_model():
    """A factory must not hand out the same object twice, or folds share state."""
    for factory in CLASSICAL_MODELS.values():
        assert factory() is not factory()


def test_vectorizer_is_fitted_inside_each_fold(one_project):
    """The pipeline must not have seen the validation fold's vocabulary.

    This guards against the most consequential mistake available here: fitting
    TF-IDF on the whole dataset before cross-validating, which leaks test-fold
    document frequencies into training.

    The comparison uses the vectoriser's *own* analyzer rather than a naive
    whitespace split, because the two tokenise differently and comparing
    across them would compare noise.
    """
    X, y = one_project.texts_and_labels()
    train_texts, held_out_texts = X[:100], X[100:]

    pipeline = logistic_regression()
    pipeline.fit(train_texts, y[:100])
    vectorizer = pipeline.named_steps["tfidf"]
    analyze = vectorizer.build_analyzer()

    learned = set(vectorizer.vocabulary_)
    from_training = {term for text in train_texts for term in analyze(text)}
    from_held_out = {term for text in held_out_texts for term in analyze(text)}

    assert learned <= from_training, "the vocabulary contains unseen terms"
    unseen = from_held_out - from_training
    assert unseen, "fixture problem: the two halves share all vocabulary"
    assert not (unseen & learned), "validation vocabulary leaked into the fit"


def test_svm_has_no_probabilities_and_that_is_handled(one_project):
    """LinearSVC has no probability head; the harness must cope, not crash."""
    result = cross_validate(linear_svm, one_project, k=3, model_name="svm")
    assert result.mean_macro_f1 > 0.3
    assert result.mean_macro_auc is None
    assert all(s.macro_auc is None for s in result.fold_scores)


def test_probabilistic_models_report_auc(one_project):
    result = cross_validate(naive_bayes, one_project, k=3, model_name="nb")
    assert result.mean_macro_auc is not None
    assert 0.0 <= result.mean_macro_auc <= 1.0


def test_classical_models_beat_the_trivial_floor(one_project):
    """Any real model must clear 0.5/3, the balanced-majority macro F1."""
    for name, factory in CLASSICAL_MODELS.items():
        result = cross_validate(factory, one_project, k=3, model_name=name)
        assert result.mean_macro_f1 > 0.45, f"{name} barely beats guessing"


def test_tuned_configuration_is_reproducible(one_project):
    """The recorded tuned settings must match what the grid search chose.

    This test exists to fail loudly when the configuration drifts: it caught
    the re-tune that followed the preprocessing ablation, which is precisely
    the moment a silently stale setting would otherwise have survived.
    """
    logistic = tuned_logistic_regression()
    assert logistic.named_steps["tfidf"].ngram_range == (1, 2)
    assert logistic.named_steps["tfidf"].min_df == 2
    assert logistic.named_steps["clf"].C == 10.0

    svm = tuned_linear_svm()
    assert svm.named_steps["tfidf"].ngram_range == (1, 2)
    assert svm.named_steps["tfidf"].min_df == 1
    assert svm.named_steps["clf"].C == 1.0


def test_tuned_preprocessing_matches_the_ablation_winner():
    """The ablation chose `light`; `full` must not creep back as the default."""
    assert TUNED_PREPROCESSING["level"] == "light"
    assert TUNED_PREPROCESSING["title_weight"] == 3


def test_parameterised_factory_does_not_capture_by_reference():
    """The classic loop-variable bug: every config must keep its own value."""
    factories = [parameterised_factory(linear_svm, C=c) for c in (0.5, 1.0, 2.0)]
    assert [f().named_steps["clf"].C for f in factories] == [0.5, 1.0, 2.0]


def test_grid_search_ranks_configurations(one_project):
    frame = grid_search(linear_svm, {"C": [0.1, 1.0]}, one_project, k=3)
    assert len(frame) == 2
    assert list(frame.columns[:1]) == ["C"]
    assert frame["macro_f1"].is_monotonic_decreasing, "results must be sorted"


# --------------------------------------------------------------------------- #
# Track C
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("architecture", ["ffnn", "cnn"])
def test_neural_models_train_and_predict(architecture, one_project):
    from ai4se.neural import CNNTextClassifier, FeedForwardClassifier

    cls = FeedForwardClassifier if architecture == "ffnn" else CNNTextClassifier
    X, y = one_project.texts_and_labels()
    model = cls(epochs=4).fit(X[:200], y[:200])

    predictions = model.predict(X[200:220])
    assert len(predictions) == 20
    assert set(predictions) <= set(LABELS)

    for row in model.predict_proba(X[200:205]):
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_neural_models_record_learning_curves(one_project):
    """The course requires per-epoch training and validation loss."""
    from ai4se.neural import FeedForwardClassifier

    X, y = one_project.texts_and_labels()
    model = FeedForwardClassifier(epochs=5, patience=None).fit(X, y)
    history = model.history_

    assert len(history.train_loss) == 5
    assert len(history.val_loss) == 5
    assert len(history.val_accuracy) == 5
    assert 1 <= history.best_epoch <= 5
    assert history.train_loss[-1] < history.train_loss[0], "training loss must fall"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_early_stopping_triggers_and_restores_best_weights(one_project):
    from ai4se.neural import FeedForwardClassifier

    X, y = one_project.texts_and_labels()
    model = FeedForwardClassifier(epochs=100, patience=2).fit(X, y)
    assert model.history_.stopped_early, "100 epochs on 300 samples must stop early"
    assert len(model.history_.train_loss) < 100
    # The restored weights come from the best epoch, not the last.
    assert model.history_.best_epoch <= len(model.history_.val_loss)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_neural_training_is_deterministic(one_project):
    from ai4se.neural import FeedForwardClassifier

    X, y = one_project.texts_and_labels()
    first = FeedForwardClassifier(epochs=3, seed=7).fit(X[:150], y[:150])
    second = FeedForwardClassifier(epochs=3, seed=7).fit(X[:150], y[:150])
    assert first.predict(X[150:180]) == second.predict(X[150:180])


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_validation_split_is_stratified(one_project):
    """Every class must appear in the held-out validation data."""
    import torch

    from ai4se.neural import FeedForwardClassifier

    _, y = one_project.texts_and_labels()
    model = FeedForwardClassifier(validation_fraction=0.2)
    model.classes_ = sorted(set(y))
    index_of = {label: i for i, label in enumerate(model.classes_)}
    targets = torch.tensor([index_of[label] for label in y])

    train_idx, val_idx = model._split(targets)
    assert not set(train_idx) & set(val_idx)
    assert len(train_idx) + len(val_idx) == len(y)
    held_out = {model.classes_[targets[i]] for i in val_idx}
    assert held_out == set(model.classes_)


# --------------------------------------------------------------------------- #
# Track D2
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed"
)
def test_embedding_cache_avoids_recomputation():
    from ai4se.embeddings import clear_embedding_cache, embed

    clear_embedding_cache()
    texts = ["a crash on startup", "please add a dark theme"]
    first = embed(texts)
    second = embed(texts)
    assert first.shape == second.shape == (2, first.shape[1])
    assert (first == second).all(), "cached embeddings must be identical"


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed"
)
def test_frozen_embedding_classifier_round_trip():
    from ai4se.embeddings import FrozenEmbeddingClassifier

    X = [
        "the application crashes on startup",
        "segmentation fault when loading",
        "please add support for dark mode",
        "it would be nice to have plugins",
        "how do I configure the build",
        "what does this option mean",
    ]
    y = ["bug", "bug", "feature", "feature", "question", "question"]
    model = FrozenEmbeddingClassifier().fit(X, y)

    assert set(model.predict(X)) <= set(y)
    for row in model.predict_proba(X[:2]):
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-5)


def test_setfit_reports_a_useful_error_without_the_extra():
    """Missing the optional dependency must explain how to install it."""
    from ai4se.embeddings import SetFitClassifier

    try:
        import setfit  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"\[dl\]"):
            SetFitClassifier().fit(["a"], ["bug"])
    else:  # pragma: no cover - only where setfit is installed
        pytest.skip("setfit is installed, so the error path cannot be exercised")


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def predictions(train, test_split):
    from ai4se import error_analysis as ea

    return ea.collect_predictions(tuned_logistic_regression, train, test_split)


def test_collect_predictions_covers_every_repository(predictions):
    assert len(predictions) == 5
    for payload in predictions.values():
        assert len(payload["issues"]) == 300
        assert len(payload["y_true"]) == len(payload["y_pred"]) == 300


def test_collected_predictions_agree_with_the_harness(train, test_split, predictions):
    """Error analysis must describe the same run the leaderboard reports."""
    from ai4se import error_analysis as ea

    harness = evaluate_competition(
        tuned_logistic_regression, train, test_split, model_name="lr"
    )
    frame = ea.per_repository_scores(predictions)
    for repo, scores in harness.per_repository.items():
        assert frame.loc[repo, "F1 avg"] == pytest.approx(scores.macro_f1, abs=1e-4)


def test_confusion_summary_accounts_for_every_error(predictions):
    from ai4se import error_analysis as ea

    frame = ea.confusion_summary(predictions)
    counted = frame["count"].sum()
    actual = sum(
        1
        for payload in predictions.values()
        for a, b in zip(payload["y_true"], payload["y_pred"], strict=True)
        if a != b
    )
    assert counted == actual
    assert frame["share of errors %"].sum() == pytest.approx(100.0, abs=0.5)


def test_worst_mistakes_are_all_genuine_errors(predictions):
    from ai4se import error_analysis as ea

    mistakes = ea.worst_mistakes(predictions, n=15)
    assert len(mistakes) == 15
    assert all(m.true_label != m.predicted_label for m in mistakes)
    confidences = [m.confidence for m in mistakes if m.confidence is not None]
    assert confidences == sorted(confidences, reverse=True), "must rank by confidence"


def test_errors_by_length_partitions_the_test_set(predictions):
    from ai4se import error_analysis as ea

    frame = ea.errors_by_length(predictions, bins=5)
    assert frame["issues"].sum() == 1500
    assert ((frame["accuracy"] >= 0) & (frame["accuracy"] <= 1)).all()


# --------------------------------------------------------------------------- #
# Ensembles
# --------------------------------------------------------------------------- #


def _light_ensemble():
    """A two-member ensemble whose members need only scikit-learn."""
    from ai4se.classical import naive_bayes
    from ai4se.ensemble import SoftVotingEnsemble

    return SoftVotingEnsemble(
        {"lr": tuned_logistic_regression, "nb": naive_bayes}
    )


def test_ensemble_averages_member_probabilities(one_project):
    from ai4se.classical import naive_bayes

    X, y = one_project.texts_and_labels()
    ensemble = _light_ensemble().fit(X[:200], y[:200])

    members = {
        "lr": tuned_logistic_regression().fit(X[:200], y[:200]),
        "nb": naive_bayes().fit(X[:200], y[:200]),
    }
    from ai4se.evaluation import class_probabilities

    combined = ensemble.predict_proba(X[200:210])
    for index in range(10):
        for label in LABELS:
            expected = sum(
                class_probabilities(m, X[200:210], LABELS)[index][label]
                for m in members.values()
            ) / 2
            assert combined[index][label] == pytest.approx(expected)


def test_ensemble_probabilities_are_normalised(one_project):
    X, y = one_project.texts_and_labels()
    ensemble = _light_ensemble().fit(X[:200], y[:200])
    for row in ensemble.predict_proba(X[200:210]):
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-6)
        assert set(row) == set(LABELS)


def test_ensemble_rejects_a_single_member():
    from ai4se.classical import naive_bayes
    from ai4se.ensemble import SoftVotingEnsemble

    with pytest.raises(ValueError, match="at least two"):
        SoftVotingEnsemble({"only": naive_bayes})


def test_ensemble_rejects_a_member_without_probabilities(one_project):
    """LinearSVC has no probability head, so soft voting must refuse it."""
    from ai4se.classical import linear_svm
    from ai4se.ensemble import SoftVotingEnsemble

    X, y = one_project.texts_and_labels()
    ensemble = SoftVotingEnsemble(
        {"svm": linear_svm, "lr": tuned_logistic_regression}
    ).fit(X[:150], y[:150])

    with pytest.raises(TypeError, match="predict_proba"):
        ensemble.predict(X[150:160])


def test_weighted_ensemble_validates_its_weights():
    from ai4se.classical import naive_bayes
    from ai4se.ensemble import WeightedVotingEnsemble

    members = {"a": naive_bayes, "b": tuned_logistic_regression}
    with pytest.raises(ValueError, match="do not match"):
        WeightedVotingEnsemble(members, {"a": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        WeightedVotingEnsemble(members, {"a": 1.0, "b": -1.0})


def test_weighting_a_member_to_zero_reproduces_the_other(one_project):
    """A sanity check on the weighting arithmetic."""
    from ai4se.classical import naive_bayes
    from ai4se.ensemble import WeightedVotingEnsemble

    X, y = one_project.texts_and_labels()
    weighted = WeightedVotingEnsemble(
        {"lr": tuned_logistic_regression, "nb": naive_bayes},
        {"lr": 1.0, "nb": 0.0},
    ).fit(X[:200], y[:200])

    alone = tuned_logistic_regression().fit(X[:200], y[:200])
    assert weighted.predict(X[200:220]) == list(alone.predict(X[200:220]))


def test_setfit_exposes_the_memory_controls():
    """The two knobs that decide whether a run fits on a given GPU.

    Added after MPNet hit CUDA OOM at MiniLM's defaults: contrastive training
    embeds both sentences of every pair, and MPNet is twelve layers of width
    768 against MiniLM's six of 384, so the same settings need several times
    the memory. ``max_seq_length`` is the dominant term, since attention cost
    grows with its square.
    """
    from ai4se.embeddings import STRONG_ENCODER, SetFitClassifier, setfit_mpnet

    default = SetFitClassifier()
    assert default.max_seq_length == 128, "the default must cap sequence length"
    assert default.batch_size == 16

    mpnet = setfit_mpnet()
    assert mpnet.model_name == STRONG_ENCODER
    assert mpnet.batch_size <= 8, "MPNet needs a smaller batch than MiniLM"
    assert mpnet.max_seq_length is not None

    # None must stay available, to fall back on the encoder's own default.
    assert SetFitClassifier(max_seq_length=None).max_seq_length is None


def test_free_gpu_memory_is_safe_without_a_gpu():
    """Called between the five per-project fits; must never raise."""
    from ai4se.embeddings import free_gpu_memory

    free_gpu_memory()
    free_gpu_memory()


def test_setfit_factories_pin_the_settings_that_produced_their_results():
    """Each recorded result must stay reproducible from its own factory.

    ``setfit_minilm`` produced 0.7982 with ``batch_size=16`` and the encoder's
    own ``max_seq_length`` of 256. The class default was later lowered to 128
    so MPNet would fit in 12 GB — which would silently have changed what that
    factory builds. Pinning both values explicitly keeps the saved number
    reproducible, and this test fails if the pin is removed.
    """
    from ai4se.embeddings import (
        DEFAULT_ENCODER,
        STRONG_ENCODER,
        setfit_minilm,
        setfit_minilm_matched,
        setfit_mpnet,
    )

    original = setfit_minilm()
    assert original.model_name == DEFAULT_ENCODER
    assert original.batch_size == 16
    assert original.max_seq_length is None, "must use the encoder's own 256"

    matched = setfit_minilm_matched()
    mpnet = setfit_mpnet()
    assert matched.model_name == DEFAULT_ENCODER
    assert mpnet.model_name == STRONG_ENCODER
    # The whole point of the matched variant: identical settings, different
    # encoder, so a difference is attributable to the encoder alone.
    assert matched.batch_size == mpnet.batch_size
    assert matched.max_seq_length == mpnet.max_seq_length


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_set_seed_requests_deterministic_kernels():
    """The determinism flags must stay set, as insurance rather than a fix.

    Measured effect on this project: none. The CNN returned 0.7438 with and
    without them, so training was already deterministic on a fixed device.
    They are kept so that a future change introducing a nondeterministic
    kernel cannot silently make runs irreproducible.

    They do not reconcile devices: the CNN scores 0.7578 on a CPU and 0.7438
    on a GPU from identical code and seed, because the two backends use
    different kernels and accumulation orders.
    """
    import torch

    from ai4se.neural import set_seed

    set_seed(123)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    # Seeding must also cover Python and NumPy, which the data-order shuffles use.
    import random

    set_seed(7)
    first = [random.random() for _ in range(3)]
    set_seed(7)
    assert [random.random() for _ in range(3)] == first
