"""Tests for the transformer classifier's contract, not its accuracy.

Fine-tuning DeBERTa-v3 needs a checkpoint download and minutes of compute, so
nothing here trains a model. What is checked is the part that corrupts results
*silently* when it breaks: that the factory resolves presets, overrides and
seeds exactly the way ``experiments/02_baselines.py`` assumes, and that an
unfitted model refuses to predict rather than returning garbage.

Run with::

    python -m pytest tests/test_transformer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4se.classifiers.base import Classifier  # noqa: E402
from ai4se.classifiers.transformer import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PRESET,
    FAST_MODEL,
    TransformerClassifier,
    make_transformer,
)

# ------------------------------------------------------------- interface --


def test_it_is_a_classifier():
    assert isinstance(TransformerClassifier(), Classifier)
    assert TransformerClassifier().name == "transformer"


def test_predicting_before_fitting_raises():
    model = TransformerClassifier()
    with pytest.raises(RuntimeError):
        model.predict(["anything"])
    with pytest.raises(RuntimeError):
        model.predict_proba(["anything"])
    with pytest.raises(RuntimeError):
        _ = model.classes_


# ---------------------------------------------------------------- presets --


def test_default_preset_is_the_preregistered_encoder():
    """H1 names DeBERTa-v3-base; the default must not drift away from it."""
    model = make_transformer()()
    assert model.model_name == DEFAULT_MODEL == DEFAULT_PRESET["model_name"]
    assert model.max_length == 512
    assert model.seed == 42


def test_fast_preset_switches_both_encoder_and_length():
    model = make_transformer(preset="fast")()
    assert model.model_name == FAST_MODEL
    assert model.max_length == 256


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError):
        make_transformer(preset="huge")


# ------------------------------------------------------- factory contract --


def test_overrides_reach_the_instance():
    model = make_transformer(learning_rate=3e-5, max_length=256)()
    assert model.learning_rate == 3e-5
    assert model.max_length == 256


def test_explicit_seed_wins_over_one_passed_as_an_override():
    """The loop seed has to win.

    02_baselines.py passes the seed explicitly while forwarding the selected
    hyper-parameters as ``**overrides``.
    """
    model = make_transformer(preset="fast", seed=43, learning_rate=3e-5)()
    assert model.seed == 43
    assert model.learning_rate == 3e-5


def test_seed_may_also_arrive_as_an_override():
    assert make_transformer(preset="fast", **{"seed": 44})().seed == 44


def test_factory_returns_a_fresh_instance_every_call():
    """Every call must hand back an untrained model.

    train_per_repo trains five of them; sharing one instance would leak across
    projects and silently invalidate the competition protocol.
    """
    factory = make_transformer(preset="fast")
    first, second = factory(), factory()
    assert first is not second
    assert first._model is None and second._model is None


def test_presets_are_not_mutated_by_overrides():
    """A factory built with overrides must not poison the next one."""
    make_transformer(learning_rate=9e-5)
    assert DEFAULT_PRESET["learning_rate"] == 2e-5
    assert make_transformer()().learning_rate == 2e-5
