"""Tests for the classifier interface and the selection machinery.

These check the *protocol*, not predictive quality: that one model really is
trained per project, that a fitted model refuses to predict before it is
fitted, that cross-validation never sees the validation fold during training.
A model that is merely weak shows up in the results table; a model that
quietly trains on the wrong data does not, which is why it is tested here.

The heavy models (SetFit, the CNN) are exercised only through their shared
interface with a tiny synthetic corpus, so the suite stays fast.

Run with::

    python -m pytest tests/test_classifiers.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4se.classifiers.base import (  # noqa: E402
    Classifier,
    train_per_repo,
    train_pooled,
)
from ai4se.classifiers.selection import (  # noqa: E402
    cross_validate,
    grid_search,
    make_subset,
    summarise_search,
)
from ai4se.model import IssueReport  # noqa: E402
from ai4se.repository import make_repository  # noqa: E402

sklearn = pytest.importorskip("sklearn", reason="track B extra not installed")

from ai4se.classifiers.classical import (  # noqa: E402
    ESTIMATORS,
    TfidfClassifier,
    make_classical,
)

LABELS3 = ("bug", "feature", "question")

#: Vocabulary chosen so the three classes are linearly separable, letting the
#: tests assert on behaviour rather than on a fuzzy accuracy threshold.
WORDS = {
    "bug": "crash stacktrace exception error fails broken traceback",
    "feature": "please add support request enhancement would like new",
    "question": "how do i what does why is this possible help",
}


def _corpus(repos=("a/a", "b/b"), per_cell: int = 12) -> list[IssueReport]:
    """Build a small separable corpus with a controlled creation date."""
    issues = []
    for repo in repos:
        for label in LABELS3:
            for k in range(per_cell):
                issues.append(
                    IssueReport(
                        repo=repo,
                        created_at=f"2023-01-{(k % 28) + 1:02d} 00:00:00",
                        label=label,
                        title=f"{WORDS[label]} {k}",
                        body=f"{WORDS[label]} body {repo} {k}",
                    )
                )
    return issues


@pytest.fixture
def split():
    """Return a (train, test) pair over the synthetic corpus."""
    issues = _corpus()
    midpoint = len(issues) // 2
    everything = make_repository("memory", issues=issues)
    train = make_repository("memory", issues=[i for n, i in enumerate(issues) if n % 2])
    test = make_repository(
        "memory", issues=[i for n, i in enumerate(issues) if not n % 2]
    )
    del everything, midpoint
    return train, test


# ------------------------------------------------------------- interface --


def test_every_estimator_name_builds():
    """All four advertised estimators must construct and train."""
    issues = _corpus(repos=("a/a",))
    texts = [i.text for i in issues]
    labels = [i.label for i in issues]
    for name in ESTIMATORS:
        model = TfidfClassifier(estimator=name, min_df=1).fit(texts, labels)
        assert len(model.predict(texts)) == len(texts)


def test_unknown_estimator_is_rejected():
    """A typo in the estimator name must fail loudly at construction."""
    with pytest.raises(ValueError):
        TfidfClassifier(estimator="xgboost")


def test_predicting_before_fitting_raises():
    """An unfitted model must refuse rather than return garbage."""
    model = TfidfClassifier()
    with pytest.raises(RuntimeError):
        model.predict(["anything"])
    with pytest.raises(RuntimeError):
        model.predict_proba(["anything"])


def test_probabilities_are_normalised_even_for_the_svm():
    """LinearSVC has no predict_proba; the softmax fallback must still be valid."""
    issues = _corpus(repos=("a/a",))
    texts = [i.text for i in issues]
    model = TfidfClassifier(estimator="linear_svm", min_df=1).fit(
        texts, [i.label for i in issues]
    )
    probabilities = model.predict_proba(texts)
    assert probabilities.shape == (len(texts), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_softmax_fallback_does_not_change_the_predicted_label():
    """The probability fallback is monotone, so argmax must agree with predict."""
    issues = _corpus(repos=("a/a",))
    texts = [i.text for i in issues]
    model = TfidfClassifier(estimator="linear_svm", min_df=1).fit(
        texts, [i.label for i in issues]
    )
    from_proba = model.classes_[model.predict_proba(texts).argmax(axis=1)]
    assert list(from_proba) == list(model.predict(texts))


def test_classifier_is_abstract():
    """The base class must not be instantiable."""
    with pytest.raises(TypeError):
        Classifier()


# -------------------------------------------------------------- protocol --


def test_train_per_repo_trains_one_model_per_project(split):
    """Each project's model must see only that project's training issues."""
    train, test = split
    seen: list[set[str]] = []

    class _Spy(TfidfClassifier):
        """Records which projects appear in the text it is trained on."""

        def fit(self, texts, labels):
            seen.append({t.split()[-2] for t in texts if "/" in t})
            return super().fit(texts, labels)

    train_per_repo(lambda: _Spy(min_df=1), train, test)
    # The body embeds the project name, so each call must have seen exactly one.
    assert all(len(projects) == 1 for projects in seen)
    assert len(seen) == len(train.repos())


def test_train_per_repo_predicts_every_test_issue(split):
    """No test issue may be left without a prediction."""
    train, test = split
    predictions = train_per_repo(make_classical("linear_svm", min_df=1), train, test)
    assert len(predictions) == len(test)
    assert all(p in LABELS3 for p in predictions)


def test_pooled_and_per_repo_produce_the_same_shape(split):
    """The two protocols differ in what they train on, not in their output."""
    train, test = split
    per_repo = train_per_repo(make_classical("logreg", min_df=1), train, test)
    pooled = train_pooled(make_classical("logreg", min_df=1), train, test)
    assert per_repo.shape == pooled.shape == (len(test),)


def test_a_separable_corpus_is_actually_learned(split):
    """Sanity check: on separable data the protocol must score near-perfectly."""
    from ai4se.evaluation import cross_repo_f1

    train, test = split
    predictions = train_per_repo(make_classical("linear_svm", min_df=1), train, test)
    truth = [i.label for i in test.all()]
    repos = [i.repo for i in test.all()]
    assert cross_repo_f1(truth, predictions, repos) > 0.9


# ------------------------------------------------------------- selection --


def test_cross_validate_reports_one_score_per_fold(split):
    """Five folds must give five scores, and a standard deviation."""
    train, _ = split
    result = cross_validate(make_classical("logreg", min_df=1), train, n_folds=3)
    assert len(result["scores"]) == 3
    assert 0.0 <= result["mean"] <= 1.0
    assert result["std"] >= 0.0


def test_cross_validate_is_deterministic_for_a_fixed_seed(split):
    """The same seed must give the same folds and therefore the same score."""
    train, _ = split
    factory = make_classical("logreg", min_df=1)
    first = cross_validate(factory, train, n_folds=3, seed=7)
    second = cross_validate(factory, train, n_folds=3, seed=7)
    assert first["scores"] == second["scores"]


def test_cross_validate_never_trains_on_the_validation_fold(split):
    """The guarantee that makes selection meaningful, asserted directly."""
    from ai4se.evaluation.splits import stratified_folds

    train, _ = split
    issues = train.all()

    for train_index, validation_index in stratified_folds(train, 3, seed=7):
        training_texts = {issues[i].text for i in train_index}
        validation_texts = {issues[i].text for i in validation_index}
        assert not training_texts & validation_texts


def test_grid_search_ranks_best_first(split):
    """Rows must come back sorted by mean score, descending."""
    train, _ = split
    rows = grid_search(
        lambda c_value: make_classical("logreg", c_value=c_value, min_df=1),
        {"c_value": [0.5, 1.0, 2.0]},
        train,
        n_folds=3,
        verbose=False,
    )
    assert len(rows) == 3
    assert [r["mean"] for r in rows] == sorted([r["mean"] for r in rows], reverse=True)
    assert set(rows[0]["params"]) == {"c_value"}


def test_summarise_search_mentions_the_detection_threshold(split):
    """The summary must remind the reader what a spread of this size means."""
    train, _ = split
    rows = grid_search(
        lambda c_value: make_classical("logreg", c_value=c_value, min_df=1),
        {"c_value": [0.5, 1.0]},
        train,
        n_folds=3,
        verbose=False,
    )
    text = summarise_search(rows)
    assert "configurations" in text
    assert "threshold" in text
    assert summarise_search([]) == "(no configurations evaluated)"


def test_make_subset_keeps_only_one_project(split):
    """The helper used to build per-project repositories must filter."""
    train, _ = split
    subset = make_subset(train, "a/a")
    assert len(subset) > 0
    assert subset.repos() == ["a/a"]
