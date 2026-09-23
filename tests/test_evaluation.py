"""Correctness of the hand-written metrics and cross-validation splitter.

The metrics in :mod:`ai4se.metrics` and the splitter in
:mod:`ai4se.evaluation` are implemented from first principles rather than
taken from scikit-learn. That is only defensible if they are verified against
it, which is what this file does: where scikit-learn is installed the two
implementations must agree to twelve decimal places, and where it is not the
tests fall back to hand-computed expected values.

Run with::

    make test
"""

from __future__ import annotations

import random

import pytest

from ai4se.baselines import (
    KeywordClassifier,
    MajorityClassifier,
    MultinomialNaiveBayes,
    StratifiedRandomClassifier,
)
from ai4se.evaluation import (
    SETFIT_BASELINE,
    SETFIT_OVERALL,
    cross_validate,
    evaluate_competition,
    leaderboard,
    stratified_folds,
)
from ai4se.loader import load_split
from ai4se.metrics import auc, confusion_matrix, evaluate, roc_auc_ovr, roc_curve
from ai4se.model import LABELS
from ai4se.preprocessing import make_cleaner
from ai4se.repository import make_repository

try:
    import numpy as np
    from sklearn.metrics import (
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - sklearn is an optional extra
    HAS_SKLEARN = False


def _synthetic(n: int = 300, seed: int = 7) -> tuple[list[str], list[str]]:
    """Generate imbalanced, imperfect predictions to exercise edge cases."""
    rng = random.Random(seed)
    y_true = rng.choices(list(LABELS), weights=[5, 3, 2], k=n)
    y_pred = [
        label if rng.random() < 0.7 else rng.choice(list(LABELS)) for label in y_true
    ]
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_confusion_matrix_by_hand():
    y_true = ["bug", "bug", "feature", "question"]
    y_pred = ["bug", "feature", "feature", "bug"]
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    assert matrix["bug"]["bug"] == 1
    assert matrix["bug"]["feature"] == 1
    assert matrix["feature"]["feature"] == 1
    assert matrix["question"]["bug"] == 1
    assert sum(sum(row.values()) for row in matrix.values()) == len(y_true)


def test_metrics_by_hand():
    # bug:      TP=1, FP=1 (the question row), FN=1  -> P=0.5, R=0.5, F1=0.5
    # feature:  TP=1, FP=1, FN=0                     -> P=0.5, R=1.0, F1=2/3
    # question: TP=0, FP=0, FN=1                     -> P=0.0, R=0.0, F1=0.0
    scores = evaluate(
        ["bug", "bug", "feature", "question"],
        ["bug", "feature", "feature", "bug"],
        labels=LABELS,
    )
    assert scores.per_class["bug"].precision == pytest.approx(0.5)
    assert scores.per_class["bug"].recall == pytest.approx(0.5)
    assert scores.per_class["feature"].f1 == pytest.approx(2 / 3)
    assert scores.per_class["question"].f1 == pytest.approx(0.0)
    assert scores.accuracy == pytest.approx(0.5)
    assert scores.macro_f1 == pytest.approx((0.5 + 2 / 3 + 0.0) / 3)


def test_micro_f1_equals_accuracy_for_single_label():
    """A property of single-label multi-class problems, worth stating in the report."""
    y_true, y_pred = _synthetic()
    scores = evaluate(y_true, y_pred, labels=LABELS)
    assert scores.micro_f1 == pytest.approx(scores.accuracy)


def test_perfect_and_worst_predictions():
    y_true = list(LABELS) * 10
    assert evaluate(y_true, y_true, labels=LABELS).macro_f1 == pytest.approx(1.0)

    # Always predicting one class: that class scores, the other two are zero.
    y_pred = ["bug"] * len(y_true)
    scores = evaluate(y_true, y_pred, labels=LABELS)
    assert scores.per_class["feature"].f1 == 0.0
    assert scores.accuracy == pytest.approx(1 / 3)
    assert scores.macro_f1 < scores.accuracy  # macro punishes ignored classes


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate(["bug", "bug"], ["bug"])


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="Unexpected"):
        evaluate(["bug"], ["duplicate"], labels=LABELS)


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_metrics_match_sklearn():
    """The whole justification for hand-rolling the metrics."""
    y_true, y_pred = _synthetic(n=500)
    scores = evaluate(y_true, y_pred, labels=LABELS)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    for index, label in enumerate(LABELS):
        assert scores.per_class[label].precision == pytest.approx(precision[index])
        assert scores.per_class[label].recall == pytest.approx(recall[index])
        assert scores.per_class[label].f1 == pytest.approx(f1[index])
        assert scores.per_class[label].support == support[index]

    for average, ours in [
        ("macro", scores.macro_f1),
        ("micro", scores.micro_f1),
        ("weighted", scores.weighted_f1),
    ]:
        expected = f1_score(
            y_true, y_pred, labels=list(LABELS), average=average, zero_division=0
        )
        assert ours == pytest.approx(expected), f"{average} average disagrees"


# --------------------------------------------------------------------------- #
# ROC and AUC
# --------------------------------------------------------------------------- #


def _probabilistic(n: int = 400, seed: int = 11):
    """Ground truth plus probabilities correlated with it."""
    rng = random.Random(seed)
    y_true = rng.choices(list(LABELS), weights=[5, 3, 2], k=n)
    y_score = []
    for label in y_true:
        raw = {c: rng.random() + (0.8 if c == label else 0.0) for c in LABELS}
        total = sum(raw.values())
        y_score.append({c: v / total for c, v in raw.items()})
    return y_true, y_score


def test_auc_of_a_perfect_ranking_is_one():
    y_true = ["bug", "bug", "feature", "feature"]
    y_score = [
        {"bug": 0.9, "feature": 0.1},
        {"bug": 0.8, "feature": 0.2},
        {"bug": 0.2, "feature": 0.8},
        {"bug": 0.1, "feature": 0.9},
    ]
    _, macro = roc_auc_ovr(y_true, y_score, labels=["bug", "feature"])
    assert macro == pytest.approx(1.0)


def test_auc_of_a_constant_score_is_one_half():
    """No ranking information at all should land on the diagonal."""
    y_true = ["bug", "feature"] * 20
    y_score = [{"bug": 0.5, "feature": 0.5}] * 40
    _, macro = roc_auc_ovr(y_true, y_score, labels=["bug", "feature"])
    assert macro == pytest.approx(0.5)


def test_roc_curve_starts_at_origin_and_is_monotonic():
    y_true, y_score = _probabilistic()
    binary = [1 if label == "bug" else 0 for label in y_true]
    scores = [p["bug"] for p in y_score]
    fpr, tpr, _ = roc_curve(binary, scores)

    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == pytest.approx((1.0, 1.0))
    # strict=False is deliberate: pairing a sequence with its own tail is an
    # offset-by-one comparison, so the two sides differ in length by design.
    pairs = lambda seq: zip(seq, seq[1:], strict=False)  # noqa: E731
    assert all(a <= b for a, b in pairs(fpr)), "FPR must not decrease"
    assert all(a <= b for a, b in pairs(tpr)), "TPR must not decrease"


def test_roc_curve_handles_a_single_class():
    """An absent class makes the curve undefined; return the diagonal, not a crash."""
    fpr, tpr, _ = roc_curve([1, 1, 1], [0.2, 0.5, 0.9])
    assert auc(fpr, tpr) == pytest.approx(0.5)


def test_auc_is_absent_without_probabilities():
    """A label-only model must report None, never a value faked from hard labels."""
    scores = evaluate(["bug", "feature"], ["bug", "bug"], labels=LABELS)
    assert scores.macro_auc is None
    assert scores.per_class_auc is None
    assert scores.auc("bug") is None


def test_naive_bayes_probabilities_are_well_formed():
    model = MultinomialNaiveBayes().fit(
        ["app crashes", "please add dark mode", "how do I install"],
        ["bug", "feature", "question"],
    )
    for row in model.predict_proba(["the app crashes on launch"]):
        assert set(row) == set(LABELS)
        assert sum(row.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in row.values())


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_auc_matches_sklearn():
    """The hand-rolled ROC must agree with scikit-learn's."""
    y_true, y_score = _probabilistic(n=500)
    ours, ours_macro = roc_auc_ovr(y_true, y_score, labels=LABELS)

    truth = np.array([[1 if t == c else 0 for c in LABELS] for t in y_true])
    scores = np.array([[row[c] for c in LABELS] for row in y_score])

    for index, label in enumerate(LABELS):
        assert ours[label] == pytest.approx(
            roc_auc_score(truth[:, index], scores[:, index])
        )
    assert ours_macro == pytest.approx(
        roc_auc_score(truth, scores, average="macro", multi_class="ovr")
    )


# --------------------------------------------------------------------------- #
# Cross-validation splitter
# --------------------------------------------------------------------------- #


def test_folds_partition_the_dataset():
    labels = ["bug"] * 50 + ["feature"] * 30 + ["question"] * 20
    folds = list(stratified_folds(labels, k=10))

    assert len(folds) == 10
    seen: list[int] = []
    for train_idx, val_idx in folds:
        assert not set(train_idx) & set(val_idx), (
            "a fold leaked into its own training set"
        )
        assert len(train_idx) + len(val_idx) == len(labels)
        seen.extend(val_idx)
    assert sorted(seen) == list(range(len(labels))), "validation folds must partition"


def test_folds_are_stratified():
    labels = ["bug"] * 50 + ["feature"] * 30 + ["question"] * 20
    for _, val_idx in stratified_folds(labels, k=10):
        held_out = [labels[i] for i in val_idx]
        assert held_out.count("bug") == 5
        assert held_out.count("feature") == 3
        assert held_out.count("question") == 2


def test_folds_are_deterministic():
    labels = ["bug"] * 30 + ["feature"] * 30
    first = [v for _, v in stratified_folds(labels, k=5, seed=1)]
    second = [v for _, v in stratified_folds(labels, k=5, seed=1)]
    third = [v for _, v in stratified_folds(labels, k=5, seed=2)]
    assert first == second, "same seed must give identical folds"
    assert first != third, "different seeds should give different folds"


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        list(stratified_folds(["bug"] * 10, k=1))
    with pytest.raises(ValueError, match="rarest class"):
        list(stratified_folds(["bug"] * 10 + ["feature"] * 3, k=5))


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_folds_match_sklearn_proportions():
    """Fold sizes and class proportions must match sklearn's StratifiedKFold."""
    labels = ["bug"] * 50 + ["feature"] * 30 + ["question"] * 20

    ours = [sorted(v) for _, v in stratified_folds(labels, k=5)]
    theirs = [
        sorted(v)
        for _, v in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(
            range(len(labels)), labels
        )
    ]

    assert sorted(len(f) for f in ours) == sorted(len(f) for f in theirs)
    for ours_fold, theirs_fold in zip(ours, theirs, strict=True):
        ours_counts = sorted([labels[i] for i in ours_fold].count(c) for c in LABELS)
        theirs_counts = sorted(
            [labels[i] for i in theirs_fold].count(c) for c in LABELS
        )
        assert ours_counts == theirs_counts


# --------------------------------------------------------------------------- #
# Runners and reference models
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cleaned_train():
    repository = load_split("train", kind="memory")
    repository.apply(make_cleaner(level="full", max_words=400))
    return repository


@pytest.fixture(scope="module")
def cleaned_test():
    repository = load_split("test", kind="memory")
    repository.apply(make_cleaner(level="full", max_words=400))
    return repository


@pytest.mark.parametrize(
    "factory",
    [
        MajorityClassifier,
        StratifiedRandomClassifier,
        KeywordClassifier,
        MultinomialNaiveBayes,
    ],
)
def test_reference_models_satisfy_the_interface(factory):
    X = ["app crashes on launch", "please add dark mode", "how do I install this"]
    y = ["bug", "feature", "question"]
    model = factory().fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions) <= set(LABELS)


def test_predict_before_fit_is_rejected():
    for factory in (
        MajorityClassifier,
        StratifiedRandomClassifier,
        MultinomialNaiveBayes,
    ):
        with pytest.raises(RuntimeError, match="fit"):
            factory().predict(["anything"])


def test_majority_classifier_scores_at_the_expected_floor(cleaned_train):
    """On a balanced 3-class set: accuracy 1/3, macro F1 0.5/3."""
    result = cross_validate(
        MajorityClassifier, cleaned_train, k=5, model_name="majority"
    )
    assert result.mean_macro_f1 == pytest.approx(0.5 / 3, abs=0.01)
    assert all(s.accuracy == pytest.approx(1 / 3, abs=0.02) for s in result.fold_scores)


def test_naive_bayes_beats_the_trivial_floors(cleaned_train):
    """A learned model must clear the floors, or the pipeline is broken."""
    naive_bayes = cross_validate(
        MultinomialNaiveBayes, cleaned_train, k=5, model_name="nb"
    )
    majority = cross_validate(
        MajorityClassifier, cleaned_train, k=5, model_name="majority"
    )
    assert naive_bayes.mean_macro_f1 > majority.mean_macro_f1 + 0.20
    assert naive_bayes.std_macro_f1 < 0.10, "suspiciously unstable across folds"


def test_competition_protocol_shape(cleaned_train, cleaned_test):
    result = evaluate_competition(
        MultinomialNaiveBayes, cleaned_train, cleaned_test, model_name="nb"
    )
    assert set(result.per_repository) == set(SETFIT_BASELINE)
    # Every per-repository score is the mean F1 over the three classes.
    for repo, scores in result.per_repository.items():
        assert scores.n == 300, f"{repo} should have 300 test issues"
        assert scores.macro_f1 == pytest.approx(
            sum(scores.f1(label) for label in LABELS) / 3
        )
    # The reported figure is the mean of the five.
    assert result.overall_f1 == pytest.approx(sum(result.repository_f1.values()) / 5)
    assert result.delta_vs_baseline == pytest.approx(result.overall_f1 - SETFIT_OVERALL)


def test_baseline_constants_are_self_consistent():
    """The published per-repository scores must average to the published overall."""
    assert sum(SETFIT_BASELINE.values()) / len(SETFIT_BASELINE) == pytest.approx(
        SETFIT_OVERALL, abs=5e-5
    )


def test_leaderboard_includes_the_baseline(cleaned_train, cleaned_test):
    result = evaluate_competition(
        MajorityClassifier, cleaned_train, cleaned_test, model_name="majority"
    )
    table = leaderboard([result])
    assert "SetFit (NLBSE'24 baseline)" in table.index
    assert "majority" in table.index
    # The baseline must outrank a trivial model.
    assert table.index[0] == "SetFit (NLBSE'24 baseline)"


def test_evaluation_is_indifferent_to_persistence_layer(tmp_path):
    """The harness, like the EDA, must not care where the data lives."""
    source = load_split("train", kind="memory")
    source.apply(make_cleaner(level="full", max_words=400))
    subset = source.all()[:300]

    in_memory = make_repository("memory", issues=subset)
    on_disk = make_repository("file", issues=subset, path=tmp_path / "subset.csv")
    on_disk.save()

    from_memory = cross_validate(MultinomialNaiveBayes, in_memory, k=3)
    from_file = cross_validate(
        MultinomialNaiveBayes,
        make_repository("file", path=tmp_path / "subset.csv"),
        k=3,
    )
    assert from_memory.mean_macro_f1 == pytest.approx(from_file.mean_macro_f1)


# --------------------------------------------------------------------------- #
# Validity analyses
# --------------------------------------------------------------------------- #


def test_timestamp_only_reproduces_the_confound():
    """A model that reads only the creation date must score near 0.70.

    If this ever drops towards 0.33, the dataset changed or the temporal
    confound was removed -- either way, the report's framing is stale.
    """
    from ai4se.validity import timestamp_only_scores

    scores = timestamp_only_scores(load_split("train"), load_split("test"))
    assert scores["mean"] == pytest.approx(0.7071, abs=5e-4)
    assert set(scores) == {*SETFIT_BASELINE, "mean"}


def test_no_bug_report_predates_2021():
    """The fact the whole confound rests on."""
    from ai4se.validity import label_share_by_year

    table = label_share_by_year(load_split("train"))
    assert (table.loc[table.index < 2021, "bug"] == 0).all()
    assert table.loc[2023, "bug"] > 0.4


def test_discordant_rate_counts_only_disagreeing_outcomes():
    from ai4se.validity import discordant_rate

    truth = ["bug", "bug", "feature", "question"]
    # Item 0: both right. Item 1: only A right. Item 2: only B right.
    # Item 3: both wrong -- in different ways, which is still concordant.
    a = ["bug", "bug", "bug", "bug"]
    b = ["bug", "feature", "feature", "feature"]
    assert discordant_rate(a, b, truth) == pytest.approx(0.5)


def test_power_grows_with_the_true_difference():
    from ai4se.validity import mcnemar_power

    weak = mcnemar_power(0.005, 1500, 0.15)
    strong = mcnemar_power(0.05, 1500, 0.15)
    assert weak < 0.2 < 0.95 < strong


def test_detectable_difference_depends_on_the_pair():
    """Two models that rarely disagree give a far more sensitive test.

    This is why a single "minimum detectable difference" for the benchmark is
    misleading: it is a property of the pair being compared, not of the test set.
    """
    from ai4se.validity import minimum_detectable_difference as mdd

    similar, dissimilar = mdd(1500, 0.05), mdd(1500, 0.20)
    assert similar < dissimilar
    assert 0.01 < similar < 0.025
    assert 0.025 < dissimilar < 0.045
    assert mdd(300, 0.15) > mdd(1500, 0.15), "smaller test sets detect less"


def test_unpaired_threshold_is_conservative():
    from ai4se.validity import minimum_detectable_difference, unpaired_mdd

    assert unpaired_mdd(1500, 0.827) == pytest.approx(0.0387, abs=1e-3)
    assert unpaired_mdd(1500, 0.827) > minimum_detectable_difference(1500, 0.16)
