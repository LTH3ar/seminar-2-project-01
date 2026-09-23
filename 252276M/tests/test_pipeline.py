"""End-to-end integration tests for the 252276M pipeline.

Run with:
    cd 252276M
    python -m pytest tests/test_pipeline.py -v

Tests use a tiny synthetic dataset (3 repos × 3 classes × 5 issues = 45 issues)
so they finish in seconds without needing the real CSV files.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from ai4se.model import LABELS, IssueReport
from ai4se.repository import InMemoryIssueRepository
from ai4se.preprocessing import clean_raw, clean_light, clean_full, extract_structural_features, make_cleaner


# ─────────────────────────────── fixtures ─────────────────────────────────

@pytest.fixture
def small_train() -> InMemoryIssueRepository:
    """45 synthetic issues: 3 repos × 3 labels × 5 issues each."""
    repo = InMemoryIssueRepository()
    repos = ["facebook/react", "tensorflow/tensorflow", "opencv/opencv"]
    bodies = {
        "bug": (
            "Application crashes with Traceback error at runtime. "
            "```python\nTraceback:\n  File main.py, line 10\nAttributeError: NoneType\n```"
        ),
        "feature": (
            "Would be great to add support for async rendering. "
            "This feature would allow users to request improvements and enhance performance."
        ),
        "question": (
            "How do I configure the build pipeline? "
            "Can anyone explain the steps to reproduce the installation?"
        ),
    }
    idx = 0
    for r in repos:
        for lbl in LABELS:
            for i in range(5):
                repo.add(IssueReport(
                    repo=r,
                    issue_id=idx,
                    created_at=datetime(2023, 1, i + 1),
                    title=f"{lbl.capitalize()} issue {i} in {r.split('/')[1]}",
                    body=bodies[lbl],
                    label=lbl,
                ))
                idx += 1
    return repo


@pytest.fixture
def small_test(small_train) -> InMemoryIssueRepository:
    """Use first 3 issues per cell as test (smaller)."""
    repo = InMemoryIssueRepository()
    seen: dict = {}
    for iss in small_train.all():
        key = (iss.repo, iss.label)
        seen.setdefault(key, 0)
        if seen[key] < 3:
            repo.add(iss)
            seen[key] += 1
    return repo


# ───────────────────────── model & entity tests ───────────────────────────

class TestIssueReport:
    def test_valid_label(self):
        iss = IssueReport(
            repo="facebook/react", issue_id=1, created_at=datetime.now(),
            title="Bug", body="crash", label="bug"
        )
        assert iss.label == "bug"

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            IssueReport(
                repo="facebook/react", issue_id=1, created_at=datetime.now(),
                title="X", body="X", label="invalid"
            )

    def test_text_combines_title_and_body(self):
        iss = IssueReport(
            repo="facebook/react", issue_id=1, created_at=datetime.now(),
            title="Title here", body="Body here", label="bug"
        )
        assert "Title here" in iss.text
        assert "Body here" in iss.text

    def test_replace_text(self):
        iss = IssueReport(
            repo="facebook/react", issue_id=1, created_at=datetime.now(),
            title="Title", body="old body", label="bug"
        )
        new_iss = iss.replace_text("new body")
        assert new_iss.body == "new body"
        assert new_iss.label == "bug"

    def test_round_trip_dict(self):
        iss = IssueReport(
            repo="facebook/react", issue_id=99, created_at=datetime(2023, 5, 1),
            title="T", body="B", label="feature"
        )
        restored = IssueReport.from_dict(iss.to_dict())
        assert restored.repo == iss.repo
        assert restored.label == iss.label
        assert restored.issue_id == iss.issue_id


# ─────────────────────── repository tests ─────────────────────────────────

class TestInMemoryRepository:
    def test_by_repo_filters_correctly(self, small_train):
        react_issues = small_train.by_repo("facebook/react")
        assert all(iss.repo == "facebook/react" for iss in react_issues)
        assert len(react_issues) == 15  # 3 labels × 5 issues

    def test_by_label_filters_correctly(self, small_train):
        bugs = small_train.by_label("bug")
        assert all(iss.label == "bug" for iss in bugs)
        assert len(bugs) == 15  # 3 repos × 5 issues

    def test_texts_and_labels_length_match(self, small_train):
        texts, labels = small_train.texts_and_labels()
        assert len(texts) == len(labels) == 45

    def test_label_distribution(self, small_train):
        dist = small_train.label_distribution()
        for lbl in LABELS:
            assert dist[lbl] == 15

    def test_apply_transform(self, small_train):
        small_train.apply(lambda body: body.upper())
        for iss in small_train.all():
            assert iss.body == iss.body.upper()


# ─────────────────────── preprocessing tests ──────────────────────────────

class TestPreprocessing:
    def test_clean_raw_normalises_whitespace(self):
        assert clean_raw("hello   world\n\t!") == "hello world !"

    def test_clean_light_removes_code_block(self):
        text = "Intro\n```python\ncode\n```\nafter"
        result = clean_light(text)
        assert "python" not in result
        assert "after" in result

    def test_clean_light_removes_url(self):
        text = "See https://github.com/issue/1 for details"
        result = clean_light(text)
        assert "https://" not in result

    def test_clean_full_lowercases(self):
        result = clean_full("Hello World ERROR")
        assert result == result.lower()

    def test_structural_features_length(self):
        features = extract_structural_features("Bug in app?", "```\ncode\n```\nTraceback")
        assert len(features) == 15
        assert all(f in (0, 1) for f in features)

    def test_cleaner_respects_max_words(self):
        text = " ".join([f"word{i}" for i in range(1000)])
        cleaner = make_cleaner(level="raw", title_weight=0, max_words=50)
        result = cleaner("", text)
        assert len(result.split()) <= 50


# ──────────────────────── floor classifier tests ──────────────────────────

class TestFloorClassifiers:
    def test_majority_predicts_most_frequent(self, small_train):
        from ai4se.classifiers.floors import MajorityClassifier
        texts, labels = small_train.texts_and_labels()
        clf = MajorityClassifier()
        clf.fit(texts, labels)
        preds = clf.predict(texts[:5])
        assert all(p in LABELS for p in preds)

    def test_keyword_rules_returns_valid_labels(self, small_train):
        from ai4se.classifiers.floors import KeywordRulesClassifier
        texts, labels = small_train.texts_and_labels()
        clf = KeywordRulesClassifier()
        clf.fit(texts, labels)
        preds = clf.predict(texts)
        assert all(p in LABELS for p in preds)

    def test_scratch_nb_fit_predict(self, small_train):
        from ai4se.classifiers.floors import ScratchNaiveBayesClassifier
        react_issues = small_train.by_repo("facebook/react")
        texts = [iss.text for iss in react_issues]
        labels = [iss.label for iss in react_issues]
        clf = ScratchNaiveBayesClassifier()
        clf.fit(texts, labels)
        preds = clf.predict(texts[:3])
        assert len(preds) == 3
        assert all(p in LABELS for p in preds)

    def test_scratch_nb_predict_proba_sums_to_one(self, small_train):
        from ai4se.classifiers.floors import ScratchNaiveBayesClassifier
        texts, labels = ["app crash", "add feature", "how to?"], ["bug", "feature", "question"]
        clf = ScratchNaiveBayesClassifier()
        clf.fit(texts, labels)
        probs = clf.predict_proba(texts)
        assert probs.shape == (3, 3)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-6)


# ─────────────────────── classical classifier tests ───────────────────────

class TestClassicalClassifiers:
    def test_tfidf_logreg_fit_predict(self, small_train):
        from ai4se.classifiers.classical import TfidfClassifier
        texts, labels = small_train.texts_and_labels()
        clf = TfidfClassifier(estimator="logreg")
        clf.fit(texts, labels)
        preds = clf.predict(texts[:10])
        assert len(preds) == 10
        assert all(p in LABELS for p in preds)

    def test_tfidf_svm_no_proba_fallback(self, small_train):
        from ai4se.classifiers.classical import TfidfClassifier
        texts, labels = small_train.texts_and_labels()
        clf = TfidfClassifier(estimator="linear_svm")
        clf.fit(texts, labels)
        probs = clf.predict_proba(texts[:5])
        assert probs.shape[1] == 3
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

    def test_complement_nb_runs(self, small_train):
        from ai4se.classifiers.classical import TfidfClassifier
        texts, labels = small_train.texts_and_labels()
        clf = TfidfClassifier(estimator="complement_nb")
        clf.fit(texts, labels)
        preds = clf.predict(texts[:5])
        assert all(p in LABELS for p in preds)


# ──────────────────── train_per_repo harness tests ────────────────────────

class TestTrainPerRepo:
    def test_per_repo_covers_all_test_issues(self, small_train, small_test):
        from ai4se.classifiers.base import train_per_repo
        from ai4se.classifiers.classical import make_classical
        preds, _ = train_per_repo(make_classical("logreg"), small_train, small_test)
        assert len(preds) == len(small_test)
        assert all(p in LABELS for p in preds)

    def test_per_repo_returns_proba_when_requested(self, small_train, small_test):
        from ai4se.classifiers.base import train_per_repo
        from ai4se.classifiers.classical import make_classical
        preds, probs = train_per_repo(
            make_classical("logreg"), small_train, small_test, return_proba=True
        )
        assert probs is not None
        assert probs.shape == (len(small_test), 3)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(small_test)), atol=1e-5)


# ──────────────────────────── metrics tests ───────────────────────────────

class TestMetrics:
    def test_confusion_matrix_correct_shape(self):
        from ai4se.evaluation.metrics import confusion_matrix
        y_true = ["bug", "feature", "question", "bug", "feature"]
        y_pred = ["bug", "bug",     "question", "bug", "feature"]
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm.sum() == 5

    def test_macro_f1_balanced_perfect(self):
        from ai4se.evaluation.metrics import confusion_matrix, per_class_metrics, macro_f1
        y = ["bug", "feature", "question"]
        cm = confusion_matrix(y, y)
        pcm = per_class_metrics(cm)
        assert macro_f1(pcm) == pytest.approx(1.0)

    def test_per_repo_f1_structure(self, small_train, small_test):
        from ai4se.evaluation.metrics import per_repo_and_macro_f1
        test_issues = list(small_test.all())
        # Use label-as-prediction (perfect oracle)
        perfect_preds = [iss.label for iss in test_issues]
        result = per_repo_and_macro_f1(test_issues, perfect_preds)
        assert "cross_repo_f1" in result
        assert result["cross_repo_f1"] == pytest.approx(1.0, abs=1e-6)
        assert len(result["per_repo"]) == 3


# ───────────────────────── ensemble tests ─────────────────────────────────

class TestEnsemble:
    def test_soft_voting_averages_probabilities(self, small_train, small_test):
        from ai4se.classifiers.ensemble import SoftVotingEnsemble
        from ai4se.classifiers.classical import TfidfClassifier

        texts, labels = small_train.texts_and_labels()
        clf1 = TfidfClassifier(estimator="logreg").fit(texts, labels)
        clf2 = TfidfClassifier(estimator="complement_nb").fit(texts, labels)

        ensemble = SoftVotingEnsemble(classifiers=[clf1, clf2])
        test_texts = [iss.text for iss in small_test.all()]
        probs = ensemble.predict_proba(test_texts)
        preds = ensemble.predict(test_texts)

        assert probs.shape == (len(test_texts), 3)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(test_texts)), atol=1e-5)
        assert all(p in LABELS for p in preds)


# ─────────────────────── stratified split tests ───────────────────────────

class TestSplits:
    def test_stratified_kfold_no_overlap(self, small_train):
        from ai4se.evaluation.splits import stratified_k_fold
        issues = list(small_train.all())
        folds = stratified_k_fold(issues, n_splits=5)
        for train_idx, val_idx in folds:
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_train_val_split_sizes(self):
        from ai4se.evaluation.splits import train_val_split
        texts = [f"text {i}" for i in range(100)]
        labels = ["bug"] * 34 + ["feature"] * 33 + ["question"] * 33
        tr_t, tr_l, v_t, v_l = train_val_split(texts, labels, val_ratio=0.2)
        assert len(tr_t) + len(v_t) == 100
        assert abs(len(v_t) - 20) <= 5  # ~20% tolerance


# ────────────────────────── significance tests ────────────────────────────

class TestSignificance:
    def test_mcnemar_identical_models_not_significant(self):
        from ai4se.evaluation.significance import exact_mcnemar
        preds = ["bug", "feature", "question"] * 10
        truth = ["bug", "feature", "question"] * 10
        result = exact_mcnemar(truth, preds, preds)
        assert result["p_value"] == 1.0

    def test_holm_bonferroni_correction(self):
        from ai4se.evaluation.significance import holm_bonferroni
        p_values = [0.01, 0.04, 0.20, 0.50]
        corrected = holm_bonferroni(p_values, alpha=0.05)
        assert len(corrected) == 4
        # First p-value should be significant
        assert corrected[0]["significant"]
