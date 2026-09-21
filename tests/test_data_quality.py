"""Tests for datdq features consolidated into the ai4se package."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ai4se.audit import audit_dataset, audit_train_test_leakage, duplicate_group
from ai4se.folds import make_folds_by_repository
from ai4se.loader import load_split
from ai4se.preprocessing import clean_text, structural_features
from ai4se.service import IssueDataService
from ai4se.validation import validate_issues

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def train():
    """Load training data once for this module."""
    return load_split("train", raw_dir=DATA_DIRECTORY).all()


@pytest.fixture(scope="module")
def test():
    """Load test data once for this module."""
    return load_split("test", raw_dir=DATA_DIRECTORY).all()


def test_validation_and_known_empty_bodies(test):
    """The test split is valid with two empty-body warnings."""
    report = validate_issues(test, split="test")
    assert report.is_valid
    assert not report.errors
    assert len(report.warnings) == 2
    assert {message.code for message in report.warnings} == {"empty_body"}


def test_audit_and_known_cross_split_leakage(train, test):
    """Audits reproduce the known duplicate and leakage findings."""
    audit = audit_dataset(train, split="train")
    leakage = audit_train_test_leakage(train, test)

    assert audit.label_counts == {"bug": 500, "feature": 500, "question": 500}
    assert audit.duplicate_group_count == 1
    assert audit.duplicate_record_count == 2
    assert leakage.overlap_group_count == 3
    assert leakage.overlapping_train_record_count == 4
    assert leakage.overlapping_test_record_count == 3
    assert leakage.conflicting_label_group_count == 1


def test_conservative_cleaning_preserves_software_information(train):
    """Conservative cleaning retains software-specific tokens."""
    marker = chr(96) * 3
    value = f"API v2 failed with 404 at https://example.com. {marker}code{marker}"
    cleaned = clean_text(value, level="conservative")
    features = structural_features(train[0])

    assert "v2" in cleaned
    assert "404" in cleaned
    assert "<URL>" in cleaned
    assert "<CODE_BLOCK>" in cleaned
    assert "body_word_count" in features


def test_service_prepares_feature_dataframe():
    """The service exposes cleaned text and structural features."""
    service = IssueDataService.from_loader(raw_dir=DATA_DIRECTORY)
    frame = service.prepared_dataframe(
        "train",
        repo="facebook/react",
        level="conservative",
    )

    assert len(frame) == 300
    assert "clean_text" in frame.columns
    assert "body_word_count" in frame.columns


def test_folds_are_stratified_and_duplicate_safe(train):
    """Each repository receives duplicate-safe stratified folds."""
    pytest.importorskip("sklearn")
    folds_by_repo = make_folds_by_repository(train)

    assert len(folds_by_repo) == 5
    for folds in folds_by_repo.values():
        assert len(folds) == 5
        validation_ids = []
        for fold in folds:
            train_groups = {duplicate_group(issue) for issue in fold.train}
            validation_groups = {duplicate_group(issue) for issue in fold.validation}
            assert train_groups.isdisjoint(validation_groups)
            assert set(Counter(issue.label for issue in fold.validation)) == {
                "bug",
                "feature",
                "question",
            }
            validation_ids.extend(issue.issue_id for issue in fold.validation)
        assert len(validation_ids) == 300
        assert len(set(validation_ids)) == 300
