from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

from issue_classifier.audit import (
    audit_dataset,
    audit_train_test_leakage,
    duplicate_group,
)
from issue_classifier.dataframe import prepared_issues_to_dataframe
from issue_classifier.domain import IssueReport
from issue_classifier.eda import cleaning_impact, overview
from issue_classifier.folds import make_folds_by_repository
from issue_classifier.loader import ensure_dataset
from issue_classifier.preprocessing import TextCleaningConfig, TextPreprocessor
from issue_classifier.repositories import (
    CsvIssueRepository,
    InMemoryIssueRepository,
    JsonIssueRepository,
)
from issue_classifier.service import IssueDataService, create_issue_repository
from issue_classifier.validation import validate_issues


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"


class RepositoryTests(unittest.TestCase):
    def test_bundled_dataset_satisfies_the_download_contract(self) -> None:
        paths = ensure_dataset(DATA_DIRECTORY)

        self.assertEqual({"train", "test"}, set(paths))
        self.assertTrue(all(path.exists() for path in paths.values()))

    def test_csv_repository_loads_the_supplied_splits(self) -> None:
        repository = CsvIssueRepository(DATA_DIRECTORY)

        train = repository.load("train")
        test = repository.load("test")

        self.assertEqual(1_500, len(train))
        self.assertEqual(1_500, len(test))
        self.assertEqual({"bug", "feature", "question"}, {x.label for x in train})
        self.assertEqual(2, sum(not issue.body for issue in test))

    def test_file_and_memory_repositories_have_the_same_contract(self) -> None:
        issue = IssueReport.create(
            repo="facebook/react",
            created_at=datetime(2024, 1, 2, 3, 4, 5),
            label="bug",
            title="Rendering fails",
            body="Steps to reproduce",
        )
        memory = InMemoryIssueRepository()
        memory.save("sample", [issue])

        with tempfile.TemporaryDirectory() as directory:
            file_repository = CsvIssueRepository(
                directory,
                filenames={"sample": "sample.csv"},
            )
            file_repository.save("sample", memory.load("sample"))

            self.assertEqual(memory.load("sample"), file_repository.load("sample"))
            self.assertEqual(
                memory.find_by_repo("sample", "facebook/react"),
                file_repository.find_by_repo("sample", "facebook/react"),
            )

    def test_factory_can_load_the_dataset_into_memory(self) -> None:
        repository = create_issue_repository("memory", DATA_DIRECTORY)
        self.assertIsInstance(repository, InMemoryIssueRepository)
        self.assertEqual(1_500, len(repository.load("train")))

    def test_json_repository_round_trip_matches_csv_records(self) -> None:
        issue = CsvIssueRepository(DATA_DIRECTORY).load("train")[0]

        with tempfile.TemporaryDirectory() as directory:
            repository = JsonIssueRepository(
                directory,
                filenames={"sample": "sample.json"},
            )
            repository.save("sample", [issue])

            self.assertEqual([issue], repository.load("sample"))

    def test_repository_query_helpers_are_backend_independent(self) -> None:
        repository = CsvIssueRepository(DATA_DIRECTORY)

        self.assertEqual(
            ["bug", "feature", "question"],
            repository.labels("train"),
        )
        self.assertEqual(5, len(repository.repositories("train")))
        self.assertEqual(
            {"bug": 100, "feature": 100, "question": 100},
            repository.label_distribution("train", repo="facebook/react"),
        )


class ValidationAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repository = CsvIssueRepository(DATA_DIRECTORY)
        cls.train = repository.load("train")
        cls.test = repository.load("test")

    def test_supplied_data_is_valid_but_reports_empty_test_bodies(self) -> None:
        report = validate_issues(self.test, split="test")

        self.assertTrue(report.is_valid)
        self.assertEqual(0, len(report.errors))
        self.assertEqual(2, len(report.warnings))
        self.assertEqual({"empty_body"}, {message.code for message in report.warnings})

    def test_audit_reports_distribution_and_duplicates(self) -> None:
        audit = audit_dataset(self.train, split="train")

        self.assertEqual(1_500, audit.record_count)
        self.assertEqual(
            {"bug": 500, "feature": 500, "question": 500},
            audit.label_counts,
        )
        self.assertEqual(1, audit.duplicate_group_count)
        self.assertEqual(2, audit.duplicate_record_count)

    def test_cross_split_audit_finds_known_text_leakage(self) -> None:
        report = audit_train_test_leakage(self.train, self.test)

        self.assertEqual(3, report.overlap_group_count)
        self.assertEqual(4, report.overlapping_train_record_count)
        self.assertEqual(3, report.overlapping_test_record_count)
        self.assertEqual(1, report.conflicting_label_group_count)


class PreprocessingAndFoldTests(unittest.TestCase):
    def test_cleaning_levels_support_model_specific_ablation(self) -> None:
        value = (
            "API v2 fails at https://example.com. "
            "\x60\x60\x60python\nraise Error()\n\x60\x60\x60"
        )

        raw = TextPreprocessor(TextCleaningConfig(level="raw")).clean(value)
        light = TextPreprocessor(TextCleaningConfig(level="light")).clean(value)
        full = TextPreprocessor(TextCleaningConfig(level="full")).clean(value)

        self.assertIn("https://example.com", raw)
        self.assertNotIn("https://example.com", light)
        self.assertNotIn("raise Error", light)
        self.assertEqual(full, full.lower())

    def test_cleaning_preserves_software_information(self) -> None:
        issue = IssueReport.create(
            repo="opencv/opencv",
            created_at=datetime(2024, 1, 1),
            label="question",
            title="Why does API v2 fail?",
            body="See https://example.com. Error 404. ```python\nraise Error()\n```",
        )

        prepared = TextPreprocessor().prepare(issue)

        self.assertIn("v2", prepared.text)
        self.assertIn("?", prepared.text)
        self.assertIn("404", prepared.text)
        self.assertIn("<URL>", prepared.text)
        self.assertIn("<CODE_BLOCK>", prepared.text)
        self.assertEqual(1.0, prepared.features["question_mark_count"])
        self.assertEqual(1.0, prepared.features["url_count"])

    def test_dataframe_adapter_includes_text_and_features(self) -> None:
        repository = CsvIssueRepository(DATA_DIRECTORY)
        service = IssueDataService(repository)

        frame = prepared_issues_to_dataframe(
            service.prepare("train", repo="facebook/react")[:2]
        )

        self.assertEqual(2, len(frame))
        self.assertIn("text", frame.columns)
        self.assertIn("body_word_count", frame.columns)

    def test_eda_helpers_accept_the_same_domain_entities(self) -> None:
        issues = CsvIssueRepository(DATA_DIRECTORY).load("train")[:30]

        self.assertEqual(1, len(overview(issues)))
        self.assertEqual(
            {"raw", "conservative", "light", "full"},
            set(cleaning_impact(issues).index),
        )

    def test_folds_are_stratified_and_duplicate_safe(self) -> None:
        issues = CsvIssueRepository(DATA_DIRECTORY).load("train")
        folds_by_repo = make_folds_by_repository(issues)

        self.assertEqual(5, len(folds_by_repo))
        for folds in folds_by_repo.values():
            self.assertEqual(5, len(folds))
            validation_ids = []
            for fold in folds:
                train_groups = {duplicate_group(issue) for issue in fold.train}
                validation_groups = {
                    duplicate_group(issue) for issue in fold.validation
                }
                self.assertTrue(train_groups.isdisjoint(validation_groups))
                self.assertEqual(
                    {"bug", "feature", "question"},
                    set(Counter(issue.label for issue in fold.validation)),
                )
                validation_ids.extend(issue.issue_id for issue in fold.validation)
            self.assertEqual(300, len(validation_ids))
            self.assertEqual(300, len(set(validation_ids)))


if __name__ == "__main__":
    unittest.main()
