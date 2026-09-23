"""Tests for the shared Track D evaluation workflow."""

from __future__ import annotations

from pathlib import Path

from ai4se.evaluation import (
    calculate_metrics,
    evaluate_cross_validation,
    evaluate_holdout_by_repository,
    load_evaluation,
    save_evaluation,
)
from ai4se.model import LABELS, IssueReport
from ai4se.reporting import result_rows, write_result_tables
from ai4se.setfit_baseline import SetFitConfig


class KeywordClassifier:
    """Tiny deterministic classifier for exercising the evaluation harness."""

    def fit(self, texts, labels):
        """Record labels observed during training."""

        self.labels = set(labels)
        return self

    def predict(self, texts):
        """Recover the label keyword embedded in each test issue."""

        return [next(label for label in LABELS if label in text) for text in texts]


class DiagnosticKeywordClassifier(KeywordClassifier):
    """Keyword classifier exposing optional training diagnostics."""

    def training_summary(self):
        return {"best_epoch": 2, "history": {"validation_loss": [0.5, 0.4]}}


def make_issues(
    *,
    suffix: str = "train",
    per_label: int = 4,
) -> list[IssueReport]:
    """Create balanced, unique issues for two repositories."""

    return [
        IssueReport(
            repo=repository,
            created_at=f"2024-01-{index + 1:02d}",
            label=label,
            title=f"{label} example {suffix} {index}",
            body=f"unique {repository} {label} {suffix} {index}",
        )
        for repository in ("owner/one", "owner/two")
        for label in LABELS
        for index in range(per_label)
    ]


def classifier_factory(repository: str, fold: int) -> KeywordClassifier:
    """Return a fresh deterministic classifier."""

    return KeywordClassifier()


def test_calculate_metrics_uses_fixed_label_order():
    metrics = calculate_metrics(
        ["bug", "feature", "question", "bug"],
        ["bug", "question", "question", "feature"],
    )

    assert metrics["accuracy"] == 0.5
    assert list(metrics["per_label"]) == list(LABELS)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 0, 1], [0, 0, 1]]


def test_cross_validation_produces_complete_oof_predictions():
    issues = make_issues()
    result = evaluate_cross_validation(
        issues,
        classifier_factory,
        model_name="keyword",
        n_splits=2,
    )

    assert result["evaluation"] == "stratified_group_k_fold"
    assert result["overall"]["sample_count"] == len(issues)
    assert result["overall"]["repository_average"]["macro_average"]["f1"] == 1
    for repository in result["repositories"].values():
        assert len(repository["folds"]) == 2
        assert repository["fold_summary"]["macro_f1_mean"] == 1
        assert repository["fold_summary"]["macro_f1_std"] == 0
        ids = [row["issue_id"] for row in repository["predictions"]]
        assert len(ids) == len(set(ids)) == 12


def test_official_holdout_uses_one_model_per_repository():
    train = make_issues(per_label=2)
    test = make_issues(suffix="test", per_label=1)
    result = evaluate_holdout_by_repository(
        train,
        test,
        classifier_factory,
        model_name="keyword",
    )

    assert result["evaluation"] == "official_holdout"
    assert result["overall"]["sample_count"] == 6
    assert result["overall"]["pooled"]["accuracy"] == 1


def test_evaluator_records_optional_training_diagnostics():
    result = evaluate_holdout_by_repository(
        make_issues(per_label=2),
        make_issues(suffix="test", per_label=1),
        lambda repository, fold: DiagnosticKeywordClassifier(),
        model_name="diagnostic-keyword",
    )

    for repository in result["repositories"].values():
        assert repository["training"]["best_epoch"] == 2


def test_result_persistence_and_tables(tmp_path: Path):
    result = evaluate_holdout_by_repository(
        make_issues(per_label=2),
        make_issues(suffix="test", per_label=1),
        classifier_factory,
        model_name="keyword",
    )
    json_path = save_evaluation(result, tmp_path / "result.json")
    loaded = load_evaluation(json_path)
    rows = result_rows(loaded)
    csv_path, markdown_path = write_result_tables(rows, tmp_path / "tables")

    assert loaded["model_name"] == "keyword"
    assert len(rows) == 3
    assert csv_path.exists()
    assert "macro_f1" in markdown_path.read_text(encoding="utf-8")


def test_supplied_baseline_format_is_supported():
    baseline = {
        "overall": {
            "average": {
                "precision": 0.8,
                "recall": 0.75,
                "f1-score": 0.77,
            }
        }
    }

    assert result_rows(baseline, model_name="setfit")[0]["macro_f1"] == 0.77


def test_setfit_defaults_match_the_supplied_notebook():
    config = SetFitConfig()

    assert config.model_id == "sentence-transformers/all-mpnet-base-v2"
    assert config.num_epochs == 1
    assert config.num_iterations == 20
    assert config.body_batch_size == 16
    assert config.classifier_batch_size == 2
    assert config.prediction_batch_size == 8
    assert config.max_sequence_length == 384
