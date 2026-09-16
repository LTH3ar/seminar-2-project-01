"""Shared, leakage-aware evaluation for every classification track."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Protocol

from .folds import make_folds_by_repository
from .model import LABELS, IssueReport

EVALUATION_SCHEMA_VERSION = 1


class TextClassifier(Protocol):
    """Small interface shared by scikit-learn, SetFit, and neural models."""

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> Any:
        """Train the classifier."""

        ...

    def predict(self, texts: Sequence[str]) -> Sequence[str]:
        """Predict one label for every input text."""

        ...


EstimatorFactory = Callable[[str, int], TextClassifier]


def calculate_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    """Calculate the common metrics and confusion matrix.

    All tracks use this function so class order, averaging, and zero-division
    behaviour cannot silently differ between experiments.
    """

    references = [str(value) for value in references]
    predictions = [str(value) for value in predictions]
    labels = tuple(labels)
    if not references:
        raise ValueError("At least one reference label is required")
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have the same length")

    allowed = set(labels)
    unknown_references = sorted(set(references) - allowed)
    unknown_predictions = sorted(set(predictions) - allowed)
    if unknown_references or unknown_predictions:
        raise ValueError(
            "Unknown labels: "
            f"references={unknown_references}, predictions={unknown_predictions}"
        )

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for shared model evaluation"
        ) from exc

    report = classification_report(
        references,
        predictions,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )

    def scores(section: Mapping[str, Any]) -> dict[str, float | int]:
        return {
            "precision": float(section["precision"]),
            "recall": float(section["recall"]),
            "f1": float(section["f1-score"]),
            "support": int(section["support"]),
        }

    return {
        "accuracy": float(accuracy_score(references, predictions)),
        "per_label": {label: scores(report[label]) for label in labels},
        "macro_average": scores(report["macro avg"]),
        "weighted_average": scores(report["weighted avg"]),
        "confusion_matrix": confusion_matrix(
            references,
            predictions,
            labels=list(labels),
        ).tolist(),
    }


def evaluate_cross_validation(
    issues: Sequence[IssueReport],
    estimator_factory: EstimatorFactory,
    *,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one fresh classifier per repository and grouped fold."""

    issue_list = list(issues)
    folds_by_repository = make_folds_by_repository(
        issue_list,
        n_splits=n_splits,
        random_state=random_state,
    )
    repositories: dict[str, dict[str, Any]] = {}

    for repository, folds in folds_by_repository.items():
        references: list[str] = []
        predictions: list[str] = []
        prediction_rows: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []

        for fold in folds:
            classifier = estimator_factory(repository, fold.number)
            started = perf_counter()
            classifier.fit(
                [issue.text for issue in fold.train],
                [issue.label for issue in fold.train],
            )
            fold_predictions = _normalise_predictions(
                classifier.predict([issue.text for issue in fold.validation])
            )
            duration = perf_counter() - started
            fold_references = [issue.label for issue in fold.validation]
            fold_metrics = calculate_metrics(fold_references, fold_predictions)

            references.extend(fold_references)
            predictions.extend(fold_predictions)
            prediction_rows.extend(
                _prediction_rows(
                    fold.validation,
                    fold_predictions,
                    fold=fold.number,
                )
            )
            fold_rows.append(
                {
                    "fold": fold.number,
                    "train_size": len(fold.train),
                    "validation_size": len(fold.validation),
                    "duration_seconds": duration,
                    "metrics": fold_metrics,
                }
            )

        repositories[repository] = {
            "sample_count": len(references),
            "metrics": calculate_metrics(references, predictions),
            "folds": fold_rows,
            "fold_summary": _summarise_folds(fold_rows),
            "predictions": prediction_rows,
        }

    return _build_result(
        model_name=model_name,
        evaluation="stratified_group_k_fold",
        repositories=repositories,
        random_state=random_state,
        n_splits=n_splits,
        metadata=metadata,
    )


def evaluate_holdout_by_repository(
    train_issues: Sequence[IssueReport],
    test_issues: Sequence[IssueReport],
    estimator_factory: EstimatorFactory,
    *,
    model_name: str,
    random_state: int = 42,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproduce the competition's one-model-per-repository evaluation."""

    train_by_repository = _group_by_repository(train_issues)
    test_by_repository = _group_by_repository(test_issues)
    if set(train_by_repository) != set(test_by_repository):
        raise ValueError(
            "Training and test data must contain the same repositories"
        )

    repositories: dict[str, dict[str, Any]] = {}
    for repository in sorted(train_by_repository):
        train = train_by_repository[repository]
        test = test_by_repository[repository]
        classifier = estimator_factory(repository, 0)
        started = perf_counter()
        classifier.fit(
            [issue.text for issue in train],
            [issue.label for issue in train],
        )
        predictions = _normalise_predictions(
            classifier.predict([issue.text for issue in test])
        )
        duration = perf_counter() - started
        references = [issue.label for issue in test]
        repositories[repository] = {
            "sample_count": len(test),
            "train_size": len(train),
            "duration_seconds": duration,
            "metrics": calculate_metrics(references, predictions),
            "predictions": _prediction_rows(test, predictions, fold=None),
        }

    return _build_result(
        model_name=model_name,
        evaluation="official_holdout",
        repositories=repositories,
        random_state=random_state,
        n_splits=None,
        metadata=metadata,
    )


def save_evaluation(result: Mapping[str, Any], path: str | Path) -> Path:
    """Write a complete evaluation result as UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return destination


def load_evaluation(path: str | Path) -> dict[str, Any]:
    """Load an evaluation result and verify its schema version."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        result = json.load(handle)
    version = result.get("schema_version")
    if version != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evaluation schema {version!r}; "
            f"expected {EVALUATION_SCHEMA_VERSION}"
        )
    return result


def plot_confusion_matrices(
    result: Mapping[str, Any],
    output_directory: str | Path,
    *,
    normalise: bool = False,
) -> list[Path]:
    """Save one confusion-matrix plot per repository and one pooled plot."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to create evaluation plots"
        ) from exc

    labels = list(result["labels"])
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    metric_sets = {
        repository: values["metrics"]
        for repository, values in result["repositories"].items()
    }
    metric_sets["overall"] = result["overall"]["pooled"]
    paths: list[Path] = []

    for repository, metrics in metric_sets.items():
        matrix = metrics["confusion_matrix"]
        display_matrix = _normalise_confusion_matrix(matrix) if normalise else matrix
        figure, axis = plt.subplots(figsize=(5.6, 4.8))
        image = axis.imshow(display_matrix, cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set(
            title=f"{result['model_name']} - {repository}",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=range(len(labels)),
            yticks=range(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
        )
        threshold = max(max(row) for row in display_matrix) / 2
        for row_index, row in enumerate(display_matrix):
            for column_index, value in enumerate(row):
                text = f"{value:.2f}" if normalise else str(value)
                axis.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )
        figure.tight_layout()
        suffix = "-normalised" if normalise else ""
        path = output_directory / f"{_slug(repository)}{suffix}.png"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _normalise_predictions(values: Sequence[Any]) -> list[str]:
    predictions = [str(value) for value in values]
    if not predictions:
        raise ValueError("The classifier returned no predictions")
    return predictions


def _prediction_rows(
    issues: Sequence[IssueReport],
    predictions: Sequence[str],
    *,
    fold: int | None,
) -> list[dict[str, Any]]:
    if len(issues) != len(predictions):
        raise ValueError(
            "The classifier must return exactly one prediction per issue"
        )
    return [
        {
            "issue_id": issue.issue_id,
            "repository": issue.repo,
            "actual": issue.label,
            "predicted": prediction,
            "fold": fold,
        }
        for issue, prediction in zip(issues, predictions, strict=True)
    ]


def _group_by_repository(
    issues: Sequence[IssueReport],
) -> dict[str, list[IssueReport]]:
    grouped: dict[str, list[IssueReport]] = {}
    for issue in issues:
        grouped.setdefault(issue.repo, []).append(issue)
    if not grouped:
        raise ValueError("At least one issue is required")
    return grouped


def _build_result(
    *,
    model_name: str,
    evaluation: str,
    repositories: Mapping[str, Mapping[str, Any]],
    random_state: int,
    n_splits: int | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pooled_references: list[str] = []
    pooled_predictions: list[str] = []
    for values in repositories.values():
        for prediction in values["predictions"]:
            pooled_references.append(prediction["actual"])
            pooled_predictions.append(prediction["predicted"])

    repository_metrics = [values["metrics"] for values in repositories.values()]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "model_name": model_name,
        "evaluation": evaluation,
        "labels": list(LABELS),
        "random_state": random_state,
        "n_splits": n_splits,
        "metadata": dict(metadata or {}),
        "repositories": dict(repositories),
        "overall": {
            "sample_count": len(pooled_references),
            "repository_average": _average_metrics(repository_metrics),
            "pooled": calculate_metrics(
                pooled_references,
                pooled_predictions,
            ),
        },
    }


def _average_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def average_scores(section: str) -> dict[str, float]:
        return {
            metric: mean(values[section][metric] for values in metrics)
            for metric in ("precision", "recall", "f1")
        }

    return {
        "accuracy": mean(values["accuracy"] for values in metrics),
        "per_label": {
            label: {
                metric: mean(
                    values["per_label"][label][metric] for values in metrics
                )
                for metric in ("precision", "recall", "f1")
            }
            for label in LABELS
        },
        "macro_average": average_scores("macro_average"),
        "weighted_average": average_scores("weighted_average"),
    }


def _normalise_confusion_matrix(matrix: Sequence[Sequence[int]]) -> list[list[float]]:
    normalised: list[list[float]] = []
    for row in matrix:
        total = sum(row)
        normalised.append(
            [value / total if total else 0.0 for value in row]
        )
    return normalised


def _summarise_folds(folds: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    macro_f1 = [fold["metrics"]["macro_average"]["f1"] for fold in folds]
    return {
        "macro_f1_mean": mean(macro_f1),
        "macro_f1_std": pstdev(macro_f1),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
