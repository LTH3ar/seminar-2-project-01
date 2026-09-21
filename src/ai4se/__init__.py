"""Consolidated issue-report classification utilities."""

from .audit import DatasetAudit, LeakageReport
from .classical import (
    AVAILABLE_MODELS,
    TfidfConfig,
    build_classical_pipeline,
    make_classical_estimator_factory,
)
from .deep_learning import (
    AVAILABLE_DEEP_MODELS,
    CNNConfig,
    CNNTextClassifier,
    DistilBERTConfig,
    DistilBERTTextClassifier,
    FeedForwardTextClassifier,
    FFNNConfig,
    TrainingConfig,
    build_deep_classifier,
    make_deep_estimator_factory,
)
from .error_analysis import analyse_evaluation
from .evaluation import (
    calculate_metrics,
    evaluate_cross_validation,
    evaluate_holdout_by_repository,
)
from .model import IssueReport
from .service import IssueDataService
from .splits import (
    DeduplicationSummary,
    deduplicate_for_evaluation,
    random_split_control,
    time_aware_split,
)
from .statistical_analysis import (
    bootstrap_confidence_interval,
    compare_evaluations,
    minimum_detectable_difference,
    summarise_repeated_results,
)
from .validation import DatasetValidationError, ValidationReport

__all__ = [
    "AVAILABLE_MODELS",
    "AVAILABLE_DEEP_MODELS",
    "CNNConfig",
    "CNNTextClassifier",
    "DatasetAudit",
    "DatasetValidationError",
    "DeduplicationSummary",
    "DistilBERTConfig",
    "DistilBERTTextClassifier",
    "FFNNConfig",
    "FeedForwardTextClassifier",
    "IssueDataService",
    "IssueReport",
    "LeakageReport",
    "TfidfConfig",
    "TrainingConfig",
    "ValidationReport",
    "analyse_evaluation",
    "bootstrap_confidence_interval",
    "build_classical_pipeline",
    "build_deep_classifier",
    "calculate_metrics",
    "compare_evaluations",
    "deduplicate_for_evaluation",
    "evaluate_cross_validation",
    "evaluate_holdout_by_repository",
    "make_classical_estimator_factory",
    "make_deep_estimator_factory",
    "minimum_detectable_difference",
    "random_split_control",
    "summarise_repeated_results",
    "time_aware_split",
]
