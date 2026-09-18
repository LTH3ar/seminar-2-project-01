"""Consolidated issue-report classification utilities."""

from .audit import DatasetAudit, LeakageReport
from .classical import (
    AVAILABLE_MODELS,
    TfidfConfig,
    build_classical_pipeline,
    make_classical_estimator_factory,
)
from .evaluation import (
    calculate_metrics,
    evaluate_cross_validation,
    evaluate_holdout_by_repository,
)
from .model import IssueReport
from .service import IssueDataService
from .validation import DatasetValidationError, ValidationReport

__all__ = [
    "AVAILABLE_MODELS",
    "DatasetAudit",
    "DatasetValidationError",
    "IssueDataService",
    "IssueReport",
    "LeakageReport",
    "TfidfConfig",
    "ValidationReport",
    "build_classical_pipeline",
    "calculate_metrics",
    "evaluate_cross_validation",
    "evaluate_holdout_by_repository",
    "make_classical_estimator_factory",
]
