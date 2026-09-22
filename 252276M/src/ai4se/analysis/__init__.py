"""Analysis sub-package for deep-dive evaluation."""

from .error_analysis import (
    confusion_summary,
    high_confidence_errors,
    length_quintile_accuracy,
    misclassification_pairs,
    per_class_per_repo_errors,
    template_lift_analysis,
)

__all__ = [
    "confusion_summary",
    "high_confidence_errors",
    "length_quintile_accuracy",
    "misclassification_pairs",
    "per_class_per_repo_errors",
    "template_lift_analysis",
]
