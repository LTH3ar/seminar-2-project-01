"""Report-oriented analysis of saved classification errors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .model import IssueReport


def analyse_evaluation(
    result: Mapping[str, Any],
    issues: Sequence[IssueReport],
    *,
    worst_count: int = 25,
) -> dict[str, Any]:
    """Build confusion, length, and high-confidence error summaries."""

    issues_by_id = {issue.issue_id: issue for issue in issues}
    errors: list[dict[str, Any]] = []
    confusion_counts: Counter[tuple[str, str]] = Counter()
    length_rows: list[dict[str, Any]] = []

    for repository, values in result["repositories"].items():
        for row in values["predictions"]:
            issue_id = str(row["issue_id"])
            try:
                issue = issues_by_id[issue_id]
            except KeyError as exc:
                raise ValueError(
                    f"No issue text is available for prediction {issue_id}"
                ) from exc
            actual = str(row["actual"])
            predicted = str(row["predicted"])
            correct = actual == predicted
            length_rows.append(
                {
                    "issue_id": issue_id,
                    "word_count": issue.word_count,
                    "correct": correct,
                }
            )
            if correct:
                continue
            confusion_counts[(actual, predicted)] += 1
            confidence = row.get("confidence")
            errors.append(
                {
                    "issue_id": issue_id,
                    "repository": repository,
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": confidence,
                    "title": issue.title.strip()[:160],
                    "excerpt": issue.body.replace("\n", " ").strip()[:240],
                    "word_count": issue.word_count,
                }
            )

    total_errors = len(errors)
    confusion_summary = [
        {
            "actual": actual,
            "predicted": predicted,
            "count": count,
            "share_of_errors": count / total_errors if total_errors else 0.0,
        }
        for (actual, predicted), count in confusion_counts.most_common()
    ]
    errors.sort(
        key=lambda row: (
            row["confidence"] is not None,
            float(row["confidence"] or 0.0),
        ),
        reverse=True,
    )
    return {
        "model_name": result["model_name"],
        "evaluation": result["evaluation"],
        "total_predictions": len(length_rows),
        "total_errors": total_errors,
        "confusions": confusion_summary,
        "errors_by_length": _errors_by_length(length_rows),
        "high_confidence_mistakes": errors[:worst_count],
    }


def _errors_by_length(
    rows: list[dict[str, Any]],
    bins: int = 5,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: int(row["word_count"]))
    summaries = []
    for bin_number in range(bins):
        start = len(ordered) * bin_number // bins
        end = len(ordered) * (bin_number + 1) // bins
        bucket = ordered[start:end]
        if not bucket:
            continue
        correct = sum(bool(row["correct"]) for row in bucket)
        word_counts = sorted(int(row["word_count"]) for row in bucket)
        middle = len(word_counts) // 2
        median = (
            float(word_counts[middle])
            if len(word_counts) % 2
            else (word_counts[middle - 1] + word_counts[middle]) / 2
        )
        summaries.append(
            {
                "bin": bin_number + 1,
                "issues": len(bucket),
                "minimum_words": word_counts[0],
                "maximum_words": word_counts[-1],
                "median_words": median,
                "accuracy": correct / len(bucket),
                "error_rate": 1.0 - correct / len(bucket),
            }
        )
    return summaries
