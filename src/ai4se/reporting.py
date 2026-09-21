"""Result-table and comparison-plot generation."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

TABLE_COLUMNS = (
    "model",
    "evaluation",
    "repository",
    "sample_count",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "fold_macro_f1_mean",
    "fold_macro_f1_std",
)


def result_rows(
    result: Mapping[str, Any],
    *,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Convert either the shared schema or a supplied baseline JSON to rows."""

    if result.get("schema_version") == 1:
        return _shared_schema_rows(result)
    return _legacy_baseline_rows(result, model_name=model_name)


def load_result_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load and combine result rows from multiple JSON files."""

    rows: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        rows.extend(result_rows(result, model_name=path.stem))
    return rows


def write_result_tables(
    rows: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
    *,
    stem: str = "model_comparison",
) -> tuple[Path, Path]:
    """Write equivalent CSV and Markdown comparison tables."""

    if not rows:
        raise ValueError("At least one result row is required")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{stem}.csv"
    markdown_path = output_directory / f"{stem}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TABLE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in TABLE_COLUMNS} for row in rows
        )

    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(TABLE_COLUMNS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |\n")
        for row in rows:
            values = [
                _format_table_value(row.get(column, "")) for column in TABLE_COLUMNS
            ]
            handle.write("| " + " | ".join(values) + " |\n")
    return csv_path, markdown_path


def plot_model_comparison(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    repository: str = "overall",
) -> Path:
    """Plot macro-F1 for each model at one comparison level."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to create result comparison plots"
        ) from exc

    selected = [row for row in rows if row["repository"] == repository]
    if not selected:
        raise ValueError(f"No result rows found for repository {repository!r}")
    labels = [str(row["model"]) for row in selected]
    values = [float(row["macro_f1"]) for row in selected]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(labels, values, color="#176B87")
    axis.set(
        title=f"Model comparison - {repository}",
        ylabel="Macro F1",
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def _shared_schema_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        _row_from_metrics(
            model=str(result["model_name"]),
            evaluation=str(result["evaluation"]),
            repository=repository,
            sample_count=values["sample_count"],
            metrics=values["metrics"],
            fold_summary=values.get("fold_summary"),
        )
        for repository, values in result["repositories"].items()
    ]
    rows.append(
        _row_from_metrics(
            model=str(result["model_name"]),
            evaluation=str(result["evaluation"]),
            repository="overall",
            sample_count=result["overall"]["sample_count"],
            metrics=result["overall"]["repository_average"],
        )
    )
    return rows


def _legacy_baseline_rows(
    result: Mapping[str, Any],
    *,
    model_name: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repository, values in result.items():
        if not isinstance(values, Mapping) or "average" not in values:
            continue
        average = values["average"]
        if not isinstance(average, Mapping) or "f1-score" not in average:
            continue
        # Every official repository split has equal class support, making the
        # supplied weighted average equal to the macro average.
        rows.append(
            {
                "model": model_name or "legacy-baseline",
                "evaluation": "official_holdout",
                "repository": repository,
                "sample_count": 1500 if repository == "overall" else 300,
                "accuracy": float(average["recall"]),
                "macro_precision": float(average["precision"]),
                "macro_recall": float(average["recall"]),
                "macro_f1": float(average["f1-score"]),
                "weighted_f1": float(average["f1-score"]),
                "fold_macro_f1_mean": "",
                "fold_macro_f1_std": "",
            }
        )
    if not rows:
        raise ValueError("The JSON file is not a supported evaluation result")
    return rows


def _row_from_metrics(
    *,
    model: str,
    evaluation: str,
    repository: str,
    sample_count: int,
    metrics: Mapping[str, Any],
    fold_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    macro = metrics["macro_average"]
    weighted = metrics["weighted_average"]
    row = {
        "model": model,
        "evaluation": evaluation,
        "repository": repository,
        "sample_count": sample_count,
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "macro_f1": float(macro["f1"]),
        "weighted_f1": float(weighted["f1"]),
    }
    row["fold_macro_f1_mean"] = (
        float(fold_summary["macro_f1_mean"]) if fold_summary else ""
    )
    row["fold_macro_f1_std"] = (
        float(fold_summary["macro_f1_std"]) if fold_summary else ""
    )
    return row


def _format_table_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")
