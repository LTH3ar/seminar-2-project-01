"""Generate LaTeX tables for the project report from saved JSON artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_DIRECTORY = PROJECT_ROOT / "results" / "classical"
DEEP_LEARNING_DIRECTORY = PROJECT_ROOT / "results" / "deep_learning"
BASELINE_DIRECTORY = PROJECT_ROOT / "results" / "baselines"
TABLE_DIRECTORY = PROJECT_ROOT / "report" / "tables"
MODEL_NAMES = {
    "naive_bayes": "Complement Naive Bayes",
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random forest",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _write_table(
    path: Path,
    *,
    columns: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        " & ".join(_escape(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(_escape(value) for value in row) + r" \\" for row in rows)
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_holdout(selection: Mapping[str, Any]) -> dict[str, Any]:
    model = str(selection["selected_model"])
    path = CLASSICAL_DIRECTORY / f"{model}-official-holdout.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run `make classical` before building report tables"
        )
    return _load(path)


def _build_cv_table(selection: Mapping[str, Any]) -> None:
    rows = [
        (
            MODEL_NAMES.get(str(item["model"]), str(item["model"])),
            f"{item['mean_cross_repository_weighted_f1']:.4f}",
            f"{item['standard_deviation']:.4f}",
        )
        for item in selection["ranking"]
    ]
    _write_table(
        TABLE_DIRECTORY / "repeated_cv.tex",
        columns="lrr",
        headers=("Model", "Mean weighted F1", "SD"),
        rows=rows,
    )


def _build_holdout_table(result: Mapping[str, Any]) -> None:
    rows = []
    for repository, values in result["repositories"].items():
        metrics = values["metrics"]
        rows.append(
            (
                repository,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['macro_average']['f1']:.4f}",
                f"{metrics['weighted_average']['f1']:.4f}",
            )
        )
    overall = result["overall"]["repository_average"]
    rows.append(
        (
            "Cross-repository mean",
            f"{overall['accuracy']:.4f}",
            f"{overall['macro_average']['f1']:.4f}",
            f"{overall['weighted_average']['f1']:.4f}",
        )
    )
    _write_table(
        TABLE_DIRECTORY / "official_holdout.tex",
        columns="lrrr",
        headers=("Repository", "Accuracy", "Macro F1", "Weighted F1"),
        rows=rows,
    )


def _build_baseline_table(result: Mapping[str, Any]) -> None:
    selected_score = result["overall"]["repository_average"]["weighted_average"]["f1"]
    rows = [(str(result["model_name"]), f"{selected_score:.4f}")]
    for filename, display_name in (
        ("setfit.json", "SetFit (Sentence Transformer)"),
        ("roberta.json", "RoBERTa"),
        ("fasttext.json", "fastText"),
    ):
        baseline = _load(BASELINE_DIRECTORY / filename)
        rows.append((display_name, f"{baseline['overall']['average']['f1-score']:.4f}"))
    rows.sort(key=lambda row: float(row[1]), reverse=True)
    _write_table(
        TABLE_DIRECTORY / "baseline_comparison.tex",
        columns="lr",
        headers=("Model", "Cross-repository weighted F1"),
        rows=rows,
    )


def _build_ablation_table() -> None:
    result = _load(CLASSICAL_DIRECTORY / "preprocessing-ablation.json")
    rows = [
        (
            item["cleaning_level"],
            item["title_weight"],
            item["max_words"] if item["max_words"] is not None else "unlimited",
            "yes" if item["use_character_ngrams"] else "no",
            f"{item['cross_repository_weighted_f1']:.4f}",
        )
        for item in result["configurations"][:8]
    ]
    _write_table(
        TABLE_DIRECTORY / "preprocessing_ablation.tex",
        columns="lrrlr",
        headers=("Cleaning", "Title weight", "Max words", "Char TF-IDF", "F1"),
        rows=rows,
    )


def _build_uncertainty_table(result: Mapping[str, Any]) -> None:
    analysis = result.get("analysis", {})
    interval = analysis.get("bootstrap_confidence_interval", {})
    detectable = analysis.get(
        "minimum_detectable_accuracy_difference",
        float("nan"),
    )
    rows = (
        ("Point estimate", f"{interval.get('point_estimate', float('nan')):.4f}"),
        ("95 percent bootstrap lower", f"{interval.get('lower', float('nan')):.4f}"),
        ("95 percent bootstrap upper", f"{interval.get('upper', float('nan')):.4f}"),
        (
            "Approximate detectable accuracy difference",
            f"{detectable:.4f}",
        ),
    )
    _write_table(
        TABLE_DIRECTORY / "uncertainty.tex",
        columns="lr",
        headers=("Quantity", "Value"),
        rows=rows,
    )


def _build_robustness_table(selection: Mapping[str, Any]) -> None:
    model = str(selection["selected_model"])
    result = _load(CLASSICAL_DIRECTORY / f"{model}-robustness.json")
    rows = (
        ("Matched random split", f"{result['random_control_weighted_f1']:.4f}"),
        ("Chronological split", f"{result['time_aware_weighted_f1']:.4f}"),
        ("Chronological minus random", f"{result['time_aware_delta']:.4f}"),
    )
    _write_table(
        TABLE_DIRECTORY / "temporal_robustness.tex",
        columns="lr",
        headers=("Protocol", "Weighted F1"),
        rows=rows,
    )


def _build_deep_learning_table() -> None:
    paths = sorted(DEEP_LEARNING_DIRECTORY.glob("*-official-holdout.json"))
    if not paths:
        return
    rows = []
    for path in paths:
        result = _load(path)
        average = result["overall"]["repository_average"]
        rows.append(
            (
                result["model_name"],
                f"{average['accuracy']:.4f}",
                f"{average['macro_average']['f1']:.4f}",
                f"{average['weighted_average']['f1']:.4f}",
            )
        )
    rows.sort(key=lambda row: float(row[-1]), reverse=True)
    _write_table(
        TABLE_DIRECTORY / "deep_learning.tex",
        columns="lrrr",
        headers=("Model", "Accuracy", "Macro F1", "Weighted F1"),
        rows=rows,
    )


def main() -> None:
    """Read experiment outputs and write every table included by the report."""

    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    selection = _load(CLASSICAL_DIRECTORY / "model-selection.json")
    holdout = _selected_holdout(selection)
    _build_cv_table(selection)
    _build_holdout_table(holdout)
    _build_baseline_table(holdout)
    _build_ablation_table()
    _build_uncertainty_table(holdout)
    _build_robustness_table(selection)
    _build_deep_learning_table()
    print(f"Generated report tables in {TABLE_DIRECTORY}")


if __name__ == "__main__":
    main()
