"""Run Track B classical model selection and official evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai4se.classical import (
    AVAILABLE_MODELS,
    TfidfConfig,
    classifier_parameters,
    make_classical_estimator_factory,
    model_display_name,
    normalise_model_name,
)
from ai4se.evaluation import (
    evaluate_cross_validation,
    evaluate_holdout_by_repository,
    plot_confusion_matrices,
    save_evaluation,
)
from ai4se.reporting import result_rows, write_result_tables
from ai4se.service import IssueDataService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse Track B experiment configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare classical TF-IDF classifiers with grouped CV and then "
            "evaluate the selected model on the official test split."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("all", "cross-validation", "official"),
        default="all",
        help=(
            "'all' selects by CV and tests only the winner; 'official' "
            "evaluates every requested model directly on the official split."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(AVAILABLE_MODELS),
        help="Model names or aliases: nb, lr, svm, rf.",
    )
    parser.add_argument(
        "--backend",
        choices=("memory", "file"),
        default="memory",
    )
    parser.add_argument(
        "--cleaning-level",
        choices=("raw", "conservative", "light", "full"),
        default="full",
    )
    parser.add_argument("--max-words", type=int, default=1_000)
    parser.add_argument("--title-weight", type=int, default=2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--word-only",
        action="store_true",
        help="Disable character TF-IDF features for an ablation run.",
    )
    parser.add_argument(
        "--official-all",
        action="store_true",
        help="In all mode, evaluate every model rather than only the CV winner.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate confusion matrices for every completed evaluation.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "results" / "classical",
    )
    parser.add_argument(
        "--table-directory",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables",
    )
    return parser.parse_args()


def _score(result: dict[str, Any]) -> float:
    return float(
        result["overall"]["repository_average"]["macro_average"]["f1"]
    )


def _metadata(
    *,
    model: str,
    args: argparse.Namespace,
    tfidf: TfidfConfig,
    overlap_groups: int,
) -> dict[str, Any]:
    return {
        "track": "B",
        "model": model,
        "cleaning_level": args.cleaning_level,
        "max_words": args.max_words or None,
        "title_weight": args.title_weight,
        "tfidf": tfidf.to_dict(),
        "classifier": classifier_parameters(
            model,
            random_state=args.random_state,
        ),
        "known_train_test_overlap_groups": overlap_groups,
    }


def _save_result_artifacts(
    result: dict[str, Any],
    *,
    model: str,
    suffix: str,
    args: argparse.Namespace,
) -> Path:
    path = args.output_directory / f"{model}-{suffix}.json"
    save_evaluation(result, path)
    if args.plots:
        plot_confusion_matrices(
            result,
            PROJECT_ROOT
            / "results"
            / "figures"
            / "classical"
            / model
            / suffix,
        )
    print(f"{model:20s} {suffix:18s} macro-F1={_score(result):.4f}")
    return path


def _write_selection(
    cross_validation_results: dict[str, dict[str, Any]],
    output_directory: Path,
) -> str:
    ranking = sorted(
        (
            {"model": model, "cross_repository_macro_f1": _score(result)}
            for model, result in cross_validation_results.items()
        ),
        key=lambda row: row["cross_repository_macro_f1"],
        reverse=True,
    )
    selection = {
        "selection_metric": "cross_repository_macro_f1",
        "selected_model": ranking[0]["model"],
        "ranking": ranking,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "model-selection.json"
    path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {selection['selected_model']} from grouped cross-validation")
    return str(selection["selected_model"])


def main() -> None:
    """Run Track B without exposing the official test set during selection."""

    args = parse_args()
    if args.max_words < 0:
        raise ValueError("--max-words must be zero (unlimited) or positive")
    if args.title_weight < 1:
        raise ValueError("--title-weight must be at least 1")

    models = list(dict.fromkeys(normalise_model_name(name) for name in args.models))
    tfidf = TfidfConfig(use_character_ngrams=not args.word_only)
    service = IssueDataService.from_loader(kind=args.backend)
    leakage = service.audit_leakage()
    train = service.prepare(
        "train",
        level=args.cleaning_level,
        max_words=args.max_words or None,
        title_weight=args.title_weight,
    )
    completed_results: list[dict[str, Any]] = []
    cross_validation_results: dict[str, dict[str, Any]] = {}

    if args.mode in {"all", "cross-validation"}:
        for model in models:
            metadata = _metadata(
                model=model,
                args=args,
                tfidf=tfidf,
                overlap_groups=leakage.overlap_group_count,
            )
            result = evaluate_cross_validation(
                train,
                make_classical_estimator_factory(
                    model,
                    tfidf=tfidf,
                    random_state=args.random_state,
                ),
                model_name=model_display_name(model),
                n_splits=args.n_splits,
                random_state=args.random_state,
                metadata=metadata,
            )
            cross_validation_results[model] = result
            completed_results.append(result)
            _save_result_artifacts(
                result,
                model=model,
                suffix="cross-validation",
                args=args,
            )

    selected_model: str | None = None
    if cross_validation_results:
        selected_model = _write_selection(
            cross_validation_results,
            args.output_directory,
        )

    if args.mode in {"all", "official"}:
        test = service.prepare(
            "test",
            level=args.cleaning_level,
            max_words=args.max_words or None,
            title_weight=args.title_weight,
        )
        official_models = models
        if args.mode == "all" and not args.official_all:
            official_models = [selected_model] if selected_model else []
        for model in official_models:
            metadata = _metadata(
                model=model,
                args=args,
                tfidf=tfidf,
                overlap_groups=leakage.overlap_group_count,
            )
            if selected_model:
                metadata["selected_by_cross_validation"] = model == selected_model
            result = evaluate_holdout_by_repository(
                train,
                test,
                make_classical_estimator_factory(
                    model,
                    tfidf=tfidf,
                    random_state=args.random_state,
                ),
                model_name=model_display_name(model),
                random_state=args.random_state,
                metadata=metadata,
            )
            completed_results.append(result)
            _save_result_artifacts(
                result,
                model=model,
                suffix="official-holdout",
                args=args,
            )

    rows = [row for result in completed_results for row in result_rows(result)]
    csv_path, markdown_path = write_result_tables(
        rows,
        args.table_directory,
        stem="classical_ml",
    )
    print(f"Saved result tables to {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
