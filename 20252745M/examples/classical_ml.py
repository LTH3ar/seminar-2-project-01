"""Run Track B classical model selection and official evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
from ai4se.error_analysis import analyse_evaluation
from ai4se.evaluation import (
    evaluate_cross_validation,
    evaluate_holdout_by_repository,
    plot_confusion_matrices,
    save_evaluation,
)
from ai4se.model import IssueReport
from ai4se.preprocessing import clean_issue
from ai4se.reporting import result_rows, write_result_tables
from ai4se.service import IssueDataService
from ai4se.splits import random_split_control, time_aware_split
from ai4se.statistical_analysis import (
    bootstrap_confidence_interval,
    minimum_detectable_difference,
    result_repository_score,
    summarise_repeated_results,
)

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
        default="raw",
    )
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--title-weight", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="CV seeds used to estimate model-selection stability.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
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
        "--skip-temporal-analysis",
        action="store_true",
        help="Skip the matched random versus chronological robustness check.",
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
    return result_repository_score(result)


def _metadata(
    *,
    model: str,
    args: argparse.Namespace,
    tfidf: TfidfConfig,
    overlap_groups: int,
    random_state: int | None = None,
) -> dict[str, Any]:
    effective_random_state = args.random_state if random_state is None else random_state
    return {
        "track": "B",
        "model": model,
        "cleaning_level": args.cleaning_level,
        "max_words": args.max_words or None,
        "title_weight": args.title_weight,
        "tfidf": tfidf.to_dict(),
        "classifier": classifier_parameters(
            model,
            random_state=effective_random_state,
        ),
        "cross_validation_seeds": list(args.seeds),
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
            PROJECT_ROOT / "results" / "figures" / "classical" / model / suffix,
        )
    print(f"{model:20s} {suffix:18s} weighted-F1={_score(result):.4f}")
    return path


def _write_selection(
    repeated_results: dict[str, dict[str, Any]],
    output_directory: Path,
    *,
    detectable_difference: float,
) -> tuple[str, dict[str, Any]]:
    ranking = sorted(
        (
            {
                "model": model,
                "mean_cross_repository_weighted_f1": result["mean"],
                "standard_deviation": result["standard_deviation"],
                "scores": result["scores"],
            }
            for model, result in repeated_results.items()
        ),
        key=lambda row: row["mean_cross_repository_weighted_f1"],
        reverse=True,
    )
    lead = (
        ranking[0]["mean_cross_repository_weighted_f1"]
        - ranking[1]["mean_cross_repository_weighted_f1"]
        if len(ranking) > 1
        else None
    )
    selection = {
        "selection_metric": "mean_cross_repository_weighted_f1",
        "selected_model": ranking[0]["model"],
        "minimum_detectable_difference": detectable_difference,
        "lead_over_runner_up": lead,
        "lead_exceeds_approximate_accuracy_mdd": (
            lead >= detectable_difference if lead is not None else None
        ),
        "note": (
            "The highest repeated-CV mean is selected operationally. A lead below "
            "the approximate holdout accuracy detection threshold is not claimed "
            "as superiority; the threshold is contextual because selection uses F1."
        ),
        "ranking": ranking,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "model-selection.json"
    path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected {selection['selected_model']} by repeated grouped cross-validation"
    )
    return str(selection["selected_model"]), selection


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _prepare_issues(
    issues: list[IssueReport],
    *,
    cleaning_level: str,
    max_words: int | None,
    title_weight: int,
) -> list[IssueReport]:
    return [
        clean_issue(
            issue,
            level=cleaning_level,
            max_words=max_words,
            title_weight=title_weight,
        )
        for issue in issues
    ]


def _run_temporal_analysis(
    *,
    model: str,
    service: IssueDataService,
    tfidf: TfidfConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    combined = [
        *service.load("train", validate=False),
        *service.load("test", validate=False),
    ]
    temporal_train, temporal_test, temporal_deduplication = time_aware_split(combined)
    random_train, random_test, random_deduplication = random_split_control(
        combined,
        random_state=args.random_state,
    )
    prepare_options = {
        "cleaning_level": args.cleaning_level,
        "max_words": args.max_words or None,
        "title_weight": args.title_weight,
    }
    factory = make_classical_estimator_factory(
        model,
        tfidf=tfidf,
        random_state=args.random_state,
    )
    temporal_result = evaluate_holdout_by_repository(
        _prepare_issues(temporal_train, **prepare_options),
        _prepare_issues(temporal_test, **prepare_options),
        factory,
        model_name=model_display_name(model),
        random_state=args.random_state,
        metadata={"split_protocol": "time_aware"},
    )
    temporal_result["evaluation"] = "time_aware_holdout"
    random_result = evaluate_holdout_by_repository(
        _prepare_issues(random_train, **prepare_options),
        _prepare_issues(random_test, **prepare_options),
        factory,
        model_name=model_display_name(model),
        random_state=args.random_state,
        metadata={"split_protocol": "matched_random_control"},
    )
    random_result["evaluation"] = "matched_random_holdout"
    save_evaluation(
        temporal_result,
        args.output_directory / f"{model}-time-aware.json",
    )
    save_evaluation(
        random_result,
        args.output_directory / f"{model}-random-control.json",
    )
    summary = {
        "model": model,
        "time_aware_weighted_f1": _score(temporal_result),
        "random_control_weighted_f1": _score(random_result),
        "time_aware_delta": _score(temporal_result) - _score(random_result),
        "temporal_deduplication": asdict(temporal_deduplication),
        "random_control_deduplication": asdict(random_deduplication),
    }
    _write_json(summary, args.output_directory / f"{model}-robustness.json")
    return summary


def main() -> None:
    """Run Track B without exposing the official test set during selection."""

    args = parse_args()
    if args.max_words < 0:
        raise ValueError("--max-words must be zero (unlimited) or positive")
    if args.title_weight < 1:
        raise ValueError("--title-weight must be at least 1")
    if not args.seeds:
        raise ValueError("At least one cross-validation seed is required")
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")

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
    repeated_results: dict[str, dict[str, Any]] = {}
    detectable_difference = minimum_detectable_difference()

    if args.mode in {"all", "cross-validation"}:
        for model in models:
            runs = []
            for seed in args.seeds:
                metadata = _metadata(
                    model=model,
                    args=args,
                    tfidf=tfidf,
                    overlap_groups=leakage.overlap_group_count,
                    random_state=seed,
                )
                runs.append(
                    evaluate_cross_validation(
                        train,
                        make_classical_estimator_factory(
                            model,
                            tfidf=tfidf,
                            random_state=seed,
                        ),
                        model_name=model_display_name(model),
                        n_splits=args.n_splits,
                        random_state=seed,
                        metadata=metadata,
                    )
                )
            result = runs[0]
            summary = summarise_repeated_results(runs)
            repeated_results[model] = summary
            completed_results.append(result)
            _save_result_artifacts(
                result,
                model=model,
                suffix="cross-validation",
                args=args,
            )
            _write_json(
                summary,
                args.output_directory / f"{model}-repeated-cross-validation.json",
            )
            print(
                f"{model:20s} repeated CV        "
                f"weighted-F1={summary['mean']:.4f} +/- "
                f"{summary['standard_deviation']:.4f}"
            )

    selected_model: str | None = None
    selection: dict[str, Any] | None = None
    if repeated_results:
        selected_model, selection = _write_selection(
            repeated_results,
            args.output_directory,
            detectable_difference=detectable_difference,
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
            result["analysis"] = {
                "bootstrap_confidence_interval": bootstrap_confidence_interval(
                    result,
                    n_resamples=args.bootstrap_resamples,
                    random_state=args.random_state,
                ),
                "minimum_detectable_accuracy_difference": detectable_difference,
                "selection": selection,
            }
            completed_results.append(result)
            _save_result_artifacts(
                result,
                model=model,
                suffix="official-holdout",
                args=args,
            )
            _write_json(
                analyse_evaluation(result, test),
                args.output_directory / f"{model}-error-analysis.json",
            )

        if args.mode == "all" and selected_model and not args.skip_temporal_analysis:
            robustness = _run_temporal_analysis(
                model=selected_model,
                service=service,
                tfidf=tfidf,
                args=args,
            )
            print(
                "Temporal robustness: "
                f"random={robustness['random_control_weighted_f1']:.4f}, "
                f"time-aware={robustness['time_aware_weighted_f1']:.4f}"
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
