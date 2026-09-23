"""Run the supplied SetFit baseline through the shared evaluator."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ai4se.evaluation import (
    evaluate_cross_validation,
    evaluate_holdout_by_repository,
    plot_confusion_matrices,
    save_evaluation,
)
from ai4se.model import IssueReport
from ai4se.service import IssueDataService
from ai4se.setfit_baseline import SetFitConfig, make_setfit_factory


def parse_args() -> argparse.Namespace:
    """Parse reproducible SetFit experiment settings."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("official", "cross-validation"),
        default="official",
        help="Use the official test split or duplicate-safe training folds.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--backend", choices=("memory", "file"), default="memory")
    parser.add_argument(
        "--cleaning-level",
        choices=("raw", "conservative", "light", "full"),
        default="raw",
    )
    parser.add_argument("--max-words", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluations/setfit-official.json"),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=Path("results/models/setfit"),
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=Path("results/figures/setfit"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train SetFit and save metrics, predictions, and plots."""

    args = parse_args()
    config = SetFitConfig(
        model_id=args.model,
        random_state=args.random_state,
    )
    service = IssueDataService.from_loader(kind=args.backend)
    train = prepare_issues(
        service,
        "train",
        cleaning_level=args.cleaning_level,
        max_words=args.max_words,
        random_state=args.random_state,
    )
    factory = make_setfit_factory(config, args.model_directory)
    metadata = {
        "setfit": config.to_dict(),
        "cleaning_level": args.cleaning_level,
        "max_words": args.max_words,
        "source": "notebooks/02_setfit_baseline.ipynb",
    }

    if args.mode == "official":
        test = prepare_issues(
            service,
            "test",
            cleaning_level=args.cleaning_level,
            max_words=args.max_words,
            random_state=args.random_state,
        )
        result = evaluate_holdout_by_repository(
            train,
            test,
            factory,
            model_name="SetFit all-mpnet-base-v2",
            random_state=args.random_state,
            metadata=metadata,
        )
    else:
        result = evaluate_cross_validation(
            train,
            factory,
            model_name="SetFit all-mpnet-base-v2",
            n_splits=args.n_splits,
            random_state=args.random_state,
            metadata=metadata,
        )

    result_path = save_evaluation(result, args.output)
    print(f"Saved evaluation to {result_path}")
    overall = result["overall"]["repository_average"]["macro_average"]
    print(f"Cross-repository macro F1: {overall['f1']:.4f}")
    if not args.no_plots:
        for path in plot_confusion_matrices(result, args.figure_directory):
            print(f"Saved figure to {path}")


def prepare_issues(
    service: IssueDataService,
    split: str,
    *,
    cleaning_level: str,
    max_words: int | None,
    random_state: int,
) -> list[IssueReport]:
    """Prepare and shuffle a split like the supplied HuggingFace notebook."""

    if cleaning_level == "raw":
        issues = [
            replace(issue, clean_text=f"{issue.title} {issue.body}".strip())
            for issue in service.load(split)
        ]
    else:
        issues = service.prepare(
            split,
            level=cleaning_level,
            max_words=max_words,
        )
    generator = np.random.default_rng(random_state)
    return [issues[index] for index in generator.permutation(len(issues))]


if __name__ == "__main__":
    main()
