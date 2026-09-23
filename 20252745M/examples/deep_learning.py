"""Run Track C neural and transformer experiments."""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from ai4se.deep_learning import (
    AVAILABLE_DEEP_MODELS,
    CNNConfig,
    DistilBERTConfig,
    FFNNConfig,
    TrainingConfig,
    deep_model_display_name,
    make_deep_estimator_factory,
    normalise_deep_model_name,
    plot_learning_history,
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
    """Parse reproducible Track C experiment settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an FFNN, TextCNN, or fine-tuned DistilBERT through the "
            "same per-repository protocol as Tracks B and D."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("official", "cross-validation"),
        default="official",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(AVAILABLE_DEEP_MODELS),
        help="Model names or aliases: ffnn, cnn, distilbert.",
    )
    parser.add_argument("--backend", choices=("memory", "file"), default="memory")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cleaning-level",
        choices=("auto", "raw", "conservative", "light", "full"),
        default="auto",
        help="Auto uses raw text for FFNN/CNN and light cleaning for DistilBERT.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=-1,
        help="-1 uses the model default; 0 disables word truncation.",
    )
    parser.add_argument(
        "--title-weight",
        type=int,
        default=0,
        help="0 uses the model default (3 for FFNN/CNN, 1 for DistilBERT).",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--transformer-model",
        default="distilbert-base-uncased",
        help="HuggingFace checkpoint used by the DistilBERT adapter.",
    )
    parser.add_argument("--transformer-max-length", type=int, default=384)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "results" / "deep_learning",
    )
    parser.add_argument(
        "--table-directory",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save confusion matrices and per-repository learning curves.",
    )
    return parser.parse_args()


def _training_config(
    default: TrainingConfig,
    args: argparse.Namespace,
) -> TrainingConfig:
    updates: dict[str, Any] = {
        "random_state": args.random_state,
        "device": args.device,
        "verbose": args.verbose,
    }
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.patience is not None:
        updates["patience"] = args.patience
    return replace(default, **updates)


def _model_config(model: str, args: argparse.Namespace):
    if model == "ffnn":
        default = FFNNConfig()
        return replace(
            default,
            training=_training_config(default.training, args),
        )
    if model == "cnn":
        default = CNNConfig()
        return replace(
            default,
            training=_training_config(default.training, args),
        )
    default = DistilBERTConfig(
        model_id=args.transformer_model,
        max_sequence_length=args.transformer_max_length,
    )
    return replace(
        default,
        training=_training_config(default.training, args),
    )


def _preprocessing(model: str, args: argparse.Namespace) -> dict[str, Any]:
    if model == "distilbert":
        defaults = {"level": "light", "max_words": None, "title_weight": 1}
    else:
        defaults = {"level": "raw", "max_words": 400, "title_weight": 3}
    if args.cleaning_level != "auto":
        defaults["level"] = args.cleaning_level
    if args.max_words >= 0:
        defaults["max_words"] = args.max_words or None
    if args.title_weight > 0:
        defaults["title_weight"] = args.title_weight
    return defaults


def _plot_training_curves(
    result: dict[str, Any],
    *,
    model: str,
    output_directory: Path,
) -> None:
    for repository, values in result["repositories"].items():
        histories = []
        if "training" in values:
            histories.append(("official", values["training"]["history"]))
        for fold in values.get("folds", []):
            if "training" in fold:
                histories.append((f"fold-{fold['fold']}", fold["training"]["history"]))
        repository_slug = re.sub(r"[^a-z0-9]+", "-", repository.casefold()).strip("-")
        for run_name, history in histories:
            figure = plot_learning_history(
                history,
                title=f"{deep_model_display_name(model)} - {repository} - {run_name}",
            )
            destination = output_directory / model / repository_slug / f"{run_name}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=200, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(figure)


def main() -> None:
    """Train requested Track C models and save shared-schema results."""

    args = parse_args()
    if args.max_words < -1:
        raise ValueError("--max-words must be -1, zero, or positive")
    if args.title_weight < 0:
        raise ValueError("--title-weight must be zero or positive")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")

    models = list(
        dict.fromkeys(normalise_deep_model_name(name) for name in args.models)
    )
    service = IssueDataService.from_loader(kind=args.backend)
    leakage = service.audit_leakage()
    completed_results = []

    for model in models:
        preprocessing = _preprocessing(model, args)
        config = _model_config(model, args)
        train = service.prepare("train", **preprocessing)
        metadata = {
            "track": "C",
            "model": model,
            "configuration": config.to_dict(),
            "preprocessing": preprocessing,
            "known_train_test_overlap_groups": leakage.overlap_group_count,
            "selection_note": (
                "Architecture and hyperparameters are fixed before official "
                "holdout evaluation. Use cross-validation mode for tuning."
            ),
        }
        factory = make_deep_estimator_factory(
            model,
            config=config,
            random_state=args.random_state,
        )
        if args.mode == "official":
            test = service.prepare("test", **preprocessing)
            result = evaluate_holdout_by_repository(
                train,
                test,
                factory,
                model_name=deep_model_display_name(model),
                random_state=args.random_state,
                metadata=metadata,
            )
            suffix = "official-holdout"
        else:
            result = evaluate_cross_validation(
                train,
                factory,
                model_name=deep_model_display_name(model),
                n_splits=args.n_splits,
                random_state=args.random_state,
                metadata=metadata,
            )
            suffix = "cross-validation"

        destination = args.output_directory / f"{model}-{suffix}.json"
        save_evaluation(result, destination)
        completed_results.append(result)
        score = result["overall"]["repository_average"]["weighted_average"]["f1"]
        print(f"{model:12s} {suffix:18s} weighted-F1={score:.4f}")
        print(f"Saved evaluation to {destination}")
        if args.plots:
            figure_directory = PROJECT_ROOT / "results" / "figures" / "deep_learning"
            plot_confusion_matrices(
                result,
                figure_directory / model / suffix / "confusion_matrices",
            )
            _plot_training_curves(
                result,
                model=model,
                output_directory=figure_directory / suffix / "learning_curves",
            )

    rows = [row for result in completed_results for row in result_rows(result)]
    csv_path, markdown_path = write_result_tables(
        rows,
        args.table_directory,
        stem="deep_learning",
    )
    print(f"Saved result tables to {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
