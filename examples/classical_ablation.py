"""Select Track B preprocessing using duplicate-safe training folds only."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ai4se.classical import TfidfConfig, make_classical_estimator_factory
from ai4se.evaluation import evaluate_cross_validation
from ai4se.service import IssueDataService
from ai4se.statistical_analysis import result_repository_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse the training-only ablation configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="logistic_regression")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-conservative",
        action="store_true",
        help="Include the software-aware conservative cleaner in the grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "classical" / "preprocessing-ablation.json",
    )
    return parser.parse_args()


def _fold_scores(result: dict[str, Any]) -> list[float]:
    repositories = list(result["repositories"].values())
    fold_count = len(repositories[0]["folds"])
    return [
        mean(
            repository["folds"][fold]["metrics"]["weighted_average"]["f1"]
            for repository in repositories
        )
        for fold in range(fold_count)
    ]


def main() -> None:
    """Evaluate cleaning, truncation, title weighting, and character features."""

    args = parse_args()
    levels = ["raw", "light", "full"]
    if args.include_conservative:
        levels.insert(1, "conservative")
    configurations = product(
        levels,
        (1, 3),
        (200, 400, None),
        (False, True),
    )
    service = IssueDataService.from_loader(kind="memory")
    rows = []

    for level, title_weight, max_words, use_character_ngrams in configurations:
        issues = service.prepare(
            "train",
            level=level,
            max_words=max_words,
            title_weight=title_weight,
        )
        tfidf = TfidfConfig(use_character_ngrams=use_character_ngrams)
        result = evaluate_cross_validation(
            issues,
            make_classical_estimator_factory(
                args.model,
                tfidf=tfidf,
                random_state=args.random_state,
            ),
            model_name=args.model,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        scores = _fold_scores(result)
        row = {
            "cleaning_level": level,
            "title_weight": title_weight,
            "max_words": max_words,
            "use_character_ngrams": use_character_ngrams,
            "cross_repository_weighted_f1": result_repository_score(result),
            "fold_mean": mean(scores),
            "fold_standard_deviation": stdev(scores) if len(scores) > 1 else 0.0,
        }
        rows.append(row)
        print(
            f"{level:12s} title={title_weight} max={str(max_words):4s} "
            f"char={str(use_character_ngrams):5s} "
            f"F1={row['cross_repository_weighted_f1']:.4f}"
        )

    rows.sort(key=lambda row: row["cross_repository_weighted_f1"], reverse=True)
    payload = {
        "selection_split": "official_training_only",
        "model": args.model,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "best": rows[0],
        "configurations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} configurations to {args.output}")


if __name__ == "__main__":
    main()
