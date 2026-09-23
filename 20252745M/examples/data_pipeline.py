"""Run validation, auditing, cleaning, EDA, and fold generation."""

from __future__ import annotations

import argparse
import json

from ai4se.eda import cleaning_impact, overview
from ai4se.service import IssueDataService


def parse_args() -> argparse.Namespace:
    """Parse pipeline configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("memory", "file"), default="memory")
    parser.add_argument(
        "--cleaning-level",
        choices=("raw", "conservative", "light", "full"),
        default="conservative",
    )
    parser.add_argument("--max-words", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    """Execute the consolidated workflow."""
    args = parse_args()
    service = IssueDataService.from_loader(kind=args.backend)
    train_repository = service.repositories["train"]
    prepared = service.prepare(
        "train",
        level=args.cleaning_level,
        max_words=args.max_words,
    )

    print(json.dumps(service.audit("train").to_dict(), indent=2))
    print(json.dumps(service.audit_leakage().to_dict(), indent=2))
    print(overview(train_repository).to_string(index=False))
    print(cleaning_impact(train_repository, sample=100).to_string())
    print(
        {
            "repo": prepared[0].repo,
            "label": prepared[0].label,
            "clean_text_preview": prepared[0].clean_text[:160],
        }
    )
    print({repo: len(folds) for repo, folds in service.make_folds().items()})


if __name__ == "__main__":
    main()
