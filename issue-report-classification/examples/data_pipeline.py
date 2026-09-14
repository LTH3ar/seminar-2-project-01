"""Run the integrated data, audit, preprocessing, and fold workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from issue_classifier.eda import cleaning_impact, overview
from issue_classifier.loader import ensure_dataset
from issue_classifier.preprocessing import TextCleaningConfig, TextPreprocessor
from issue_classifier.service import IssueDataService, create_issue_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("csv", "memory"), default="csv")
    parser.add_argument(
        "--cleaning-level",
        choices=("conservative", "raw", "light", "full"),
        default="conservative",
    )
    parser.add_argument("--max-words", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_directory = PROJECT_ROOT / "data"
    ensure_dataset(data_directory)

    repository = create_issue_repository(args.backend, data_directory)
    preprocessor = TextPreprocessor(
        TextCleaningConfig(
            level=args.cleaning_level,
            max_words=args.max_words,
        )
    )
    service = IssueDataService(repository, preprocessor)

    train = service.load("train")
    prepared = service.prepare("train")
    folds = service.make_folds(n_splits=5, random_state=42)

    print(json.dumps(service.audit("train").to_dict(), indent=2))
    print(json.dumps(service.audit_leakage().to_dict(), indent=2))
    print(overview(train).to_string(index=False))
    print(cleaning_impact(train[:100]).to_string())
    print(prepared[0].to_dict())
    print({repo: len(repo_folds) for repo, repo_folds in folds.items()})


if __name__ == "__main__":
    main()
