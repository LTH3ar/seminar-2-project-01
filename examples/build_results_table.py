"""Build shared tables and a comparison plot from experiment JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai4se.reporting import (
    load_result_rows,
    plot_model_comparison,
    write_result_tables,
)

DEFAULT_RESULTS = (
    Path("results/baselines/setfit.json"),
    Path("results/baselines/roberta.json"),
    Path("results/baselines/fasttext.json"),
)


def parse_args() -> argparse.Namespace:
    """Parse result artifact paths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/tables"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/figures/model-comparison.png"),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Create report-ready result artifacts."""

    args = parse_args()
    rows = load_result_rows(args.results)
    csv_path, markdown_path = write_result_tables(rows, args.output_directory)
    print(f"Saved table to {csv_path}")
    print(f"Saved table to {markdown_path}")
    if not args.no_plot:
        figure_path = plot_model_comparison(rows, args.figure)
        print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
