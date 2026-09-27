"""Phase 1 -- reproduce the published SetFit baseline.

Nothing else in this project means anything until this number exists. The
competition releases aggregate scores but not per-issue predictions, so no
paired significance test against the published figure is possible. Running the
baseline ourselves is what makes every later comparison testable.

    python run_phase1.py                 # one seed, the official configuration
    python run_phase1.py --seeds 41 42 43 44 45

Expect about 15 minutes per seed on a 24 GB card. Predictions are written to
``results/`` after every seed, so an interruption costs one seed, not the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import audit, metrics, stats  # noqa: E402
from src.data import describe, load_dataset  # noqa: E402
from src.runner import OFFICIAL, Config, gpu_report, run  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / "data"


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Training seeds. 42 is the organisers'; use five for a variance estimate.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Do not re-run a seed whose results file already exists.",
    )
    return parser.parse_args()


def evaluate(record: dict, test) -> dict:
    """Score one run and attach the comparison against the published figures."""
    truth = test.frame.label.tolist()
    repos = test.frame.repo.tolist()
    report = metrics.full_report(truth, record["predictions"], repos)
    point, low, high = stats.bootstrap_ci(
        truth, record["predictions"], repos, n_resamples=2000
    )
    report["ci95"] = [low, high]
    report["verdict_vs_file"] = metrics.verdict(point, metrics.PUBLISHED_OVERALL_FILE)
    report["verdict_vs_readme"] = metrics.verdict(
        point, metrics.PUBLISHED_OVERALL_README
    )
    return report


def show(seed: int, report: dict, minutes: float) -> None:
    """Print one seed's result against the published per-project figures."""
    print(f"\n  seed {seed} -- {minutes:.1f} min")
    print(f"    {'project':<24} {'ours':>8} {'published':>10} {'delta':>9}")
    for repo, value in report["per_repo"].items():
        reference = metrics.PUBLISHED[repo]
        print(f"    {repo:<24} {value:>8.4f} {reference:>10.4f} "
              f"{value - reference:>+9.4f}")
    overall = report["cross_repo_f1"]
    low, high = report["ci95"]
    print(f"    {'-' * 54}")
    print(f"    {'cross-repository':<24} {overall:>8.4f}")
    print(f"    95% CI [{low:.4f}, {high:.4f}]")
    print(f"    vs committed 0.8240 : {report['vs_published_file']:+.4f}"
          f"   -> {report['verdict_vs_file']}")
    print(f"    vs README    0.8270 : {report['vs_published_readme']:+.4f}"
          f"   -> {report['verdict_vs_readme']}")


def main() -> None:
    """Run the official configuration for each requested seed."""
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)

    print("=" * 74)
    print("PHASE 1 -- REPRODUCING THE PUBLISHED SETFIT BASELINE")
    print("=" * 74)
    print(f"  GPU: {gpu_report()}")
    print(f"  {OFFICIAL.notes}\n")

    data = load_dataset(CACHE, results_dir=RESULTS, reason="phase1-baseline")
    train, test = data["train"], data["test"]
    print(f"  loaded {len(train)} train / {len(test)} test")
    print("  " + describe(train).to_string().replace("\n", "\n  "))

    print("\n  leakage audit (train <-> test)")
    overlap = audit.leakage(train.frame, test.frame)
    audit.write(RESULTS, "leakage_audit", overlap)
    print(f"    exact duplicates       : {overlap['exact_duplicates_train_test']}")
    print(f"    identical titles       : {overlap['identical_titles']}")
    for key, value in overlap["near_duplicates"].items():
        print(f"    {key:<22} : {value['n']:4d}  ({value['share']:.2%})")
    print("    -> belongs in Threats to Validity; see results/leakage_audit.json")

    every = {}
    for seed in args.seeds:
        path = RESULTS / f"phase1_seed{seed}.json"
        if args.skip_existing and path.exists():
            print(f"\n  seed {seed}: reusing {path.name}")
            every[seed] = json.loads(path.read_text(encoding="utf-8"))
            continue

        print(f"\n  training, seed {seed}")
        config = Config(**{**OFFICIAL.as_dict(), "seed": seed})
        record = run(config, train, test)
        record["report"] = evaluate(record, test)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        every[seed] = record
        show(seed, record["report"], record["minutes"])

    scores = [r["report"]["cross_repo_f1"] for r in every.values()]
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print("  " + "  ".join(f"seed {s}: {v:.4f}" for s, v in zip(every, scores)))
    if len(scores) > 1:
        spread = max(scores) - min(scores)
        print(f"  mean {np.mean(scores):.4f} +/- {np.std(scores, ddof=1):.4f}")
        print(f"  spread {spread * 100:.2f} points from seed alone")
        print(f"  reporting only the best seed would inflate this by "
              f"{(max(scores) - np.mean(scores)) * 100:.2f} points")
    print(f"\n  published: 0.8240 (committed file) / 0.8270 (README)")
    print(f"  best published result on this task: {metrics.PUBLISHED_BEST_KNOWN:.4f}")
    print(f"\n  Predictions are saved. Phase 2 compares against them with a "
          f"paired test,\n  which is the only way an improvement here can be "
          f"shown rather than asserted.")


if __name__ == "__main__":
    main()
