"""Phase 2 -- try to improve on the baseline, without cheating to do it.

Two stages, in this order and never the other way round.

**select** carves a validation set out of the *training* split and ranks
candidate configurations on it. The test split is not read. Every choice --
encoder, preprocessing, number of contrastive iterations -- is made here.

**final** takes the winner, trains it on the full training split, predicts the
test split once, and compares it against the phase 1 reproduction with a
paired McNemar test. That comparison is possible only because phase 1 gave us
the baseline's per-issue predictions; the organisers published aggregate
scores only.

    python run_phase2.py select
    python run_phase2.py final --config <name>

Three rules this script enforces rather than trusts:

* A difference below 3.0 F1 points is reported as no detectable difference.
  The best published result on this task beat the baseline by 0.1 points, so
  anything in that range is noise, not news.
* ``--allow-global`` is required for the configuration that fine-tunes one
  encoder on all 1,500 issues. The rules say five classifiers must be
  submitted and say nothing about how much data each may see, so it is legal
  by the letter and ambiguous in spirit. Ask the organisers before using it.
* Encoders released after 2024 may have seen this test set during
  pretraining. Those configurations print a contamination warning and carry it
  into the results file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import audit, metrics, preprocess, stats  # noqa: E402
from src.data import (  # noqa: E402
    LABELS,
    REPOSITORIES,
    Split,
    load_dataset,
    load_split,
)
from src.runner import OFFICIAL, Config, gpu_report, run  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / "data"

#: Encoders whose pretraining data postdates the competition's test set going
#: public. Legal under the rules, but a contamination risk that has to be
#: declared rather than quietly enjoyed.
POSSIBLY_CONTAMINATED = {"BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"}

#: Candidates. Each names one lever so an effect can be attributed to a cause.
CANDIDATES: dict[str, dict] = {
    # -- control -----------------------------------------------------------
    "baseline": {"config": {}, "text": "raw"},
    # -- lever: issue-template boilerplate ---------------------------------
    "strip_templates": {"config": {}, "text": "strip_templates"},
    "mask_structure": {"config": {}, "text": "mask_structure"},
    "strip_and_mask": {"config": {}, "text": "strip_and_mask"},
    # -- lever: contrastive budget (the organisers tuned none of this) -----
    "iters_10": {"config": {"num_iterations": 10}, "text": "raw"},
    "iters_40": {"config": {"num_iterations": 40}, "text": "raw"},
    "epochs_2": {"config": {"num_epochs": 2}, "text": "raw"},
    # -- lever: encoder ----------------------------------------------------
    "encoder_bge": {"config": {"encoder": "BAAI/bge-base-en-v1.5"}, "text": "raw"},
    "encoder_e5": {"config": {"encoder": "intfloat/e5-base-v2"}, "text": "raw"},
    # -- lever: training data scope (NEEDS CONFIRMATION) -------------------
    "global_pairs": {"config": {"scope": "global"}, "text": "raw", "gated": True},
}


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "final"))
    parser.add_argument("--config", help="Candidate name, for the final stage.")
    parser.add_argument("--only", nargs="+", help="Restrict the select stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction", type=float, default=0.25,
        help="Share of each (project, class) cell held out for selection.",
    )
    parser.add_argument(
        "--allow-global", action="store_true",
        help="Permit the configuration that trains on all 1,500 issues.",
    )
    return parser.parse_args()


def split_for_selection(
    train: Split, fraction: float, seed: int, verbose: bool = True
) -> tuple[Split, Split]:
    """Carve a validation set out of the training split.

    Stratified within each (project, class) cell, so both sides keep the
    balance the metric assumes. The test split is untouched.

    Splitting is done over *distinct texts* rather than over rows. The training
    split contains one duplicated (title, body) pair, and a row-wise split puts
    one copy on each side -- a small leak, but selecting a configuration on a
    validation set that contains a training item is exactly the kind of error
    this project exists to avoid.

    Args:
        train: The official training split.
        fraction: Share of each cell held out.
        seed: Seed of the shuffle.
        verbose: Report any duplicates that had to be kept together.

    Returns:
        ``(inner_train, validation)``.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    held, kept, tied = [], [], 0

    for _, cell in train.frame.groupby(["repo", "label"]):
        groups = [g for _, g in cell.groupby(cell.text, sort=False)]
        tied += len(cell) - len(groups)
        order = rng.permutation(len(groups))
        target = int(round(len(cell) * fraction))
        taken = 0
        for position in order:
            group = groups[position]
            if taken < target:
                held.append(group)
                taken += len(group)
            else:
                kept.append(group)

    inner = Split("inner_train", pd.concat(kept).reset_index(drop=True))
    validation = Split("validation", pd.concat(held).reset_index(drop=True))

    overlap = set(inner.frame.text) & set(validation.frame.text)
    if overlap:  # pragma: no cover - the grouping above prevents this
        raise RuntimeError(f"{len(overlap)} texts straddle the selection split.")
    if verbose and tied:
        print(f"  {tied} duplicated text(s) kept on one side of the split")
    return inner, validation


def build(name: str, seed: int) -> tuple[Config, str, bool]:
    """Return the configuration, text variant and gating flag for a candidate.

    Raises:
        KeyError: If ``name`` is not a known candidate.
    """
    if name not in CANDIDATES:
        raise KeyError(f"Unknown candidate {name!r}; use one of {sorted(CANDIDATES)}.")
    spec = CANDIDATES[name]
    settings = {**OFFICIAL.as_dict(), **spec["config"], "name": name, "seed": seed}
    return Config(**settings), spec["text"], bool(spec.get("gated"))


def warn_if_contaminated(config: Config) -> str | None:
    """Print and return a contamination warning when one applies."""
    if config.encoder in POSSIBLY_CONTAMINATED:
        warning = (
            f"{config.encoder} postdates the public release of this test set; "
            "any gain may be pretraining contamination rather than a better "
            "encoder. Report alongside a pre-2024 encoder."
        )
        print(f"    !! {warning}")
        return warning
    return None


def evaluate(record: dict, target: Split) -> dict:
    """Score a run against a split's ground truth."""
    truth = target.frame.label.tolist()
    repos = target.frame.repo.tolist()
    report = metrics.full_report(truth, record["predictions"], repos)
    point, low, high = stats.bootstrap_ci(
        truth, record["predictions"], repos, n_resamples=2000
    )
    report["ci95"] = [low, high]
    return report


def stage_select(args) -> None:
    """Rank candidates on validation data carved from the training split."""
    # Only the training split is loaded. Loading both here would put a read
    # into the test-access log that the stage does not need, and would make
    # the claim printed below untrue in the one way that matters.
    train_only = load_split("train", CACHE, args.seed)
    inner, validation = split_for_selection(
        train_only, args.validation_fraction, args.seed
    )
    print(f"  selection set: {len(inner)} train / {len(validation)} validation")
    print("  the test split is NOT read in this stage\n")

    names = args.only or list(CANDIDATES)
    rows = []
    for name in names:
        config, variant, gated = build(name, args.seed)
        if gated and not args.allow_global:
            print(f"  {name}: skipped -- needs --allow-global, and the "
                  f"organisers' confirmation first")
            continue
        print(f"  {name}  (encoder={config.encoder.split('/')[-1]}, "
              f"text={variant}, iters={config.num_iterations}, "
              f"epochs={config.num_epochs}, scope={config.scope})")
        warning = warn_if_contaminated(config)

        record = run(
            config,
            Split("inner", preprocess.apply(inner.frame, variant)),
            Split("validation", preprocess.apply(validation.frame, variant)),
            verbose=False,
        )
        report = evaluate(record, validation)
        record["report"] = report
        record["text_variant"] = variant
        record["contamination_warning"] = warning
        (RESULTS / f"select_{name}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        rows.append((name, report["cross_repo_f1"], report["ci95"], record["minutes"]))
        print(f"    validation F1 {report['cross_repo_f1']:.4f}  "
              f"({record['minutes']:.1f} min)\n")

    rows.sort(key=lambda r: -r[1])
    control = next((r for r in rows if r[0] == "baseline"), None)
    print("=" * 74)
    print("SELECTION RESULT (validation data -- not comparable with 0.8270)")
    print("=" * 74)
    print(f"  {'candidate':<20} {'val F1':>8} {'vs control':>11}  95% CI")
    for name, score, ci, _ in rows:
        delta = f"{score - control[1]:+.4f}" if control else "     --"
        flag = "" if not control or abs(score - control[1]) >= stats.DECISION_THRESHOLD \
            else "  (within noise)"
        print(f"  {name:<20} {score:>8.4f} {delta:>11}  "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]{flag}")
    if rows:
        print(f"\n  Next: python run_phase2.py final --config {rows[0][0]}")
        print("  Choose on this table only. Do not look at test scores first.")


def stage_final(args) -> None:
    """Run one selected configuration on the test split, once, and test it."""
    if not args.config:
        sys.exit("--config is required for the final stage.")
    baseline_path = RESULTS / f"phase1_seed{args.seed}.json"
    if not baseline_path.exists():
        sys.exit(
            f"{baseline_path.name} not found. Run phase 1 first -- without the "
            "baseline's own predictions there is no paired test, and a "
            "leaderboard delta on its own proves nothing."
        )

    config, variant, gated = build(args.config, args.seed)
    if gated and not args.allow_global:
        sys.exit(f"{args.config} needs --allow-global and the organisers' confirmation.")

    data = load_dataset(
        CACHE, results_dir=RESULTS, reason=f"phase2-final:{args.config}"
    )
    train, test = data["train"], data["test"]
    print(f"  {args.config}: encoder={config.encoder}, text={variant}, "
          f"scope={config.scope}")
    warning = warn_if_contaminated(config)
    print("  reading the test split now -- once.\n")

    record = run(
        config,
        Split("train", preprocess.apply(train.frame, variant)),
        Split("test", preprocess.apply(test.frame, variant)),
    )
    report = evaluate(record, test)
    record["report"] = report
    record["contamination_warning"] = warning

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    truth = test.frame.label.tolist()
    repos = test.frame.repo.tolist()
    comparison = stats.compare(
        baseline["predictions"], record["predictions"], truth, repos
    )
    rate = comparison["pooled"]["discordant"] / len(truth)
    mdd = stats.minimum_detectable_difference(rate, n_items=len(truth))
    tried = sorted(p.stem.replace("select_", "") for p in RESULTS.glob("select_*.json"))
    record["comparison_vs_reproduction"] = comparison
    record["mdd"] = mdd
    record["configurations_tried_in_selection"] = tried
    record["test_access_log"] = audit.test_access_count(RESULTS)
    (RESULTS / f"final_{args.config}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    base_f1 = baseline["report"]["cross_repo_f1"]
    ours = report["cross_repo_f1"]
    print("=" * 74)
    print(f"FINAL -- {args.config}")
    print("=" * 74)
    print(f"  {'project':<24} {'ours':>8} {'our repro':>10} {'published':>10}")
    for repo in REPOSITORIES:
        if repo not in report["per_repo"]:
            continue
        print(f"  {repo:<24} {report['per_repo'][repo]:>8.4f} "
              f"{baseline['report']['per_repo'][repo]:>10.4f} "
              f"{metrics.PUBLISHED[repo]:>10.4f}")
    print(f"  {'-' * 56}")
    print(f"  {'cross-repository':<24} {ours:>8.4f} {base_f1:>10.4f} "
          f"{metrics.PUBLISHED_OVERALL_FILE:>10.4f}")
    print(f"  95% CI [{report['ci95'][0]:.4f}, {report['ci95'][1]:.4f}]")

    print(f"\n  paired against our own reproduction:")
    for row in comparison["per_repo"]:
        mark = "*" if row["significant"] else "ns"
        print(f"    {row['repo']:<24} n01={row['n01']:<4} n10={row['n10']:<4} "
              f"p_holm={row['p_holm']:.3f}  {mark}")
    print(f"    discordant rate {rate:.3f} -> this test set resolves "
          f"{mdd * 100:.1f} points")
    print(f"\n  CLAIM: {stats.describe_claim(comparison['delta_cross_repo_f1'], comparison)}")
    if ours > metrics.PUBLISHED_BEST_KNOWN:
        print(f"  note: also above the best published result on this task "
              f"({metrics.PUBLISHED_BEST_KNOWN:.4f}).")
        if warning:
            print("        that result used an encoder flagged for possible "
                  "contamination -- see above before making anything of it.")


def main() -> None:
    """Dispatch to the requested stage."""
    RESULTS.mkdir(exist_ok=True)
    args = parse_args()
    print("=" * 74)
    print(f"PHASE 2 -- {args.stage.upper()}")
    print("=" * 74)
    print(f"  GPU: {gpu_report()}")
    print(f"  labels: {', '.join(LABELS)}\n")
    (stage_select if args.stage == "select" else stage_final)(args)


if __name__ == "__main__":
    main()
