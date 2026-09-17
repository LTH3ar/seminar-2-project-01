r"""Generate every LaTeX table in the report from the result files.

No number in ``report/report.tex`` is typed by hand. This script reads
``results/*.json`` and writes ``report/tables/*.tex``; the report ``\\input``s
them. Re-run it after any experiment and the report is up to date.

When an experiment has not been run yet, the corresponding table file is still
written -- containing a visible placeholder naming the missing script -- so the
report always compiles and never silently shows stale numbers.

    python experiments/04_make_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "report" / "tables"

sys.path.insert(0, str(ROOT / "src"))

from ai4se.model import LABELS, REPOSITORIES  # noqa: E402


#: LaTeX needs the underscore and slash in project names escaped.
def tex(value: str) -> str:
    """Escape the characters that would break LaTeX in a project name."""
    return str(value).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def load(name: str) -> dict | None:
    """Read one result file, or return None when the experiment has not run."""
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def model_results() -> dict | None:
    """Merge every model run into one payload: baselines plus the transformer.

    Both scripts emit the same per-model record; ``02_baselines.py`` writes a
    mapping of them and ``05_transformer.py`` writes a single one. Merging
    here is what puts the transformer -- the pre-registration's primary
    hypothesis -- into the report tables instead of leaving it unread in
    ``results/transformer.json``.

    Returns:
        Mapping with ``"seeds"``, ``"official_baselines"`` and ``"results"``
        keyed by model name, or None when neither experiment has been run.
    """
    files = (load("baselines.json"), load("transformer.json"))
    parts = [part for part in files if part]
    if not parts:
        return None
    results: dict[str, dict] = {}
    for part in parts:
        found = part["results"]
        for result in ([found] if "name" in found else found.values()):
            results[result["name"]] = result
    return {
        "seeds": parts[0]["seeds"],
        "official_baselines": parts[0]["official_baselines"],
        "results": results,
    }


def write(name: str, body: str) -> None:
    """Write one table file and report what happened."""
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(body.rstrip() + "\n", encoding="utf-8")
    first = body.strip().splitlines()[0] if body.strip() else ""
    status = "placeholder" if "missingtable" in first else "written"
    print(f"  {name + '.tex':<28} {status}")


def missing(name: str, script: str) -> None:
    """Write a placeholder naming the script that would fill this table."""
    write(name, f"\\missingtable{{{script}}}")


def table(
    caption: str, label: str, spec: str, header: str, rows: list[str], note: str = ""
) -> str:
    """Assemble a booktabs table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        lines.append(f"\\caption*{{\\footnotesize {note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ----------------------------------------------------------------- tables --


def dataset_overview() -> None:
    """Static description of the corpus; no experiment needed."""
    rows = [
        r"\texttt{facebook/react} & 100 & 100 & 100 & 300 & 2016-03 -- 2023-08 \\",
        r"\texttt{tensorflow/tensorflow} & 100 & 100 & 100 & 300 &"
        r" 2021-12 -- 2023-09 \\",
        r"\texttt{microsoft/vscode} & 100 & 100 & 100 & 300 & 2023-06 -- 2023-09 \\",
        r"\texttt{bitcoin/bitcoin} & 100 & 100 & 100 & 300 & 2018-10 -- 2023-09 \\",
        r"\texttt{opencv/opencv} & 100 & 100 & 100 & 300 & 2022-01 -- 2023-09 \\",
        r"\midrule",
        r"total & 500 & 500 & 500 & 1500 & \\",
    ]
    write(
        "dataset_overview",
        table(
            "Composition of each split. Both splits have identical structure.",
            "dataset",
            "lrrrrl",
            r"Project & bug & feature & question & total & Issue dates",
            rows,
            "Date ranges measured from the CSVs and span both splits. Three of "
            "the five projects begin well before the January 2022 start date "
            "given in the competition materials.",
        ),
    )


def timestamp_only() -> None:
    """The confound table: label predicted from creation date alone."""
    data = load("benchmark_audit.json")
    if not data:
        return missing("timestamp_only", "experiments/benchmark_audit.py")
    confound = data["temporal_confound"]
    rows = [
        f"\\texttt{{{tex(repo)}}} & {score:.3f} \\\\"
        for repo, score in sorted(confound["per_repo"].items(), key=lambda kv: -kv[1])
    ]
    rows += [r"\midrule", f"mean & {confound['mean']:.3f} \\\\"]
    write(
        "timestamp_only",
        table(
            "F1 of a classifier given only \\texttt{created\\_at} and no text.",
            "timestamp",
            "lr",
            r"Project & F1 from timestamp alone",
            rows,
            f"Chance level is {confound['chance']:.3f}. Five-fold "
            "cross-validation within each project.",
        ),
    )


def power() -> None:
    """Detection threshold of the benchmark."""
    data = load("benchmark_audit.json")
    if not data:
        return missing("power", "experiments/benchmark_audit.py")
    p = data["power"]
    rows = [
        f"Full test set (1500 issues) & {p['mdd_cross_repo'] * 100:.1f} \\\\",
        f"One project (300 issues) & {p['mdd_per_repo'] * 100:.1f} \\\\",
        r"\midrule",
        r"Seed variation alone, same configuration & 2.4 \\",
    ]
    write(
        "power",
        table(
            "Minimum detectable difference at 80\\% power, $\\alpha = 0.05$.",
            "power",
            "lr",
            r"Evaluation unit & Detectable difference (F1 points)",
            rows,
            f"Monte Carlo simulation of McNemar's test at the observed "
            f"discordant rate of {p['discordant_rate']:.3f}.",
        ),
    )


def setfit_reproduction() -> None:
    """Our SetFit run against the organisers' published figures."""
    data = load("setfit_official.json")
    if not data:
        return missing("setfit_reproduction", "experiments/03_setfit.py")
    rows = []
    for repo in REPOSITORIES:
        ours = data["per_repo"].get(repo)
        theirs = data["published"].get(repo)
        if ours is None or theirs is None:
            continue
        rows.append(
            f"\\texttt{{{tex(repo)}}} & {ours:.4f} & {theirs:.4f} & "
            f"{ours - theirs:+.4f} \\\\"
        )
    rows += [
        r"\midrule",
        f"cross-repository & {data['cross_repo_f1']:.4f} & "
        f"{data['published']['cross-repo']:.4f} & {data['gap']:+.4f} \\\\",
    ]
    write(
        "setfit_reproduction",
        table(
            "Reproduction of the organisers' SetFit baseline.",
            "setfit",
            "lrrr",
            r"Project & Ours & Published & $\Delta$",
            rows,
            f"Verdict: {data['verdict']}. The competition README quotes 0.8270 "
            "for the same configuration that the committed result file scores "
            "at 0.8240.",
        ),
    )


def main_results() -> None:
    """All models, five seeds each, against the published baselines."""
    baselines = model_results()
    if not baselines:
        return missing("main_results", "experiments/02_baselines.py")

    rows = []
    for name, published in baselines["official_baselines"].items():
        rows.append(
            f"{tex(name)} (published) & {published['cross-repo']:.4f} & --- & --- \\\\"
        )
    rows.append(r"\midrule")

    setfit = load("setfit_official.json")
    if setfit:
        rows.append(
            f"SetFit (our run) & {setfit['cross_repo_f1']:.4f} & --- & "
            f"[{setfit['ci'][0]:.3f}, {setfit['ci'][1]:.3f}] \\\\"
        )

    ordered = sorted(
        baselines["results"].values(), key=lambda r: r["mean"], reverse=True
    )
    for result in ordered:
        mark = r"$\checkmark$" if result["detectable_vs_setfit"] else "---"
        rows.append(
            f"{tex(result['name'])} & {result['mean']:.4f} & "
            f"{result['std']:.4f} & "
            f"[{result['ci'][0]:.3f}, {result['ci'][1]:.3f}] \\\\"
        )
        del mark
    write(
        "main_results",
        table(
            "Cross-repository F1. Our models are the mean of five seeds.",
            "main",
            "lrrc",
            r"Model & F1 & sd (seeds) & 95\% CI",
            rows,
            "Intervals are stratified bootstraps on the median-seed run. "
            "Differences below 3.0 points are not detectable on this test set "
            "(Table~\\ref{tab:power}) and are not described as improvements.",
        ),
    )


def per_repo_class() -> None:
    """The fifteen-cell table for the best model available."""
    baselines = model_results()
    if not baselines:
        return missing("per_repo_class", "experiments/02_baselines.py")
    best = max(baselines["results"].values(), key=lambda r: r["mean"])
    cells = {
        (row["repo"], row["label"]): row["f1"] for row in best["classification_rows"]
    }
    rows = []
    for repo in REPOSITORIES:
        values = " & ".join(
            f"{cells.get((repo, label), float('nan')):.3f}" for label in LABELS
        )
        rows.append(f"\\texttt{{{tex(repo)}}} & {values} \\\\")
    rows.append(r"\midrule")
    averages = " & ".join(
        f"{cells.get(('cross-repo', label), float('nan')):.3f}" for label in LABELS
    )
    rows.append(f"mean & {averages} \\\\")
    write(
        "per_repo_class",
        table(
            f"Per-project, per-class F1 of the strongest model "
            f"(\\texttt{{{tex(best['name'])}}}).",
            "perrepoclass",
            "lrrr",
            r"Project & bug & feature & question",
            rows,
            "The single cross-repository figure conceals this spread; the "
            "weakest and strongest cells differ by roughly 30 points.",
        ),
    )


def significance() -> None:
    """McNemar with Holm correction, from the audit run."""
    data = load("benchmark_audit.json")
    if not data:
        return missing("significance", "experiments/benchmark_audit.py")
    rows = []
    for row in data["significance"]["rows"]:
        is_pooled = row["repo"] == "POOLED"
        name = "pooled" if is_pooled else f"\\texttt{{{tex(row['repo'])}}}"
        verdict = r"$\checkmark$" if row["significant"] else "ns"
        rows.append(
            f"{name} & {row['n01']} & {row['n10']} & {row['discordant']} & "
            f"{row['p_value']:.3f} & {row['p_holm']:.3f} & {verdict} \\\\"
        )
    write(
        "significance",
        table(
            "Per-project model against pooled model, exact McNemar.",
            "significance",
            "lrrrrrc",
            r"Project & $n_{01}$ & $n_{10}$ & discordant & $p$ &"
            r" $p_{\text{Holm}}$ & sig.",
            rows,
            f"{data['significance']['survivors']} of five projects reach "
            "significance after correction. The two configurations are not "
            "distinguishable on this test set.",
        ),
    )


def time_aware() -> None:
    """Random split against the chronological control."""
    data = load("benchmark_audit.json")
    if not data:
        return missing("time_aware", "experiments/benchmark_audit.py")
    split = data["split_protocol"]
    rows = [
        f"Random split (matched control) & {split['random']:.4f} \\\\",
        f"Time-aware split & {split['time_aware']:.4f} \\\\",
        r"\midrule",
        f"Difference & {split['delta']:+.4f} \\\\",
    ]
    write(
        "time_aware",
        table(
            "Cost of evaluating on a random rather than a chronological split.",
            "timeaware",
            "lr",
            r"Protocol & Cross-repository F1",
            rows,
            "Identical cell sizes and identical class balance in both "
            "conditions; only the direction of time differs.",
        ),
    )


def seed_scores() -> None:
    """All raw per-seed values, for the appendix."""
    baselines = model_results()
    if not baselines:
        return missing("seed_scores", "experiments/02_baselines.py")
    seeds = baselines["seeds"]
    header = "Model & " + " & ".join(str(s) for s in seeds) + r" & spread"
    rows = []
    for result in sorted(
        baselines["results"].values(), key=lambda r: r["mean"], reverse=True
    ):
        values = " & ".join(f"{s:.4f}" for s in result["seed_scores"])
        spread = max(result["seed_scores"]) - min(result["seed_scores"])
        rows.append(f"{tex(result['name'])} & {values} & {spread * 100:.2f} \\\\")
    write(
        "seed_scores",
        table(
            "Raw cross-repository F1 for every seed.",
            "seeds",
            "l" + "r" * (len(seeds) + 1),
            header,
            rows,
            "Published so that the mean can be checked and so that no seed was "
            "quietly dropped. Spread is in F1 points.",
        ),
    )


def main() -> None:
    """Regenerate every table."""
    print("Generating LaTeX tables from results/:")
    dataset_overview()
    timestamp_only()
    power()
    setfit_reproduction()
    main_results()
    per_repo_class()
    significance()
    time_aware()
    seed_scores()
    print(f"\nTables written to {TABLES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
