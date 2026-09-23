"""Exploratory data analysis for the NLBSE'24 issue report dataset.

Every function here takes an :class:`~ai4se.repository.IssueRepository` and is
therefore indifferent to whether the data is held in memory or read from a
file. That is the point of the abstraction: the analysis code was written once
and runs unchanged against both persistence layers.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .model import LABELS
from .repository import IssueRepository


def _pyplot():
    """Import plotting dependencies only when a figure is requested."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def overview(repository: IssueRepository) -> pd.DataFrame:
    """One-row summary of the dataset: size, projects, classes, missing values."""
    frame = repository.to_dataframe()
    return pd.DataFrame(
        [
            {
                "issues": len(frame),
                "projects": frame["repo"].nunique(),
                "classes": frame["label"].nunique(),
                "empty_titles": int((frame["title"].str.strip() == "").sum()),
                "empty_bodies": int((frame["body"].str.strip() == "").sum()),
                "duplicate_texts": int(
                    frame.duplicated(subset=["title", "body"]).sum()
                ),
                "unexpected_labels": int((~frame["label"].isin(LABELS)).sum()),
            }
        ]
    )


def label_distribution(repository: IssueRepository) -> pd.DataFrame:
    """Cross-tabulation of project by class, with row and column totals."""
    frame = repository.to_dataframe()
    table = pd.crosstab(frame["repo"], frame["label"])
    table = table.reindex(columns=[c for c in LABELS if c in table.columns])
    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum(axis=0)
    return table


def length_statistics(repository: IssueRepository) -> pd.DataFrame:
    """Word-count statistics of title + body, broken down by class.

    The spread here justifies the truncation decision in the preprocessing
    pipeline and should be reported in the Dataset section of the report.
    """
    frame = repository.to_dataframe()
    frame["words"] = (
        frame["title"].fillna("") + " " + frame["body"].fillna("")
    ).str.split().str.len()
    stats = frame.groupby("label")["words"].describe()
    overall = frame["words"].describe().to_frame().T
    overall.index = ["overall"]
    return pd.concat([stats, overall])[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ].round(1)


def structural_noise(repository: IssueRepository) -> pd.DataFrame:
    """Share of issues containing code blocks, URLs, images and template headings.

    Quantifies how much of the raw text is not natural language, which is the
    empirical justification for the cleaning pipeline.
    """
    from . import preprocessing as pp

    rows = []
    for issue in repository:
        body = issue.body
        rows.append(
            {
                "label": issue.label,
                "code_block": bool(pp.RE_FENCED_CODE.search(body)),
                "stack_trace": bool(pp.RE_STACK_FRAME.search(body)),
                "url": bool(pp.RE_URL.search(body)),
                "image": bool(pp.RE_MD_IMAGE.search(body)),
                "html": bool(pp.RE_HTML_TAG.search(body)),
                "template_heading": bool(pp.RE_TEMPLATE_HEADING.search(body)),
            }
        )
    frame = pd.DataFrame(rows)
    by_label = frame.groupby("label").mean(numeric_only=True)
    by_label.loc["overall"] = frame.mean(numeric_only=True)
    return (by_label * 100).round(1)


def cleaning_impact(
    repository: IssueRepository,
    sample: int | None = None,
) -> pd.DataFrame:
    """Average word count before and after each cleaning level.

    Args:
        repository: Source of issues.
        sample: Analyse only the first ``sample`` issues (faster during
            development); ``None`` uses all of them.
    """
    from .preprocessing import clean_text

    issues = repository.all()[: sample or None]
    rows = []
    for issue in issues:
        raw = issue.raw_text
        rows.append(
            {
                "raw": len(raw.split()),
                "conservative": len(clean_text(raw, "conservative").split()),
                "light": len(clean_text(raw, "light").split()),
                "full": len(clean_text(raw, "full").split()),
            }
        )
    frame = pd.DataFrame(rows)
    summary = frame.mean().to_frame("mean_words").round(1)
    summary["retained_%"] = (100 * frame.mean() / frame["raw"].mean()).round(1)
    return summary


def top_terms(
    repository: IssueRepository, label: str, n: int = 20
) -> pd.DataFrame:
    """Most frequent cleaned terms for one class.

    Requires the preprocessing pipeline to have been applied first, otherwise
    the raw text is used and the result is dominated by stop words.
    """
    counter: Counter[str] = Counter()
    for issue in repository.by_label(label):
        counter.update(issue.text.split())
    return pd.DataFrame(counter.most_common(n), columns=["term", "frequency"])


def distinctive_terms(
    repository: IssueRepository,
    n: int = 15,
    min_documents: int = 15,
    smoothing: float = 1.0,
) -> pd.DataFrame:
    """Terms most characteristic of each class, by document frequency.

    Document frequency (the number of *issues* containing a term) is used
    rather than raw token counts, because a handful of very long issues --
    the longest one in the training set exceeds 21,000 words -- would
    otherwise dominate the ranking with vocabulary from a single log dump.
    Add-one smoothing keeps the ratio finite for terms absent from a class.

    A quick sanity check that the classes are separable at all with a
    bag-of-words representation, before any model is trained.

    Args:
        repository: Source of issues, preferably already preprocessed.
        n: Number of terms to report per class.
        min_documents: Ignore terms appearing in fewer issues than this.
        smoothing: Additive constant applied to every document frequency.
    """
    labels = repository.labels()
    document_frequency = {
        label: Counter(
            term
            for issue in repository.by_label(label)
            for term in set(issue.text.split())  # set() -> count each issue once
        )
        for label in labels
    }
    class_sizes = {label: len(repository.by_label(label)) or 1 for label in labels}
    vocabulary = set().union(*(set(c) for c in document_frequency.values()))

    rows = []
    for term in vocabulary:
        documents = sum(document_frequency[label][term] for label in labels)
        if documents < min_documents:
            continue
        rates = {
            label: (document_frequency[label][term] + smoothing)
            / (class_sizes[label] + smoothing * len(labels))
            for label in labels
        }
        best = max(rates, key=rates.get)
        runner_up = max(rate for label, rate in rates.items() if label != best)
        rows.append(
            {
                "term": term,
                "class": best,
                "lift": rates[best] / runner_up,
                "issues": documents,
                "coverage_%": 100 * document_frequency[best][term] / class_sizes[best],
            }
        )

    frame = pd.DataFrame(rows).sort_values("lift", ascending=False)
    return (
        frame.groupby("class", group_keys=False)
        .head(n)
        .sort_values(["class", "lift"], ascending=[True, False])
        .round(2)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Figures. Each returns the Matplotlib figure so the notebook can display it
# and the caller can save it into results/ for inclusion in the LaTeX report.
# --------------------------------------------------------------------------- #


def plot_label_distribution(repository: IssueRepository):
    """Grouped bar chart of class counts per project."""
    plt = _pyplot()
    table = label_distribution(repository).drop(index="total", columns="total")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Issue reports per project and class (NLBSE'24)")
    ax.set_xlabel("")
    ax.set_ylabel("number of issues")
    ax.legend(title="class")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_length_distribution(repository: IssueRepository, clip: int = 800):
    """Overlaid histograms of issue length per class, on a log-count axis."""
    plt = _pyplot()
    frame = repository.to_dataframe()
    frame["words"] = (
        frame["title"].fillna("") + " " + frame["body"].fillna("")
    ).str.split().str.len().clip(upper=clip)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label in [c for c in LABELS if c in set(frame["label"])]:
        ax.hist(
            frame.loc[frame["label"] == label, "words"],
            bins=50,
            alpha=0.55,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_title(f"Issue length by class (clipped at {clip} words)")
    ax.set_xlabel("words in title + body")
    ax.set_ylabel("number of issues (log scale)")
    ax.legend(title="class")
    fig.tight_layout()
    return fig


def save_figure(fig, path: str) -> str:
    """Write a figure to ``results/`` at a resolution suitable for LaTeX."""
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    return str(destination)
