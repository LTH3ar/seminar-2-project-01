"""Exploratory analysis helpers shared by notebooks and reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from issue_classifier.dataframe import issues_to_dataframe
from issue_classifier.domain import IssueReport, PreparedIssue
from issue_classifier.preprocessing import TextCleaningConfig, TextPreprocessor


def overview(issues: Sequence[IssueReport]):
    """Summarize record counts, labels, and text lengths by repository."""

    import pandas as pd

    rows = []
    for repo in sorted({issue.repo for issue in issues}):
        repo_issues = [issue for issue in issues if issue.repo == repo]
        labels = Counter(issue.label for issue in repo_issues)
        word_counts = [
            len(f"{issue.title} {issue.body}".split())
            for issue in repo_issues
        ]
        rows.append(
            {
                "repo": repo,
                "records": len(repo_issues),
                "bug": labels.get("bug", 0),
                "feature": labels.get("feature", 0),
                "question": labels.get("question", 0),
                "median_words": float(pd.Series(word_counts).median()),
                "max_words": max(word_counts, default=0),
            }
        )
    return pd.DataFrame(rows)


def label_distribution(issues: Sequence[IssueReport]):
    """Return repository-by-label counts with totals."""

    import pandas as pd

    frame = issues_to_dataframe(issues)
    table = pd.crosstab(frame["repo"], frame["label"])
    table = table.reindex(columns=["bug", "feature", "question"], fill_value=0)
    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum(axis=0)
    return table


def length_statistics(issues: Sequence[IssueReport]):
    """Return word-count statistics for each label and the whole dataset."""

    import pandas as pd

    frame = issues_to_dataframe(issues)
    frame["words"] = (
        frame["title"].fillna("") + " " + frame["body"].fillna("")
    ).str.split().str.len()
    statistics = frame.groupby("label")["words"].describe()
    overall = frame["words"].describe().to_frame().T
    overall.index = ["overall"]
    return pd.concat([statistics, overall]).round(1)


def structural_noise(issues: Sequence[IssueReport]):
    """Measure the prevalence of structural signals in issue reports."""

    import pandas as pd

    feature_names = (
        "url_count",
        "code_block_count",
        "checkbox_count",
        "has_stack_trace",
        "body_is_empty",
    )
    rows = []
    for issue in issues:
        features = TextPreprocessor.structural_features(issue)
        rows.append(
            {
                "label": issue.label,
                **{
                    name: float(features[name] > 0)
                    for name in feature_names
                },
            }
        )
    frame = pd.DataFrame(rows)
    by_label = frame.groupby("label").mean(numeric_only=True)
    by_label.loc["overall"] = frame.mean(numeric_only=True)
    return (by_label * 100).round(1)


def cleaning_impact(issues: Sequence[IssueReport]):
    """Compare retained word counts across all cleaning levels."""

    import pandas as pd

    rows = []
    for issue in issues:
        combined = f"{issue.title}\n{issue.body}"
        rows.append(
            {
                level: len(
                    TextPreprocessor(TextCleaningConfig(level=level))
                    .clean(combined)
                    .split()
                )
                for level in ("raw", "conservative", "light", "full")
            }
        )
    frame = pd.DataFrame(rows)
    summary = frame.mean().to_frame("mean_words").round(1)
    raw_mean = frame["raw"].mean()
    summary["retained_percent"] = (
        100 * frame.mean() / raw_mean if raw_mean else 0
    ).round(1)
    return summary


def distinctive_terms(
    issues: Sequence[PreparedIssue],
    *,
    n: int = 15,
    min_documents: int = 15,
):
    """Find terms whose document frequency is concentrated in one class."""

    import pandas as pd

    labels = sorted({issue.label for issue in issues})
    document_frequency = {
        label: Counter(
            term
            for issue in issues
            if issue.label == label
            for term in set(issue.text.split())
        )
        for label in labels
    }
    class_sizes = {
        label: sum(issue.label == label for issue in issues) or 1
        for label in labels
    }
    vocabulary = set().union(
        *(set(counts) for counts in document_frequency.values())
    )

    if len(labels) < 2:
        return pd.DataFrame(columns=["term", "class", "lift", "documents"])

    rows = []
    for term in vocabulary:
        documents = sum(document_frequency[label][term] for label in labels)
        if documents < min_documents:
            continue
        rates = {
            label: (document_frequency[label][term] + 1)
            / (class_sizes[label] + len(labels))
            for label in labels
        }
        best = max(rates, key=rates.get)
        other_rates = [rate for label, rate in rates.items() if label != best]
        rows.append(
            {
                "term": term,
                "class": best,
                "lift": rates[best] / max(other_rates),
                "documents": documents,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["term", "class", "lift", "documents"])
    frame = pd.DataFrame(rows).sort_values("lift", ascending=False)
    return (
        frame.groupby("class", group_keys=False)
        .head(n)
        .sort_values(["class", "lift"], ascending=[True, False])
        .round(2)
        .reset_index(drop=True)
    )


def plot_label_distribution(issues: Sequence[IssueReport]):
    """Create a grouped label-distribution chart."""

    table = label_distribution(issues).drop(index="total", columns="total")
    axis = table.plot(kind="bar", figsize=(9, 4.5), width=0.75)
    axis.set_title("Issue reports per repository and class")
    axis.set_xlabel("")
    axis.set_ylabel("Number of issues")
    axis.figure.tight_layout()
    return axis.figure


def save_figure(figure, path: str) -> str:
    """Save a report-ready figure and return its path."""

    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    return str(destination)
