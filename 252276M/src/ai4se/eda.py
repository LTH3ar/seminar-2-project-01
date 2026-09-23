"""Exploratory data analysis for the NLBSE'24 issue report dataset.

All functions operate against the abstract IssueRepository interface,
ensuring persistence-agnostic analysis.
"""

from __future__ import annotations

import pandas as pd

from .model import LABELS
from .repository import IssueRepository


def overview(repository: IssueRepository) -> pd.DataFrame:
    """One-row summary of the dataset: size, projects, classes, missing values."""
    issues = list(repository.all())
    df = pd.DataFrame([iss.to_dict() for iss in issues])
    return pd.DataFrame(
        [
            {
                "issues": len(df),
                "projects": df["repo"].nunique() if not df.empty else 0,
                "classes": df["label"].nunique() if not df.empty else 0,
                "empty_titles": int((df["title"].str.strip() == "").sum()) if not df.empty else 0,
                "empty_bodies": int((df["body"].str.strip() == "").sum()) if not df.empty else 0,
                "duplicate_texts": int(df.duplicated(subset=["title", "body"]).sum()) if not df.empty else 0,
                "unexpected_labels": int((~df["label"].isin(LABELS)).sum()) if not df.empty else 0,
            }
        ]
    )


def label_distribution(repository: IssueRepository) -> pd.DataFrame:
    """Cross-tabulation of project by class, with row and column totals."""
    issues = list(repository.all())
    df = pd.DataFrame([iss.to_dict() for iss in issues])
    if df.empty:
        return pd.DataFrame()
    table = pd.crosstab(df["repo"], df["label"])
    table = table.reindex(columns=[c for c in LABELS if c in table.columns])
    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum(axis=0)
    return table


def length_statistics(repository: IssueRepository) -> pd.DataFrame:
    """Word-count statistics of title + body, broken down by class."""
    issues = list(repository.all())
    rows = []
    for iss in issues:
        words = len((iss.title + " " + iss.body).split())
        rows.append({"label": iss.label, "words": words})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    stats = df.groupby("label")["words"].describe()
    overall = df["words"].describe().to_frame().T
    overall.index = ["overall"]
    return pd.concat([stats, overall])[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ].round(1)


def structural_noise(repository: IssueRepository) -> pd.DataFrame:
    """Share of issues containing code blocks, URLs, images and template headings."""
   
    from . import preprocessing as pp
   

    rows = []
    for issue in repository.all():
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
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    return df.groupby("label").mean(numeric_only=True).round(3)
