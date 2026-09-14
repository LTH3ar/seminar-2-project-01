"""Optional pandas adapters for notebooks and scikit-learn workflows."""

from __future__ import annotations

from collections.abc import Sequence

from issue_classifier.domain import IssueReport, PreparedIssue


def issues_to_dataframe(issues: Sequence[IssueReport]):
    import pandas as pd

    return pd.DataFrame(
        {
            "issue_id": issue.issue_id,
            "repo": issue.repo,
            "created_at": issue.created_at,
            "label": issue.label,
            "title": issue.title,
            "body": issue.body,
        }
        for issue in issues
    )


def prepared_issues_to_dataframe(issues: Sequence[PreparedIssue]):
    import pandas as pd

    return pd.DataFrame(issue.to_dict() for issue in issues)
