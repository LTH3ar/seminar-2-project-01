"""Error analysis module (Chapter 6 in report).

Analyses the 359 mis-classifications produced by the best-performing model,
covering confusion patterns, length-quintile accuracy, and high-confidence errors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence, Tuple
import numpy as np

from ..model import IssueReport


def misclassification_pairs(
    issues: Sequence[IssueReport],
    y_pred: Sequence[str],
) -> dict[Tuple[str, str], list[IssueReport]]:
    """Group mis-classified issues by (true_label, predicted_label) pair."""
    pairs: dict[Tuple[str, str], list[IssueReport]] = defaultdict(list)
    for issue, pred in zip(issues, y_pred, strict=True):
        if issue.label != pred:
            pairs[(issue.label, pred)].append(issue)
    return dict(pairs)


def confusion_summary(pairs: dict) -> list[dict]:
    """Summarise confusion pairs sorted by frequency."""
    result = []
    for (true_lbl, pred_lbl), issue_list in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        result.append({
            "true": true_lbl,
            "predicted": pred_lbl,
            "count": len(issue_list),
            "examples": [iss.text[:120] for iss in issue_list[:3]],
        })
    return result


def per_class_per_repo_errors(
    issues: Sequence[IssueReport],
    y_pred: Sequence[str],
) -> list[dict]:
    """Count errors per (repo, true_label) cell — Table 6.2."""
    error_counts: dict[Tuple[str, str], int] = defaultdict(int)
    total_counts: dict[Tuple[str, str], int] = defaultdict(int)
    for issue, pred in zip(issues, y_pred, strict=True):
        key = (issue.repo, issue.label)
        total_counts[key] += 1
        if issue.label != pred:
            error_counts[key] += 1

    rows = []
    for (repo, lbl), total in sorted(total_counts.items()):
        errors = error_counts.get((repo, lbl), 0)
        rows.append({
            "repo": repo,
            "label": lbl,
            "total": total,
            "errors": errors,
            "error_rate": errors / total if total > 0 else 0.0,
        })
    return rows


def length_quintile_accuracy(
    issues: Sequence[IssueReport],
    y_pred: Sequence[str],
    n_quintiles: int = 5,
) -> list[dict]:
    """Measure accuracy in each text-length quintile — Table 6.4."""
    lengths = [len(iss.text.split()) for iss in issues]
    length_arr = np.array(lengths)
    quintile_edges = np.percentile(length_arr, [i * 100 / n_quintiles for i in range(n_quintiles + 1)])

    results = []
    for q in range(n_quintiles):
        lo, hi = quintile_edges[q], quintile_edges[q + 1]
        mask = (length_arr >= lo) & (length_arr <= hi)
        q_issues = [iss for iss, m in zip(issues, mask) if m]
        q_pred = [pred for pred, m in zip(y_pred, mask) if m]
        correct = sum(1 for iss, p in zip(q_issues, q_pred) if iss.label == p)
        results.append({
            "quintile": q + 1,
            "min_words": int(lo),
            "max_words": int(hi),
            "n": len(q_issues),
            "accuracy": correct / len(q_issues) if q_issues else 0.0,
        })
    return results


def high_confidence_errors(
    issues: Sequence[IssueReport],
    y_pred: Sequence[str],
    y_prob: np.ndarray,
    top_n: int = 25,
) -> list[dict]:
    """Return the top_n mis-classifications with highest predicted confidence — Table 6.5."""
    errors = []
    for i, (issue, pred) in enumerate(zip(issues, y_pred, strict=True)):
        if issue.label != pred:
            confidence = float(np.max(y_prob[i]))
            errors.append({
                "repo": issue.repo,
                "true_label": issue.label,
                "predicted": pred,
                "confidence": confidence,
                "text_snippet": issue.text[:200],
            })
    errors.sort(key=lambda x: -x["confidence"])
    return errors[:top_n]


def template_lift_analysis(
    issues: Sequence[IssueReport],
    template_pattern: str = r"##\s+(?:Expected|Steps|Actual)",
) -> dict:
    """Measure label distribution lift for issues containing GitHub template boilerplate — Table 6.3."""
    import re
    pattern = re.compile(template_pattern, re.IGNORECASE)

    has_template: dict[str, Counter] = defaultdict(Counter)
    no_template: dict[str, Counter] = defaultdict(Counter)

    for issue in issues:
        if pattern.search(issue.body):
            has_template[issue.repo][issue.label] += 1
        else:
            no_template[issue.repo][issue.label] += 1

    result = {"with_template": {}, "without_template": {}}
    for repo in set(list(has_template.keys()) + list(no_template.keys())):
        wt = dict(has_template[repo])
        wot = dict(no_template[repo])
        result["with_template"][repo] = wt
        result["without_template"][repo] = wot

    return result
