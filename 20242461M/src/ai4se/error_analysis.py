"""Error analysis over a model's predictions.

The project brief's final bullet for Project 1 asks for analysis of the
obtained results, and a results table alone does not explain *why* a model
fails. This module turns predictions into the material the report's discussion
section needs: which confusions dominate, what the misclassified issues look
like, which terms mislead the model, and whether errors correlate with
document length.

Everything here takes an :class:`~ai4se.repository.IssueRepository` and a
:class:`~ai4se.evaluation.CompetitionResult`-style prediction mapping, so it
works for any model that ran through the evaluation harness.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .metrics import evaluate
from .model import LABELS
from .repository import IssueRepository


@dataclass
class Misprediction:
    """One misclassified issue, with the context needed to judge it."""

    repo: str
    true_label: str
    predicted_label: str
    confidence: float | None
    title: str
    excerpt: str
    word_count: int

    def as_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "repository": self.repo,
            "true": self.true_label,
            "predicted": self.predicted_label,
            "confidence": self.confidence,
            "title": self.title,
            "words": self.word_count,
            "excerpt": self.excerpt,
        }


def collect_predictions(
    model_factory,
    train: IssueRepository,
    test: IssueRepository,
) -> dict[str, dict]:
    """Train per project and return predictions alongside the ground truth.

    Mirrors :func:`~ai4se.evaluation.evaluate_competition` exactly, but keeps
    the individual predictions instead of collapsing them into scores.

    Args:
        model_factory: Zero-argument callable returning a fresh model.
        train: The training split.
        test: The test split.

    Returns:
        ``{repo: {"issues": [...], "y_true": [...], "y_pred": [...],
        "y_score": [...] or None}}``.
    """
    from .evaluation import class_probabilities
    from .repository import make_repository

    output: dict[str, dict] = {}
    for repo in sorted(set(train.repos()) & set(test.repos())):
        train_subset = make_repository("memory", issues=train.by_repo(repo))
        test_issues = test.by_repo(repo)

        X_train, y_train = train_subset.texts_and_labels()
        X_test = [issue.text for issue in test_issues]

        model = model_factory()
        model.fit(X_train, y_train)

        output[repo] = {
            "issues": test_issues,
            "y_true": [issue.label for issue in test_issues],
            "y_pred": list(model.predict(X_test)),
            "y_score": class_probabilities(model, X_test),
        }
    return output


def confusion_summary(predictions: dict[str, dict]):
    """Rank the confusion types by how often they occur.

    Answers the first question of any error analysis: of everything the model
    gets wrong, which mistake dominates?

    Returns:
        A pandas DataFrame of (true, predicted) pairs with counts and shares.
    """
    import pandas as pd

    errors: Counter[tuple[str, str]] = Counter()
    total = 0
    for payload in predictions.values():
        for true, predicted in zip(payload["y_true"], payload["y_pred"], strict=True):
            total += 1
            if true != predicted:
                errors[(true, predicted)] += 1

    n_errors = sum(errors.values()) or 1
    rows = [
        {
            "true": true,
            "predicted": predicted,
            "count": count,
            "share of errors %": round(100 * count / n_errors, 1),
            "share of all %": round(100 * count / max(total, 1), 1),
        }
        for (true, predicted), count in errors.most_common()
    ]
    return pd.DataFrame(rows)


def per_repository_scores(predictions: dict[str, dict]):
    """Score each repository from collected predictions.

    Returns:
        A pandas DataFrame indexed by repository, with per-class F1 and the
        average, plus an error count.
    """
    import pandas as pd

    rows = []
    for repo, payload in predictions.items():
        scores = evaluate(payload["y_true"], payload["y_pred"], labels=LABELS)
        errors = sum(
            1
            for a, b in zip(payload["y_true"], payload["y_pred"], strict=True)
            if a != b
        )
        rows.append(
            {
                "repository": repo,
                **{f"F1 {label}": scores.f1(label) for label in LABELS},
                "F1 avg": scores.macro_f1,
                "errors": errors,
            }
        )
    return pd.DataFrame(rows).set_index("repository").round(4)


def worst_mistakes(
    predictions: dict[str, dict],
    n: int = 20,
    excerpt_chars: int = 160,
    confident_only: bool = True,
) -> list[Misprediction]:
    """Return the errors the model was most confident about.

    Confident errors are the informative ones. A near-tie between two classes
    usually means a genuinely ambiguous issue; a confident mistake means the
    model learned something wrong, or the ground-truth label is questionable.

    Args:
        predictions: Output of :func:`collect_predictions`.
        n: How many to return.
        excerpt_chars: Characters of body text to include.
        confident_only: Sort by confidence. When the model produced no
            probabilities, errors are returned in dataset order instead.

    Returns:
        A list of :class:`Misprediction`.
    """
    mistakes: list[Misprediction] = []
    for repo, payload in predictions.items():
        scores = payload.get("y_score")
        for index, (issue, true, predicted) in enumerate(
            zip(payload["issues"], payload["y_true"], payload["y_pred"], strict=True)
        ):
            if true == predicted:
                continue
            confidence = scores[index].get(predicted) if scores else None
            body = issue.body.replace("\n", " ").strip()
            mistakes.append(
                Misprediction(
                    repo=repo,
                    true_label=true,
                    predicted_label=predicted,
                    confidence=confidence,
                    title=issue.title.strip()[:110],
                    excerpt=body[:excerpt_chars],
                    word_count=issue.word_count,
                )
            )

    if confident_only and any(m.confidence is not None for m in mistakes):
        mistakes.sort(key=lambda m: m.confidence or 0.0, reverse=True)
    return mistakes[:n]


def mistakes_frame(mistakes: Sequence[Misprediction]):
    """Render mispredictions as a pandas DataFrame for inspection."""
    import pandas as pd

    return pd.DataFrame([m.as_dict() for m in mistakes])


def errors_by_length(predictions: dict[str, dict], bins: int = 5):
    """Break the error rate down by document length.

    Issue length is extremely skewed in this dataset, and truncation is a
    hyperparameter, so whether long issues are harder is worth measuring
    rather than assuming.

    Returns:
        A pandas DataFrame with one row per length quantile.
    """
    import pandas as pd

    rows = []
    for payload in predictions.values():
        for issue, true, predicted in zip(
            payload["issues"], payload["y_true"], payload["y_pred"], strict=True
        ):
            rows.append({"words": issue.word_count, "correct": int(true == predicted)})

    frame = pd.DataFrame(rows)
    frame["bucket"] = pd.qcut(frame["words"], q=bins, duplicates="drop")
    summary = frame.groupby("bucket", observed=True).agg(
        issues=("correct", "size"),
        accuracy=("correct", "mean"),
        median_words=("words", "median"),
    )
    return summary.round(4)


def misleading_terms(
    predictions: dict[str, dict],
    true_label: str,
    predicted_label: str,
    n: int = 15,
    min_documents: int = 3,
):
    """Terms over-represented in one specific confusion.

    Given that the model called a ``true_label`` issue a ``predicted_label``
    one, which words does that group of issues share? This is what turns
    "question is often called bug" into a statement about *why*.

    Returns:
        A pandas DataFrame of terms with their document frequency in the
        confused group and in correctly classified issues of the same class.
    """
    import pandas as pd

    confused: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    n_confused = n_correct = 0

    for payload in predictions.values():
        for issue, true, predicted in zip(
            payload["issues"], payload["y_true"], payload["y_pred"], strict=True
        ):
            if true != true_label:
                continue
            terms = set(issue.text.lower().split())
            if predicted == predicted_label:
                confused.update(terms)
                n_confused += 1
            elif predicted == true_label:
                correct.update(terms)
                n_correct += 1

    if not n_confused:
        return pd.DataFrame(columns=["term", "in_errors_%", "in_correct_%", "lift"])

    rows = []
    for term, count in confused.items():
        if count < min_documents:
            continue
        error_rate = count / n_confused
        correct_rate = (correct[term] + 1) / (n_correct + 1)
        rows.append(
            {
                "term": term,
                "in_errors_%": round(100 * error_rate, 1),
                "in_correct_%": round(100 * correct[term] / max(n_correct, 1), 1),
                "lift": round(error_rate / correct_rate, 2),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("lift", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
