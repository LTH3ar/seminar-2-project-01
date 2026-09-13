"""Reproduce the benchmark audit end to end.

Runs the five measurements that ``docs/benchmark-audit.md`` reports, using
only the project's own package so that the result is reproducible by anyone
who can run ``make data``:

1. A classical baseline, scored under the official protocol with a bootstrap
   confidence interval.
2. The confound: how well the label can be predicted from ``created_at``
   alone, with no text at all.
3. The cost of the random split, measured against a matched time-aware split.
4. Whether observed differences survive McNemar with Holm correction.
5. The minimum difference this test set can detect.

Needs the track B extra for scikit-learn::

    pip install -e ".[ml]"
    python experiments/benchmark_audit.py

Expected runtime is about two minutes on a laptop CPU. Results are written to
``results/benchmark_audit.json`` so the report can cite exact numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4se.evaluation import (  # noqa: E402
    bootstrap_ci,
    classification_rows,
    compare_per_repo,
    cross_repo_f1,
    minimum_detectable_difference,
    observed_discordant_rate,
    random_split_control,
    time_aware_split,
)
from ai4se.loader import load_split  # noqa: E402
from ai4se.model import REPOSITORIES  # noqa: E402
from ai4se.repository import IssueRepository, make_repository  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "benchmark_audit.json"
SEED = 42


def _banner(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _fit_predict(
    train: IssueRepository,
    test: IssueRepository,
    seed: int = SEED,
) -> np.ndarray:
    """Train a TF-IDF + linear SVM on ``train`` and predict ``test``.

    Args:
        train: Training issues.
        test: Issues to predict.
        seed: Seed passed to the estimator for reproducibility.

    Returns:
        Predicted labels, aligned with ``test.all()``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    x_train, y_train = train.texts_and_labels()
    x_test, _ = test.texts_and_labels()
    pipeline = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
        LinearSVC(class_weight="balanced", random_state=seed),
    )
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_test), dtype=object)


def _predict_per_repo(train: IssueRepository, test: IssueRepository) -> np.ndarray:
    """Train one model per project, as the competition requires.

    Args:
        train: Training issues of all projects.
        test: Test issues of all projects.

    Returns:
        Predicted labels, aligned with ``test.all()``.
    """
    issues = test.all()
    predictions = np.empty(len(issues), dtype=object)
    for repo in REPOSITORIES:
        positions = [i for i, issue in enumerate(issues) if issue.repo == repo]
        if not positions:
            continue
        subset = make_repository("memory", issues=[issues[i] for i in positions])
        subset_predictions = _fit_predict(
            make_repository("memory", issues=train.by_repo(repo)), subset
        )
        for position, prediction in zip(positions, subset_predictions, strict=True):
            predictions[position] = prediction
    return predictions


def _labels_and_repos(repository: IssueRepository) -> tuple[list[str], list[str]]:
    """Return the ground-truth labels and source projects of a repository."""
    issues = repository.all()
    return [i.label for i in issues], [i.repo for i in issues]


def step_1_baseline(train: IssueRepository, test: IssueRepository) -> dict:
    """Score the classical baseline under the official protocol."""
    _banner("1. BASELINE UNDER THE OFFICIAL PROTOCOL")
    y_true, repos = _labels_and_repos(test)

    global_pred = _fit_predict(train, test)
    per_repo_pred = _predict_per_repo(train, test)

    point, low, high = bootstrap_ci(y_true, global_pred, repos, n_resamples=2000)
    print(f"  one model for all five projects : {point:.4f}")
    print(f"    95% CI [{low:.4f}, {high:.4f}]  width {high - low:.4f}")
    per_repo_score = cross_repo_f1(y_true, per_repo_pred, repos)
    print(f"  one model per project           : {per_repo_score:.4f}")
    print(f"  official SetFit baseline        : 0.8270\n")

    print("  per project and class (the table the report needs):")
    for row in classification_rows(y_true, global_pred, repos):
        if row["repo"] == "cross-repo":
            continue
        print(
            f"    {row['repo']:<24} {row['label']:<9} "
            f"P {row['precision']:.3f}  R {row['recall']:.3f}  F1 {row['f1']:.3f}"
        )
    return {
        "global": {"f1": point, "ci": [low, high]},
        "per_repo": {"f1": per_repo_score},
        "setfit_official": 0.8270,
    }


def step_2_temporal_confound(full: IssueRepository) -> dict:
    """Predict the label from the creation timestamp alone."""
    _banner("2. THE CONFOUND: LABEL FROM TIMESTAMP ALONE, NO TEXT")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from ai4se.evaluation import micro_f1

    scores = {}
    for repo in REPOSITORIES:
        issues = full.by_repo(repo)
        if not issues:
            continue
        # Seconds since epoch as the single feature. No text is read.
        times = np.array(
            [np.datetime64(i.created_at.replace(" ", "T")).astype("int64") for i in issues],
            dtype=float,
        ).reshape(-1, 1)
        labels = [i.label for i in issues]
        predicted = cross_val_predict(
            GradientBoostingClassifier(random_state=SEED),
            times,
            labels,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        )
        scores[repo] = micro_f1(labels, predicted)
        print(f"  {repo:<24} F1 = {scores[repo]:.3f}")
    mean = float(np.mean(list(scores.values())))
    print(f"  {'MEAN':<24} F1 = {mean:.3f}   (chance = 0.333)")
    return {"per_repo": scores, "mean": mean, "chance": 1 / 3}


def step_3_split_protocol(full: IssueRepository) -> dict:
    """Compare a time-aware split against a matched random control."""
    _banner("3. WHAT THE RANDOM SPLIT COSTS")
    time_train, time_test = time_aware_split(full)
    rand_train, rand_test = random_split_control(full, seed=SEED)

    time_truth, time_repos = _labels_and_repos(time_test)
    rand_truth, rand_repos = _labels_and_repos(rand_test)

    time_score = cross_repo_f1(
        time_truth, _fit_predict(time_train, time_test), time_repos
    )
    rand_score = cross_repo_f1(
        rand_truth, _fit_predict(rand_train, rand_test), rand_repos
    )
    print(f"  random split (matched control) : {rand_score:.4f}")
    print(f"  time-aware split               : {time_score:.4f}")
    print(f"  difference                     : {time_score - rand_score:+.4f}")
    print(f"  class balance preserved        : {time_train.label_distribution()}")
    return {
        "random": rand_score,
        "time_aware": time_score,
        "delta": time_score - rand_score,
    }


def step_4_significance(train: IssueRepository, test: IssueRepository) -> dict:
    """Test whether the baseline differences survive correction."""
    _banner("4. DO THE DIFFERENCES SURVIVE McNEMAR + HOLM?")
    y_true, repos = _labels_and_repos(test)
    global_pred = _fit_predict(train, test)
    per_repo_pred = _predict_per_repo(train, test)

    rows = compare_per_repo(per_repo_pred, global_pred, y_true, repos)
    print("  one model per project  vs  one model for all")
    for row in rows:
        verdict = "significant" if row["significant"] else "ns"
        print(
            f"    {row['repo']:<24} n01={row['n01']:<4} n10={row['n10']:<4} "
            f"discordant={row['discordant']:<4} p={row['p_value']:.3e} "
            f"p_holm={row['p_holm']:.3e}  {verdict}"
        )
    survivors = sum(1 for r in rows[:-1] if r["significant"])
    print(f"\n  projects reaching significance after correction: {survivors}/5")
    return {"rows": rows, "survivors": survivors}


def step_5_power(train: IssueRepository, test: IssueRepository) -> dict:
    """Measure the smallest difference this test set can detect."""
    _banner("5. WHAT CAN THIS TEST SET DETECT AT ALL?")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    y_true, _ = _labels_and_repos(test)
    word_pred = _fit_predict(train, test)

    x_train, y_train = train.texts_and_labels()
    x_test, _ = test.texts_and_labels()
    char_model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    ).fit(x_train, y_train)
    char_pred = np.asarray(char_model.predict(x_test), dtype=object)

    rate = observed_discordant_rate(word_pred, char_pred, y_true)
    mdd_full = minimum_detectable_difference(n_items=len(y_true), discordant_rate=rate)
    mdd_repo = minimum_detectable_difference(n_items=300, discordant_rate=rate)
    print(f"  discordant rate between two reasonable models : {rate:.3f}")
    print(f"  minimum detectable difference, whole test set : {mdd_full * 100:.1f} pts")
    print(f"  minimum detectable difference, one project    : {mdd_repo * 100:.1f} pts")
    print("\n  Any reported gain below that threshold is not evidence.")
    return {
        "discordant_rate": rate,
        "mdd_cross_repo": mdd_full,
        "mdd_per_repo": mdd_repo,
    }


def main() -> None:
    """Run every step and write the results to ``results/``."""
    train = load_split("train", kind="memory")
    test = load_split("test", kind="memory")
    full = make_repository("memory", issues=train.all() + test.all())
    print(f"Loaded {len(train)} training and {len(test)} test issues.")

    results = {
        "baseline": step_1_baseline(train, test),
        "temporal_confound": step_2_temporal_confound(full),
        "split_protocol": step_3_split_protocol(full),
        "significance": step_4_significance(train, test),
        "power": step_5_power(train, test),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten to {RESULTS_PATH.relative_to(RESULTS_PATH.parents[1])}")


if __name__ == "__main__":
    main()
