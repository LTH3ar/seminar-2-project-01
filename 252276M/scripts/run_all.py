#!/usr/bin/env python
"""Main entrypoint: run all experiments and generate all results.

Student ID: 252276M  |  Group 10 - HUST  |  Seminar 2, 2026

Usage:
    python scripts/run_all.py               # Run all stages
    python scripts/run_all.py --stage floors,classical
    python scripts/run_all.py --stage classical
    python scripts/run_all.py --stage neural
    python scripts/run_all.py --stage frozen
    python scripts/run_all.py --stage ensemble
    python scripts/run_all.py --stage error
    python scripts/run_all.py --stage figures

Stages (in order):
    1. floors     - Reference floor classifiers (Majority, Random, Keyword, Scratch NB)
    2. classical  - TF-IDF classical models (NB, Complement NB, LogReg, SVM, RF)
    3. neural     - Neural networks (FFNN, TextCNN) with early stopping
    4. frozen     - Frozen Sentence Encoder + LogReg head (MiniLM, MPNet)
    5. ensemble   - Soft-voting ensembles of the above models
    6. error      - Error analysis on best classical model
    7. figures    - Learning curve and EDA figures

Results are written to results/tables/ (JSON) and results/figures/ (PNG).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ─────────────────────────────── path setup ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai4se.loader import load_split
from ai4se.preprocessing import make_cleaner
from ai4se.evaluation.metrics import evaluate_predictions, per_repo_and_macro_f1
from ai4se.classifiers.base import train_per_repo, train_pooled
from ai4se.classifiers.floors import (
    MajorityClassifier, StratifiedRandomClassifier,
    KeywordRulesClassifier, ScratchNaiveBayesClassifier,
)
from ai4se.classifiers.classical import make_tuned_logreg, make_tuned_linear_svm, make_classical
from ai4se.classifiers.neural import make_ffnn, make_text_cnn
from ai4se.classifiers.frozen import make_frozen_minilm, make_frozen_mpnet
from ai4se.classifiers.ensemble import SoftVotingEnsemble
from export_figures import export_figure_2_1, export_figure_2_2, export_figure_neural_curves


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =============================== helpers ===============================

SEEDS = [41, 42, 43, 44, 45]

def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def save_json(name: str, data: dict) -> None:
    path = TABLES_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"  -> Saved {path.relative_to(ROOT)}")


def apply_preprocessing(repo, cleaner_fn):
    """Apply cleaning to all issues in a repository in-place."""
    from ai4se.repository import InMemoryIssueRepository
    issues = repo.all()
    cleaned = InMemoryIssueRepository()
    for iss in issues:
        cleaned.add(iss.replace_text(cleaner_fn(iss.title, iss.body)))
    return cleaned


def run_model_multi_seed(
    name: str,
    factory_fn,
    train_repo,
    test_repo,
    test_issues,
    seeds: list[int] = SEEDS,
    pooled: bool = False,
    return_proba: bool = False,
) -> dict:
    """Run a factory model over multiple seeds and aggregate results."""
    seed_scores = []
    seed_preds = []
    seed_probs = []

    for seed in seeds:
        def _factory(s=seed):
            clf = factory_fn()
            if hasattr(clf, "seed"):
                clf.seed = s
            return clf

        runner = train_pooled if pooled else train_per_repo
        preds, probs = runner(
            _factory, train_repo, test_repo, return_proba=return_proba
        )
        seed_preds.append(preds)
        if probs is not None:
            seed_probs.append(probs)

        repo_result = per_repo_and_macro_f1(test_issues, preds)
        seed_scores.append(repo_result["cross_repo_f1"])

    import numpy as np
    mean_score = float(np.mean(seed_scores))
    std_score = float(np.std(seed_scores))
    # Use median-seed predictions for detailed breakdown
    median_idx = int(np.argmin(np.abs(np.array(seed_scores) - mean_score)))
    best_preds = seed_preds[median_idx]
    best_probs = seed_probs[median_idx] if seed_probs else None

    repo_result = per_repo_and_macro_f1(test_issues, best_preds)
    eval_result = evaluate_predictions(
        [iss.label for iss in test_issues], list(best_preds),
        y_prob=best_probs
    )

    return {
        "name": name,
        "seeds": seeds,
        "seed_scores": seed_scores,
        "mean_f1": mean_score,
        "std_f1": std_score,
        "cross_repo_f1": repo_result["cross_repo_f1"],
        "per_repo_f1": repo_result["per_repo"],
        "macro_f1": eval_result["macro_f1"],
        "weighted_f1": eval_result["weighted_f1"],
        "accuracy": eval_result["accuracy"],
        "auc": eval_result["auc"],
        "per_class": eval_result["per_class"],
        "predictions": list(best_preds),
        "probabilities": best_probs.tolist() if best_probs is not None else None,
    }


# ================================= stages =================================

def stage_floors(train_repo, test_repo, test_issues):
    banner("Stage 1: Reference Floors")

    cleaner = make_cleaner(level="light", title_weight=2, max_words=300)
    train_c = apply_preprocessing(train_repo, cleaner)
    test_c  = apply_preprocessing(test_repo, cleaner)

    floors = [
        ("majority_class",      lambda: MajorityClassifier()),
        ("random_stratified",   lambda: StratifiedRandomClassifier(seed=42)),
        ("keyword_rules",       lambda: KeywordRulesClassifier()),
        ("naive_bayes_scratch", lambda: ScratchNaiveBayesClassifier()),
    ]

    results = {}
    for name, factory in floors:
        print(f"\n  Running {name}...")
        t0 = time.time()
        preds, probs = train_per_repo(factory, train_c, test_c, return_proba=True)
        elapsed = time.time() - t0

        true_labels = [iss.label for iss in test_issues]
        repo_res = per_repo_and_macro_f1(test_issues, preds)
        eval_res = evaluate_predictions(true_labels, list(preds), y_prob=probs)

        results[name] = {
            "name": name,
            "cross_repo_f1": repo_res["cross_repo_f1"],
            "per_repo_f1": repo_res["per_repo"],
            "macro_f1": eval_res["macro_f1"],
            "accuracy": eval_res["accuracy"],
            "auc": eval_res["auc"],
            "elapsed_s": round(elapsed, 2),
        }
        print(f"    Cross-repo F1: {repo_res['cross_repo_f1']:.4f}  ({elapsed:.1f}s)")

    save_json("floors", results)
    return results


def stage_classical(train_repo, test_repo, test_issues):
    banner("Stage 2: Classical Models (TF-IDF)")

    cleaner = make_cleaner(level="full", title_weight=3, max_words=400)
    train_c = apply_preprocessing(train_repo, cleaner)
    test_c  = apply_preprocessing(test_repo, cleaner)

    models = [
        ("tfidf_naive_bayes",   make_classical("naive_bayes")),
        ("tfidf_complement_nb", make_classical("complement_nb")),
        ("tfidf_logreg",        make_classical("logreg")),
        ("tfidf_logreg_tuned",  make_tuned_logreg()),
        ("tfidf_linear_svm",    make_classical("linear_svm")),
        ("tfidf_svm_tuned",     make_tuned_linear_svm()),
        ("tfidf_random_forest", make_classical("random_forest")),
    ]

    results = {}
    for name, factory in models:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_model_multi_seed(name, factory, train_c, test_c, test_issues)
        elapsed = time.time() - t0
        res["elapsed_s"] = round(elapsed, 2)
        results[name] = res
        print(f"    Mean F1: {res['mean_f1']:.4f} +/- {res['std_f1']:.4f}  ({elapsed:.1f}s)")

    save_json("classical", results)
    return results


def stage_neural(train_repo, test_repo, test_issues):
    banner("Stage 3: Neural Networks (FFNN + TextCNN)")

    cleaner = make_cleaner(level="light", title_weight=2, max_words=256)
    train_c = apply_preprocessing(train_repo, cleaner)
    test_c  = apply_preprocessing(test_repo, cleaner)

    models = [
        ("ffnn_tfidf", make_ffnn(seed=42)),
        ("text_cnn",   make_text_cnn(seed=42)),
    ]

    results = {}
    for name, factory in models:
        print(f"\n  Running {name} ...")
        t0 = time.time()
        res = run_model_multi_seed(name, factory, train_c, test_c, test_issues, seeds=[41, 42, 43])
        elapsed = time.time() - t0

        # Capture learning curve from last run
        try:
            clf = factory()
            sample_train = train_c.by_repo("facebook/react")
            sample_texts = [iss.text for iss in sample_train]
            sample_labels = [iss.label for iss in sample_train]
            clf.fit(sample_texts, sample_labels)
            res["history"] = clf.history
            res["best_epoch"] = clf.best_epoch
            res["stopped_early"] = clf.stopped_early
        except Exception:
            pass

        res["elapsed_s"] = round(elapsed, 2)
        results[name] = res
        print(f"    Mean F1: {res['mean_f1']:.4f} +/- {res['std_f1']:.4f}  ({elapsed:.1f}s)")

    save_json("neural", results)
    return results


def stage_frozen(train_repo, test_repo, test_issues):
    banner("Stage 4: Frozen Sentence Encoders")

    cleaner = make_cleaner(level="light", title_weight=2, max_words=300)
    train_c = apply_preprocessing(train_repo, cleaner)
    test_c  = apply_preprocessing(test_repo, cleaner)

    models = [
        ("frozen_minilm", make_frozen_minilm()),
        ("frozen_mpnet",  make_frozen_mpnet()),
    ]

    results = {}
    for name, factory in models:
        print(f"\n  Running {name} ...")
        t0 = time.time()
        res = run_model_multi_seed(name, factory, train_c, test_c, test_issues,
                                   seeds=[41, 42, 43], return_proba=True)
        elapsed = time.time() - t0
        res["elapsed_s"] = round(elapsed, 2)
        results[name] = res
        print(f"    Mean F1: {res['mean_f1']:.4f} +/- {res['std_f1']:.4f}  ({elapsed:.1f}s)")

    save_json("frozen", results)
    return results


def stage_ensemble(train_repo, test_repo, test_issues, classical_results, frozen_results):
    banner("Stage 5: Soft-Voting Ensembles")
    import numpy as np

    cleaner = make_cleaner(level="light", title_weight=2, max_words=300)
    train_c = apply_preprocessing(train_repo, cleaner)
    test_c  = apply_preprocessing(test_repo, cleaner)

    true_labels = [iss.label for iss in test_issues]

    # Build best individual fitted models (tuned logreg + frozen mpnet)
    logreg_factory  = make_tuned_logreg()
    mpnet_factory   = make_frozen_mpnet()

    def make_two_model_ensemble():
        logreg = logreg_factory()
        mpnet  = mpnet_factory()
        return SoftVotingEnsemble(classifiers=[logreg, mpnet])

    def make_three_model_ensemble():
        logreg = logreg_factory()
        mpnet  = mpnet_factory()
        minilm = make_frozen_minilm()()
        return SoftVotingEnsemble(classifiers=[logreg, mpnet, minilm])

    ensembles = [
        ("ensemble_logreg_mpnet",         make_two_model_ensemble),
        ("ensemble_logreg_mpnet_minilm",  make_three_model_ensemble),
    ]

    results = {}
    for name, factory in ensembles:
        print(f"\n  Running {name} ...")
        t0 = time.time()
        preds, probs = train_per_repo(factory, train_c, test_c, return_proba=True)
        elapsed = time.time() - t0

        repo_res  = per_repo_and_macro_f1(test_issues, preds)
        eval_res  = evaluate_predictions(true_labels, list(preds), y_prob=probs)

        results[name] = {
            "name": name,
            "cross_repo_f1": repo_res["cross_repo_f1"],
            "per_repo_f1": repo_res["per_repo"],
            "macro_f1": eval_res["macro_f1"],
            "weighted_f1": eval_res["weighted_f1"],
            "accuracy": eval_res["accuracy"],
            "auc": eval_res["auc"],
            "elapsed_s": round(elapsed, 2),
            "predictions": list(preds),
        }
        print(f"    Cross-repo F1: {repo_res['cross_repo_f1']:.4f}  AUC: {eval_res['auc']}  ({elapsed:.1f}s)")

    save_json("ensembles", results)
    return results


def stage_error_analysis(train_repo, test_repo, test_issues, classical_results):
    banner("Stage 6: Error Analysis on Best Classical Model")

    from ai4se.analysis import (
        misclassification_pairs, confusion_summary,
        per_class_per_repo_errors, length_quintile_accuracy,
        template_lift_analysis,
    )

    cleaner = make_cleaner(level="full", title_weight=3, max_words=400)
    train_c = apply_preprocessing(train_repo, cleaner)
    test_c  = apply_preprocessing(test_repo, cleaner)

    # Use tuned LogReg as best classical model
    factory = make_tuned_logreg()
    preds, probs = train_per_repo(factory, train_c, test_c, return_proba=True)

    pairs = misclassification_pairs(test_issues, list(preds))
    confusion = confusion_summary(pairs)
    per_repo_errs = per_class_per_repo_errors(test_issues, list(preds))
    quintile_acc = length_quintile_accuracy(test_issues, list(preds))
    template_lift = template_lift_analysis(list(train_repo.all()))

    n_errors = sum(len(v) for v in pairs.values())

    results = {
        "model": "tfidf_logreg_tuned",
        "n_errors": n_errors,
        "confusion_pairs": confusion,
        "per_repo_class_errors": per_repo_errs,
        "length_quintile_accuracy": quintile_acc,
        "template_lift": template_lift,
    }

    save_json("error_analysis", results)
    print(f"  Total mis-classifications: {n_errors}")
    print(f"  Confusion pairs: {len(confusion)}")
    return results


def stage_figures(neural_results=None, train_repo=None):
    banner("Stage 7: Generating Figures")
    try:
        if train_repo is not None:
            export_figure_2_1(train_repo)
            export_figure_2_2(train_repo)
        export_figure_neural_curves()
        print("  All publication figures generated in results/figures/")
    except Exception as e:
        print(f"  Error generating figures: {e}")


# ================================= main =================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run 252276M experiments")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Comma-separated list of stages to run: floors, classical, neural, frozen, ensemble, error, figures, all"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    requested = set(s.strip().lower() for s in args.stage.split(","))
    run_all = "all" in requested

    banner("Loading dataset...")
    print("  Loading train split...")
    train_repo = load_split("train", kind="memory")
    print(f"  -> {len(train_repo)} training issues loaded")

    print("  Loading test split...")
    test_repo = load_split("test", kind="memory")
    test_issues = list(test_repo.all())
    print(f"  -> {len(test_repo)} test issues loaded")

    classical_results = {}
    frozen_results = {}
    neural_results = {}

    if run_all or "floors" in requested:
        stage_floors(train_repo, test_repo, test_issues)

    if run_all or "classical" in requested:
        classical_results = stage_classical(train_repo, test_repo, test_issues)

    if run_all or "neural" in requested:
        neural_results = stage_neural(train_repo, test_repo, test_issues)

    if run_all or "frozen" in requested:
        frozen_results = stage_frozen(train_repo, test_repo, test_issues)

    if run_all or "ensemble" in requested:
        stage_ensemble(train_repo, test_repo, test_issues, classical_results, frozen_results)

    if run_all or "error" in requested:
        stage_error_analysis(train_repo, test_repo, test_issues, classical_results)

    if run_all or "figures" in requested:
        stage_figures(neural_results, train_repo=train_repo)

    banner("All requested stages completed.")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
