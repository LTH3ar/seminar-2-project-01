"""Run every experiment in the project and save the results.

One entry point that reproduces every number in the report, so the notebooks
can read saved results instead of re-training, and so a reviewer can
regenerate everything with a single command::

    python scripts/run_experiments.py --all
    python scripts/run_experiments.py --classical --neural
    python scripts/run_experiments.py --list

Results are written to ``results/tables/*.json`` and figures to
``results/figures/``. Each stage is independent and skips work that is already
on disk unless ``--force`` is given, so an interrupted run resumes cheaply.

Stages
------
``floors``      reference baselines (majority, random, keyword, naive Bayes)
``ablation``    cleaning level x title weight x truncation, on the training split
``grid``        hyperparameter search for the linear models
``classical``   Track B, tuned, on the competition protocol
``neural``      Track C, with learning curves
``embeddings``  Track D2, sentence-transformer baselines (``--setfit`` adds the
                full contrastive reproduction)
``ensemble``    soft-voting ensembles of the complementary models
``errors``      error analysis of the best model
``validity``    temporal confound, statistical power, per-project vs pooled
"""

from __future__ import annotations

import argparse
import json
import time

from ai4se.baselines import REFERENCE_MODELS
from ai4se.classical import (
    CLASSICAL_MODELS,
    TUNED_MODELS,
    TUNED_PREPROCESSING,
    linear_svm,
    logistic_regression,
    tuned_logistic_regression,
)
from ai4se.evaluation import (
    cross_validate,
    evaluate_competition,
    grid_search,
    result_slug,
    save_result,
)
from ai4se.loader import PROJECT_ROOT, load_split
from ai4se.preprocessing import make_cleaner

RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

#: Preprocessing for models that do their own sub-word tokenisation.
TRANSFORMER_PREPROCESSING = {"level": "light", "max_words": 256, "title_weight": 1}


#: Canonical file stem for a model's saved result. Defined in the package so
#: the notebooks and this runner cannot produce different names for one model.
slug = result_slug


def load(preprocessing: dict) -> tuple:
    """Load both splits with the given preprocessing applied."""
    cleaner = make_cleaner(**preprocessing)
    train = load_split("train", kind="memory")
    test = load_split("test", kind="memory")
    train.apply(cleaner)
    test.apply(cleaner)
    return train, test


def announce(stage: str) -> float:
    """Print a stage header and return the start time."""
    print(f"\n{'=' * 70}\n{stage}\n{'=' * 70}")
    return time.time()


def done(started: float) -> None:
    """Print the elapsed time of a stage."""
    print(f"-- {time.time() - started:.0f}s")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def run_floors(force: bool = False) -> None:
    """Reference baselines, on both the CV and competition protocols."""
    started = announce("Reference floors")
    train, test = load({"level": "full", "max_words": 400})

    for name, factory in REFERENCE_MODELS.items():
        cv = cross_validate(factory, train, k=10, model_name=name)
        save_result(cv, TABLES / f"cv_{slug(name)}.json")
        competition = evaluate_competition(factory, train, test, model_name=name)
        save_result(competition, TABLES / f"competition_{slug(name)}.json")
        print(
            f"  {name:<30} CV {cv.mean_macro_f1:.4f}  "
            f"test {competition.overall_f1:.4f}"
        )
    done(started)


def run_ablation(force: bool = False) -> None:
    """Cleaning level, title weight and truncation, cross-validated."""
    import pandas as pd

    destination = TABLES / "ablation_preprocessing.json"
    if destination.exists() and not force:
        print("Ablation already on disk; use --force to recompute.")
        return

    started = announce("Preprocessing ablation")
    from ai4se.evaluation import parameterised_factory

    factory = parameterised_factory(
        logistic_regression, C=5.0, ngram_range=(1, 2), min_df=1
    )

    rows = []
    for level in ("raw", "light", "full"):
        for title_weight in (1, 3):
            for max_words in (200, 400, None):
                train = load_split("train", kind="memory")
                train.apply(
                    make_cleaner(
                        level=level, title_weight=title_weight, max_words=max_words
                    )
                )
                result = cross_validate(factory, train, k=5)
                rows.append(
                    {
                        "level": level,
                        "title_weight": title_weight,
                        "max_words": max_words,
                        "macro_f1": round(result.mean_macro_f1, 4),
                        "std": round(result.std_macro_f1, 4),
                    }
                )
                print(
                    f"  {level:<6} tw={title_weight} mw={str(max_words):<5} "
                    f"-> {result.mean_macro_f1:.4f}"
                )

    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    destination.write_text(frame.to_json(orient="records", indent=2))
    done(started)


def run_grid(force: bool = False) -> None:
    """Hyperparameter search for the two linear models."""
    destination = TABLES / "grid_search.json"
    if destination.exists() and not force:
        print("Grid search already on disk; use --force to recompute.")
        return

    started = announce("Hyperparameter grid search")
    train, _ = load(TUNED_PREPROCESSING)

    grids = {
        "logistic_regression": (
            logistic_regression,
            {"C": [1.0, 5.0, 10.0, 25.0], "ngram_range": [(1, 1), (1, 2)],
             "min_df": [1, 2]},
        ),
        "linear_svm": (
            linear_svm,
            {"C": [0.25, 0.5, 1.0, 2.0], "ngram_range": [(1, 1), (1, 2)],
             "min_df": [1, 2]},
        ),
    }

    output = {}
    for name, (factory, grid) in grids.items():
        frame = grid_search(factory, grid, train, k=5)
        frame = frame.astype({"ngram_range": str})
        output[name] = json.loads(frame.to_json(orient="records"))
        best = frame.iloc[0]
        print(f"  {name}: best {best['macro_f1']:.4f} at C={best['C']}")

    destination.write_text(json.dumps(output, indent=2))
    done(started)


def run_classical(force: bool = False) -> None:
    """Track B on the competition protocol, untuned and tuned."""
    started = announce("Track B -- classical machine learning")
    train, test = load(TUNED_PREPROCESSING)

    for name, factory in {**CLASSICAL_MODELS, **TUNED_MODELS}.items():
        cv = cross_validate(factory, train, k=10, model_name=name)
        save_result(cv, TABLES / f"cv_{slug(name)}.json")
        competition = evaluate_competition(factory, train, test, model_name=name)
        save_result(competition, TABLES / f"competition_{slug(name)}.json")
        auc = f"{competition.overall_auc:.4f}" if competition.overall_auc else "--"
        print(
            f"  {name:<38} CV {cv.mean_macro_f1:.4f}  "
            f"test {competition.overall_f1:.4f}  AUC {auc}"
        )
    done(started)


def run_neural(force: bool = False) -> None:
    """Track C, saving learning curves as well as scores."""
    started = announce("Track C -- neural networks")
    from ai4se.neural import NEURAL_MODELS, plot_learning_curves

    train, test = load(TUNED_PREPROCESSING)

    for name, factory in NEURAL_MODELS.items():
        competition = evaluate_competition(factory, train, test, model_name=name)
        save_result(competition, TABLES / f"competition_{slug(name)}.json")

        # Refit on one project purely to capture a representative curve.
        subset = train.by_repo("facebook/react")
        model = factory()
        model.fit([i.text for i in subset], [i.label for i in subset])
        (TABLES / f"history_{slug(name)}.json").write_text(
            json.dumps(model.history_.as_dict(), indent=2)
        )
        figure = plot_learning_curves(model.history_, title=f"{name} -- facebook/react")
        figure.savefig(
            FIGURES / f"learning_curve_{slug(name)}.png", dpi=200, bbox_inches="tight"
        )

        print(
            f"  {name:<24} test {competition.overall_f1:.4f}  "
            f"(best epoch {model.history_.best_epoch}, "
            f"early stop {model.history_.stopped_early})"
        )
    done(started)


def run_embeddings(force: bool = False, with_setfit: bool = False) -> None:
    """Track D2, the frozen sentence-transformer baselines.

    Pass ``--setfit`` to additionally run the full contrastive reproduction.
    It is off by default because fine-tuning the encoder five times needs a
    GPU; on a CPU it takes hours rather than minutes.
    """
    started = announce("Track D2 -- sentence transformers")
    from ai4se.embeddings import EMBEDDING_MODELS

    train, test = load(TRANSFORMER_PREPROCESSING)

    models = dict(EMBEDDING_MODELS)
    if with_setfit:
        # Imported only when asked for: the module is importable without the
        # 'setfit' extra, but pulling the class in unconditionally would make
        # this stage's imports imply a dependency it does not need.
        from ai4se.embeddings import SETFIT_MODELS

        models.update(SETFIT_MODELS)

    for name, factory in models.items():
        destination = TABLES / f"competition_{slug(name)}.json"
        if destination.exists() and not force:
            print(f"  {name:<28} already on disk (use --force to recompute)")
            continue
        if name.startswith("SetFit"):
            print(f"  {name}: fine-tuning five encoders, this will take a while ...")
        competition = evaluate_competition(factory, train, test, model_name=name)
        save_result(competition, destination)
        auc = f"{competition.overall_auc:.4f}" if competition.overall_auc else "--"
        print(f"  {name:<28} test {competition.overall_f1:.4f}  AUC {auc}")
    done(started)


def run_ensemble(force: bool = False, with_setfit: bool = False) -> None:
    """Soft-voting ensembles of the complementary models.

    Motivated by the per-repository profiles: TF-IDF wins on the two easiest
    projects, frozen MPNet on the three hardest, with half the spread. Needs
    the 'embeddings' extra, and downloads the MPNet weights on first use.
    """
    started = announce("Ensembles")
    from ai4se.ensemble import ENSEMBLE_MODELS

    models = dict(ENSEMBLE_MODELS)
    if with_setfit:
        from ai4se.ensemble import SETFIT_ENSEMBLES

        models.update(SETFIT_ENSEMBLES)

    train, test = load(TUNED_PREPROCESSING)

    for name, factory in models.items():
        destination = TABLES / f"competition_{slug(name)}.json"
        if destination.exists() and not force:
            print(f"  {name:<34} already on disk (use --force to recompute)")
            continue
        competition = evaluate_competition(factory, train, test, model_name=name)
        save_result(competition, destination)
        auc = f"{competition.overall_auc:.4f}" if competition.overall_auc else "--"
        print(f"  {name:<34} test {competition.overall_f1:.4f}  AUC {auc}")
    done(started)


def run_validity(force: bool = False) -> None:
    """Temporal confound, statistical power, and per-project versus pooled training."""
    started = announce("Validity -- confound, power, pooling")
    from ai4se.baselines import MultinomialNaiveBayes
    from ai4se.classical import naive_bayes, tuned_linear_svm
    from ai4se.error_analysis import collect_predictions
    from ai4se.evaluation import SETFIT_BASELINE, SETFIT_OVERALL
    from ai4se.validity import (
        discordant_rate,
        label_share_by_year,
        minimum_detectable_difference,
        pooling_comparison,
        timestamp_only_scores,
        unpaired_mdd,
    )

    raw_train = load_split("train", kind="memory")
    raw_test = load_split("test", kind="memory")
    timestamp = timestamp_only_scores(raw_train, raw_test)
    by_year = label_share_by_year(raw_train)
    print(f"  timestamp-only cross-repository F1: {timestamp['mean']:.4f}")

    train, test = load(TUNED_PREPROCESSING)
    flat = lambda preds, key: [y for r in preds for y in preds[r][key]]  # noqa: E731
    reference = collect_predictions(tuned_logistic_regression, train, test)
    truth, ref_pred = flat(reference, "y_true"), flat(reference, "y_pred")

    pairs = {}
    for name, factory in [
        ("TF-IDF + Linear SVM (tuned)", tuned_linear_svm),
        ("TF-IDF + Naive Bayes", naive_bayes),
        ("TF-IDF + Random Forest", CLASSICAL_MODELS["TF-IDF + Random Forest"]),
    ]:
        other = collect_predictions(factory, train, test)
        rate = discordant_rate(ref_pred, flat(other, "y_pred"), truth)
        pairs[name] = {
            "discordant_rate": round(rate, 4),
            "mdd_overall": minimum_detectable_difference(1500, rate),
            "mdd_per_project": minimum_detectable_difference(300, rate),
        }
        print(f"  LR vs {name:<30} disagree {rate:.3f} -> MDD "
              f"{pairs[name]['mdd_overall'] * 100:.1f} / "
              f"{pairs[name]['mdd_per_project'] * 100:.1f} points")

    sensitivity = {
        f"{rate:.2f}": {
            "overall": minimum_detectable_difference(1500, rate),
            "per_project": minimum_detectable_difference(300, rate),
        }
        for rate in (0.05, 0.10, 0.15, 0.20)
    }
    published = {"overall": unpaired_mdd(1500, SETFIT_OVERALL)}
    published.update({r: unpaired_mdd(300, a) for r, a in SETFIT_BASELINE.items()})

    full = {"level": "full", "max_words": 400}
    pooling = {}
    for name, factory, prep in [
        ("naive Bayes (from scratch), full", MultinomialNaiveBayes, full),
        ("naive Bayes (from scratch), light", MultinomialNaiveBayes,
         TUNED_PREPROCESSING),
        ("TF-IDF + Naive Bayes", naive_bayes, TUNED_PREPROCESSING),
        ("TF-IDF + Logistic Regression (tuned)", tuned_logistic_regression,
         TUNED_PREPROCESSING),
        ("TF-IDF + Linear SVM (tuned)", tuned_linear_svm, TUNED_PREPROCESSING),
    ]:
        tr, te = load(prep)
        result = pooling_comparison(factory, tr, te)
        pooling[name] = result
        wins = result["per_project_wins"]
        print(f"  pooling {name:<38} per-project {result['per_project_mean']:.4f}  "
              f"pooled {result['pooled_mean']:.4f}  wins {wins}/5")

    payload = {
        "timestamp_only": timestamp,
        "label_share_by_year": json.loads(by_year.to_json(orient="index")),
        "discordance_vs_tuned_logreg": pairs,
        "mdd_sensitivity": sensitivity,
        "mdd_vs_published_unpaired": published,
        "pooling": pooling,
    }
    (TABLES / "validity.json").write_text(json.dumps(payload, indent=2))
    done(started)


def run_errors(force: bool = False) -> None:
    """Error analysis of the best classical model."""
    started = announce("Error analysis")
    from ai4se import error_analysis as ea

    train, test = load(TUNED_PREPROCESSING)
    predictions = ea.collect_predictions(tuned_logistic_regression, train, test)

    payload = {
        "confusions": json.loads(
            ea.confusion_summary(predictions).to_json(orient="records")
        ),
        "by_length": json.loads(
            ea.errors_by_length(predictions).reset_index().astype(str).to_json(
                orient="records"
            )
        ),
        "worst": [m.as_dict() for m in ea.worst_mistakes(predictions, n=25)],
    }
    (TABLES / "error_analysis.json").write_text(json.dumps(payload, indent=2))

    top = payload["confusions"][0]
    print(
        f"  dominant confusion: {top['true']} -> {top['predicted']} "
        f"({top['share of errors %']}% of all errors)"
    )
    done(started)


STAGES = {
    "floors": run_floors,
    "ablation": run_ablation,
    "grid": run_grid,
    "classical": run_classical,
    "neural": run_neural,
    "embeddings": run_embeddings,
    "ensemble": run_ensemble,
    "errors": run_errors,
    "validity": run_validity,
}


def main() -> None:
    """Parse arguments and run the requested stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run every stage")
    parser.add_argument("--list", action="store_true", help="list the stages")
    parser.add_argument(
        "--force", action="store_true", help="recompute stages already on disk"
    )
    parser.add_argument(
        "--setfit",
        action="store_true",
        help="include the full SetFit reproduction in the embeddings stage "
        "(needs a GPU; hours on a CPU)",
    )
    for stage in STAGES:
        parser.add_argument(f"--{stage}", action="store_true", help=f"run {stage}")
    args = parser.parse_args()

    if args.list:
        for stage, function in STAGES.items():
            print(f"  {stage:<12} {(function.__doc__ or '').splitlines()[0]}")
        return

    # --setfit selects the embeddings stage on its own, so the common case is
    # a single flag rather than two.
    selected = [s for s in STAGES if getattr(args, s)] or (
        list(STAGES) if args.all else (["embeddings"] if args.setfit else [])
    )
    if args.setfit and "embeddings" not in selected:
        selected.append("embeddings")
    if not selected:
        parser.print_help()
        return

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    overall = time.time()
    for stage in selected:
        if stage in {"embeddings", "ensemble"}:
            STAGES[stage](force=args.force, with_setfit=args.setfit)
        else:
            STAGES[stage](force=args.force)
    print(f"\nAll requested stages finished in {time.time() - overall:.0f}s")
    print(f"Results in {TABLES}")


if __name__ == "__main__":
    main()
