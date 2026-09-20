"""AI4SE course project -- Project 1: Issue Report Classification.

The package is layered by dependency weight, and this module re-exports only
the layers that need nothing beyond pandas and the standard library:

    model, repository, loader   domain and persistence
    preprocessing, eda          cleaning and dataset characterisation
    metrics, evaluation         scoring and the competition protocol
    baselines                   dependency-free reference models

The model tracks pull in heavy optional dependencies and are therefore *not*
imported here. Import them from their submodules, which keeps ``import ai4se``
usable in an environment with none of them installed::

    from ai4se.classical import CLASSICAL_MODELS      # needs scikit-learn
    from ai4se.neural import FeedForwardClassifier    # needs torch
    from ai4se.embeddings import frozen_minilm        # needs sentence-transformers

Install them per track: ``pip install -e ".[ml]"``, ``".[dl]"``,
``".[embeddings]"``, or ``".[all]"``.
"""

from .baselines import (
    REFERENCE_MODELS,
    KeywordClassifier,
    MajorityClassifier,
    MultinomialNaiveBayes,
    StratifiedRandomClassifier,
)
from .evaluation import (
    RANDOM_SEED,
    SETFIT_BASELINE,
    SETFIT_OVERALL,
    CompetitionResult,
    CrossValidationResult,
    cross_validate,
    cross_validate_per_project,
    evaluate_competition,
    grid_search,
    leaderboard,
    leaderboard_from_disk,
    load_result,
    parameterised_factory,
    result_slug,
    save_result,
    stratified_folds,
    to_latex,
)
from .loader import load_dataset, load_split
from .metrics import (
    ClassScore,
    Scores,
    auc,
    confusion_matrix,
    evaluate,
    roc_auc_ovr,
    roc_curve,
)
from .model import LABELS, REPOSITORIES, IssueReport
from .preprocessing import clean_issue, clean_text, make_cleaner
from .repository import (
    FileIssueRepository,
    InMemoryIssueRepository,
    IssueRepository,
    make_repository,
)

__all__ = [
    # domain
    "IssueReport",
    "LABELS",
    "REPOSITORIES",
    # persistence
    "IssueRepository",
    "InMemoryIssueRepository",
    "FileIssueRepository",
    "make_repository",
    # acquisition
    "load_split",
    "load_dataset",
    # preprocessing
    "clean_text",
    "clean_issue",
    "make_cleaner",
    # metrics
    "evaluate",
    "confusion_matrix",
    "roc_curve",
    "roc_auc_ovr",
    "auc",
    "Scores",
    "ClassScore",
    # evaluation protocol
    "stratified_folds",
    "cross_validate",
    "cross_validate_per_project",
    "evaluate_competition",
    "grid_search",
    "parameterised_factory",
    "leaderboard",
    "leaderboard_from_disk",
    "to_latex",
    "save_result",
    "load_result",
    "result_slug",
    "CrossValidationResult",
    "CompetitionResult",
    "RANDOM_SEED",
    "SETFIT_BASELINE",
    "SETFIT_OVERALL",
    # reference models (no third-party dependencies)
    "REFERENCE_MODELS",
    "MajorityClassifier",
    "StratifiedRandomClassifier",
    "KeywordClassifier",
    "MultinomialNaiveBayes",
]

__version__ = "0.1.0"