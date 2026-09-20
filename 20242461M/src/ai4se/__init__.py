"""AI4SE course project -- Project 1: Issue Report Classification.

Track A (data & persistence) public surface.
"""

from .loader import load_dataset, load_split
from .model import LABELS, REPOSITORIES, IssueReport
from .preprocessing import clean_issue, clean_text, make_cleaner
from .repository import (
    FileIssueRepository,
    InMemoryIssueRepository,
    IssueRepository,
    make_repository,
)

__all__ = [
    "IssueReport",
    "LABELS",
    "REPOSITORIES",
    "IssueRepository",
    "InMemoryIssueRepository",
    "FileIssueRepository",
    "make_repository",
    "load_split",
    "load_dataset",
    "clean_text",
    "clean_issue",
    "make_cleaner",
    # metrics
    "evaluate",
    "confusion_matrix",
    "Scores",
    # evaluation protocol
    "stratified_folds",
    "cross_validate",
    "cross_validate_per_project",
    "evaluate_competition",
    "leaderboard",
    "CrossValidationResult",
    "CompetitionResult",
    "SETFIT_BASELINE",
    "SETFIT_OVERALL",
    "grid_search",
    "roc_curve",
    "roc_auc_ovr",
    # models
    "CLASSICAL_MODELS",
    "TUNED_MODELS",
    "TUNED_PREPROCESSING",
]

# Track C (torch) and Track D2 (sentence-transformers) are imported on demand
# rather than here: they pull in heavy optional dependencies, and the data and
# evaluation layers must stay usable without them.

__version__ = "0.1.0"
