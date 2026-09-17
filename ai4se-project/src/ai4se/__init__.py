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
]

__version__ = "0.1.0"
