"""Acquisition of the NLBSE'24 issue report classification dataset.

The competition publishes two balanced CSV files of 1,500 issues each
(a 50/50 train/test split of 3,000 issues, 300 per project per split).
This module downloads them once, caches them under ``data/raw/`` and hands
them to the persistence layer defined in :mod:`ai4se.repository`.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .repository import IssueRepository, make_repository

BASE_URL = (
    "https://raw.githubusercontent.com/nlbse2024/issue-report-classification/main/data"
)

SPLIT_FILES = {
    "train": "issues_train.csv",
    "test": "issues_test.csv",
}


def _project_root() -> Path:
    """Locate the project root so paths do not depend on the working directory.

    Without this, running the notebook from ``notebooks/`` would create a
    second copy of the dataset under ``notebooks/data/raw/``.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "src" / "ai4se").is_dir():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _project_root()
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def download_split(split: str, raw_dir: Path | str = DEFAULT_RAW_DIR) -> Path:
    """Download one split of the dataset if it is not already cached.

    Args:
        split: ``"train"`` or ``"test"``.
        raw_dir: Directory where the raw CSV files are cached.

    Returns:
        Path of the local CSV file.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; use 'train' or 'test'.")

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / SPLIT_FILES[split]

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    url = f"{BASE_URL}/{SPLIT_FILES[split]}"
    print(f"Downloading {split} split from {url} ...")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved to {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return destination


def load_split(
    split: str,
    kind: str = "memory",
    raw_dir: Path | str = DEFAULT_RAW_DIR,
) -> IssueRepository:
    """Load one split into a repository of the requested persistence kind.

    Args:
        split: ``"train"`` or ``"test"``.
        kind: ``"memory"`` to hold the issues in RAM, ``"file"`` to keep the
            CSV on disk as the backing store.
        raw_dir: Directory where the raw CSV files are cached.

    Returns:
        An :class:`~ai4se.repository.IssueRepository`.

    Example:
        >>> train = load_split("train", kind="memory")
        >>> len(train)
        1500
    """
    path = download_split(split, raw_dir)

    # The file-backed repository reads the CSV directly.
    file_repo = make_repository("file", path=path)

    if kind == "file":
        return file_repo
    # For the in-memory variant we simply hand the same entities over; from
    # this point on nothing in the application can tell the difference.
    return make_repository("memory", issues=file_repo.all())


def load_dataset(
    kind: str = "memory",
    raw_dir: Path | str = DEFAULT_RAW_DIR,
) -> dict[str, IssueRepository]:
    """Load both splits.

    Returns:
        Mapping ``{"train": repository, "test": repository}``.
    """
    return {split: load_split(split, kind, raw_dir) for split in SPLIT_FILES}
