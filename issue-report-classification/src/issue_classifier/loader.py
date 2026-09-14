"""Acquire the official dataset while preferring the bundled local files."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from issue_classifier.repositories.csv_repository import DEFAULT_FILENAMES


BASE_URL = (
    "https://raw.githubusercontent.com/nlbse2024/"
    "issue-report-classification/main/data"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data"


def download_split(
    split: str,
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    *,
    force: bool = False,
) -> Path:
    """Return a local split, downloading it atomically when it is absent."""

    try:
        filename = DEFAULT_FILENAMES[split]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_FILENAMES))
        raise ValueError(f"Unknown split {split!r}; expected one of: {known}") from exc

    data_directory = Path(data_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    destination = data_directory / filename
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=data_directory,
            prefix=f".{filename}.",
            suffix=".download",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", str(temporary_path))
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def ensure_dataset(
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    splits: Iterable[str] = ("train", "test"),
) -> dict[str, Path]:
    """Ensure every requested split is available locally."""

    return {split: download_split(split, data_directory) for split in splits}
