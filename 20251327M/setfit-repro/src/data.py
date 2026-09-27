"""Dataset loading, replicating the organisers' pipeline step for step.

The order of operations matters for a faithful reproduction. The organisers'
notebook shuffles the *whole* dataset with seed 42, then maps the text field,
then filters per repository. Shuffling after filtering would hand SetFit a
different set of contrastive pairs and quietly change the result, so the order
here mirrors theirs exactly.

Reference: ``2-Template-SetFit.ipynb`` in
https://github.com/nlbse2024/issue-report-classification
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_URL = (
    "https://raw.githubusercontent.com/nlbse2024/"
    "issue-report-classification/main/data"
)
SPLITS = {"train": "issues_train.csv", "test": "issues_test.csv"}

#: Project order as the organisers' notebook discovers it, i.e. order of first
#: appearance in the training CSV. Kept explicit so results tables are stable.
REPOSITORIES = (
    "facebook/react",
    "tensorflow/tensorflow",
    "microsoft/vscode",
    "bitcoin/bitcoin",
    "opencv/opencv",
)
LABELS = ("bug", "feature", "question")

#: The organisers' seed, used for the shuffle and for training.
OFFICIAL_SEED = 42


@dataclass(frozen=True)
class Split:
    """One split of the dataset, already shuffled and with text assembled.

    Attributes:
        name: ``"train"`` or ``"test"``.
        frame: Columns ``repo``, ``label``, ``text``, plus the original fields.
    """

    name: str
    frame: pd.DataFrame

    def for_repo(self, repo: str) -> pd.DataFrame:
        """Return only the rows belonging to one project."""
        return self.frame[self.frame.repo == repo]

    def texts_and_labels(self, repo: str | None = None) -> tuple[list[str], list[str]]:
        """Return ``(texts, labels)`` for one project, or for all of them."""
        frame = self.frame if repo is None else self.for_repo(repo)
        return frame.text.tolist(), frame.label.tolist()

    def __len__(self) -> int:
        """Number of issues in the split."""
        return len(self.frame)


def download(split: str, cache_dir: Path) -> Path:
    """Download one CSV if it is not already cached.

    Args:
        split: ``"train"`` or ``"test"``.
        cache_dir: Directory to cache the raw CSVs in.

    Returns:
        Path of the local file.

    Raises:
        ValueError: If ``split`` is not a known split name.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; use 'train' or 'test'.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / SPLITS[split]
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    url = f"{BASE_URL}/{SPLITS[split]}"
    print(f"  downloading {split} from {url}")
    urllib.request.urlretrieve(url, destination)
    return destination


def _assemble_text(frame: pd.DataFrame) -> pd.Series:
    """Concatenate title and body exactly as the organisers' notebook does.

    Their ``process_dataset`` is ``(title or "") + " " + (body or "")`` -- a
    single space, no stripping, no cleaning. Reproducing the baseline means
    reproducing this too, including the trailing space when the body is empty.
    """
    title = frame.title.fillna("")
    body = frame.body.fillna("")
    return title.astype(str) + " " + body.astype(str)


def load_split(
    split: str,
    cache_dir: Path,
    seed: int = OFFICIAL_SEED,
    shuffle: bool = True,
    results_dir: Path | None = None,
    reason: str | None = None,
) -> Split:
    """Load one split, shuffled and with the text column assembled.

    The file is checksummed against the published dataset on every load, so a
    truncated download or a silently republished file cannot pass unnoticed.
    Reading the *test* split is recorded in ``results/test_access.log`` when
    ``results_dir`` is given -- the log is what lets a reader verify that the
    test split was read once per configuration rather than tuned against.

    Args:
        split: ``"train"`` or ``"test"``.
        cache_dir: Where the raw CSVs live.
        seed: Shuffle seed; the organisers use 42.
        shuffle: Set to False only when a deterministic unshuffled order is
            needed, e.g. for the leakage audit.
        results_dir: Where to append the test-access log.
        reason: Why the test split is being read.

    Returns:
        A :class:`Split`.
    """
    from . import audit

    path = download(split, cache_dir)
    audit.verify(path)
    if split == "test" and results_dir is not None:
        audit.log_test_access(results_dir, reason or "unspecified")
    frame = pd.read_csv(path)
    frame["text"] = _assemble_text(frame)
    if shuffle:
        # datasets.Dataset.shuffle and pandas.sample do not produce the same
        # permutation. That is acceptable: the organisers' seed fixes *their*
        # permutation, not a universal one, and SetFit's own pair sampling is
        # reseeded by TrainingArguments. What matters is that ours is fixed.
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return Split(name=split, frame=frame)


def load_dataset(
    cache_dir: Path,
    seed: int = OFFICIAL_SEED,
    results_dir: Path | None = None,
    reason: str | None = None,
) -> dict[str, Split]:
    """Load both splits.

    Args:
        cache_dir: Where the raw CSVs live.
        seed: Shuffle seed.
        results_dir: Where to append the test-access log.
        reason: Why the test split is being read.

    Returns:
        Mapping ``{"train": Split, "test": Split}``.
    """
    return {
        name: load_split(name, cache_dir, seed, results_dir=results_dir, reason=reason)
        for name in SPLITS
    }


def describe(split: Split) -> pd.DataFrame:
    """Return the project-by-class counts, for a sanity check at startup."""
    return (
        split.frame.groupby(["repo", "label"]).size().unstack(fill_value=0)
    )
