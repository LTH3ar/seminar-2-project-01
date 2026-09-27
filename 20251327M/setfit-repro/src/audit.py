"""Checks that make the numbers auditable rather than merely produced.

Three things a reader of the report is entitled to, and which nothing else in
the pipeline provides:

**The data is the data.** The CSVs are downloaded from GitHub at run time. A
checksum turns "we used the official dataset" from a claim into a fact, and
catches the silent failure where a network hiccup leaves a truncated file.

**The test split was read when we say it was.** Every read is appended to
``results/test_access.log`` with a reason. If a configuration shows six reads,
the reader sees six reads. This is deliberately harder to circumvent than a
promise in a methods section.

**The benchmark leaks a little.** 60 of the 1,500 test issues (4.0%) have a
training issue at cosine similarity 0.90 or above, and 3 are exact duplicates.
That is small, but it is the same order of magnitude as the differences people
argue over on this task, so it belongs in Threats to Validity rather than in a
footnote nobody computes.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

#: SHA-256 of the official CSVs as published. Verified 2026-09-27 against
#: https://github.com/nlbse2024/issue-report-classification/tree/main/data
EXPECTED_SHA256 = {
    "issues_train.csv": (
        "18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5"
    ),
    "issues_test.csv": (
        "4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85"
    ),
}


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, strict: bool = True) -> str:
    """Check a downloaded CSV against its published checksum.

    Args:
        path: The local file.
        strict: Raise on mismatch rather than warn. Leave this on unless the
            organisers have republished the data, in which case update
            :data:`EXPECTED_SHA256` and say so in the report.

    Returns:
        The computed digest.

    Raises:
        ValueError: If the digest differs and ``strict`` is set.
    """
    actual = sha256(path)
    expected = EXPECTED_SHA256.get(path.name)
    if expected is None:
        print(f"  ! no published checksum on record for {path.name}")
        return actual
    if actual != expected:
        message = (
            f"{path.name} does not match the published dataset.\n"
            f"  expected {expected}\n  got      {actual}\n"
            "Delete the file and let it download again. If the organisers have "
            "republished the data, update EXPECTED_SHA256 and record the change."
        )
        if strict:
            raise ValueError(message)
        print(f"  ! {message}")
    return actual


def log_test_access(results_dir: Path, reason: str) -> None:
    """Append one line to the test-set access log.

    Args:
        results_dir: Directory holding ``test_access.log``.
        reason: Why the test split is being read, e.g. a configuration name.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (results_dir / "test_access.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{reason}\n")


def test_access_count(results_dir: Path) -> dict[str, int]:
    """Return how many times the test split has been read, per reason."""
    path = results_dir / "test_access.log"
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            reason = line.split("\t", 1)[1]
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _normalise(text: str) -> str:
    """Case- and punctuation-insensitive key for duplicate detection."""
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def leakage(
    train: pd.DataFrame, test: pd.DataFrame, thresholds=(0.99, 0.95, 0.90, 0.80)
) -> dict:
    """Measure how much of the test split is echoed in the training split.

    Exact duplicates are found by hashing a normalised form. Near-duplicates
    are the maximum TF-IDF cosine similarity of each test issue against every
    training issue -- fitted on the two splits together, which is legitimate
    here because this is an audit of the data, not a model being trained.

    Args:
        train: Training frame with a ``text`` column.
        test: Test frame with ``text`` and ``repo`` columns.
        thresholds: Cosine cut-offs to report.

    Returns:
        A record of exact and near-duplicate counts, overall and per project.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    train_keys = set(train.text.map(_normalise).map(hash))
    exact = int(test.text.map(_normalise).map(hash).isin(train_keys).sum())
    titles = set(train.title.map(_normalise)) if "title" in train else set()
    same_title = (
        int(test.title.map(_normalise).isin(titles).sum()) if titles else None
    )

    vectoriser = TfidfVectorizer(sublinear_tf=True, min_df=2)
    vectoriser.fit(pd.concat([train.text, test.text]))
    train_matrix = vectoriser.transform(train.text)
    best = np.zeros(len(test))
    test_matrix = vectoriser.transform(test.text)
    for start in range(0, len(test), 300):
        chunk = test_matrix[start : start + 300]
        best[start : start + 300] = cosine_similarity(chunk, train_matrix).max(axis=1)

    near = {
        f"cosine_ge_{t:.2f}": {
            "n": int((best >= t).sum()),
            "share": float((best >= t).mean()),
        }
        for t in thresholds
    }
    per_repo = {
        repo: int((best[(test.repo == repo).to_numpy()] >= 0.90).sum())
        for repo in sorted(test.repo.unique())
    }
    return {
        "exact_duplicates_train_test": exact,
        "identical_titles": same_title,
        "duplicates_within_train": int(train.text.map(_normalise).duplicated().sum()),
        "near_duplicates": near,
        "near_duplicates_per_repo_at_090": per_repo,
        "note": (
            "Small, but the same order of magnitude as the score differences "
            "reported on this benchmark. Report it rather than discover it."
        ),
    }


def write(results_dir: Path, name: str, record: dict) -> Path:
    """Write an audit record as JSON and return its path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
