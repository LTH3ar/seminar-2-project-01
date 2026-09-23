"""Evidence that the two persistence layers are interchangeable.

The project specification requires that switching between the in-memory and
the file-based persistence layer involves few changes to the application and
leaves the representation and the interaction with the business logic
unchanged. This test file is the proof, and the table it prints is meant to be
pasted into the report.

Run with::

    python -m pytest tests/ -v
    python tests/test_persistence_equivalence.py    # standalone, prints a table
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4se.eda import label_distribution, length_statistics, overview
from ai4se.loader import load_split
from ai4se.model import LABELS, REPOSITORIES, IssueReport
from ai4se.preprocessing import make_cleaner
from ai4se.repository import IssueRepository, make_repository


def _business_logic(repository: IssueRepository) -> dict:
    """A stand-in for the real pipeline.

    Note that it only ever calls methods declared on ``IssueRepository``; it
    has no way of knowing which concrete implementation it was handed.
    """
    repository.apply(make_cleaner(level="full", max_words=400))
    texts, labels = repository.texts_and_labels()
    return {
        "n": len(repository),
        "projects": tuple(repository.repos()),
        "labels": tuple(repository.labels()),
        "distribution": tuple(sorted(repository.label_distribution().items())),
        "per_project": tuple(
            (r, len(repository.by_repo(r))) for r in repository.repos()
        ),
        "checksum": sum(len(t) for t in texts),
        "first_label": labels[0],
        "first_text": texts[0][:120],
    }


def test_dataset_shape():
    """The official training split is balanced: 1,500 issues, 100 per cell."""
    train = load_split("train", kind="memory")
    assert len(train) == 1500
    assert set(train.labels()) == set(LABELS)
    assert set(train.repos()) == set(REPOSITORIES)
    for repo in REPOSITORIES:
        assert train.label_distribution(repo) == dict.fromkeys(LABELS, 100)


def test_memory_and_file_produce_identical_results():
    """The same computation over both layers must return the same answer."""
    from_memory = _business_logic(load_split("train", kind="memory"))
    from_file = _business_logic(load_split("train", kind="file"))
    assert from_memory == from_file


def test_round_trip_through_csv_and_json():
    """Saving and reloading must not alter a single entity."""
    original = load_split("train", kind="memory")
    original.apply(make_cleaner(level="full", max_words=400))

    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("csv", "json"):
            path = Path(tmp) / f"issues.{suffix}"
            written = make_repository("file", issues=original.all(), path=path)
            written.save()

            reloaded = make_repository("file", path=path)
            assert len(reloaded) == len(original)
            assert [i.to_dict() for i in reloaded] == [i.to_dict() for i in original]


def test_repository_switch_is_one_argument():
    """Both layers implement the full abstract interface and nothing less.

    Extra attributes on the file implementation (``path``, ``fmt``) are
    implementation detail: the business logic never touches them, because it
    is typed against ``IssueRepository``.
    """
    interface = {m for m in dir(IssueRepository) if not m.startswith("_")}
    memory = make_repository("memory", issues=[])
    file_backed = make_repository(
        "file", issues=[], path=Path(tempfile.mkdtemp()) / "empty.csv"
    )

    for implementation in (memory, file_backed):
        assert isinstance(implementation, IssueRepository)
        missing = {
            method
            for method in interface
            if not callable(getattr(implementation, method, None))
        }
        assert not missing, f"{type(implementation).__name__} is missing {missing}"

    # Both are usable through the interface without any type check.
    for implementation in (memory, file_backed):
        implementation.add(IssueReport("facebook/react", "2023-01-01", "bug", "t", "b"))
        assert len(implementation) == 1
        assert implementation.label_distribution() == {"bug": 1}


def test_entity_validation():
    """Malformed records are detectable through the domain model."""
    good = IssueReport("facebook/react", "2023-01-01", "bug", "crash", "it crashes")
    empty = IssueReport("facebook/react", "2023-01-01", "bug", "", "")
    wrong = IssueReport("facebook/react", "2023-01-01", "duplicate", "t", "b")
    assert good.is_valid()
    assert not empty.is_valid()
    assert not wrong.is_valid()


if __name__ == "__main__":
    print("=" * 72)
    print("Persistence layer equivalence check")
    print("=" * 72)

    memory_repo = load_split("train", kind="memory")
    file_repo = load_split("train", kind="file")
    print(f"\nin-memory layer : {memory_repo!r}")
    print(f"file layer      : {file_repo!r}")

    a = _business_logic(load_split("train", kind="memory"))
    b = _business_logic(load_split("train", kind="file"))

    print("\n{:<18}{:<24}{:<24}{}".format("property", "in-memory", "file", "equal"))
    print("-" * 72)
    for key in ("n", "distribution", "checksum", "first_label"):
        va, vb = str(a[key])[:22], str(b[key])[:22]
        print(f"{key:<18}{va:<24}{vb:<24}{'yes' if a[key] == b[key] else 'NO'}")
    print("-" * 72)
    print(f"\nall properties identical: {a == b}")

    print("\nOverview (computed through the abstract interface):")
    print(overview(memory_repo).to_string(index=False))
    print("\nLabel distribution:")
    print(label_distribution(memory_repo).to_string())
    print("\nLength statistics (words):")
    print(length_statistics(memory_repo).to_string())
