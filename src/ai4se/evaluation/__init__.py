"""Evaluation, significance testing and statistical power for track D.

The competition ranks submissions by the *cross-repository F1*: one classifier
is trained per project, scored on that project's test issues, and the five
scores are averaged. Every function here is written against that protocol.

Three concerns are separated on purpose:

``metrics``
    What the number is (:func:`cross_repo_f1`) and how uncertain it is
    (:func:`bootstrap_ci`).
``significance``
    Whether two classifiers actually differ (:func:`mcnemar_exact`), corrected
    for the fact that the comparison is repeated over five projects
    (:func:`holm`).
``power``
    How small a difference this test set can detect at all
    (:func:`minimum_detectable_difference`). Measured at 3.0 F1 points, which
    is larger than most gaps reported in the competition -- see
    ``docs/benchmark-audit.md``.

Nothing in this package depends on scikit-learn or scipy, so it imports with
the base requirements and can be used from any track.
"""

from .metrics import (
    bootstrap_ci,
    classification_rows,
    cross_repo_f1,
    macro_f1,
    micro_f1,
    per_repo_f1,
)
from .power import (
    is_conclusive,
    minimum_detectable_difference,
    observed_discordant_rate,
    power_at,
    power_curve,
)
from .significance import (
    McNemarResult,
    cliffs_delta,
    compare_per_repo,
    holm,
    mcnemar_exact,
)
from .splits import random_split_control, stratified_folds, time_aware_split

__all__ = [
    "McNemarResult",
    "bootstrap_ci",
    "classification_rows",
    "cliffs_delta",
    "compare_per_repo",
    "cross_repo_f1",
    "holm",
    "is_conclusive",
    "macro_f1",
    "mcnemar_exact",
    "micro_f1",
    "minimum_detectable_difference",
    "observed_discordant_rate",
    "per_repo_f1",
    "power_at",
    "power_curve",
    "random_split_control",
    "stratified_folds",
    "time_aware_split",
]
