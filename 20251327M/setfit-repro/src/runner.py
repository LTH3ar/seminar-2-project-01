"""Training and prediction for SetFit, in the competition's protocol.

One model per project, five per submission. The official configuration is
recorded in :data:`OFFICIAL` and is not to be edited -- variants belong in
phase 2, as separate configurations with their own names, so that every
reported number can be traced back to the settings that produced it.

One deviation from the organisers' notebook is deliberate and is documented
rather than hidden: ``max_seq_length`` is set explicitly to 384. That is
``all-mpnet-base-v2``'s own configured limit, which older sentence-transformers
releases applied automatically and newer ones do not. Leaving it unset is not
more faithful, it is differently unfaithful -- and it puts a 21,595-word issue
through the encoder, which exhausts any GPU. Setting it restores the behaviour
the published run had.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .data import REPOSITORIES, Split

# Keep the libraries quiet and offline-ish; the organisers' notebook logs to
# Weights & Biases, which we neither need nor want in a batch run.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass(frozen=True)
class Config:
    """A SetFit configuration.

    Attributes:
        name: Short identifier used in results files and tables.
        encoder: Sentence-transformer checkpoint.
        body_batch: Batch size for contrastive fine-tuning of the encoder.
        head_batch: Batch size for the logistic-regression head.
        num_epochs: Epochs over the generated pairs.
        num_iterations: Pairs generated per training example.
        max_seq_length: Tokens kept per document. None leaves the library's
            behaviour alone, which is not recommended -- see the module
            docstring.
        seed: Training seed.
        scope: ``"per_repo"`` trains each model on its own project's 300
            issues; ``"global"`` fine-tunes one encoder on all 1,500 and fits a
            separate head per project. The second is only permitted if the
            organisers confirm it -- see ``README.md``.
        notes: Free text carried into the results file.
    """

    name: str
    encoder: str = "sentence-transformers/all-mpnet-base-v2"
    body_batch: int = 16
    head_batch: int = 2
    num_epochs: int = 1
    num_iterations: int = 20
    max_seq_length: int | None = 384
    seed: int = 42
    scope: str = "per_repo"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return the configuration as a plain dictionary for the results file."""
        return asdict(self)


#: The published baseline, reproduced exactly. Do not edit.
OFFICIAL = Config(
    name="setfit_official_reproduction",
    notes=(
        "Organisers' configuration: all-mpnet-base-v2, batch (16, 2), 1 epoch, "
        "20 contrastive iterations, seed 42, one model per project, text = "
        "title + ' ' + body. max_seq_length pinned to the checkpoint's own 384."
    ),
)


def gpu_report() -> str:
    """Return a one-line description of the available GPU."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement
        return "torch not installed"
    if not torch.cuda.is_available():
        return "NO GPU -- training will take hours instead of minutes"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name}, {total:.1f} GiB"


def versions() -> dict[str, str]:
    """Record the library versions a run was produced with.

    The organisers ran an earlier generation of these packages and it is not
    installable alongside a current torch. Recording what we actually used is
    what lets a reader judge how much of any gap is method and how much is
    library drift -- see ``requirements.txt``.
    """
    import platform
    import sys

    found = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("torch", "transformers", "sentence_transformers", "setfit",
                   "datasets", "sklearn", "numpy"):
        try:
            found[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - a missing version is not fatal
            found[module] = "unknown"
    return found


def _build_model(config: Config):
    """Instantiate a SetFit model at the requested configuration."""
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(config.encoder)
    if config.max_seq_length:
        model.model_body.max_seq_length = config.max_seq_length
    return model


def _train(model, texts, labels, config: Config):
    """Run SetFit's two-stage training on one set of labelled texts."""
    from datasets import Dataset
    from setfit import Trainer, TrainingArguments

    arguments = TrainingArguments(
        output_dir=f"/tmp/setfit/{config.name}",
        save_strategy="no",
        report_to="none",
        seed=config.seed,
        batch_size=(config.body_batch, config.head_batch),
        num_epochs=config.num_epochs,
        num_iterations=config.num_iterations,
    )
    dataset = Dataset.from_dict({"text": list(texts), "label": list(labels)})
    Trainer(model=model, args=arguments, train_dataset=dataset).train()
    return model


def run(config: Config, train: Split, test: Split, verbose: bool = True) -> dict:
    """Train under ``config`` and predict the whole test split.

    Args:
        config: The configuration to run.
        train: Training split.
        test: Test split.
        verbose: Print per-project progress.

    Returns:
        A record with ``predictions`` (aligned with ``test.frame``),
        ``probabilities`` where available, the wall-clock time, and the
        configuration itself.

    Raises:
        ValueError: If ``config.scope`` is not a supported value.
    """
    if config.scope not in {"per_repo", "global"}:
        raise ValueError(f"Unknown scope {config.scope!r}.")

    started = time.time()
    frame = test.frame
    predictions = np.empty(len(frame), dtype=object)
    probabilities = np.full((len(frame), 3), np.nan)
    classes: list[str] = []

    if config.scope == "global":
        # One encoder adapted on all 1,500 training issues, then a separate
        # head per project. Still five classifiers; still only the provided
        # training data. See README.md for why this needs confirming.
        if verbose:
            print(f"  fine-tuning one encoder on all {len(train)} issues")
        shared = _train(_build_model(config), *train.texts_and_labels(), config)

    for repo in REPOSITORIES:
        rows = np.flatnonzero((frame.repo == repo).to_numpy())
        if rows.size == 0:
            continue
        train_texts, train_labels = train.texts_and_labels(repo)
        if verbose:
            print(
                f"  {repo:<24} train={len(train_texts):<5} test={rows.size:<5}",
                flush=True,
            )

        if config.scope == "per_repo":
            model = _train(_build_model(config), train_texts, train_labels, config)
        else:
            # Re-fit only the head on this project's issues; the encoder keeps
            # what it learned from all five.
            model = shared
            model.fit(list(train_texts), list(train_labels))

        texts = frame.text.to_numpy()[rows].tolist()
        predictions[rows] = np.asarray(list(model.predict(texts)), dtype=object)
        try:
            proba = np.asarray(model.predict_proba(texts))
            probabilities[rows] = proba
            classes = [str(c) for c in model.model_head.classes_]
        except (AttributeError, NotImplementedError, ValueError):
            pass

    if any(p is None for p in predictions):
        missing = sum(1 for p in predictions if p is None)
        raise RuntimeError(f"{missing} test issues were left unpredicted.")

    return {
        "config": config.as_dict(),
        "predictions": [str(p) for p in predictions],
        "probabilities": None if not classes else probabilities.tolist(),
        "classes": classes,
        "minutes": (time.time() - started) / 60,
        "gpu": gpu_report(),
        "versions": versions(),
    }
