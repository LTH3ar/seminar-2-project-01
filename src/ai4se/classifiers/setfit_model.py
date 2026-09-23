"""Reproduction of the official SetFit baseline (track D).

The organisers publish a SetFit notebook as the reference to beat. Its
configuration is copied here exactly, because a reproduction that quietly
changes a hyper-parameter is not a reproduction:

============================  ====================================
base model                    ``sentence-transformers/all-mpnet-base-v2``
batch size                    ``(16, 2)`` -- body then head
epochs                        1
contrastive iterations        20
seed                          42
input text                    ``title + " " + body``
protocol                      one model per project
============================  ====================================

Two reference numbers exist for this configuration and they disagree: the
competition README quotes **0.8270** while the result file committed in the
same repository gives **0.8240**. A reproduction landing anywhere in that
neighbourhood has succeeded; chasing the third decimal place would be
chasing noise, which the power analysis in ``docs/benchmark-audit.md``
quantifies at 3.0 F1 points.

Requires the deep-learning extra::

    pip install -e ".[dl]"
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Classifier

#: The base encoder used by the official baseline. 110M parameters, so
#: training is slow without a GPU -- see :data:`FAST_PRESET`.
OFFICIAL_BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"

#: Exact hyper-parameters of the organisers' notebook.
OFFICIAL_PRESET: dict = {
    "base_model": OFFICIAL_BASE_MODEL,
    "batch_size": (16, 2),
    "num_epochs": 1,
    "num_iterations": 20,
}

#: A cheaper configuration for iterating on CPU. **Not** the official
#: baseline: a smaller encoder and a quarter of the contrastive pairs. Results
#: obtained with it must never be compared against the published 0.8240 as if
#: they were a reproduction.
FAST_PRESET: dict = {
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": (16, 2),
    "num_epochs": 1,
    "num_iterations": 5,
}


class SetFitClassifier(Classifier):
    """Few-shot classifier built on a sentence-transformer encoder.

    SetFit works in two stages: the encoder is first fine-tuned with a
    contrastive objective on pairs drawn from the training set, then a
    logistic-regression head is fitted on the resulting embeddings. With only
    300 labelled issues per project that is a better use of the data than
    fine-tuning a full classifier head, which is why it is a strong baseline
    here.

    Args:
        base_model: Sentence-transformer checkpoint to start from.
        batch_size: ``(body, head)`` batch sizes, as in the official notebook.
        num_epochs: Epochs over the generated contrastive pairs.
        num_iterations: Contrastive pairs generated per training example.
        seed: Seed for reproducibility.
        output_dir: Where the trainer writes its scratch files.
    """

    def __init__(  # noqa: D107 -- arguments documented on the class
        self,
        base_model: str = OFFICIAL_BASE_MODEL,
        batch_size: tuple[int, int] = (16, 2),
        num_epochs: int = 1,
        num_iterations: int = 20,
        seed: int = 42,
        output_dir: str = "results/setfit_runs",
    ) -> None:
        self.name = "setfit"
        self.base_model = base_model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.seed = seed
        self.output_dir = output_dir
        self._model = None

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> SetFitClassifier:
        """Fine-tune the encoder and fit the classification head."""
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        dataset = Dataset.from_dict({"text": list(texts), "label": list(labels)})
        self._model = SetFitModel.from_pretrained(self.base_model)
        arguments = TrainingArguments(
            output_dir=self.output_dir,
            save_strategy="no",
            report_to="none",  # the official notebook logs to W&B; not needed here
            seed=self.seed,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            num_iterations=self.num_iterations,
        )
        Trainer(model=self._model, args=arguments, train_dataset=dataset).train()
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Predict a label for each text.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("SetFitClassifier must be fitted before predicting.")
        predicted = self._model.predict(list(texts), batch_size=8)
        # SetFit returns a torch tensor of indices for integer labels and a
        # plain list for string ones; the dataset here uses strings.
        return np.asarray(list(predicted), dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return the head's class probabilities.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("SetFitClassifier must be fitted before predicting.")
        return np.asarray(self._model.predict_proba(list(texts), batch_size=8))

    @property
    def classes_(self) -> np.ndarray:
        """Label order of the probability columns.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("SetFitClassifier must be fitted before predicting.")
        return np.asarray(self._model.model_head.classes_, dtype=object)


def make_setfit(preset: str = "official", **overrides):
    """Return a factory producing fresh :class:`SetFitClassifier` instances.

    Args:
        preset: ``"official"`` for the published configuration, ``"fast"`` for
            the cheaper CPU variant.
        **overrides: Individual hyper-parameters to override.

    Returns:
        A zero-argument callable suitable for
        :func:`~ai4se.classifiers.base.train_per_repo`.

    Raises:
        ValueError: If ``preset`` is not recognised.
    """
    if preset == "official":
        settings = dict(OFFICIAL_PRESET)
    elif preset == "fast":
        settings = dict(FAST_PRESET)
    else:
        raise ValueError(f"Unknown preset {preset!r}; use 'official' or 'fast'.")
    settings.update(overrides)
    return lambda: SetFitClassifier(**settings)
