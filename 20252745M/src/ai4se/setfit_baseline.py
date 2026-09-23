"""SetFit adapter matching the supplied NLBSE'24 baseline notebook."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evaluation import EstimatorFactory
from .model import LABELS


@dataclass(frozen=True, slots=True)
class SetFitConfig:
    """Hyperparameters used by the supplied SetFit reproduction."""

    model_id: str = "sentence-transformers/all-mpnet-base-v2"
    random_state: int = 42
    body_batch_size: int = 16
    classifier_batch_size: int = 2
    prediction_batch_size: int = 8
    num_epochs: int = 1
    num_iterations: int = 20
    max_sequence_length: int = 384

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable experiment metadata."""

        return asdict(self)


class SetFitTextClassifier:
    """Expose SetFit through the shared text-classifier interface."""

    def __init__(self, config: SetFitConfig, output_directory: str | Path) -> None:
        """Store configuration without importing optional ML dependencies."""

        self.config = config
        self.output_directory = Path(output_directory)
        self.model: Any | None = None

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> SetFitTextClassifier:
        """Fine-tune SetFit using the original notebook hyperparameters."""

        try:
            from datasets import Dataset
            from setfit import SetFitModel, Trainer, TrainingArguments
        except ImportError as exc:
            raise ImportError(
                "SetFit reproduction requires `pip install -e '.[setfit]'`"
            ) from exc

        self.output_directory.mkdir(parents=True, exist_ok=True)
        train_dataset = Dataset.from_dict({"text": list(texts), "label": list(labels)})
        self.model = SetFitModel.from_pretrained(
            self.config.model_id,
            labels=list(LABELS),
        )
        model_body = getattr(self.model, "model_body", None)
        if model_body is None:
            raise RuntimeError(
                "The installed SetFit version does not expose model_body; "
                "cannot enforce the encoder sequence limit safely"
            )
        model_body.max_seq_length = self.config.max_sequence_length
        arguments = TrainingArguments(
            output_dir=str(self.output_directory),
            save_strategy="no",
            report_to="none",
            seed=self.config.random_state,
            batch_size=(
                self.config.body_batch_size,
                self.config.classifier_batch_size,
            ),
            num_epochs=self.config.num_epochs,
            num_iterations=self.config.num_iterations,
        )
        trainer = Trainer(
            model=self.model,
            args=arguments,
            train_dataset=train_dataset,
        )
        trainer.train()
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict labels with the notebook's batch size."""

        if self.model is None:
            raise RuntimeError("The SetFit classifier must be fitted first")
        predictions = self.model.predict(
            list(texts),
            batch_size=self.config.prediction_batch_size,
            show_progress_bar=True,
        )
        return [str(value) for value in predictions]


def make_setfit_factory(
    config: SetFitConfig,
    output_directory: str | Path,
) -> EstimatorFactory:
    """Create isolated SetFit models for repositories and folds."""

    output_directory = Path(output_directory)

    def factory(repository: str, fold: int) -> SetFitTextClassifier:
        repository_slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            repository.casefold(),
        ).strip("-")
        run_name = "official" if fold == 0 else f"fold-{fold}"
        return SetFitTextClassifier(
            config,
            output_directory / repository_slug / run_name,
        )

    return factory
