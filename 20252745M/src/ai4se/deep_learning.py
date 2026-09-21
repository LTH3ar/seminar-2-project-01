"""Deep-learning classifiers for Track C.

The module provides three estimators behind the same ``fit``/``predict``
contract used by the classical and SetFit tracks:

* a feed-forward network over TF-IDF features;
* a TextCNN over learned word embeddings;
* a fine-tuned DistilBERT sequence classifier.

Optional dependencies are imported only when a model is fitted. This keeps the
data pipeline and classical experiments usable in lightweight environments.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .evaluation import EstimatorFactory, calculate_metrics
from .model import LABELS

AVAILABLE_DEEP_MODELS: tuple[str, ...] = ("ffnn", "cnn", "distilbert")

DEEP_MODEL_DISPLAY_NAMES = {
    "ffnn": "FFNN (TF-IDF)",
    "cnn": "TextCNN (learned embeddings)",
    "distilbert": "Fine-tuned DistilBERT",
}

_MODEL_ALIASES = {
    "ffnn": "ffnn",
    "feed-forward": "ffnn",
    "feed_forward": "ffnn",
    "mlp": "ffnn",
    "cnn": "cnn",
    "textcnn": "cnn",
    "text-cnn": "cnn",
    "distilbert": "distilbert",
    "distil-bert": "distilbert",
    "transformer": "distilbert",
}

_TOKEN = re.compile(r"[a-z0-9_]+")
_WHITESPACE = re.compile(r"\s+")
PAD_INDEX = 0
UNKNOWN_INDEX = 1


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Shared optimisation and early-stopping settings."""

    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    patience: int | None = 6
    minimum_improvement: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    random_state: int = 42
    device: str = "auto"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be positive or None")
        if self.minimum_improvement < 0:
            raise ValueError("minimum_improvement cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")


def _default_neural_training() -> TrainingConfig:
    return TrainingConfig()


def _default_transformer_training() -> TrainingConfig:
    return TrainingConfig(
        epochs=4,
        batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        patience=2,
    )


@dataclass(frozen=True, slots=True)
class FFNNConfig:
    """Architecture and feature settings for the TF-IDF feed-forward model."""

    training: TrainingConfig = field(default_factory=_default_neural_training)
    hidden_sizes: tuple[int, ...] = (256, 64)
    dropout: float = 0.5
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    min_document_frequency: int = 2

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.min_document_frequency < 1:
            raise ValueError("min_document_frequency must be positive")
        if self.ngram_range[0] < 1 or self.ngram_range[0] > self.ngram_range[1]:
            raise ValueError("ngram_range must contain increasing positive values")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable experiment metadata."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class CNNConfig:
    """Architecture and vocabulary settings for the TextCNN model."""

    training: TrainingConfig = field(default_factory=_default_neural_training)
    embedding_dimension: int = 128
    filter_sizes: tuple[int, ...] = (2, 3, 4, 5)
    filters_per_size: int = 64
    dropout: float = 0.5
    max_sequence_length: int = 400
    maximum_vocabulary_size: int = 30_000
    minimum_token_frequency: int = 2

    def __post_init__(self) -> None:
        if self.embedding_dimension < 1 or self.filters_per_size < 1:
            raise ValueError("embedding and filter dimensions must be positive")
        if not self.filter_sizes or min(self.filter_sizes) < 1:
            raise ValueError("filter_sizes must contain positive values")
        if max(self.filter_sizes) > self.max_sequence_length:
            raise ValueError("filter sizes cannot exceed max_sequence_length")
        if self.maximum_vocabulary_size < 2:
            raise ValueError("maximum_vocabulary_size must be at least 2")
        if self.minimum_token_frequency < 1:
            raise ValueError("minimum_token_frequency must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable experiment metadata."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DistilBERTConfig:
    """Fine-tuning settings for the selected transformer architecture."""

    training: TrainingConfig = field(default_factory=_default_transformer_training)
    model_id: str = "distilbert-base-uncased"
    max_sequence_length: int = 384
    prediction_batch_size: int = 16
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        if self.max_sequence_length < 8:
            raise ValueError("max_sequence_length must be at least 8")
        if self.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable experiment metadata."""

        return asdict(self)


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch diagnostics retained in the evaluation artifact."""

    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)
    validation_weighted_f1: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible history dictionary."""

        return {
            **asdict(self),
            "epochs_run": len(self.training_loss),
        }


def normalise_deep_model_name(model_name: str) -> str:
    """Resolve a model alias to a canonical Track C name."""

    key = model_name.strip().lower()
    try:
        return _MODEL_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(AVAILABLE_DEEP_MODELS)
        raise ValueError(
            f"Unknown deep-learning model {model_name!r}; choose one of: {choices}"
        ) from exc


def deep_model_display_name(model_name: str) -> str:
    """Return the report-friendly name of a Track C model."""

    return DEEP_MODEL_DISPLAY_NAMES[normalise_deep_model_name(model_name)]


def make_validation_split(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Create a stratified split while keeping duplicate texts together.

    The split happens before TF-IDF fitting or vocabulary construction, so the
    internal validation set cannot influence either representation.
    """

    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("at least one training example is required")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise ImportError(
            "Track C requires scikit-learn; install `pip install -e '.[dl]'`"
        ) from exc

    groups = [_normalised_text_group(text) for text in texts]
    groups_per_label = {
        label: len(
            {
                group
                for group, value in zip(groups, labels, strict=True)
                if value == label
            }
        )
        for label in set(labels)
    }
    desired_splits = max(2, round(1 / validation_fraction))
    n_splits = min(desired_splits, min(groups_per_label.values(), default=0))
    if n_splits < 2:
        raise ValueError(
            "Each observed class needs at least two distinct text groups for "
            "internal validation"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    candidates = list(splitter.split(texts, labels, groups))
    target_size = len(texts) * validation_fraction
    train_indices, validation_indices = min(
        candidates,
        key=lambda split: abs(len(split[1]) - target_size),
    )
    return train_indices.tolist(), validation_indices.tolist()


class _TorchTextClassifier:
    """Shared training loop for the FFNN and TextCNN estimators."""

    def __init__(self, training: TrainingConfig) -> None:
        self.training = training
        self.classes_ = list(LABELS)
        self.network_: Any | None = None
        self.history_ = TrainingHistory()
        self.device_: str | None = None
        self.training_size_: int = 0
        self.validation_size_: int = 0

    def _fit_features(self, texts: Sequence[str]) -> Any:
        raise NotImplementedError

    def _transform_features(self, texts: Sequence[str]) -> Any:
        raise NotImplementedError

    def _build_network(self, input_dimension: int) -> Any:
        raise NotImplementedError

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> _TorchTextClassifier:
        """Fit features on the training subset and optimise with early stopping."""

        torch, nn, data_loader, tensor_dataset = _require_torch()
        text_list, label_list = _validate_training_data(texts, labels)
        _set_seed(torch, self.training.random_state)
        train_indices, validation_indices = make_validation_split(
            text_list,
            label_list,
            validation_fraction=self.training.validation_fraction,
            random_state=self.training.random_state,
        )
        train_texts = [text_list[index] for index in train_indices]
        validation_texts = [text_list[index] for index in validation_indices]
        train_features = self._fit_features(train_texts)
        validation_features = self._transform_features(validation_texts)
        train_targets = torch.tensor(
            [LABELS.index(label_list[index]) for index in train_indices],
            dtype=torch.long,
        )
        validation_targets = torch.tensor(
            [LABELS.index(label_list[index]) for index in validation_indices],
            dtype=torch.long,
        )

        device = _resolve_device(torch, self.training.device)
        self.device_ = str(device)
        self.training_size_ = len(train_indices)
        self.validation_size_ = len(validation_indices)
        self.network_ = self._build_network(train_features.shape[-1]).to(device)
        optimiser = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        loader = data_loader(
            tensor_dataset(train_features, train_targets),
            batch_size=self.training.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.training.random_state),
        )
        validation_features = validation_features.to(device)
        validation_targets = validation_targets.to(device)

        self.history_ = TrainingHistory()
        best_loss = math.inf
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.training.epochs + 1):
            self.network_.train()
            cumulative_loss = 0.0
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                optimiser.zero_grad()
                loss = criterion(self.network_(batch_features), batch_targets)
                loss.backward()
                if self.training.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.network_.parameters(),
                        self.training.gradient_clip_norm,
                    )
                optimiser.step()
                cumulative_loss += loss.item() * len(batch_targets)

            self.network_.eval()
            with torch.no_grad():
                validation_logits = self.network_(validation_features)
                validation_loss = criterion(
                    validation_logits,
                    validation_targets,
                ).item()
            training_loss = cumulative_loss / len(train_indices)
            predictions = validation_logits.argmax(dim=1).cpu().tolist()
            references = validation_targets.cpu().tolist()
            metrics = calculate_metrics(
                [LABELS[index] for index in references],
                [LABELS[index] for index in predictions],
            )
            self._record_epoch(training_loss, validation_loss, metrics)
            if self.training.verbose:
                print(
                    f"epoch={epoch:02d} train_loss={training_loss:.4f} "
                    f"validation_loss={validation_loss:.4f} "
                    f"weighted_f1={metrics['weighted_average']['f1']:.4f}"
                )

            if validation_loss < best_loss - self.training.minimum_improvement:
                best_loss = validation_loss
                self.history_.best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.network_.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if (
                    self.training.patience is not None
                    and epochs_without_improvement >= self.training.patience
                ):
                    self.history_.stopped_early = True
                    break

        if best_state is not None:
            self.network_.load_state_dict(best_state)
        return self

    def _record_epoch(
        self,
        training_loss: float,
        validation_loss: float,
        metrics: Mapping[str, Any],
    ) -> None:
        self.history_.training_loss.append(float(training_loss))
        self.history_.validation_loss.append(float(validation_loss))
        self.history_.validation_accuracy.append(float(metrics["accuracy"]))
        self.history_.validation_weighted_f1.append(
            float(metrics["weighted_average"]["f1"])
        )

    def _logits(self, texts: Sequence[str]) -> Any:
        torch, _, data_loader, tensor_dataset = _require_torch()
        if self.network_ is None:
            raise RuntimeError("The classifier must be fitted before prediction")
        text_list = [str(text) for text in texts]
        if not text_list:
            return torch.empty((0, len(LABELS)), dtype=torch.float32)
        features = self._transform_features(text_list)
        loader = data_loader(
            tensor_dataset(features),
            batch_size=self.training.batch_size,
            shuffle=False,
        )
        device = _resolve_device(torch, self.training.device)
        self.network_.to(device)
        self.network_.eval()
        batches = []
        with torch.no_grad():
            for (batch_features,) in loader:
                batches.append(self.network_(batch_features.to(device)).cpu())
        return torch.cat(batches, dim=0)

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict one canonical label for each text."""

        indices = self._logits(texts).argmax(dim=1).tolist()
        return [LABELS[index] for index in indices]

    def predict_proba(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Return softmax probabilities aligned to :data:`ai4se.model.LABELS`."""

        torch, _, _, _ = _require_torch()
        probabilities = torch.softmax(self._logits(texts), dim=1)
        return [dict(zip(LABELS, row.tolist(), strict=True)) for row in probabilities]

    def training_summary(self) -> dict[str, Any]:
        """Return diagnostics that the shared evaluator stores with results."""

        parameter_count = (
            sum(parameter.numel() for parameter in self.network_.parameters())
            if self.network_ is not None
            else 0
        )
        return {
            "device": self.device_,
            "training_size": self.training_size_,
            "validation_size": self.validation_size_,
            "parameter_count": parameter_count,
            "history": self.history_.to_dict(),
        }


class FeedForwardTextClassifier(_TorchTextClassifier):
    """Multi-layer feed-forward network over training-only TF-IDF features."""

    def __init__(self, config: FFNNConfig | None = None) -> None:
        self.config = config or FFNNConfig()
        super().__init__(self.config.training)
        self.vectorizer_: Any | None = None

    def _fit_features(self, texts: Sequence[str]) -> Any:
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError(
                "The FFNN requires numpy and scikit-learn; install "
                "`pip install -e '.[dl]'`"
            ) from exc

        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_document_frequency,
            max_features=self.config.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        return _sparse_to_tensor(self.vectorizer_.fit_transform(texts))

    def _transform_features(self, texts: Sequence[str]) -> Any:
        if self.vectorizer_ is None:
            raise RuntimeError("The FFNN feature extractor has not been fitted")
        return _sparse_to_tensor(self.vectorizer_.transform(texts))

    def _build_network(self, input_dimension: int) -> Any:
        _, nn, _, _ = _require_torch()
        layers: list[Any] = []
        previous_size = input_dimension
        for hidden_size in self.config.hidden_sizes:
            layers.extend(
                (
                    nn.Linear(previous_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                )
            )
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, len(LABELS)))
        return nn.Sequential(*layers)


class CNNTextClassifier(_TorchTextClassifier):
    """TextCNN with learned embeddings and parallel n-gram convolutions."""

    def __init__(self, config: CNNConfig | None = None) -> None:
        self.config = config or CNNConfig()
        super().__init__(self.config.training)
        self.vocabulary_: dict[str, int] = {}

    def _fit_features(self, texts: Sequence[str]) -> Any:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(_tokenise(text))
        self.vocabulary_ = {"<pad>": PAD_INDEX, "<unk>": UNKNOWN_INDEX}
        maximum_terms = self.config.maximum_vocabulary_size - len(self.vocabulary_)
        for token, count in counts.most_common(maximum_terms):
            if count < self.config.minimum_token_frequency:
                break
            self.vocabulary_[token] = len(self.vocabulary_)
        return self._transform_features(texts)

    def _transform_features(self, texts: Sequence[str]) -> Any:
        torch, _, _, _ = _require_torch()
        if not self.vocabulary_:
            raise RuntimeError("The CNN vocabulary has not been fitted")
        rows = []
        for text in texts:
            token_ids = [
                self.vocabulary_.get(token, UNKNOWN_INDEX)
                for token in _tokenise(text)[: self.config.max_sequence_length]
            ]
            if not token_ids:
                token_ids = [UNKNOWN_INDEX]
            token_ids.extend(
                [PAD_INDEX] * (self.config.max_sequence_length - len(token_ids))
            )
            rows.append(token_ids)
        return torch.tensor(rows, dtype=torch.long)

    def _build_network(self, input_dimension: int) -> Any:
        del input_dimension
        torch, nn, _, _ = _require_torch()
        config = self.config
        vocabulary_size = len(self.vocabulary_)

        class TextCNNNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(
                    vocabulary_size,
                    config.embedding_dimension,
                    padding_idx=PAD_INDEX,
                )
                self.convolutions = nn.ModuleList(
                    nn.Conv1d(
                        config.embedding_dimension,
                        config.filters_per_size,
                        kernel_size=filter_size,
                    )
                    for filter_size in config.filter_sizes
                )
                self.dropout = nn.Dropout(config.dropout)
                self.output = nn.Linear(
                    config.filters_per_size * len(config.filter_sizes),
                    len(LABELS),
                )

            def forward(self, token_ids):
                embedded = self.embedding(token_ids).transpose(1, 2)
                pooled = [
                    convolution(embedded).relu().amax(dim=2)
                    for convolution in self.convolutions
                ]
                activations = nn.functional.relu(torch.cat(pooled, 1))
                return self.output(self.dropout(activations))

        return TextCNNNetwork()


class DistilBERTTextClassifier:
    """Full fine-tuning of DistilBERT with duplicate-safe early stopping."""

    def __init__(self, config: DistilBERTConfig | None = None) -> None:
        self.config = config or DistilBERTConfig()
        self.classes_ = list(LABELS)
        self.model_: Any | None = None
        self.tokenizer_: Any | None = None
        self.history_ = TrainingHistory()
        self.device_: str | None = None
        self.training_size_: int = 0
        self.validation_size_: int = 0

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> DistilBERTTextClassifier:
        """Download the checkpoint and fine-tune all encoder parameters."""

        torch, nn, data_loader, _ = _require_torch()
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                DataCollatorWithPadding,
                get_linear_schedule_with_warmup,
            )
        except ImportError as exc:
            raise ImportError(
                "DistilBERT requires transformers; install `pip install -e '.[dl]'`"
            ) from exc

        text_list, label_list = _validate_training_data(texts, labels)
        training = self.config.training
        _set_seed(torch, training.random_state)
        train_indices, validation_indices = make_validation_split(
            text_list,
            label_list,
            validation_fraction=training.validation_fraction,
            random_state=training.random_state,
        )
        self.training_size_ = len(train_indices)
        self.validation_size_ = len(validation_indices)
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.config.model_id)
        label_to_id = {label: index for index, label in enumerate(LABELS)}
        id_to_label = {index: label for label, index in label_to_id.items()}
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            num_labels=len(LABELS),
            label2id=label_to_id,
            id2label=id_to_label,
        )

        train_dataset = _TransformerDataset(
            self.tokenizer_,
            [text_list[index] for index in train_indices],
            [label_to_id[label_list[index]] for index in train_indices],
            max_length=self.config.max_sequence_length,
        )
        validation_dataset = _TransformerDataset(
            self.tokenizer_,
            [text_list[index] for index in validation_indices],
            [label_to_id[label_list[index]] for index in validation_indices],
            max_length=self.config.max_sequence_length,
        )
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer_)
        train_loader = data_loader(
            train_dataset,
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=torch.Generator().manual_seed(training.random_state),
        )
        validation_loader = data_loader(
            validation_dataset,
            batch_size=self.config.prediction_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        device = _resolve_device(torch, training.device)
        self.device_ = str(device)
        self.model_.to(device)
        optimiser = torch.optim.AdamW(
            self.model_.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )
        updates_per_epoch = math.ceil(
            len(train_loader) / self.config.gradient_accumulation_steps
        )
        total_updates = max(1, updates_per_epoch * training.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimiser,
            num_warmup_steps=round(total_updates * self.config.warmup_ratio),
            num_training_steps=total_updates,
        )

        self.history_ = TrainingHistory()
        best_loss = math.inf
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        accumulation_steps = self.config.gradient_accumulation_steps

        for epoch in range(1, training.epochs + 1):
            self.model_.train()
            optimiser.zero_grad()
            cumulative_loss = 0.0
            observed = 0
            for batch_number, batch in enumerate(train_loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = self.model_(**batch)
                batch_size = len(batch["labels"])
                cumulative_loss += outputs.loss.item() * batch_size
                observed += batch_size
                (outputs.loss / accumulation_steps).backward()
                should_update = (
                    batch_number % accumulation_steps == 0
                    or batch_number == len(train_loader)
                )
                if should_update:
                    if training.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model_.parameters(),
                            training.gradient_clip_norm,
                        )
                    optimiser.step()
                    scheduler.step()
                    optimiser.zero_grad()

            validation_loss, metrics = self._evaluate_loader(
                validation_loader,
                device,
            )
            training_loss = cumulative_loss / observed
            self._record_epoch(training_loss, validation_loss, metrics)
            if training.verbose:
                print(
                    f"epoch={epoch:02d} train_loss={training_loss:.4f} "
                    f"validation_loss={validation_loss:.4f} "
                    f"weighted_f1={metrics['weighted_average']['f1']:.4f}"
                )

            if validation_loss < best_loss - training.minimum_improvement:
                best_loss = validation_loss
                self.history_.best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model_.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if (
                    training.patience is not None
                    and epochs_without_improvement >= training.patience
                ):
                    self.history_.stopped_early = True
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def _record_epoch(
        self,
        training_loss: float,
        validation_loss: float,
        metrics: Mapping[str, Any],
    ) -> None:
        self.history_.training_loss.append(float(training_loss))
        self.history_.validation_loss.append(float(validation_loss))
        self.history_.validation_accuracy.append(float(metrics["accuracy"]))
        self.history_.validation_weighted_f1.append(
            float(metrics["weighted_average"]["f1"])
        )

    def _evaluate_loader(
        self,
        loader: Any,
        device: Any,
    ) -> tuple[float, dict[str, Any]]:
        torch, _, _, _ = _require_torch()
        self.model_.eval()
        cumulative_loss = 0.0
        observed = 0
        references: list[str] = []
        predictions: list[str] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = self.model_(**batch)
                batch_references = batch["labels"].cpu().tolist()
                batch_predictions = outputs.logits.argmax(dim=1).cpu().tolist()
                cumulative_loss += outputs.loss.item() * len(batch_references)
                observed += len(batch_references)
                references.extend(LABELS[index] for index in batch_references)
                predictions.extend(LABELS[index] for index in batch_predictions)
        return cumulative_loss / observed, calculate_metrics(references, predictions)

    def _logits(self, texts: Sequence[str]) -> Any:
        torch, _, _, _ = _require_torch()
        if self.model_ is None or self.tokenizer_ is None:
            raise RuntimeError("The DistilBERT classifier must be fitted first")
        text_list = [str(text) for text in texts]
        if not text_list:
            return torch.empty((0, len(LABELS)), dtype=torch.float32)
        device = _resolve_device(torch, self.config.training.device)
        self.model_.to(device)
        self.model_.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(text_list), self.config.prediction_batch_size):
                encoded = self.tokenizer_(
                    text_list[start : start + self.config.prediction_batch_size],
                    truncation=True,
                    max_length=self.config.max_sequence_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                batches.append(self.model_(**encoded).logits.cpu())
        self.model_.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return torch.cat(batches, dim=0)

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict one canonical label for each text."""

        indices = self._logits(texts).argmax(dim=1).tolist()
        return [LABELS[index] for index in indices]

    def predict_proba(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Return softmax probabilities aligned to the canonical labels."""

        torch, _, _, _ = _require_torch()
        probabilities = torch.softmax(self._logits(texts), dim=1)
        return [dict(zip(LABELS, row.tolist(), strict=True)) for row in probabilities]

    def training_summary(self) -> dict[str, Any]:
        """Return fine-tuning diagnostics for the result artifact."""

        parameter_count = (
            sum(
                parameter.numel()
                for parameter in self.model_.parameters()
                if parameter.requires_grad
            )
            if self.model_ is not None
            else 0
        )
        return {
            "device": self.device_,
            "training_size": self.training_size_,
            "validation_size": self.validation_size_,
            "parameter_count": parameter_count,
            "history": self.history_.to_dict(),
        }


class _TransformerDataset:
    """Pre-tokenised examples consumed by a HuggingFace data collator."""

    def __init__(
        self,
        tokenizer: Any,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            **{key: values[index] for key, values in self.encodings.items()},
            "label": self.labels[index],
        }


DeepConfig = FFNNConfig | CNNConfig | DistilBERTConfig


def build_deep_classifier(
    model_name: str,
    *,
    config: DeepConfig | None = None,
):
    """Build one unfitted Track C estimator."""

    canonical_name = normalise_deep_model_name(model_name)
    expected_types = {
        "ffnn": FFNNConfig,
        "cnn": CNNConfig,
        "distilbert": DistilBERTConfig,
    }
    expected_type = expected_types[canonical_name]
    if config is not None and not isinstance(config, expected_type):
        raise TypeError(
            f"{canonical_name} expects {expected_type.__name__}, "
            f"not {type(config).__name__}"
        )
    if canonical_name == "ffnn":
        return FeedForwardTextClassifier(config)
    if canonical_name == "cnn":
        return CNNTextClassifier(config)
    return DistilBERTTextClassifier(config)


def make_deep_estimator_factory(
    model_name: str,
    *,
    config: DeepConfig | None = None,
    random_state: int = 42,
) -> EstimatorFactory:
    """Return fresh, independently seeded models for the shared evaluator."""

    canonical_name = normalise_deep_model_name(model_name)
    base_config = config or _default_config(canonical_name)

    def factory(repository: str, fold: int):
        del repository
        run_training = replace(
            base_config.training,
            random_state=random_state + fold,
        )
        run_config = replace(base_config, training=run_training)
        return build_deep_classifier(canonical_name, config=run_config)

    return factory


def plot_learning_history(
    history: Mapping[str, Any],
    *,
    title: str = "Learning curves",
):
    """Plot training/validation loss and validation weighted F1."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for learning-curve plots") from exc

    training_loss = history["training_loss"]
    validation_loss = history["validation_loss"]
    validation_f1 = history["validation_weighted_f1"]
    epochs = range(1, len(training_loss) + 1)
    figure, (loss_axis, score_axis) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    loss_axis.plot(epochs, training_loss, label="training")
    loss_axis.plot(epochs, validation_loss, label="validation")
    best_epoch = int(history.get("best_epoch", 0))
    if best_epoch:
        loss_axis.axvline(best_epoch, color="#B04A3A", linestyle=":")
    loss_axis.set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    loss_axis.legend()
    score_axis.plot(epochs, validation_f1, color="#176B87")
    score_axis.set(
        xlabel="Epoch",
        ylabel="Weighted F1",
        title="Internal validation",
        ylim=(0, 1),
    )
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def _default_config(model_name: str) -> DeepConfig:
    if model_name == "ffnn":
        return FFNNConfig()
    if model_name == "cnn":
        return CNNConfig()
    return DistilBERTConfig()


def _validate_training_data(
    texts: Sequence[str],
    labels: Sequence[str],
) -> tuple[list[str], list[str]]:
    text_list = [str(text) for text in texts]
    label_list = [str(label) for label in labels]
    if len(text_list) != len(label_list):
        raise ValueError("texts and labels must have the same length")
    if not text_list:
        raise ValueError("at least one training example is required")
    unknown = sorted(set(label_list) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return text_list, label_list


def _normalised_text_group(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text).casefold()).strip()


def _tokenise(text: str) -> list[str]:
    return _TOKEN.findall(str(text).casefold())


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "Track C requires PyTorch; install `pip install -e '.[dl]'`"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _set_seed(torch: Any, random_state: int) -> None:
    random.seed(random_state)
    try:
        import numpy as np

        np.random.seed(random_state)
    except ImportError:
        pass
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def _sparse_to_tensor(matrix: Any) -> Any:
    torch, _, _, _ = _require_torch()
    return torch.from_numpy(matrix.toarray())
