"""Track C -- neural classifiers.

Two architectures from the course, wrapped in the same ``fit``/``predict``
interface as everything else so they run through the unchanged evaluation
harness:

**Feed-forward network** (course lecture 03). Input is the TF-IDF vector of a
document; one or two hidden layers with ReLU; softmax output over the three
classes. This is the simplest neural model that can be compared directly with
the Track B pipelines, because it consumes exactly the same features.

**1-D convolutional network** (course lecture 04). Input is a sequence of
learned word embeddings; parallel convolution filters of several widths act as
learned n-gram detectors; global max-pooling keeps the strongest activation of
each filter; a dense layer classifies. Convolution over text is the same
operation as over images, with the kernel sliding in one dimension only.

Both track training and validation loss at every epoch, which the course
requires: validation loss must be reported and compared against training loss
per epoch, and the resulting validation curves are the standard overfitting
diagnostic. :attr:`NeuralClassifier.history_` holds those curves, and
:func:`plot_learning_curves` renders them for the report.

Early stopping is implemented for the same reason -- monitoring validation
performance and stopping when it degrades is the first remedy the course lists
for overfitting.

With 300 training issues per project these models are small by necessity. That
is itself a finding worth reporting rather than a limitation to apologise for.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import RANDOM_SEED
from .model import LABELS

TOKEN = re.compile(r"[a-z0-9]+")

#: Reserved vocabulary indices.
PAD, UNK = 0, 1


def set_seed(seed: int = RANDOM_SEED, deterministic: bool = True) -> None:
    """Seed every source of randomness the training loop can reach.

    The loop itself draws only from torch: the global RNG for weight
    initialisation, and two explicitly seeded ``torch.Generator`` objects for
    the validation split and the batch shuffle. Python's ``random`` and NumPy
    are seeded anyway so that adding a component which uses them cannot
    silently make runs irreproducible.

    ``deterministic=True`` disables the cuDNN autotuner and requests
    deterministic kernels. Measured effect on this project: none -- the CNN
    returned 0.7438 both with and without it, so training here was already
    deterministic on a fixed device. It is kept as cheap insurance against a
    future change that introduces a nondeterministic kernel, not because it
    fixed an observed problem.

    What it does **not** fix is the gap between devices. The CNN scores 0.7578
    on a CPU and 0.7438 on a GPU from identical code and seed. That is the two
    backends computing the same operations with different kernels and
    accumulation orders, not nondeterminism: each device reproduces its own
    number exactly. No seeding or determinism flag reconciles them, so a
    reported neural result is only meaningful alongside the device that
    produced it.

    Args:
        seed: Seed for Python, NumPy and torch.
        deterministic: Request deterministic cuDNN kernels.
    """
    import random as _random

    _random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def default_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingHistory:
    """Per-epoch losses and validation scores.

    The course requires training and validation loss to be reported and
    compared at each epoch; this is the object that holds them.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    def as_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "epochs_run": len(self.train_loss),
        }

    def to_frame(self):
        """Return the curves as a pandas DataFrame, indexed by epoch."""
        import pandas as pd

        return pd.DataFrame(
            {
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "val_accuracy": self.val_accuracy,
            },
            index=pd.RangeIndex(1, len(self.train_loss) + 1, name="epoch"),
        )


class NeuralClassifier:
    """Base class holding everything the two architectures share.

    Subclasses implement :meth:`_build_features` and :meth:`_build_network`;
    training, early stopping, prediction and history tracking live here.

    Args:
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        validation_fraction: Share of the training data held out to monitor
            overfitting. The course suggests roughly 20%.
        patience: Stop when validation loss has not improved for this many
            epochs. ``None`` disables early stopping.
        seed: Seed for weight initialisation and shuffling.
        device: Torch device; inferred when omitted.
        verbose: Print the loss at every epoch.
    """

    def __init__(
        self,
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        patience: int | None = 8,
        seed: int = RANDOM_SEED,
        device: torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        """Store the hyperparameters; nothing is built until :meth:`fit`."""
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.seed = seed
        self.device = device or default_device()
        self.verbose = verbose

        self.classes_: list[str] = []
        self.network_: nn.Module | None = None
        self.history_: TrainingHistory = TrainingHistory()

    # ------------------------------------------------------- to implement --

    def _build_features(self, X: Sequence[str], fitting: bool) -> torch.Tensor:
        """Turn raw texts into the tensor the network consumes."""
        raise NotImplementedError

    def _build_network(self, input_dim: int, n_classes: int) -> nn.Module:
        """Construct the architecture."""
        raise NotImplementedError

    # ---------------------------------------------------------- training --

    def fit(self, X: Sequence[str], y: Sequence[str]) -> NeuralClassifier:
        """Train the network, holding out a validation split to monitor loss."""
        set_seed(self.seed)
        self.classes_ = sorted(set(y)) or list(LABELS)
        index_of = {label: i for i, label in enumerate(self.classes_)}

        features = self._build_features(X, fitting=True)
        targets = torch.tensor([index_of[label] for label in y], dtype=torch.long)

        train_idx, val_idx = self._split(targets)
        train_loader = DataLoader(
            TensorDataset(features[train_idx], targets[train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_features = features[val_idx].to(self.device)
        val_targets = targets[val_idx].to(self.device)

        self.network_ = self._build_network(
            features.shape[-1], len(self.classes_)
        ).to(self.device)
        optimiser = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        self.history_ = TrainingHistory()
        best_loss, best_state, waited = math.inf, None, 0

        for epoch in range(1, self.epochs + 1):
            self.network_.train()
            running = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.network_(batch_features), batch_targets)
                loss.backward()
                optimiser.step()
                running += loss.item() * len(batch_targets)
            train_loss = running / max(len(train_idx), 1)

            # Validation pass: this is the comparison the course asks for.
            self.network_.eval()
            with torch.no_grad():
                logits = self.network_(val_features)
                val_loss = criterion(logits, val_targets).item()
                val_accuracy = (
                    (logits.argmax(dim=1) == val_targets).float().mean().item()
                )

            self.history_.train_loss.append(train_loss)
            self.history_.val_loss.append(val_loss)
            self.history_.val_accuracy.append(val_accuracy)
            if self.verbose:
                print(
                    f"    epoch {epoch:>3}  train {train_loss:.4f}  "
                    f"val {val_loss:.4f}  acc {val_accuracy:.4f}"
                )

            if val_loss < best_loss - 1e-4:
                best_loss, waited = val_loss, 0
                self.history_.best_epoch = epoch
                best_state = {
                    k: v.detach().clone() for k, v in self.network_.state_dict().items()
                }
            else:
                waited += 1
                if self.patience is not None and waited >= self.patience:
                    self.history_.stopped_early = True
                    break

        # Restore the weights from the best epoch, not the last one.
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        return self

    def _split(self, targets: torch.Tensor) -> tuple[list[int], list[int]]:
        """Stratified train/validation split of the training data."""
        generator = torch.Generator().manual_seed(self.seed)
        train_idx: list[int] = []
        val_idx: list[int] = []
        for class_index in targets.unique().tolist():
            positions = (targets == class_index).nonzero(as_tuple=True)[0]
            shuffled = positions[torch.randperm(len(positions), generator=generator)]
            n_val = max(1, int(round(len(shuffled) * self.validation_fraction)))
            val_idx.extend(shuffled[:n_val].tolist())
            train_idx.extend(shuffled[n_val:].tolist())
        return sorted(train_idx), sorted(val_idx)

    # -------------------------------------------------------- prediction --

    def _logits(self, X: Sequence[str]) -> torch.Tensor:
        """Forward pass over already-trained weights."""
        if self.network_ is None:
            raise RuntimeError("Call fit() before predict().")
        features = self._build_features(X, fitting=False).to(self.device)
        self.network_.eval()
        with torch.no_grad():
            return self.network_(features).cpu()

    def predict(self, X: Sequence[str]) -> list[str]:
        """Return the highest-scoring class for each text."""
        return [self.classes_[i] for i in self._logits(X).argmax(dim=1).tolist()]

    def predict_proba(self, X: Sequence[str]) -> list[dict[str, float]]:
        """Return softmax class probabilities, enabling ROC and AUC."""
        probabilities = torch.softmax(self._logits(X), dim=1)
        return [
            dict(zip(self.classes_, row.tolist(), strict=True))
            for row in probabilities
        ]


class FeedForwardClassifier(NeuralClassifier):
    """Feed-forward network over TF-IDF features.

    Directly comparable with the Track B pipelines: same input representation,
    different classifier. The vectoriser is fitted inside :meth:`fit`, so
    cross-validation folds stay honest.

    Args:
        hidden_sizes: Width of each hidden layer.
        dropout: Dropout probability between layers.
        max_features: Vocabulary cap for the TF-IDF vectoriser.
        **kwargs: Forwarded to :class:`NeuralClassifier`.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (256, 64),
        dropout: float = 0.5,
        max_features: int = 20000,
        **kwargs,
    ) -> None:
        """Store architecture settings and defer construction to ``fit``."""
        super().__init__(**kwargs)
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout
        self.max_features = max_features
        self.vectorizer_ = None

    def _build_features(self, X: Sequence[str], fitting: bool) -> torch.Tensor:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if fitting:
            self.vectorizer_ = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                max_features=self.max_features,
            )
            matrix = self.vectorizer_.fit_transform(X)
        else:
            matrix = self.vectorizer_.transform(X)
        return torch.tensor(matrix.toarray(), dtype=torch.float32)

    def _build_network(self, input_dim: int, n_classes: int) -> nn.Module:
        layers: list[nn.Module] = []
        previous = input_dim
        for width in self.hidden_sizes:
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(self.dropout)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        return nn.Sequential(*layers)


class CNNTextClassifier(NeuralClassifier):
    """1-D convolutional network over learned word embeddings.

    Filters of several widths act as learned n-gram detectors; global max
    pooling keeps the strongest activation of each filter regardless of where
    in the document it occurred, which suits issue reports where the
    informative phrase may appear anywhere in a long body.

    Args:
        embedding_dim: Width of the learned embeddings.
        filter_sizes: Convolution kernel widths, i.e. the n-gram lengths.
        n_filters: Number of filters per width.
        dropout: Dropout probability before the classifier.
        max_length: Tokens kept per document.
        min_count: Minimum corpus frequency for a token to enter the vocabulary.
        **kwargs: Forwarded to :class:`NeuralClassifier`.
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        filter_sizes: Sequence[int] = (2, 3, 4),
        n_filters: int = 64,
        dropout: float = 0.5,
        max_length: int = 200,
        min_count: int = 2,
        **kwargs,
    ) -> None:
        """Store architecture settings and defer construction to ``fit``."""
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.filter_sizes = tuple(filter_sizes)
        self.n_filters = n_filters
        self.dropout = dropout
        self.max_length = max_length
        self.min_count = min_count
        self.vocabulary_: dict[str, int] = {}

    def _build_features(self, X: Sequence[str], fitting: bool) -> torch.Tensor:
        if fitting:
            counts: Counter[str] = Counter()
            for text in X:
                counts.update(TOKEN.findall(text.lower()))
            self.vocabulary_ = {"<pad>": PAD, "<unk>": UNK}
            for token, count in counts.most_common():
                if count >= self.min_count:
                    self.vocabulary_[token] = len(self.vocabulary_)

        rows = []
        for text in X:
            ids = [
                self.vocabulary_.get(token, UNK)
                for token in TOKEN.findall(text.lower())[: self.max_length]
            ]
            ids += [PAD] * (self.max_length - len(ids))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long)

    def _build_network(self, input_dim: int, n_classes: int) -> nn.Module:
        # input_dim is the sequence length here; the vocabulary decides the
        # embedding table's size.
        return _CNN(
            vocab_size=max(len(self.vocabulary_), 2),
            embedding_dim=self.embedding_dim,
            filter_sizes=self.filter_sizes,
            n_filters=self.n_filters,
            dropout=self.dropout,
            n_classes=n_classes,
        )


class _CNN(nn.Module):
    """The convolutional architecture used by :class:`CNNTextClassifier`."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filter_sizes: Sequence[int],
        n_filters: int,
        dropout: float,
        n_classes: int,
    ) -> None:
        """Build the embedding table, the parallel convolutions and the head."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, n_filters, kernel_size=size, padding=size - 1)
                for size in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_filters * len(filter_sizes), n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed, convolve, pool over time, then classify."""
        # (batch, length) -> (batch, embedding, length) for Conv1d
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [
            torch.relu(convolution(embedded)).max(dim=2).values
            for convolution in self.convolutions
        ]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


def plot_learning_curves(history: TrainingHistory, title: str = ""):
    """Plot training against validation loss per epoch.

    This is the validation curve the course prescribes for diagnosing
    overfitting: the gap opening between the two lines is the symptom, and the
    marked epoch is where early stopping restored the weights from.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.train_loss) + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    left.plot(epochs, history.train_loss, label="training loss")
    left.plot(epochs, history.val_loss, label="validation loss")
    if history.best_epoch:
        left.axvline(
            history.best_epoch,
            color="grey",
            linestyle=":",
            label=f"best epoch ({history.best_epoch})",
        )
    left.set_xlabel("epoch")
    left.set_ylabel("cross-entropy loss")
    left.set_title("Loss")
    left.legend()

    right.plot(epochs, history.val_accuracy, color="seagreen")
    right.set_xlabel("epoch")
    right.set_ylabel("accuracy")
    right.set_title("Validation accuracy")
    right.set_ylim(0, 1)

    fig.suptitle(title or "Learning curves")
    fig.tight_layout()
    return fig


#: Zero-argument factories for the Track C models.
NEURAL_MODELS: dict = {
    "FFNN (TF-IDF)": FeedForwardClassifier,
    "CNN (embeddings)": CNNTextClassifier,
}
