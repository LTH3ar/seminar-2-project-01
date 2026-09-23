"""Neural baselines that train on CPU in seconds (track C).

Two architectures, both small enough that five projects times five seeds
finishes in a couple of minutes without a GPU:

``FeedForwardClassifier``
    A two-layer perceptron over TF-IDF features. The simplest thing that can
    be called deep learning, and a useful check on whether the classical
    models are leaving anything on the table.
``TextCnnClassifier``
    The convolutional sentence classifier of Kim (2014): an embedding layer
    trained from scratch, parallel convolutions of width 3, 4 and 5,
    max-over-time pooling, and a linear head.

Neither is expected to beat SetFit -- 300 labelled issues per project is far
too little to learn embeddings from scratch, and saying so up front is part
of the point. They are here because the brief asks for a range of techniques
and because a cheap model with an honest number is worth more than an
expensive one with no number.

Requires the deep-learning extra::

    pip install -e ".[dl]"
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .base import Classifier

#: Tokeniser shared by the CNN's vocabulary builder and its encoder. Kept
#: deliberately simple: the point of the CNN is the architecture, not the
#: preprocessing, which ``ai4se.preprocessing`` already covers.
_TOKEN = re.compile(r"[a-z0-9_]+")


def _tokenise(text: str) -> list[str]:
    """Lowercase and split a document into word-like tokens."""
    return _TOKEN.findall(text.lower())


def _set_seed(seed: int) -> None:
    """Seed numpy and torch so a run can be repeated exactly."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


class FeedForwardClassifier(Classifier):
    """A multi-layer perceptron over TF-IDF features.

    Args:
        max_features: Vocabulary size of the TF-IDF vectoriser.
        hidden_size: Width of the single hidden layer.
        dropout: Dropout applied after the hidden layer.
        epochs: Passes over the training set.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        seed: Seed for reproducibility.
    """

    def __init__(  # noqa: D107 -- arguments documented on the class
        self,
        max_features: int = 20_000,
        hidden_size: int = 256,
        dropout: float = 0.3,
        epochs: int = 15,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.name = "ffnn"
        self.max_features = max_features
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self._vectoriser = None
        self._network = None
        self._classes: np.ndarray | None = None

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> FeedForwardClassifier:
        """Vectorise, then train the network with Adam and cross-entropy."""
        import torch
        from sklearn.feature_extraction.text import TfidfVectorizer
        from torch import nn

        _set_seed(self.seed)
        self._vectoriser = TfidfVectorizer(
            max_features=self.max_features, sublinear_tf=True, min_df=2
        )
        features = self._vectoriser.fit_transform(list(texts)).toarray()
        self._classes = np.array(sorted(set(labels)), dtype=object)
        index = {label: i for i, label in enumerate(self._classes)}
        targets = np.array([index[label] for label in labels])

        self._network = nn.Sequential(
            nn.Linear(features.shape[1], self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, len(self._classes)),
        )
        self._train_loop(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
        )
        return self

    def _train_loop(self, features, targets) -> None:
        """Run the optimisation loop shared by both neural models."""
        import torch
        from torch import nn

        optimiser = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        n = features.shape[0]
        self._network.train()
        for _ in range(self.epochs):
            order = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimiser.zero_grad()
                loss = loss_function(self._network(features[batch]), targets[batch])
                loss.backward()
                optimiser.step()
        self._network.eval()

    def _logits(self, texts: Sequence[str]):
        """Return raw network outputs for a batch of documents.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        import torch

        if self._network is None or self._vectoriser is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting.")
        features = self._vectoriser.transform(list(texts)).toarray()
        with torch.no_grad():
            return self._network(torch.tensor(features, dtype=torch.float32))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Predict a label for each text."""
        indices = self._logits(texts).argmax(dim=1).numpy()
        return np.asarray(self._classes[indices], dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return softmax class probabilities."""
        import torch

        return torch.softmax(self._logits(texts), dim=1).numpy()

    @property
    def classes_(self) -> np.ndarray:
        """Label order of the probability columns.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._classes is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting.")
        return self._classes


class TextCnnClassifier(Classifier):
    """Kim's convolutional sentence classifier, trained from scratch.

    Args:
        vocab_size: Maximum vocabulary size, most frequent tokens kept.
        embedding_dim: Width of the learned embeddings.
        n_filters: Filters per convolution width.
        filter_sizes: Convolution widths, applied in parallel.
        max_length: Documents are truncated or padded to this many tokens.
            The default of 256 covers about three quarters of the dataset; see
            ``docs/benchmark-audit.md`` for the length distribution.
        dropout: Dropout before the linear head.
        epochs: Passes over the training set.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        seed: Seed for reproducibility.
    """

    def __init__(  # noqa: D107 -- arguments documented on the class
        self,
        vocab_size: int = 20_000,
        embedding_dim: int = 100,
        n_filters: int = 100,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        max_length: int = 256,
        dropout: float = 0.5,
        epochs: int = 12,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.name = "text_cnn"
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.max_length = max_length
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self._vocab: dict[str, int] = {}
        self._network = None
        self._classes: np.ndarray | None = None

    def _build_vocab(self, texts: Sequence[str]) -> None:
        """Keep the most frequent tokens; index 0 is padding, 1 is unknown."""
        from collections import Counter

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(_tokenise(text))
        common = counts.most_common(self.vocab_size - 2)
        self._vocab = {token: i + 2 for i, (token, _) in enumerate(common)}

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Turn documents into fixed-length integer sequences."""
        encoded = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for row, text in enumerate(texts):
            tokens = _tokenise(text)[: self.max_length]
            for column, token in enumerate(tokens):
                encoded[row, column] = self._vocab.get(token, 1)
        return encoded

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> TextCnnClassifier:
        """Build the vocabulary, then train the convolutional network."""
        import torch
        from torch import nn

        _set_seed(self.seed)
        self._build_vocab(texts)
        self._classes = np.array(sorted(set(labels)), dtype=object)
        index = {label: i for i, label in enumerate(self._classes)}

        n_classes = len(self._classes)
        embedding_dim, n_filters = self.embedding_dim, self.n_filters
        filter_sizes, dropout = self.filter_sizes, self.dropout
        vocab_size = self.vocab_size

        class _Network(nn.Module):
            """Embedding, parallel convolutions, max-pooling, linear head."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.convolutions = nn.ModuleList(
                    nn.Conv1d(embedding_dim, n_filters, size) for size in filter_sizes
                )
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(n_filters * len(filter_sizes), n_classes)

            def forward(self, batch):
                """Map a batch of token-index sequences to class logits."""
                embedded = self.embedding(batch).transpose(1, 2)
                pooled = [
                    torch.relu(convolution(embedded)).max(dim=2).values
                    for convolution in self.convolutions
                ]
                return self.head(self.dropout(torch.cat(pooled, dim=1)))

        self._network = _Network()
        features = torch.tensor(self._encode(list(texts)))
        targets = torch.tensor([index[label] for label in labels], dtype=torch.long)

        optimiser = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        self._network.train()
        for _ in range(self.epochs):
            order = torch.randperm(features.shape[0])
            for start in range(0, features.shape[0], self.batch_size):
                batch = order[start : start + self.batch_size]
                optimiser.zero_grad()
                loss = loss_function(self._network(features[batch]), targets[batch])
                loss.backward()
                optimiser.step()
        self._network.eval()
        return self

    def _logits(self, texts: Sequence[str]):
        """Return raw network outputs for a batch of documents.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        import torch

        if self._network is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting.")
        with torch.no_grad():
            return self._network(torch.tensor(self._encode(list(texts))))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Predict a label for each text."""
        indices = self._logits(texts).argmax(dim=1).numpy()
        return np.asarray(self._classes[indices], dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return softmax class probabilities."""
        import torch

        return torch.softmax(self._logits(texts), dim=1).numpy()

    @property
    def classes_(self) -> np.ndarray:
        """Label order of the probability columns.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._classes is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting.")
        return self._classes


def make_ffnn(**kwargs):
    """Return a factory producing :class:`FeedForwardClassifier` instances."""
    return lambda: FeedForwardClassifier(**kwargs)


def make_text_cnn(**kwargs):
    """Return a factory producing :class:`TextCnnClassifier` instances."""
    return lambda: TextCnnClassifier(**kwargs)
