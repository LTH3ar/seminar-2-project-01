"""Neural network baselines (Chapter 3.4.3 & Table 5.4):
- FeedForwardClassifier (FFNN): 2 hidden layers (256, 64), ReLU, Dropout 0.5 over TF-IDF.
- TextCnnClassifier (Kim 2014): learned 100d embeddings, parallel filters (2, 3, 4) x 64 filters.
Both include 20% validation split, early stopping with patience 8, and training history tracking.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
import numpy as np


from ..model import LABELS
from .base import Classifier

_TOKEN = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _set_seed(seed: int) -> None:
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeedForwardClassifier(Classifier):
    """Feed-forward multi-layer perceptron over TF-IDF features (256 -> 64 -> 3)."""

    name: str = "ffnn_tfidf"

    def __init__(
        self,
        max_features: int = 20_000,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        dropout: float = 0.5,
        max_epochs: int = 40,
        patience: int = 8,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.max_features = max_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.val_ratio = val_ratio
        self.seed = seed

        self._vectorizer = None
        self._network = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_epoch: int = 0
        self.stopped_early: bool = False

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> FeedForwardClassifier:
        import torch
        from torch import nn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from ..evaluation.splits import train_val_split

        _set_seed(self.seed)

        # 20% validation split for early stopping and diagnostics
        tr_texts, tr_labels, val_texts, val_labels = train_val_split(
            texts, labels, val_ratio=self.val_ratio, seed=self.seed
        )

        self._vectorizer = TfidfVectorizer(max_features=self.max_features, sublinear_tf=True, min_df=2)
        X_tr = self._vectorizer.fit_transform(tr_texts).toarray()
        X_val = self._vectorizer.transform(val_texts).toarray()

        label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
        y_tr = np.array([label_to_idx[l] for l in tr_labels])
        y_val = np.array([label_to_idx[l] for l in val_labels])

        in_dim = X_tr.shape[1]
        self._network = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim2, len(LABELS)),
        )

        t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr, dtype=torch.long)
        t_X_val = torch.tensor(X_val, dtype=torch.float32)
        t_y_val = torch.tensor(y_val, dtype=torch.long)

        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.max_epochs + 1):
            self._network.train()
            perm = torch.randperm(len(t_X_tr))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(t_X_tr), self.batch_size):
                b_idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                out = self._network(t_X_tr[b_idx])
                loss = criterion(out, t_y_tr[b_idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            train_loss = epoch_loss / max(1, batches)

            # Validation
            self._network.eval()
            with torch.no_grad():
                val_out = self._network(t_X_val)
                v_loss = criterion(val_out, t_y_val).item()
                v_acc = (val_out.argmax(dim=1) == t_y_val).float().mean().item()

            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(v_loss))
            self.history["val_acc"].append(float(v_acc))

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                self.best_epoch = epoch
                best_state = copy.deepcopy(self._network.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.stopped_early = True
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)
        self._network.eval()
        return self

    def _get_logits(self, texts: Sequence[str]):
        import torch
        if self._network is None or self._vectorizer is None:
            raise RuntimeError("Model is not fitted.")
        feats = self._vectorizer.transform(list(texts)).toarray()
        with torch.no_grad():
            return self._network(torch.tensor(feats, dtype=torch.float32))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        logits = self._get_logits(texts)
        idx = logits.argmax(dim=1).numpy()
        return np.asarray([LABELS[i] for i in idx], dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        logits = self._get_logits(texts)
        probs = torch.softmax(logits, dim=1).numpy()
        return probs

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(LABELS, dtype=object)


class TextCnnClassifier(Classifier):
    """1-D CNN over learned 100d embeddings with parallel filters of widths (2, 3, 4) x 64 filters."""

    name: str = "text_cnn"

    def __init__(
        self,
        vocab_size: int = 20_000,
        embedding_dim: int = 100,
        n_filters: int = 64,
        filter_sizes: tuple[int, ...] = (2, 3, 4),
        max_length: int = 256,
        dropout: float = 0.5,
        max_epochs: int = 40,
        patience: int = 8,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.max_length = max_length
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.val_ratio = val_ratio
        self.seed = seed

        self._vocab: dict[str, int] = {}
        self._network = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_epoch: int = 0
        self.stopped_early: bool = False

    def _build_vocab(self, texts: Sequence[str]) -> None:
        from collections import Counter
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(_tokenize(t))
        common = counts.most_common(self.vocab_size - 2)
        self._vocab = {tok: i + 2 for i, (tok, _) in enumerate(common)}

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for r, t in enumerate(texts):
            tokens = _tokenize(t)[: self.max_length]
            for c, tok in enumerate(tokens):
                arr[r, c] = self._vocab.get(tok, 1)  # 1 is <UNK>
        return arr

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> TextCnnClassifier:
        import torch
        from torch import nn
        from ..evaluation.splits import train_val_split

        _set_seed(self.seed)

        tr_texts, tr_labels, val_texts, val_labels = train_val_split(
            texts, labels, val_ratio=self.val_ratio, seed=self.seed
        )

        self._build_vocab(tr_texts)
        X_tr = self._encode(tr_texts)
        X_val = self._encode(val_texts)

        label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
        y_tr = np.array([label_to_idx[l] for l in tr_labels])
        y_val = np.array([label_to_idx[l] for l in val_labels])

        v_size = len(self._vocab) + 2
        emb_dim = self.embedding_dim
        n_filt = self.n_filters
        f_sizes = self.filter_sizes
        drop = self.dropout

        class _CnnNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(v_size, emb_dim, padding_idx=0)
                self.convs = nn.ModuleList(
                    [nn.Conv1d(emb_dim, n_filt, size) for size in f_sizes]
                )
                self.dropout = nn.Dropout(drop)
                self.fc = nn.Linear(n_filt * len(f_sizes), len(LABELS))

            def forward(self, x):
                # x: (batch, seq_len) -> (batch, seq_len, emb_dim) -> (batch, emb_dim, seq_len)
                emb = self.embedding(x).transpose(1, 2)
                pooled = [torch.relu(c(emb)).max(dim=2).values for c in self.convs]
                h = self.dropout(torch.cat(pooled, dim=1))
                return self.fc(h)

        self._network = _CnnNet()
        t_X_tr = torch.tensor(X_tr, dtype=torch.long)
        t_y_tr = torch.tensor(y_tr, dtype=torch.long)
        t_X_val = torch.tensor(X_val, dtype=torch.long)
        t_y_val = torch.tensor(y_val, dtype=torch.long)

        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.max_epochs + 1):
            self._network.train()
            perm = torch.randperm(len(t_X_tr))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(t_X_tr), self.batch_size):
                b_idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                out = self._network(t_X_tr[b_idx])
                loss = criterion(out, t_y_tr[b_idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            train_loss = epoch_loss / max(1, batches)

            self._network.eval()
            with torch.no_grad():
                val_out = self._network(t_X_val)
                v_loss = criterion(val_out, t_y_val).item()
                v_acc = (val_out.argmax(dim=1) == t_y_val).float().mean().item()

            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(v_loss))
            self.history["val_acc"].append(float(v_acc))

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                self.best_epoch = epoch
                best_state = copy.deepcopy(self._network.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.stopped_early = True
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)
        self._network.eval()
        return self

    def _get_logits(self, texts: Sequence[str]):
        import torch
        if self._network is None:
            raise RuntimeError("Model is not fitted.")
        feats = self._encode(texts)
        with torch.no_grad():
            return self._network(torch.tensor(feats, dtype=torch.long))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        logits = self._get_logits(texts)
        idx = logits.argmax(dim=1).numpy()
        return np.asarray([LABELS[i] for i in idx], dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        logits = self._get_logits(texts)
        probs = torch.softmax(logits, dim=1).numpy()
        return probs

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(LABELS, dtype=object)


def make_ffnn(**kwargs):
    return lambda: FeedForwardClassifier(**kwargs)


def make_text_cnn(**kwargs):
    return lambda: TextCnnClassifier(**kwargs)
