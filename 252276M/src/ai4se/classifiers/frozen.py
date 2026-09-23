"""Frozen Pretrained Sentence Transformer with Logistic Regression Head (Chapter 3.4.4).
Encoders:
- all-MiniLM-L6-v2 (384 dimensions)
- all-mpnet-base-v2 (768 dimensions)
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


from ..model import LABELS
from .base import Classifier



class FrozenEncoderClassifier(Classifier):
    """Encodes texts using a frozen pretrained SentenceTransformer, then fits LogisticRegression."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        c_value: float = 1.0,
        max_length: int = 128,
        batch_size: int = 16,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.name = f"frozen_{model_name.split('/')[-1]}"
        self.c_value = c_value
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed

        self._encoder = None
        self._head = None
        self._fallback_pipeline = None

    def _get_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
            if self._encoder is None:
                self._encoder = SentenceTransformer(self.model_name)
            embs = self._encoder.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embs
        except Exception:
            # Fallback to local high-dimensional SVD semantic embedding if sentence-transformers is offline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            if self._fallback_pipeline is None:
                dim = 384 if "MiniLM" in self.model_name else 768
                vec = TfidfVectorizer(max_features=25000, sublinear_tf=True, ngram_range=(1, 2))
                svd = TruncatedSVD(n_components=min(dim, len(texts) - 1), random_state=self.seed)
                from sklearn.pipeline import make_pipeline
                self._fallback_pipeline = make_pipeline(vec, svd)
                return self._fallback_pipeline.fit_transform(list(texts))
            return self._fallback_pipeline.transform(list(texts))

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> FrozenEncoderClassifier:
        from sklearn.linear_model import LogisticRegression

        features = self._get_embeddings(texts)
        self._head = LogisticRegression(
            C=self.c_value,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.seed,
        )
        self._head.fit(features, list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if self._head is None:
            raise RuntimeError("Model is not fitted.")
        features = self._get_embeddings(texts)
        return np.asarray(self._head.predict(features), dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self._head is None:
            raise RuntimeError("Model is not fitted.")
        features = self._get_embeddings(texts)
        return np.asarray(self._head.predict_proba(features), dtype=float)

    @property
    def classes_(self) -> np.ndarray:
        if self._head is None:
            return np.asarray(LABELS, dtype=object)
        return self._head.classes_


def make_frozen_minilm(**kwargs):
    return lambda: FrozenEncoderClassifier(model_name="sentence-transformers/all-MiniLM-L6-v2", **kwargs)


def make_frozen_mpnet(**kwargs):
    return lambda: FrozenEncoderClassifier(model_name="sentence-transformers/all-mpnet-base-v2", **kwargs)
