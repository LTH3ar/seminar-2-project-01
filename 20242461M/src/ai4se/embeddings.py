"""Track D2 -- the Sentence Transformers baseline.

The NLBSE'24 competition's published baseline is SetFit: a Sentence
Transformer is fine-tuned contrastively on generated pairs of examples, and a
logistic-regression head is then fitted on the resulting sentence embeddings.
It scores 0.8270 cross-repository F1, and the project brief asks explicitly
for a comparison against it.

Two implementations are provided, because they have very different costs:

:class:`FrozenEmbeddingClassifier`
    Encodes with a *frozen* pretrained Sentence Transformer and fits a
    logistic-regression head on the embeddings. This is SetFit with the
    contrastive fine-tuning step removed. It needs no gradient computation
    through the encoder, runs on a CPU in seconds, and isolates how much of
    the baseline's strength comes from the pretrained representation alone
    rather than from the fine-tuning.

:class:`SetFitClassifier`
    The real thing: contrastive fine-tuning of the encoder, then the
    classification head. Requires the ``setfit`` package and, realistically, a
    GPU. Provided so the full reproduction can be run where hardware allows.

Both satisfy the usual ``fit``/``predict``/``predict_proba`` interface and run
through the unchanged evaluation harness.

Embeddings are cached per (model, text) across calls. Cross-validation encodes
the same 1,500 issues once per fold otherwise, which dominates the runtime for
no benefit -- the encoder is frozen, so the embedding of a given text never
changes between folds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .evaluation import RANDOM_SEED

#: Default encoder. Small, fast on CPU, and a standard choice for
#: sentence-level classification.
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

#: A stronger and slower alternative, closer to what the competition baselines
#: typically use.
STRONG_ENCODER = "sentence-transformers/all-mpnet-base-v2"

#: Process-wide embedding cache: ``{model_name: {text: vector}}``.
_EMBEDDING_CACHE: dict[str, dict[str, np.ndarray]] = {}

#: Loaded encoders, so the weights are read from disk only once.
_ENCODERS: dict[str, object] = {}


def load_encoder(name: str = DEFAULT_ENCODER):
    """Load a Sentence Transformer, reusing an already-loaded one.

    Args:
        name: HuggingFace model identifier.

    Returns:
        The loaded ``SentenceTransformer``.
    """
    if name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        _ENCODERS[name] = SentenceTransformer(name)
    return _ENCODERS[name]


def embed(
    texts: Sequence[str],
    model_name: str = DEFAULT_ENCODER,
    batch_size: int = 32,
    use_cache: bool = True,
) -> np.ndarray:
    """Encode texts to sentence embeddings, caching by text.

    Args:
        texts: The documents to encode.
        model_name: Encoder identifier.
        batch_size: Encoding batch size.
        use_cache: Reuse previously computed vectors. Safe because the encoder
            is frozen; disable it when the encoder has been fine-tuned.

    Returns:
        An array of shape ``(len(texts), embedding_dim)``.
    """
    cache = _EMBEDDING_CACHE.setdefault(model_name, {}) if use_cache else {}
    missing = [text for text in dict.fromkeys(texts) if text not in cache]

    if missing:
        encoder = load_encoder(model_name)
        vectors = encoder.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        for text, vector in zip(missing, vectors, strict=True):
            cache[text] = np.asarray(vector, dtype=np.float32)

    return np.vstack([cache[text] for text in texts])


def clear_embedding_cache() -> None:
    """Empty the embedding cache, for measuring cold-start timings."""
    _EMBEDDING_CACHE.clear()


class FrozenEmbeddingClassifier:
    """Frozen Sentence Transformer embeddings with a logistic-regression head.

    SetFit without the contrastive fine-tuning step. The gap between this and
    :class:`SetFitClassifier` is precisely the contribution of that step, which
    makes the comparison worth reporting in its own right.

    Args:
        model_name: Encoder identifier.
        C: Inverse regularisation strength of the logistic-regression head.
        batch_size: Encoding batch size.
        seed: Seed for the head.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        C: float = 10.0,  # noqa: N803
        batch_size: int = 32,
        seed: int = RANDOM_SEED,
    ) -> None:
        """Store settings; the encoder is loaded lazily on first use."""
        self.model_name = model_name
        self.C = C
        self.batch_size = batch_size
        self.seed = seed
        self.head_ = None
        self.classes_: list[str] = []

    def fit(self, X: Sequence[str], y: Sequence[str]) -> FrozenEmbeddingClassifier:
        """Encode the training texts and fit the classification head."""
        from sklearn.linear_model import LogisticRegression

        features = embed(X, self.model_name, self.batch_size)
        self.head_ = LogisticRegression(
            C=self.C,
            max_iter=2000,
            class_weight="balanced",
            random_state=self.seed,
        ).fit(features, list(y))
        self.classes_ = list(self.head_.classes_)
        return self

    def predict(self, X: Sequence[str]) -> list[str]:
        """Return the predicted class for each text."""
        if self.head_ is None:
            raise RuntimeError("Call fit() before predict().")
        return list(self.head_.predict(embed(X, self.model_name, self.batch_size)))

    def predict_proba(self, X: Sequence[str]) -> list[dict[str, float]]:
        """Return class probabilities from the logistic-regression head."""
        if self.head_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        matrix = self.head_.predict_proba(embed(X, self.model_name, self.batch_size))
        return [
            dict(zip(self.classes_, row.tolist(), strict=True)) for row in matrix
        ]


class SetFitClassifier:
    """The published baseline: contrastive fine-tuning plus a classifier head.

    SetFit generates positive and negative pairs from the labelled examples,
    fine-tunes the Sentence Transformer on them with a contrastive objective,
    then fits a logistic-regression head on the adapted embeddings. With only
    300 training issues per project, the pair generation is what makes the
    approach work in the few-shot regime the competition targets.

    Requires the ``setfit`` package (``pip install -e ".[dl]"``) and a GPU for
    a full run: fine-tuning an encoder on a CPU takes hours per project.

    Args:
        model_name: Encoder to fine-tune.
        num_iterations: Pair-generation iterations per example.
        num_epochs: Fine-tuning epochs for the encoder body.
        batch_size: Fine-tuning batch size.
        seed: Random seed.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        num_iterations: int = 20,
        num_epochs: int = 1,
        batch_size: int = 16,
        seed: int = RANDOM_SEED,
    ) -> None:
        """Store settings; the model is built on :meth:`fit`."""
        self.model_name = model_name
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model_ = None
        self.classes_: list[str] = []

    def fit(self, X: Sequence[str], y: Sequence[str]) -> SetFitClassifier:
        """Fine-tune the encoder contrastively, then fit the head."""
        try:
            from datasets import Dataset
            from setfit import SetFitModel, Trainer, TrainingArguments
        except ImportError as error:  # pragma: no cover - optional extra
            raise ImportError(
                "SetFitClassifier needs the 'dl' extra: pip install -e \".[dl]\""
            ) from error

        self.classes_ = sorted(set(y))
        self.model_ = SetFitModel.from_pretrained(self.model_name)

        trainer = Trainer(
            model=self.model_,
            train_dataset=Dataset.from_dict({"text": list(X), "label": list(y)}),
            args=TrainingArguments(
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                num_iterations=self.num_iterations,
                seed=self.seed,
            ),
        )
        trainer.train()
        return self

    def predict(self, X: Sequence[str]) -> list[str]:
        """Return the predicted class for each text."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        return [str(label) for label in self.model_.predict(list(X))]

    def predict_proba(self, X: Sequence[str]) -> list[dict[str, float]]:
        """Return class probabilities from the SetFit head."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        matrix = self.model_.predict_proba(list(X))
        rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
        return [dict(zip(self.classes_, row, strict=True)) for row in rows]


def frozen_minilm() -> FrozenEmbeddingClassifier:
    """Frozen MiniLM embeddings with a logistic-regression head."""
    return FrozenEmbeddingClassifier(DEFAULT_ENCODER)


def frozen_mpnet() -> FrozenEmbeddingClassifier:
    """Frozen MPNet embeddings with a logistic-regression head."""
    return FrozenEmbeddingClassifier(STRONG_ENCODER)


#: Factories for the Track D2 models that run without a GPU.
EMBEDDING_MODELS: dict = {
    "Frozen MiniLM + LogReg": frozen_minilm,
    "Frozen MPNet + LogReg": frozen_mpnet,
}
