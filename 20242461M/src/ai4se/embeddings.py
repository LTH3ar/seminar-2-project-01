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


def free_gpu_memory() -> None:
    """Release cached CUDA memory, if torch is present and a GPU is in use.

    The competition protocol fits five models in one process. Without this,
    each fine-tuned encoder stays resident in the allocator's cache and the
    third or fourth project runs out of memory even when the first fitted
    comfortably.
    """
    try:
        import gc

        import torch
    except ImportError:  # pragma: no cover - torch is an optional extra
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
        batch_size: Fine-tuning batch size. Halve it if training runs out of
            memory; peak usage is roughly linear in it.
        max_seq_length: Tokens kept per sentence. **The main memory control**:
            attention cost grows with the square of this, so halving it cuts
            peak memory by about four times. Encoder defaults are 256 for
            MiniLM and 384 for MPNet; 128 is ample here, since the median
            issue is around 150 words and the text is already truncated by
            the preprocessing. ``None`` keeps the encoder's own default.
        seed: Random seed.

    Note:
        Contrastive training embeds *both* sentences of every pair, so peak
        memory is roughly twice what fine-tuning the same encoder for plain
        classification would need. Combined with MPNet being twelve layers of
        width 768 against MiniLM's six of 384, MPNet at MiniLM's settings needs
        several times the memory -- which is why the defaults here are
        conservative rather than matching the SetFit library's.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        num_iterations: int = 20,
        num_epochs: int = 1,
        batch_size: int = 16,
        max_seq_length: int | None = 128,
        seed: int = RANDOM_SEED,
    ) -> None:
        """Store settings; the model is built on :meth:`fit`."""
        self.model_name = model_name
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
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

        # Release whatever the previous project's encoder was holding before
        # allocating this one.
        free_gpu_memory()

        self.classes_ = sorted(set(y))
        self.model_ = SetFitModel.from_pretrained(self.model_name)

        if self.max_seq_length is not None:
            body = getattr(self.model_, "model_body", None)
            if body is not None:
                body.max_seq_length = self.max_seq_length

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
        free_gpu_memory()
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


def setfit_minilm() -> SetFitClassifier:
    """The published baseline's method on the small MiniLM encoder.

    Measured at 0.7982 cross-repository F1, against 0.8270 for the published
    baseline -- a consistent shortfall of 0.009 to 0.045 on every project.

    Note this uses the library's own ``batch_size`` of 16 and the encoder's
    ``max_seq_length`` of 256. :func:`setfit_minilm_matched` runs the same
    encoder at MPNet's reduced settings, which is the comparison to quote when
    attributing a difference to the encoder rather than to the settings.

    Both values are pinned explicitly rather than left to the class defaults,
    so this factory keeps reproducing the recorded 0.7982 even though the
    default ``max_seq_length`` was later lowered to 128 to make MPNet fit in
    12 GB.
    """
    return SetFitClassifier(DEFAULT_ENCODER, batch_size=16, max_seq_length=None)


def setfit_minilm_matched() -> SetFitClassifier:
    """MiniLM at MPNet's memory settings, so the two can be compared fairly.

    The first MiniLM run used ``batch_size=16`` and the encoder's own
    ``max_seq_length`` of 256; MPNet had to be run at 8 and 128 to fit in
    12 GB. Comparing those two conflates the encoder with the training
    settings, and the handicap falls on MPNet, so its measured advantage is a
    lower bound. This variant removes the confound by handicapping MiniLM
    identically.
    """
    return SetFitClassifier(DEFAULT_ENCODER, batch_size=8, max_seq_length=128)


def setfit_mpnet() -> SetFitClassifier:
    """The published baseline's method on the larger MPNet encoder.

    The experiment the measurements point at. Contrastive fine-tuning is worth
    +0.0925 on MiniLM (0.7058 frozen -> 0.7982 fine-tuned), and frozen MPNet
    already starts 0.050 above frozen MiniLM. If the gain transfers, this
    reaches roughly 0.848 -- above the published 0.8270 -- which would suggest
    the reproduction gap is the encoder rather than the method.

    Measured at 0.8093 -- better than MiniLM's 0.7982, but far short of that
    projection. The encoder is worth +0.050 frozen and only +0.011 after
    fine-tuning, so the two effects are strongly sub-additive: adapting a small
    encoder recovers most of what a larger one provides. (Part of that shrinkage
    may be the reduced settings this variant needs; see
    :func:`setfit_minilm_matched`.)

    Slower and far hungrier than :func:`setfit_minilm`: MPNet is twelve layers
    of width 768 against MiniLM's six of 384, and contrastive training embeds
    both sentences of every pair. The settings here are chosen to fit a 12 GB
    card; raise ``batch_size`` towards 16 if there is room to spare.
    """
    return SetFitClassifier(STRONG_ENCODER, batch_size=8, max_seq_length=128)


#: Frozen-encoder models. No gradient computation, so these run on a CPU.
EMBEDDING_MODELS: dict = {
    "Frozen MiniLM + LogReg": frozen_minilm,
    "Frozen MPNet + LogReg": frozen_mpnet,
}

#: Full SetFit reproductions. Each fine-tunes an encoder five times, so these
#: need a GPU and the ``setfit`` extra.
SETFIT_MODELS: dict = {
    "SetFit (reproduction)": setfit_minilm,
    "SetFit (MiniLM, matched settings)": setfit_minilm_matched,
    "SetFit (MPNet)": setfit_mpnet,
}
