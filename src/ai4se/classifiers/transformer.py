"""Fine-tuned transformer classifier (track C — the main proposal).

Wraps a HuggingFace sequence-classification model behind the same
:class:`~ai4se.classifiers.base.Classifier` interface used by every other
model in the project, so the evaluation code, cross-validation harness and
per-project training protocol work without modification.

The default encoder is DeBERTa-v3-base, which the pre-registration
(``preregistration.md``, H1) names as the primary hypothesis.  The choice
is motivated by two properties:

1. **Disentangled attention** — relative-position information is kept in a
   separate attention head, which helps with the variable-length Markdown
   bodies in this dataset.
2. **Replaced-token detection pre-training** — the model sees more
   training signal per token than a masked-LM objective, which matters when
   fine-tuning on only 300 issues per project.

With 300 training issues and a 512-token maximum, fine-tuning one project
takes about 90 seconds on a T4 GPU and roughly 10 minutes on CPU.

Requires the deep-learning extra::

    pip install -e ".[dl]"
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence

import numpy as np

from ..model import LABELS
from .base import Classifier

# Default checkpoint — the pre-registration names DeBERTa-v3.
DEFAULT_MODEL = "microsoft/deberta-v3-base"

# A lighter alternative for CPU iteration or testing.
FAST_MODEL = "microsoft/deberta-v3-small"


def _set_seed(seed: int) -> None:
    """Seed every PRNG so a training run can be reproduced exactly."""
    import torch
    from transformers import set_seed as hf_set_seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)


class TransformerClassifier(Classifier):
    """Sequence classification head on a pretrained transformer encoder.

    The ``fit`` method fine-tunes the full model (encoder + new linear head)
    with the HuggingFace Trainer API, then ``predict`` and ``predict_proba``
    run inference through the same pipeline.

    Args:
        model_name: HuggingFace checkpoint to start from.
        max_length: Tokeniser truncation length.  Measured on the 3,000
            issues with the DeBERTa-v3 tokeniser, 512 tokens leaves 67% of
            the dataset untruncated (median 324, p95 2,125, max 79,064);
            256 leaves 41% and 1024 leaves 86% at four times the attention
            cost.  Searched rather than assumed -- see
            ``experiments/05_transformer.py``.
        epochs: Number of fine-tuning epochs.  With only 300 examples per
            project, more than 5 risks over-fitting; fewer than 3 may
            under-fit.
        batch_size: Training and evaluation batch size.
        learning_rate: Peak learning rate for the linear warm-up schedule.
        weight_decay: L2 regularisation applied to all parameters except
            biases and layer-norms.
        warmup_ratio: Fraction of total steps used for the linear warm-up.
        seed: Random seed for full reproducibility.
        output_dir: Scratch directory for the Trainer's checkpoints.
    """

    def __init__(  # noqa: D107 -- arguments documented on the class
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 512,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        seed: int = 42,
        output_dir: str | None = None,
        **extra_kwargs,
    ) -> None:
        self.name = "transformer"
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.seed = seed
        self.output_dir = output_dir
        self.extra_kwargs = extra_kwargs

        self._tokeniser = None
        self._model = None
        self._label2id: dict[str, int] = {}
        self._id2label: dict[int, str] = {}

    def fit(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> TransformerClassifier:
        """Fine-tune the transformer on labelled texts.

        The entire encoder is updated (no frozen layers) because the dataset
        is small enough that the Trainer finishes in a few minutes.
        """
        import torch
        from torch.utils.data import Dataset as TorchDataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        _set_seed(self.seed)

        # ---- label mapping --------------------------------------------------
        # Always use the canonical LABELS tuple to guarantee consistent 3-class
        # output across all per-project models and cross-validation folds.
        self._label2id = {label: idx for idx, label in enumerate(LABELS)}
        self._id2label = {idx: label for label, idx in self._label2id.items()}
        num_labels = len(LABELS)

        # ---- tokeniser -------------------------------------------------------
        # DeBERTa-v3 ships a SentencePiece tokeniser; building the fast
        # variant needs `sentencepiece` and `protobuf` (both in the `dl`
        # extra). Fall back to the slow one if that conversion fails, but let
        # network, auth and typo-in-the-checkpoint errors propagate.
        try:
            self._tokeniser = AutoTokenizer.from_pretrained(self.model_name)
        except (ImportError, ValueError):
            self._tokeniser = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=False
            )

        # ---- PyTorch dataset (avoids HuggingFace datasets PyArrow caching bugs) ----
        tokeniser = self._tokeniser
        max_len = self.max_length
        label2id = self._label2id

        class _IssueDataset(TorchDataset):
            def __init__(self, texts_seq: Sequence[str], labels_seq: Sequence[str]):
                clean_texts = [str(t) for t in texts_seq]
                self.encodings = tokeniser(
                    clean_texts,
                    truncation=True,
                    padding=False,
                    max_length=max_len,
                )
                try:
                    self.labels = [
                        label2id[str(lbl).strip().lower()] for lbl in labels_seq
                    ]
                except KeyError as exc:
                    raise ValueError(
                        f"Label {exc.args[0]!r} is not one of {LABELS}."
                    ) from exc

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                item = {
                    key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()
                }
                item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

            def __len__(self) -> int:
                return len(self.labels)

        dataset = _IssueDataset(texts, labels)

        # ---- model -----------------------------------------------------------
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            label2id=self._label2id,
            id2label=self._id2label,
        )

        # ---- output directory (isolated temp dir per run to prevent lock errors) ----
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        run_output_dir = (
            tempfile.mkdtemp(prefix="ai4se_trans_")
            if self.output_dir is None
            else tempfile.mkdtemp(dir=self.output_dir, prefix="run_")
        )

        # ---- training --------------------------------------------------------
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            seed=self.seed,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            disable_tqdm=True,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorWithPadding(self._tokeniser),
        )
        trainer.train()

        # Clean up temporary scratch directory
        shutil.rmtree(run_output_dir, ignore_errors=True)

        # Move model to eval mode and CPU (if on GPU) to free VRAM between
        # per-project models.
        self._model.eval()
        self._model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def _get_logits(self, texts: Sequence[str]):
        """Tokenise and forward in mini-batches, returning raw logits.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        # Guard before the import, so an unfitted model says so even in an
        # environment where the deep-learning extra is missing.
        if self._model is None or self._tokeniser is None:
            raise RuntimeError("TransformerClassifier must be fitted first.")

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()

        text_list = list(texts)
        eval_batch_size = self.batch_size * 2
        all_logits = []

        with torch.no_grad():
            for i in range(0, len(text_list), eval_batch_size):
                batch_texts = text_list[i : i + eval_batch_size]
                encoded = self._tokeniser(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                outputs = self._model(**encoded)
                all_logits.append(outputs.logits.cpu())

        # Move model back to CPU to save VRAM when done
        self._model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return torch.cat(all_logits, dim=0)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Predict a label for each text.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        logits = self._get_logits(texts)
        indices = logits.argmax(dim=-1).cpu().numpy()
        return np.array([self._id2label[idx] for idx in indices], dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return softmax class probabilities.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        import torch

        logits = self._get_logits(texts)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    @property
    def classes_(self) -> np.ndarray:
        """Label order of the probability columns.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if not self._id2label:
            raise RuntimeError("TransformerClassifier must be fitted first.")
        return np.array(
            [self._id2label[i] for i in range(len(self._id2label))],
            dtype=object,
        )


# ── factory helpers ──────────────────────────────────────────────────────────

#: Preset for the primary hypothesis (DeBERTa-v3-base).
DEFAULT_PRESET: dict = {
    "model_name": DEFAULT_MODEL,
    "max_length": 512,
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-5,
}

#: Cheaper preset for quick iteration on CPU or CI.
FAST_PRESET: dict = {
    "model_name": FAST_MODEL,
    "max_length": 256,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
}


def make_transformer(preset: str = "default", seed: int | None = None, **overrides):
    """Return a factory producing fresh :class:`TransformerClassifier` instances.

    Args:
        preset: ``"default"`` for DeBERTa-v3-base, ``"fast"`` for the
            smaller variant used during development.
        seed: Random seed for reproducibility.
        **overrides: Individual hyper-parameters to override.

    Returns:
        A zero-argument callable suitable for
        :func:`~ai4se.classifiers.base.train_per_repo`.

    Raises:
        ValueError: If ``preset`` is not recognised.
    """
    if preset == "default":
        settings = dict(DEFAULT_PRESET)
    elif preset == "fast":
        settings = dict(FAST_PRESET)
    else:
        raise ValueError(f"Unknown preset {preset!r}; use 'default' or 'fast'.")

    # The explicit `seed` argument wins over one passed in **overrides**;
    # with neither, TransformerClassifier's own default (42) applies.
    settings.update(overrides)
    if seed is not None:
        settings["seed"] = seed
    return lambda: TransformerClassifier(**settings)
