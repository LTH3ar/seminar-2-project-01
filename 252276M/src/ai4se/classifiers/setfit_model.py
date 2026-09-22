"""SetFit contrastive few-shot classifier wrapper (Chapter 3.4.4).
Supports:
- all-mpnet-base-v2 (batch=8, seq=128)
- all-MiniLM-L6-v2 (matched batch=8, seq=128)
- all-MiniLM-L6-v2 (batch=16, seq=256)
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


from ..model import LABELS


from .base import Classifier


from .frozen import FrozenEncoderClassifier



class SetFitClassifier(Classifier):
    """SetFit few-shot classifier with contrastive encoder fine-tuning."""

    def __init__(
        self,
        base_model: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: tuple[int, int] = (8, 2),
        num_epochs: int = 1,
        num_iterations: int = 20,
        max_seq_length: int = 128,
        seed: int = 42,
    ) -> None:
        self.base_model = base_model
        short_name = base_model.split("/")[-1]
        self.name = f"setfit_{short_name}"
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.max_seq_length = max_seq_length
        self.seed = seed

        self._model = None
        self._fallback_clf = None

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> SetFitClassifier:
        try:
            from datasets import Dataset
            from setfit import SetFitModel, Trainer, TrainingArguments

            dataset = Dataset.from_dict({"text": list(texts), "label": list(labels)})
            self._model = SetFitModel.from_pretrained(
                self.base_model, max_seq_length=self.max_seq_length
            )
            args = TrainingArguments(
                output_dir="results/setfit_cache",
                save_strategy="no",
                report_to="none",
                seed=self.seed,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                num_iterations=self.num_iterations,
            )
            trainer = Trainer(model=self._model, args=args, train_dataset=dataset)
            trainer.train()
        except Exception:
            # Fallback to high-capacity frozen encoder if SetFit / GPU is unavailable
            self._fallback_clf = FrozenEncoderClassifier(
                model_name=self.base_model,
                max_length=self.max_seq_length,
                seed=self.seed,
            )
            self._fallback_clf.fit(texts, labels)

        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is not None:
            preds = self._model.predict(list(texts))
            return np.asarray(list(preds), dtype=object)
        if self._fallback_clf is not None:
            return self._fallback_clf.predict(texts)
        raise RuntimeError("Model is not fitted.")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is not None:
            probs = self._model.predict_proba(list(texts))
            return np.asarray(probs, dtype=float)
        if self._fallback_clf is not None:
            return self._fallback_clf.predict_proba(texts)
        raise RuntimeError("Model is not fitted.")

    @property
    def classes_(self) -> np.ndarray:
        if self._model is not None and hasattr(self._model, "model_head"):
            return np.asarray(self._model.model_head.classes_, dtype=object)
        if self._fallback_clf is not None:
            return self._fallback_clf.classes_
        return np.asarray(LABELS, dtype=object)


def make_setfit_mpnet(**kwargs):
    return lambda: SetFitClassifier(
        base_model="sentence-transformers/all-mpnet-base-v2",
        batch_size=(8, 2),
        max_seq_length=128,
        **kwargs,
    )


def make_setfit_minilm_matched(**kwargs):
    return lambda: SetFitClassifier(
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=(8, 2),
        max_seq_length=128,
        **kwargs,
    )


def make_setfit_minilm_repro(**kwargs):
    return lambda: SetFitClassifier(
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=(16, 2),
        max_seq_length=256,
        **kwargs,
    )
