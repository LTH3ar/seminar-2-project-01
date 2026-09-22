"""Reference baseline floors (Chapter 3.4.1 in report).
Models with no external dependencies establishing what costs no effort:
1. Majority class
2. Stratified random
3. Keyword rules
4. Multinomial Naive Bayes from scratch
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
import numpy as np


from ..model import LABELS
from .base import Classifier



class MajorityClassifier(Classifier):
    """Always predicts the most frequent training class."""

    name: str = "majority_class"

    def __init__(self) -> None:
        self.majority_label: str = LABELS[0]

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> MajorityClassifier:
        counts = Counter(labels)
        self.majority_label = counts.most_common(1)[0][0]
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self.majority_label] * len(texts), dtype=object)


class StratifiedRandomClassifier(Classifier):
    """Draws class predictions according to empirical class priors."""

    name: str = "random_stratified"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.priors: np.ndarray = np.ones(len(LABELS)) / len(LABELS)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> StratifiedRandomClassifier:
        counts = Counter(labels)
        total = len(labels)
        self.priors = np.array([counts.get(lbl, 0) / total for lbl in LABELS])
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(len(LABELS), size=len(texts), p=self.priors)
        return np.array([LABELS[i] for i in indices], dtype=object)


class KeywordRulesClassifier(Classifier):
    """Heuristic rule-based classifier using cue words from Section 2.4 term analysis."""

    name: str = "keyword_rules"

    # Distinctive words identified by document frequency lift
    BUG_WORDS = {
        "bug", "crash", "error", "fail", "failure", "broken", "traceback",
        "exception", "panic", "segmentation", "segfault", "cudatoolkit",
        "onednn", "geforce", "rebuild", "critical", "issue", "nullpointer"
    }
    FEATURE_WORDS = {
        "feature", "support", "add", "allow", "request", "please", "enhance",
        "enhancement", "willing", "nice", "considered", "easily", "approach",
        "great", "proposal", "option", "implement", "provide"
    }
    QUESTION_WORDS = {
        "question", "how", "why", "what", "where", "can", "is", "help",
        "forum", "overflow", "explain", "understand", "usage", "liveserver",
        "canvas", "argv", "reporter"
    }

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> KeywordRulesClassifier:
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        predictions = []
        for text in texts:
            words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
            bug_score = len(words & self.BUG_WORDS)
            feat_score = len(words & self.FEATURE_WORDS)
            quest_score = len(words & self.QUESTION_WORDS)

            scores = [
                (bug_score, "bug"),
                (feat_score, "feature"),
                (quest_score, "question"),
            ]
            scores.sort(key=lambda s: s[0], reverse=True)
            if scores[0][0] > 0:
                predictions.append(scores[0][1])
            else:
                predictions.append("bug")  # default fallback
        return np.array(predictions, dtype=object)


class ScratchNaiveBayesClassifier(Classifier):
    """Multinomial Naive Bayes implemented from scratch in log space with Laplace smoothing."""

    name: str = "naive_bayes_scratch"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_priors: dict[str, float] = {}
        self.vocab: set[str] = set()
        self.word_log_probs: dict[str, dict[str, float]] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9_]{2,}\b", text.lower())

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> ScratchNaiveBayesClassifier:
        n_docs = len(texts)
        label_counts = Counter(labels)
        self.class_priors = {lbl: math.log(cnt / n_docs) for lbl, cnt in label_counts.items()}

        word_counts_by_class: dict[str, Counter[str]] = defaultdict(Counter)
        total_words_by_class: dict[str, int] = defaultdict(int)

        self.vocab = set()
        for text, lbl in zip(texts, labels, strict=True):
            tokens = self._tokenize(text)
            for tok in tokens:
                self.vocab.add(tok)
                word_counts_by_class[lbl][tok] += 1
                total_words_by_class[lbl] += 1

        v_size = len(self.vocab)
        self.word_log_probs = {}
        for lbl in LABELS:
            denom = total_words_by_class[lbl] + self.alpha * v_size
            self.word_log_probs[lbl] = {
                w: math.log((word_counts_by_class[lbl][w] + self.alpha) / denom)
                for w in self.vocab
            }
            # Log prob for unseen words
            self.word_log_probs[lbl]["<UNK>"] = math.log(self.alpha / denom)

        return self

    def _log_posteriors(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        scores = {}
        for lbl in LABELS:
            score = self.class_priors.get(lbl, math.log(1.0 / len(LABELS)))
            unk_prob = self.word_log_probs[lbl].get("<UNK>", -10.0)
            for tok in tokens:
                score += self.word_log_probs[lbl].get(tok, unk_prob)
            scores[lbl] = score
        return scores

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        preds = []
        for text in texts:
            scores = self._log_posteriors(text)
            best_lbl = max(scores.items(), key=lambda kv: kv[1])[0]
            preds.append(best_lbl)
        return np.array(preds, dtype=object)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        probs = np.zeros((len(texts), len(LABELS)), dtype=float)
        for i, text in enumerate(texts):
            scores = self._log_posteriors(text)
            vals = np.array([scores[lbl] for lbl in LABELS])
            # Softmax with max subtraction for numerical stability
            exp_vals = np.exp(vals - np.max(vals))
            probs[i] = exp_vals / exp_vals.sum()
        return probs

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(LABELS, dtype=object)
