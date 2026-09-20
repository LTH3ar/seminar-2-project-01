"""Reference classifiers with no third-party dependencies.

These exist for two reasons. First, they make the evaluation harness runnable
before tracks B and C deliver anything, so the protocol can be validated
independently of any real model. Second, they establish the floor the report
needs: a result is only meaningful against a trivial alternative, and "our
SVM reaches 0.79" means little until the reader knows that always predicting
one class reaches 0.17.

All four satisfy the ``fit``/``predict`` interface expected by
:mod:`ai4se.evaluation`, so they slot into exactly the same runners as a
scikit-learn pipeline or a fine-tuned transformer.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from .evaluation import RANDOM_SEED

TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    return TOKEN.findall(text.lower())


class MajorityClassifier:
    """Always predicts the most frequent class in the training data.

    The absolute floor. On a balanced three-class dataset it reaches roughly
    0.33 accuracy but only about 0.17 macro F1, because two of the three
    classes score zero. The gap between those two numbers is itself a useful
    illustration for the report of why accuracy alone is a poor metric.
    """

    def __init__(self) -> None:
        """Create an untrained classifier."""
        self.majority_: str | None = None

    def fit(self, X: Sequence[str], y: Sequence[str]) -> MajorityClassifier:
        """Record the most frequent label; the texts are ignored."""
        self.majority_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X: Sequence[str]) -> list[str]:
        """Return the majority label for every input."""
        if self.majority_ is None:
            raise RuntimeError("Call fit() before predict().")
        return [self.majority_] * len(X)


class StratifiedRandomClassifier:
    """Predicts randomly, in proportion to the training class frequencies.

    A stronger floor than the majority classifier: it scores on every class,
    so it shows what macro F1 a model achieves by chance alone.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        """Create an untrained classifier with a fixed seed."""
        self.seed = seed
        self.labels_: list[str] = []
        self.weights_: list[int] = []

    def fit(self, X: Sequence[str], y: Sequence[str]) -> StratifiedRandomClassifier:
        """Record the class distribution; the texts are ignored."""
        counts = Counter(y)
        self.labels_ = sorted(counts)
        self.weights_ = [counts[label] for label in self.labels_]
        return self

    def predict(self, X: Sequence[str]) -> list[str]:
        """Draw one label per input from the training distribution."""
        if not self.labels_:
            raise RuntimeError("Call fit() before predict().")
        rng = random.Random(self.seed)
        return rng.choices(self.labels_, weights=self.weights_, k=len(X))


class KeywordClassifier:
    """Hand-written rules over a few discriminative cue words.

    Included because it is the approach a developer would reach for without
    machine learning, and because beating it is the minimum bar any trained
    model must clear before the effort is justified. The cues come from the
    ``distinctive_terms`` analysis in the Track A notebook.
    """

    RULES: dict[str, tuple[str, ...]] = {
        "bug": (
            "bug",
            "crash",
            "error",
            "fail",
            "failure",
            "broken",
            "exception",
            "traceback",
            "regression",
            "wrong",
            "incorrect",
            "unexpected",
            "reproduce",
            "stacktrace",
            "segfault",
        ),
        "feature": (
            "feature",
            "request",
            "add",
            "support",
            "enhancement",
            "allow",
            "would",
            "nice",
            "propose",
            "proposal",
            "suggestion",
            "improve",
            "ability",
            "please",
            "consider",
        ),
        "question": (
            "how",
            "why",
            "question",
            "possible",
            "anyone",
            "help",
            "doubt",
            "confused",
            "understand",
            "documentation",
            "clarify",
            "what",
            "does",
            "explain",
        ),
    }

    def __init__(self) -> None:
        """Create the classifier; the rules are fixed, so fit only sets a fallback."""
        self.fallback_: str = "bug"

    def fit(self, X: Sequence[str], y: Sequence[str]) -> KeywordClassifier:
        """Record the majority class, used when no rule matches."""
        if len(y):
            self.fallback_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X: Sequence[str]) -> list[str]:
        """Assign the class whose cue words appear most often in each text."""
        predictions = []
        for text in X:
            tokens = set(_tokenize(text))
            hits = {
                label: len(tokens & set(cues)) for label, cues in self.RULES.items()
            }
            best = max(hits, key=hits.get)
            predictions.append(best if hits[best] > 0 else self.fallback_)
        return predictions


class MultinomialNaiveBayes:
    """Multinomial naive Bayes over word counts, implemented from scratch.

    A genuine learned baseline with no third-party dependency, so the
    evaluation harness can be exercised end to end before track B delivers its
    scikit-learn pipelines. The course introduced naive Bayes as a
    probabilistic algorithm based on Bayes' theorem commonly used for text
    classification, which is exactly the role it plays here.

    Probabilities are accumulated as logs to avoid underflow, and Laplace
    smoothing handles words unseen for a class.

    Args:
        alpha: Laplace smoothing constant added to every word count.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Create an untrained classifier."""
        self.alpha = alpha
        self.log_prior_: dict[str, float] = {}
        self.log_likelihood_: dict[str, dict[str, float]] = {}
        self.log_unseen_: dict[str, float] = {}
        self.vocabulary_: set[str] = set()

    def fit(self, X: Sequence[str], y: Sequence[str]) -> MultinomialNaiveBayes:
        """Estimate class priors and per-class word likelihoods."""
        counts_by_class: dict[str, Counter] = defaultdict(Counter)
        documents_by_class: Counter = Counter()

        for text, label in zip(X, y, strict=True):
            tokens = _tokenize(text)
            counts_by_class[label].update(tokens)
            documents_by_class[label] += 1
            self.vocabulary_.update(tokens)

        total_documents = sum(documents_by_class.values()) or 1
        vocabulary_size = len(self.vocabulary_) or 1

        for label, counts in counts_by_class.items():
            total_words = sum(counts.values())
            denominator = total_words + self.alpha * vocabulary_size
            self.log_prior_[label] = math.log(
                documents_by_class[label] / total_documents
            )
            self.log_likelihood_[label] = {
                word: math.log((count + self.alpha) / denominator)
                for word, count in counts.items()
            }
            # Likelihood of a vocabulary word this class never saw.
            self.log_unseen_[label] = math.log(self.alpha / denominator)
        return self

    def _log_posteriors(self, text: str) -> dict[str, float]:
        """Unnormalised log posterior of every class for one text."""
        tokens = [t for t in _tokenize(text) if t in self.vocabulary_]
        return {
            label: log_prior
            + sum(
                self.log_likelihood_[label].get(t, self.log_unseen_[label])
                for t in tokens
            )
            for label, log_prior in self.log_prior_.items()
        }

    def predict(self, X: Sequence[str]) -> list[str]:
        """Return the maximum a posteriori class for each text."""
        if not self.log_prior_:
            raise RuntimeError("Call fit() before predict().")
        return [
            max(self._log_posteriors(text).items(), key=lambda kv: kv[1])[0]
            for text in X
        ]

    def predict_proba(self, X: Sequence[str]) -> list[dict[str, float]]:
        """Return normalised class probabilities, so ROC and AUC can be computed.

        The log posteriors are converted with a numerically stable softmax:
        the maximum is subtracted before exponentiating, because a document of
        a few hundred tokens produces log scores far below the underflow point
        of ``exp``.
        """
        if not self.log_prior_:
            raise RuntimeError("Call fit() before predict_proba().")

        probabilities = []
        for text in X:
            posteriors = self._log_posteriors(text)
            highest = max(posteriors.values())
            exponentiated = {
                label: math.exp(value - highest) for label, value in posteriors.items()
            }
            total = sum(exponentiated.values()) or 1.0
            probabilities.append({k: v / total for k, v in exponentiated.items()})
        return probabilities


#: Factories for every reference model, ready to hand to the evaluation runners.
REFERENCE_MODELS: dict[str, type] = {
    "majority": MajorityClassifier,
    "random (stratified)": StratifiedRandomClassifier,
    "keyword rules": KeywordClassifier,
    "naive Bayes (from scratch)": MultinomialNaiveBayes,
}
