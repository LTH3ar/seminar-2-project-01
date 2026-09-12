"""Text cleaning pipeline for GitHub issue reports.

Issue bodies are Markdown documents that contain a large amount of material
which is not natural language: fenced code blocks, stack traces, URLs,
images, HTML comments and the boilerplate headings injected by GitHub issue
templates (``### Repro steps``, ``### Expected behaviour`` and so on). Feeding
that noise to a bag-of-words model dilutes the vocabulary, so it is removed
before vectorisation.

The pipeline follows the sequence presented in the course
(Feed-Forward Neural Networks, "Spam Detection: Data Processing"):
parse the text, remove punctuation and stop words, lemmatise, then vectorise.
The vectorisation step itself belongs to the modelling track, not here.

Three levels are offered so that the ablation study required by the report can
be run by changing a single string:

    ``"raw"``    title + body untouched (control condition)
    ``"light"``  structural noise removed, casing and words preserved
                 (the right input for transformer models)
    ``"full"``   light + lowercasing, punctuation removal, stop words,
                 lemmatisation (the right input for TF-IDF models)
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache
from typing import Callable, Iterable

from .model import IssueReport

# --------------------------------------------------------------------------- #
# Regular expressions, compiled once at import time.
# Order matters: fenced code blocks are stripped before inline code, and URLs
# before generic punctuation, otherwise the patterns interfere with each other.
# --------------------------------------------------------------------------- #

RE_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
RE_INDENTED_CODE = re.compile(r"(?m)^(?: {4}|\t).*$")
RE_INLINE_CODE = re.compile(r"`[^`\n]+`")
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_HTML_TAG = re.compile(r"<[^>\n]{1,120}>")
RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
RE_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
RE_TEMPLATE_HEADING = re.compile(r"(?m)^#{1,6}\s.*$")
RE_ISSUE_REF = re.compile(r"(?<![\w])#\d+\b")
RE_MENTION = re.compile(r"(?<![\w])@[\w-]+")
RE_SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
RE_VERSION = re.compile(r"\bv?\d+(?:\.\d+){1,3}(?:-[\w.]+)?\b")
RE_PATH = re.compile(r"(?:[\w.-]+[/\\]){1,}[\w.-]+")
RE_STACK_FRAME = re.compile(
    r"(?m)^\s*(?:at\s+\S+|File \"[^\"]+\", line \d+|Traceback \(most recent).*$"
)
RE_NON_WORD = re.compile(r"[^a-z\s]")
RE_WHITESPACE = re.compile(r"\s+")

#: Domain stop words that appear in nearly every issue regardless of its class
#: and therefore carry no discriminative signal.
DOMAIN_STOPWORDS = frozenset(
    {
        "issue", "github", "repo", "repository", "please", "thanks", "thank",
        "hi", "hello", "would", "could", "also", "using", "use", "used",
        "code", "line", "file", "version", "system", "info", "information",
        "description", "describe", "steps", "step", "reproduce", "expected",
        "actual", "behaviour", "behavior", "current", "screenshot",
        "screenshots", "log", "logs", "output", "example", "checklist",
    }
)


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    """Load NLTK's English stop word list, downloading it on first use.

    Falls back to a small built-in list if NLTK data cannot be fetched, so the
    pipeline still runs in an offline environment.
    """
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            words = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = stopwords.words("english")
        return frozenset(words)
    except Exception:  # pragma: no cover - offline fallback
        return frozenset(
            "a an the and or but if then else of to in on at for with without "
            "is are was were be been being do does did have has had i you he "
            "she it we they this that these those not no so as by from up down "
            "out over under again further once here there all any both each "
            "few more most other some such only own same than too very can "
            "will just should now".split()
        )


@lru_cache(maxsize=1)
def _lemmatizer():
    """Return an NLTK WordNet lemmatiser, or ``None`` if unavailable."""
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        try:
            lemmatizer.lemmatize("tests")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            lemmatizer.lemmatize("tests")
        return lemmatizer
    except Exception:  # pragma: no cover - offline fallback
        return None


# --------------------------------------------------------------------------- #
# Individual cleaning steps. Each is a pure str -> str function so the pipeline
# can be reordered, extended or partially disabled for the ablation study.
# --------------------------------------------------------------------------- #


def strip_code(text: str) -> str:
    """Remove fenced, indented and inline code as well as stack traces."""
    text = RE_FENCED_CODE.sub(" ", text)
    text = RE_STACK_FRAME.sub(" ", text)
    text = RE_INDENTED_CODE.sub(" ", text)
    text = RE_INLINE_CODE.sub(" ", text)
    return text


def strip_markup(text: str) -> str:
    """Remove HTML comments/tags, Markdown images, links and template headings."""
    text = RE_HTML_COMMENT.sub(" ", text)
    text = RE_MD_IMAGE.sub(" ", text)
    text = RE_MD_LINK.sub(r"\1", text)  # keep the anchor text, drop the target
    text = RE_TEMPLATE_HEADING.sub(" ", text)
    text = RE_HTML_TAG.sub(" ", text)
    return text


def strip_identifiers(text: str) -> str:
    """Remove URLs, e-mails, commit hashes, versions, paths, refs and mentions."""
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_SHA.sub(" ", text)
    text = RE_VERSION.sub(" ", text)
    text = RE_PATH.sub(" ", text)
    text = RE_ISSUE_REF.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    return text


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim."""
    return RE_WHITESPACE.sub(" ", text).strip()


def to_tokens(text: str) -> list[str]:
    """Lowercase, drop non-alphabetic characters and split into tokens."""
    return RE_NON_WORD.sub(" ", text.lower()).split()


def remove_stopwords(
    tokens: Iterable[str],
    extra: frozenset[str] = DOMAIN_STOPWORDS,
    min_length: int = 2,
) -> list[str]:
    """Drop English stop words, domain stop words and very short tokens."""
    blocked = _english_stopwords() | extra
    return [t for t in tokens if len(t) >= min_length and t not in blocked]


def lemmatize(tokens: Iterable[str]) -> list[str]:
    """Reduce tokens to their dictionary form where WordNet is available."""
    lemmatizer = _lemmatizer()
    if lemmatizer is None:
        return list(tokens)
    return [lemmatizer.lemmatize(t) for t in tokens]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def clean_text(
    text: str,
    level: str = "full",
    max_words: int | None = None,
) -> str:
    """Run the cleaning pipeline over a single string.

    Args:
        text: Raw title + body.
        level: ``"raw"``, ``"light"`` or ``"full"`` (see module docstring).
        max_words: Truncate the result to this many tokens. Useful because the
            longest issue in the training set exceeds 21,000 words while the
            median is around 150, and transformer encoders accept 512 tokens.

    Returns:
        The cleaned text.
    """
    if level not in {"raw", "light", "full"}:
        raise ValueError(f"Unknown cleaning level: {level!r}")

    if level == "raw":
        result = normalise_whitespace(text)
    else:
        stripped = normalise_whitespace(
            strip_identifiers(strip_markup(strip_code(text)))
        )
        if level == "light":
            result = stripped
        else:
            tokens = lemmatize(remove_stopwords(to_tokens(stripped)))
            result = " ".join(tokens)

    if max_words is not None:
        result = " ".join(result.split()[:max_words])
    return result


def clean_issue(
    issue: IssueReport,
    level: str = "full",
    max_words: int | None = None,
    use_title: bool = True,
    title_weight: int = 1,
) -> IssueReport:
    """Return a copy of ``issue`` with ``clean_text`` populated.

    Args:
        issue: The issue to preprocess.
        level: Cleaning level passed to :func:`clean_text`.
        max_words: Optional truncation length.
        use_title: Include the title in the classifier input.
        title_weight: Repeat the title this many times. Titles are short and
            highly informative for issue typing, so repeating them is a cheap
            way of up-weighting them in a bag-of-words representation.

    Returns:
        A new :class:`~ai4se.model.IssueReport`; the input is not mutated.
    """
    parts: list[str] = []
    if use_title:
        parts.extend([issue.title] * max(1, title_weight))
    parts.append(issue.body)
    combined = "\n".join(parts)
    return replace(issue, clean_text=clean_text(combined, level, max_words))


def make_cleaner(**kwargs) -> Callable[[IssueReport], IssueReport]:
    """Build a one-argument cleaning function for ``IssueRepository.apply``.

    Example:
        >>> train.apply(make_cleaner(level="full", max_words=400))
    """

    def _cleaner(issue: IssueReport) -> IssueReport:
        return clean_issue(issue, **kwargs)

    return _cleaner
