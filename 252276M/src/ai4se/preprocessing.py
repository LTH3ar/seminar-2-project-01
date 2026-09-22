"""Markdown-aware text cleaning, ablation levels, and structural feature extraction.
"""

from __future__ import annotations

import re
import string
from typing import Callable, Literal

# Regex patterns for structural noise
_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_STACK_TRACE = re.compile(
    r"(?:Traceback \(most recent call last\):|at [a-zA-Z0-9_$.]+\([^\)]+\)|File \".*?\", line \d+)"
)
_URL = re.compile(r"https?://\S+|ftp://\S+|www\.\S+")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MARKDOWN_HEADER = re.compile(r"^#+\s+.*$", flags=re.MULTILINE)
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
_COMMIT_HASH = re.compile(r"\b[0-9a-f]{7,40}\b")
_USER_MENTION = re.compile(r"@[a-zA-Z0-9_\-]+")
_ISSUE_REF = re.compile(r"#\d+\b")
_WHITESPACE = re.compile(r"\s+")

#: Domain-specific stop words occurring universally across issue classes
DOMAIN_STOPWORDS: frozenset[str] = frozenset({
    "issue", "reproduce", "expected", "actual", "behavior", "steps",
    "version", "screenshot", "log", "error", "description", "details",
    "using", "please", "thanks", "hello", "hi", "help"
})


def _get_stopwords() -> set[str]:
    """Retrieve combined English and domain stopwords."""
    try:
        from nltk.corpus import stopwords
        words = set(stopwords.words("english"))
    except Exception:
        words = {
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
            "what", "which", "who", "whom", "this", "that", "these", "those", "am",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "a", "an", "the", "and", "but", "if", "or", "because",
            "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
            "between", "into", "through", "during", "before", "after", "above", "below",
            "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "s", "t", "can", "will", "just", "don", "should", "now"
        }
    return words | DOMAIN_STOPWORDS


def _get_lemmatizer():
    """Retrieve NLTK WordNetLemmatizer if available."""
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("testing")
        return lemmatizer
    except Exception:
        return None


def clean_raw(text: str) -> str:
    """Level: raw - whitespace normalisation only."""
    return _WHITESPACE.sub(" ", text).strip()


def clean_light(text: str) -> str:
    """Level: light - strips structural code, stack traces, markup, URLs, mentions."""
    t = _HTML_COMMENT.sub(" ", text)
    t = _FENCED_CODE.sub(" ", t)
    t = _STACK_TRACE.sub(" ", t)
    t = _INLINE_CODE.sub(" ", t)
    t = _MARKDOWN_LINK.sub(r"\1", t)
    t = _MARKDOWN_HEADER.sub(" ", t)
    t = _HTML_TAG.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _COMMIT_HASH.sub(" ", t)
    t = _USER_MENTION.sub(" ", t)
    t = _ISSUE_REF.sub(" ", t)
    return clean_raw(t)


def clean_full(text: str) -> str:
    """Level: full - light + lowercasing, punctuation removal, stop words, lemmatisation."""
    t = clean_light(text).lower()
    # Remove punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    words = t.split()
    stops = _get_stopwords()
    filtered = [w for w in words if w not in stops and len(w) > 1 and not w.isdigit()]

    lemmatizer = _get_lemmatizer()
    if lemmatizer:
        filtered = [lemmatizer.lemmatize(w) for w in filtered]

    return " ".join(filtered)


_CLEANERS = {"raw": clean_raw, "light": clean_light, "full": clean_full}


def clean_text(text: str, level: Literal["raw", "light", "full"] = "light") -> str:
    """Clean a single string at the given level."""
    return _CLEANERS[level](text)


def make_cleaner(
    level: Literal["raw", "light", "full"] = "light",
    title_weight: int = 3,
    max_words: int | None = 200,
) -> Callable[[str, str], str]:
    """Return a cleaner callable that processes (title, body) into final model input.

    Args:
        level: Cleaning strategy ('raw', 'light', 'full').
        title_weight: How many times to repeat the title.
        max_words: Maximum number of words to keep (None for no truncation).
    """
    clean_fn = _CLEANERS[level]

    def cleaner(title: str, body: str) -> str:
        c_title = clean_fn(title)
        c_body = clean_fn(body)
        weighted_title = " ".join([c_title] * title_weight) if title_weight > 0 else ""
        combined = f"{weighted_title} {c_body}".strip()
        if max_words is not None and max_words > 0:
            words = combined.split()
            if len(words) > max_words:
                combined = " ".join(words[:max_words])
        return combined

    return cleaner


def extract_structural_features(title: str, body: str) -> list[int]:
    """Extract 15 non-prose boolean structural flags (Section 2.3 in report).

    Reaches ~0.556 F1 without using any words.
    """
    has_code_block = int(bool(_FENCED_CODE.search(body)))
    has_stack_trace = int(bool(_STACK_TRACE.search(body)))
    has_url = int(bool(_URL.search(body)))
    has_image = int(bool(re.search(r"!\[.*?\]\(.*?\)|<img", body)))
    has_html_comment = int(bool(_HTML_COMMENT.search(body)))
    has_template_heading = int(bool(re.search(r"##\s+(?:Expected|Steps|Actual|Description|Environment)", body, re.I)))
    has_question_in_title = int("?" in title)
    title_starts_with_how = int(bool(re.match(r"^(?:how|why|where|can|is|does)\b", title.strip(), re.I)))
    has_inline_code = int(bool(_INLINE_CODE.search(body)))
    has_commit_hash = int(bool(_COMMIT_HASH.search(body)))
    has_user_mention = int(bool(_USER_MENTION.search(body)))
    has_issue_ref = int(bool(_ISSUE_REF.search(body)))
    has_bug_in_title = int(bool(re.search(r"\b(?:bug|fix|error|crash|fail)\b", title, re.I)))
    has_feat_in_title = int(bool(re.search(r"\b(?:feat|feature|add|request|support|allow)\b", title, re.I)))
    has_version = int(bool(re.search(r"\bv?\d+\.\d+(?:\.\d+)?\b", body)))

    return [
        has_code_block,
        has_stack_trace,
        has_url,
        has_image,
        has_html_comment,
        has_template_heading,
        has_question_in_title,
        title_starts_with_how,
        has_inline_code,
        has_commit_hash,
        has_user_mention,
        has_issue_ref,
        has_bug_in_title,
        has_feat_in_title,
        has_version,
    ]
