"""Configurable text cleaning and structural feature extraction."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from issue_classifier.domain import IssueReport, PreparedIssue


VALID_CLEANING_LEVELS = frozenset({"conservative", "raw", "light", "full"})

_FENCED_CODE = re.compile(r"\x60\x60\x60.*?\x60\x60\x60", flags=re.DOTALL)
_INDENTED_CODE = re.compile(r"(?m)^(?: {4}|\t).*$")
_INLINE_CODE = re.compile(r"\x60[^\x60\n]+\x60")
_HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_HTML_TAG = re.compile(r"<[^>\n]{1,120}>")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TEMPLATE_HEADING = re.compile(r"(?m)^#{1,6}\s.*$")
_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_ISSUE_REFERENCE = re.compile(r"(?<![\w])#\d+\b")
_MENTION = re.compile(r"(?<![\w])@[\w-]+")
_SHA = re.compile(r"\b[0-9a-f]{7,40}\b", flags=re.IGNORECASE)
_VERSION = re.compile(r"\bv?\d+(?:\.\d+){1,3}(?:-[\w.]+)?\b")
_PATH = re.compile(r"(?:[\w.-]+[/\\]){1,}[\w.-]+")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\b\w+\b", flags=re.UNICODE)
_NON_ALPHABETIC = re.compile(r"[^a-z\s]")
_CHECKBOX = re.compile(r"(?:^|\s)-\s*\[[ xX]\]")
_STACK_TRACE = re.compile(
    r"(?:traceback \(most recent call last\)|\bat\s+[\w.$]+\([^\n]+:\d+\))",
    flags=re.IGNORECASE,
)
_STACK_FRAME = re.compile(
    r"(?m)^\s*(?:at\s+\S+|File \"[^\"]+\", line \d+|Traceback \(most recent).*$"
)
_ERROR_WORD = re.compile(
    r"\b(?:bug|crash|error|exception|fail(?:ed|ure)?|incorrect|broken)\b",
    flags=re.IGNORECASE,
)
_REQUEST_WORD = re.compile(
    r"\b(?:feature|proposal|request|support|enhancement|would like)\b",
    flags=re.IGNORECASE,
)
_QUESTION_WORD = re.compile(
    r"\b(?:how|why|what|where|when|can|could|does|is it possible)\b",
    flags=re.IGNORECASE,
)

_DOMAIN_STOPWORDS = frozenset(
    {
        "issue",
        "github",
        "repo",
        "repository",
        "please",
        "thanks",
        "thank",
        "using",
        "use",
        "used",
        "code",
        "line",
        "file",
        "version",
        "description",
        "steps",
        "expected",
        "actual",
        "screenshot",
        "log",
        "output",
        "example",
    }
)
_FALLBACK_STOPWORDS = frozenset(
    "a an the and or but if then else of to in on at for with without is are "
    "was were be been being do does did have has had i you he she it we they "
    "this that these those not no so as by from can could will would".split()
)


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except (ImportError, LookupError):
        return _FALLBACK_STOPWORDS


@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")
        return lemmatizer
    except (ImportError, LookupError):
        return None


@dataclass(frozen=True, slots=True)
class TextCleaningConfig:
    level: str = "conservative"
    replace_code_blocks: bool = True
    replace_urls: bool = True
    lowercase: bool = False
    title_body_separator: str = " [SEP] "
    max_words: int | None = None
    title_weight: int = 1

    def __post_init__(self) -> None:
        if self.level not in VALID_CLEANING_LEVELS:
            known = ", ".join(sorted(VALID_CLEANING_LEVELS))
            raise ValueError(
                f"Unknown cleaning level {self.level!r}; expected: {known}"
            )
        if self.max_words is not None and self.max_words < 1:
            raise ValueError("max_words must be positive")
        if self.title_weight < 1:
            raise ValueError("title_weight must be positive")


class TextPreprocessor:
    def __init__(self, config: TextCleaningConfig | None = None) -> None:
        self.config = config or TextCleaningConfig()

    @staticmethod
    def _normalise(value: str) -> str:
        value = html.unescape(value or "")
        value = unicodedata.normalize("NFKC", value)
        return value.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _strip_code(value: str) -> str:
        value = _FENCED_CODE.sub(" ", value)
        value = _STACK_FRAME.sub(" ", value)
        value = _INDENTED_CODE.sub(" ", value)
        return _INLINE_CODE.sub(" ", value)

    @staticmethod
    def _strip_markup(value: str) -> str:
        value = _HTML_COMMENT.sub(" ", value)
        value = _MARKDOWN_IMAGE.sub(" ", value)
        value = _MARKDOWN_LINK.sub(r"\1", value)
        value = _TEMPLATE_HEADING.sub(" ", value)
        return _HTML_TAG.sub(" ", value)

    @staticmethod
    def _strip_identifiers(value: str) -> str:
        for pattern in (
            _URL,
            _EMAIL,
            _SHA,
            _VERSION,
            _PATH,
            _ISSUE_REFERENCE,
            _MENTION,
        ):
            value = pattern.sub(" ", value)
        return value

    def clean(self, value: str) -> str:
        value = self._normalise(value)
        level = self.config.level

        if level == "conservative":
            if self.config.replace_code_blocks:
                value = _FENCED_CODE.sub(" <CODE_BLOCK> ", value)
            if self.config.replace_urls:
                value = _URL.sub(" <URL> ", value)
            value = _WHITESPACE.sub(" ", value).strip()
        elif level == "raw":
            value = _WHITESPACE.sub(" ", value).strip()
        else:
            value = self._strip_identifiers(
                self._strip_markup(self._strip_code(value))
            )
            value = _WHITESPACE.sub(" ", value).strip()
            if level == "full":
                tokens = _NON_ALPHABETIC.sub(" ", value.lower()).split()
                blocked = _english_stopwords() | _DOMAIN_STOPWORDS
                tokens = [
                    token
                    for token in tokens
                    if len(token) >= 2 and token not in blocked
                ]
                lemmatizer = _lemmatizer()
                if lemmatizer is not None:
                    tokens = [lemmatizer.lemmatize(token) for token in tokens]
                value = " ".join(tokens)

        if self.config.lowercase and level != "full":
            value = value.lower()
        if self.config.max_words is not None:
            value = " ".join(value.split()[: self.config.max_words])
        return value

    def prepare(self, issue: IssueReport) -> PreparedIssue:
        cleaned_title = self.clean(issue.title)
        cleaned_body = self.clean(issue.body)
        text_parts = [
            part
            for part in (
                *([cleaned_title] * self.config.title_weight),
                cleaned_body,
            )
            if part
        ]
        text = self.config.title_body_separator.join(text_parts)
        if self.config.max_words is not None:
            text = " ".join(text.split()[: self.config.max_words])
        return PreparedIssue(
            issue=issue,
            cleaned_title=cleaned_title,
            cleaned_body=cleaned_body,
            text=text,
            features=self.structural_features(issue),
        )

    def prepare_many(self, issues: list[IssueReport]) -> list[PreparedIssue]:
        return [self.prepare(issue) for issue in issues]

    @staticmethod
    def structural_features(issue: IssueReport) -> dict[str, float]:
        raw_text = f"{issue.title}\n{issue.body}"
        title_words = _WORD.findall(issue.title)
        body_words = _WORD.findall(issue.body)
        return {
            "title_char_count": float(len(issue.title)),
            "body_char_count": float(len(issue.body)),
            "title_word_count": float(len(title_words)),
            "body_word_count": float(len(body_words)),
            "body_line_count": float(len(issue.body.splitlines())),
            "question_mark_count": float(raw_text.count("?")),
            "url_count": float(len(_URL.findall(raw_text))),
            "code_block_count": float(raw_text.count("\x60\x60\x60") // 2),
            "checkbox_count": float(len(_CHECKBOX.findall(raw_text))),
            "has_stack_trace": float(bool(_STACK_TRACE.search(raw_text))),
            "has_error_keyword": float(bool(_ERROR_WORD.search(raw_text))),
            "has_request_keyword": float(bool(_REQUEST_WORD.search(raw_text))),
            "has_question_keyword": float(bool(_QUESTION_WORD.search(raw_text))),
            "body_is_empty": float(not issue.body.strip()),
        }
