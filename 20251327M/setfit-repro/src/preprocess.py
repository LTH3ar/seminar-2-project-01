"""Text variants for the preprocessing lever.

The error analysis in the team's report found the classifier keying on issue
templates rather than on content: OpenCV's forum-referral line appears at
15.6x lift inside one specific confusion. That is boilerplate every new issue
in that project receives, so it identifies the project, not the class -- and
the per-project protocol lets each classifier overfit to its own project's
conventions.

The counter-evidence is that structural noise is *class-correlated*: code
blocks appear in 57.6% of bug reports against 34.8% of feature requests. So
stripping is not obviously good, and the team's own ablation found the most
aggressive cleaning actively hurt. Hence variants rather than a decision, each
selected on validation data.

Every function here is ``str -> str`` so variants compose and can be disabled
one at a time.
"""

from __future__ import annotations

import re
from collections.abc import Callable

#: Boilerplate lines GitHub issue templates insert, which say nothing about
#: the class of the issue. Matched case-insensitively, line by line.
TEMPLATE_LINES = re.compile(
    r"(?im)^\s*(?:"
    r"(?:please\s+)?(?:use|ask|post).{0,40}"
    r"(?:forum\.opencv\.org|stackoverflow|stack\s*overflow|discussions?)"
    r"|<!--.*?-->"
    r"|#{1,6}\s*(?:steps?\s+to\s+reproduce|expected\s+behaviou?r"
    r"|actual\s+behaviou?r|system\s+information|describe\s+the\s+bug"
    r"|to\s+reproduce|additional\s+context|checklist|environment)\s*:?\s*"
    r"|\*\*(?:steps?\s+to\s+reproduce|expected|actual|behaviou?r)\*\*:?\s*"
    r"|-\s*\[[ xX]\]\s*I\s+(?:have|checked|read|searched).*"
    r")\s*$",
    re.DOTALL,
)

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
STACK_TRACE = re.compile(r"(?m)^\s*(?:at\s+[\w$.]+\(|File\s+\"|Traceback).*$")
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|\S+\.(?:png|jpe?g|gif)\b", re.IGNORECASE)
URL = re.compile(r"https?://\S+")
WHITESPACE = re.compile(r"\s+")


def identity(text: str) -> str:
    """Return the text unchanged -- the control condition."""
    return text


def strip_templates(text: str) -> str:
    """Remove issue-template boilerplate, keeping everything the author wrote.

    This is the targeted fix for the defect the error analysis identified. It
    deliberately does *not* touch code blocks or stack traces, because those
    carry class signal.
    """
    text = HTML_COMMENT.sub(" ", text)
    text = TEMPLATE_LINES.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def mask_structure(text: str) -> str:
    """Replace structural blocks with typed placeholders.

    Deleting a code block removes the evidence that the issue contained one,
    which is itself predictive. A placeholder keeps that bit and discards the
    vocabulary inside, which is mostly project-specific.
    """
    text = HTML_COMMENT.sub(" ", text)
    text = STACK_TRACE.sub(" xxtrace ", text)
    text = FENCED_CODE.sub(" xxcode ", text)
    text = IMAGE.sub(" xximg ", text)
    text = URL.sub(" xxurl ", text)
    return WHITESPACE.sub(" ", text).strip()


def strip_templates_and_mask(text: str) -> str:
    """Both of the above: remove boilerplate, then type the remaining blocks."""
    return mask_structure(strip_templates(text))


#: Variants offered to the selection stage. ``identity`` must stay first so a
#: results table always shows the control it is being compared against.
VARIANTS: dict[str, Callable[[str], str]] = {
    "raw": identity,
    "strip_templates": strip_templates,
    "mask_structure": mask_structure,
    "strip_and_mask": strip_templates_and_mask,
}


def apply(frame, variant: str):
    """Return a copy of ``frame`` with ``text`` rewritten by ``variant``.

    Args:
        frame: A DataFrame carrying a ``text`` column.
        variant: A key of :data:`VARIANTS`.

    Returns:
        A new DataFrame; the input is not modified.

    Raises:
        KeyError: If ``variant`` is not a known variant name.
    """
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}; use one of {sorted(VARIANTS)}.")
    out = frame.copy()
    out["text"] = out.text.map(VARIANTS[variant])
    return out
