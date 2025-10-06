"""Text processing helpers shared across source implementations."""

from __future__ import annotations

import re
from typing import Iterable


_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b")


def tokenize_casefold(text: str) -> set[str]:
    """Return a casefolded token set for exact token matching."""

    if not text:
        return set()
    return set(_TOKEN_PATTERN.findall(text.casefold()))


def build_term_pattern(terms: Iterable[str]) -> re.Pattern:
    """Compile a case-insensitive regex that matches any of the supplied terms."""

    filtered_terms = [term for term in terms if term]
    if not filtered_terms:
        return re.compile(r"(?!)")  # Always-false pattern

    unique_terms = sorted(set(filtered_terms), key=len, reverse=True)
    escaped_terms = [re.escape(term) for term in unique_terms]
    return re.compile(r"(?:{})".format("|".join(escaped_terms)), flags=re.IGNORECASE)
