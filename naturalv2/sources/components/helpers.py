"""Helpers shared across source stages.

Utilities in this module support common operations needed by curation stages,
including tokenization for exact-match filtering, building case-insensitive
regex patterns from term lists, and filtering tabular data by a date cutoff.
"""

import logging
import re
from typing import Iterable

import pandas as pd


logger = logging.getLogger(__name__)


# Match runs of unicode word characters and hyphens (letters, digits, underscore, hyphen)
_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b", flags=re.UNICODE)


def tokenize_casefold(text: str) -> set[str]:
    """Return a casefolded token set for exact token matching.

    Parameters
    ----------
    text : str
        Input text to split into tokens. Tokens are sequences of word
        characters or hyphens (regex ``\b[\w-]+\b``), matched case-insensitively.

    Returns
    -------
    set[str]
        A set of unique, lowercased tokens. Empty set if ``text`` is falsy.

    """

    if not text:
        return set()
    return set(_TOKEN_PATTERN.findall(text.casefold()))


def build_term_pattern(terms: Iterable[str]) -> re.Pattern:
    """Compile a case-insensitive regex that matches any of the supplied terms.

    Parameters
    ----------
    terms : Iterable[str]
        Terms to match. Falsy entries (e.g., empty strings) are ignored. Terms
        are de-duplicated and escaped before pattern construction.

    Returns
    -------
    re.Pattern
        A compiled regular expression that matches any provided term
        case-insensitively. If no valid terms are provided, an always-false
        pattern is returned.
    """

    filtered_terms = [term for term in terms if term]
    if not filtered_terms:
        return re.compile(r"(?!)")  # Always-false pattern

    unique_terms = sorted(set(filtered_terms), key=len, reverse=True)
    escaped_terms = [re.escape(term) for term in unique_terms]
    return re.compile(r"(?:{})".format("|".join(escaped_terms)), flags=re.IGNORECASE)


def filter_by_date(
    adf: pd.DataFrame, cutoff_dt: pd.Timestamp, date_col: str
) -> pd.DataFrame:
    """Filter a DataFrame by a date cutoff.

    Parameters
    ----------
    adf : pandas.DataFrame
        The DataFrame to filter.
    cutoff_dt : pandas.Timestamp
        The cutoff timestamp. Only rows with dates before this date will be kept.
    date_col : str
        The name of the column in the DataFrame containing date information.

    Returns
    -------
    pandas.DataFrame
        A DataFrame filtered to include only rows with dates on or before the cutoff
        date.

    Notes
    -----
    Rows with unparsable dates (``NaT``) are dropped. The result is index-reset.
    """
    if adf.empty:
        return pd.DataFrame()

    date_series: pd.Series = pd.to_datetime(adf[date_col], errors="coerce")
    num_no_date = date_series.isna().sum()
    if num_no_date > 0:
        logger.debug(
            "Found %d rows with NaN values in '%s' column.", num_no_date, date_col
        )

    mask = (date_series.notna()) & (date_series < cutoff_dt)
    return adf.loc[mask].reset_index(drop=True)
