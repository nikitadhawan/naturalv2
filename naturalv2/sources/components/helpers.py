"""Helpers shared across source stages.

Utilities in this module support common operations needed by curation stages,
including tokenization for exact-match filtering, building case-insensitive
regex patterns from term lists, and filtering tabular data by a date cutoff.
"""

import logging
import unicodedata
from functools import lru_cache
from typing import Sequence

import ahocorasick
import pandas as pd
import regex as re


logger = logging.getLogger(__name__)

_HYPHEN_RUNES = "\u2010\u2011\u2012\u2013\u2014\u2212-"
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060"
_TRANSLATION_TABLE = {
    **{ord(c): "-" for c in _HYPHEN_RUNES},
    **{ord(c): None for c in _ZERO_WIDTH},
    # Map superscript digits down to normal digits so “m²” and “m2” line up.
    ord("¹"): "1",
    ord("²"): "2",
    ord("³"): "3",
    ord("⁴"): "4",
    ord("⁵"): "5",
    ord("⁶"): "6",
    ord("⁷"): "7",
    ord("⁸"): "8",
    ord("⁹"): "9",
    ord("⁰"): "0",
    # Replace common oddities (non-breaking spaces, ®, ™, hair spaces) before we
    # normalize further
    ord("\xa0"): " ",  # non-breaking space; `&nbsp` in HTML
    ord("\u2007"): " ",  # figure space, often used in tables
    ord("\u202f"): " ",  # narrow no-break space, common in SI
    ord("\u200a"): " ",  # hair space
    ord("\ufeff"): None,
    ord("®"): None,
    ord("™"): None,
    ord("^"): None,
    ord("•"): " ",
    ord("→"): " ",
    ord("×"): "x",
    ord("±"): "+/-",
    # Different “micro” symbols both become a simple “u”
    ord("µ"): "u",
    ord("μ"): "u",
}

# Swap Greek letters for simple words so Unicode and ASCII versions match up
GREEK_CHAR_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "Δ": "delta",
    "ε": "epsilon",
    "κ": "kappa",
}

_RE_SPACES = re.compile(r"\s+")  # regex to match one or more spaces
_RE_NUM_TO_LETTER = re.compile(r"(?<=\d)(?=[a-z])")
_RE_LETTER_TO_NUM = re.compile(r"(?<=[a-z])(?=\d)")

# Break aliases on the connectors above so we can rebuild them with the flexible
# pattern e.g. "calcium-magnesium+citrate" -> ["calcium", "magnesium", "citrate"]
CONNECTOR_SPLIT = re.compile(r"[-_/+,;:\s]+")

# Strip simple parenthetical notes (e.g., “(oral)”) when we need a shorter fallback
# alias. For example, "sitagliptin (oral tablet)" -> "sitagliptin "
# The stripped form will be added to the dictionary as a backup when the text with
# the parenthesis is not present
PAREN_STRIP = re.compile(r"[\[(][^)\]]+[\])]")

# Units that most often follow a dose
_DOSE_UNITS = (
    "mg",
    "g",
    "ug",
    "µg",
    "mcg",
    "ml",
    "iu",
    "unit",
    "units",
    "%",
    "mg/m2",
    "mg/m^2",
    "mg/kg",
    "mg/day",
    "mg/ml",
)
_DOSE_UNIT_PATTERN = "|".join(re.escape(unit) for unit in _DOSE_UNITS)

# Let one dose snippet trail the name (examples: "25 mg", "125mg/5ml", "amlodipine, 10mg")
# so the match keeps the dosage text when it’s present.
_DOSE_FRAGMENT = (
    r"(?:[,;/\-\s]*+\d+(?:\.\d+)?"
    r"(?:\s*(?:" + _DOSE_UNIT_PATTERN + r"))?"
    r"(?:\s*/\s*\d+(?:\.\d+)?(?:\s*(?:day|week|wk|kg|m2|m\^2))?)?"
    r")"
)

# Let that dose fragment repeat a few times because compound entries often list
# several strengths back to back
_DOSE_PATTERN = rf"(?:{_DOSE_FRAGMENT}){{0,3}}+"

# Allow one short parenthetical hint after the name, since arm labels and routes
# show up in parentheses
_QUALIFIER_PATTERN = r"(?:\s*[\[(][^)\]]{1,40}[\])])?"

_DOSE_TAIL_REGEX = re.compile(
    rf"(?:{_QUALIFIER_PATTERN}{_DOSE_PATTERN})", flags=re.IGNORECASE | re.UNICODE
)

CONNECTOR_CHAR_SET = set("-_/+,;:")
CONNECTOR_CHAR_SET.add(" ")


@lru_cache(maxsize=50_000)
def normalize_text_for_matching(text: str) -> str:
    """Normalize text for matching.

    Parameters
    ----------
    text : str
        Raw alias or surface-form text.

    Returns
    -------
    str
        Normalized representation used for dictionary matching.
    """
    if not text:
        return ""

    # Use the translation table to swap common symbol lookalikes for plain text
    pre_normalized = text.translate(_TRANSLATION_TABLE)

    if pre_normalized.isascii():
        # ASCII only, skip NFKC/NFKD
        normalized = pre_normalized.casefold()
    else:
        # Make sure the text uses a standard Unicode form and lowercase
        normalized = unicodedata.normalize("NFKC", pre_normalized).casefold()

        # Clean up any new codepoints that were added after NFKC normalization
        normalized = normalized.translate(_TRANSLATION_TABLE)

    # Remove accent marks (so “crème” and “creme” normalize the same way) only
    # if non-ASCII remains
    if not normalized.isascii():
        normalized = unicodedata.normalize("NFKD", normalized)
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))

    # Replace Greek letters and micro symbols with simple ASCII text
    normalized = "".join(GREEK_CHAR_MAP.get(char, char) for char in normalized)

    # Shrink repeated spaces and trim loose punctuation on the ends
    normalized = _RE_SPACES.sub(" ", normalized).strip(" -_/+,;:")

    # Ensure numbers glued to units (“10mg”) match spaced forms (“10 mg”)
    normalized = _RE_NUM_TO_LETTER.sub(" ", normalized)
    return _RE_LETTER_TO_NUM.sub(" ", normalized)


def build_treatment_automaton(aliases: Sequence[str]) -> ahocorasick.Automaton:
    """Build an Aho–Corasick automaton over normalised treatment aliases.

    Parameters
    ----------
    aliases : Sequence[str]
        Raw treatment names and aliases (brand names, doses, etc.).

    Returns
    -------
    ahocorasick.Automaton
        Automaton that can be iterated to find every alias occurrence in one
        pass over canonicalised text.
    """
    canonical_aliases: set[str] = set()

    for alias in aliases:
        normalised_alias = normalize_text_for_matching(alias)
        if not normalised_alias:
            continue  # Empty or whitespace-only alias

        # Replace runs of connector punctuation with a single space so
        # variants like "sandoz-topiramate" and "Sandoz Topiramate" match
        # the same alias
        collapsed_alias = " ".join(
            token for token in CONNECTOR_SPLIT.split(normalised_alias) if token
        )
        if collapsed_alias:
            canonical_aliases.add(collapsed_alias)

        # Also keep a version without trailing qualifiers in parentheses,
        # because posts often drop the parenthetical note
        stripped_alias = PAREN_STRIP.sub(" ", normalised_alias)
        stripped_alias = re.sub(r"\s+", " ", stripped_alias).strip(" -_/+,;:")
        if stripped_alias and stripped_alias != normalised_alias:
            collapsed_stripped_alias = " ".join(
                token for token in CONNECTOR_SPLIT.split(stripped_alias) if token
            )
            if collapsed_stripped_alias:
                canonical_aliases.add(collapsed_stripped_alias)

    automaton = ahocorasick.Automaton()
    for canonical_alias in sorted(canonical_aliases):
        automaton.add_word(canonical_alias, canonical_alias)

    automaton.make_automaton()
    return automaton


def canonicalize_reports_for_matching(text: str) -> tuple[str, list[int]]:
    """Collapse connector characters to single spaces and track original indices.

    Parameters
    ----------
    text : str
        Normalised post text to prepare for automaton matching.

    Returns
    -------
    canonical_text : str
        Text where each run of connector characters (hyphen, slash, space, etc.)
        has been replaced by a single space.  This mirrors how we canonicalise
        aliases when building the automaton.
    index_map : list[int]
        For every character position in ``canonical_text`` we record the
        corresponding index in the original ``text`` so we can recover the exact
        substring once a match is reported.
    """
    canonical_characters: list[str] = []
    canonical_to_original_indices: list[int] = []
    just_added_space = False

    for original_index, character in enumerate(text):
        # Treat both explicit connectors (e.g., '-') and any whitespace the same
        connector_character = character in CONNECTOR_CHAR_SET or character.isspace()

        # Add a space after a digit/letter boundary to catch numbers glued to units
        # e.g. "10mg" should match "10 mg"
        if canonical_characters:
            prev_char = canonical_characters[-1]
            digit_letter_boundary = (prev_char.isalpha() and character.isdigit()) or (
                prev_char.isdigit() and character.isalpha()
            )
        else:
            digit_letter_boundary = False

        if digit_letter_boundary and not just_added_space:
            canonical_characters.append(" ")
            canonical_to_original_indices.append(original_index)
            just_added_space = True

        if connector_character:
            if just_added_space:
                continue  # Skip repeated connector characters

            canonical_characters.append(" ")
            canonical_to_original_indices.append(original_index)
            just_added_space = True
        else:
            canonical_characters.append(character)
            canonical_to_original_indices.append(original_index)
            just_added_space = False

    canonical_text = "".join(canonical_characters)
    return canonical_text, canonical_to_original_indices


def extract_mentions(text: str, automaton: ahocorasick.Automaton) -> list[str]:
    """Extract mentions from a text string using an ahocorasick automaton.

    Parameters
    ----------
    text : str
        The text to extract mentions from.
    automaton : ahocorasick.Automaton
        The automaton to use for extracting mentions.

    Returns
    -------
    list[str]
        A list of mentions found in the text.
    """
    # Collapse punctuation connectors to spaces so aliases match regardless
    # of hyphens/underscores
    canonical_text, canonical_to_original = canonicalize_reports_for_matching(text)
    if not canonical_text:
        return []

    found_mentions: list[str] = []
    emitted_mentions: set[str] = set()  # For quick deduplication

    for end_index, matched_alias in automaton.iter(canonical_text):
        start_index = end_index - len(matched_alias) + 1
        if start_index < 0:
            # Safety guard in case the automaton reports an unexpected position
            continue

        # Translate the canonical span back to the original text indices
        original_start = canonical_to_original[start_index]
        original_end = canonical_to_original[end_index] + 1

        # Grab the exact substring from the original text
        mention_text = text[original_start:original_end]

        # Look right after the alias for an optional qualifier/dose string
        # and include it
        dose_tail = _DOSE_TAIL_REGEX.match(text[original_end:])
        if dose_tail:
            mention_text += dose_tail.group(0)

        cleaned_mention = mention_text.strip()
        if cleaned_mention and cleaned_mention not in emitted_mentions:
            emitted_mentions.add(cleaned_mention)
            found_mentions.append(cleaned_mention)

    return sorted(found_mentions)


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
