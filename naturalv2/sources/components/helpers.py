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


# Treat all dash-like characters as the same so “drug--name” and “drug-name” both
# match
HYPHEN_CLASS = r"[\u2010\u2011\u2012\u2013\u2014\u2212-]"

# Let words reconnect with a mix of spaces, dashes, slashes, commas, or colons
# because trial text often swaps them e.g. sandoz topiramate, sandoz-topiramate,
# sandoz/topiramate, sandoz_topiramate
CONNECTOR_PATTERN = r"[-_/+,;:\s]*+"

# Define the characters that count as part of a word when we set boundaries;
# hyphenated names should stay intact
WORD_CLASS = r"[\w-]"

# Let apostrophes appear or disappear inside tokens so “children's” and “childrens”
# both hit
OPTIONAL_APOSTROPHE = r"['’]?"

# Spell check for apostrophes when building token patterns.
APOSTROPHE_CHARS = {"'", "’"}

# Break aliases on the connectors above so we can rebuild them with the flexible
# pattern e.g. "calcium-magnesium+citrate" -> ["calcium", "magnesium", "citrate"]
CONNECTOR_SPLIT = re.compile(r"[-_/+,;:\s]+")

# Strip simple parenthetical notes (e.g., “(oral)”) when we need a shorter fallback
# alias. For example, "sitagliptin (oral tablet)" -> "sitagliptin "
# The stripped form will be added to the dictionary as a backup when the text with
# the parenthesis is not present
PAREN_STRIP = re.compile(r"[\[(][^)\]]+[\])]")

# Units that most often follow a dose
DOSE_UNITS = (
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
DOSE_UNIT_PATTERN = "|".join(re.escape(unit) for unit in DOSE_UNITS)

# Let one dose snippet trail the name (examples: "25 mg", "125mg/5ml", "amlodipine, 10mg")
# so the match keeps the dosage text when it’s present.
DOSE_FRAGMENT = (
    r"(?:[,;/\-\s]*+\d+(?:\.\d+)?"
    r"(?:\s*(?:" + DOSE_UNIT_PATTERN + r"))?"
    r"(?:\s*/\s*\d+(?:\.\d+)?(?:\s*(?:day|week|wk|kg|m2|m\^2))?)?"
    r")"
)

# Let that dose fragment repeat a few times because compound entries often list
# several strengths back to back
DOSE_PATTERN = rf"(?:{DOSE_FRAGMENT}){{0,3}}+"

# Allow one short parenthetical hint after the name, since arm labels and routes
# show up in parentheses
QUALIFIER_PATTERN = r"(?:\s*[\[(][^)\]]{1,40}[\])])?"

# Map superscript digits down to normal digits so “m²” and “m2” line up.
SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁰": "0",
    }
)

# Replace common oddities (non-breaking spaces, ®, ™, hair spaces) before we
# normalize further
CHAR_TRANSLATION = {
    ord("\xa0"): " ",  # non-breaking space; `&nbsp` in HTML
    ord("\u2007"): " ",  # figure space, often used in tables
    ord("\u202f"): " ",  # narrow no-break space, common in SI
    ord("\u200a"): " ",  # hair space
    ord("\ufeff"): None,
    ord("®"): None,
    ord("™"): None,
    ord("•"): " ",
    ord("→"): " ",
    ord("^"): None,
    ord("×"): "x",
    ord("±"): "+/-",
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

# Different “micro” symbols both become a simple “u”
MICRO_FORMS = ("µ", "μ")


CONNECTOR_CHAR_SET = set("-_/+,;:")
CONNECTOR_CHAR_SET.add(" ")


@lru_cache(maxsize=50_000)
def normalize_text_for_matching(text: str) -> str:
    """Normalize a treatment alias for tolerant matching.

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

    # Make sure the text uses a standard Unicode form and lowercase
    normalized = unicodedata.normalize("NFKC", text).casefold()

    # Swap common symbol lookalikes for plain text before removing accents
    normalized = normalized.translate(SUPERSCRIPT_TRANSLATION)
    normalized = normalized.translate(CHAR_TRANSLATION)

    # Turn every dash-like character into '-' and drop hidden spaces
    normalized = re.sub(HYPHEN_CLASS, "-", normalized)
    normalized = re.sub(r"[\u200b\u200c\u200d\u2060]", "", normalized)

    # Remove accent marks so “crème” and “creme” normalize the same way
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))

    # Replace Greek letters and micro symbols with simple ASCII text
    normalized = "".join(GREEK_CHAR_MAP.get(ch, ch) for ch in normalized)
    for micro in MICRO_FORMS:
        normalized = normalized.replace(micro, "u")

    # Shrink repeated spaces and trim loose punctuation on the ends
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" -_/+,;:")

    # Ensure numbers glued to units (“10mg”) match spaced forms (“10 mg”)
    normalized = re.sub(r"(?<=\d)(?=[a-zA-Zµμ])", " ", normalized)
    return re.sub(r"(?<=[a-zA-Zµμ])(?=\d)", " ", normalized)


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


def canonicalize_for_matching(text: str) -> tuple[str, list[int]]:
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
