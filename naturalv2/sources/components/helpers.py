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

_HYPHEN_RUNS = "\u2010\u2011\u2012\u2013\u2014\u2212-"
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060"

# Characters to remove before NFKC normalization
# Without this, ™ and ® can get converted to TM and R, and match attempts fail
PRE_NFKC_TRANSLATION_TABLE = {ord("™"): None, ord("®"): None, ord("\ufeff"): None}

POST_NFKC_TRANSLATION_TABLE = {
    **{ord(c): "-" for c in _HYPHEN_RUNS},
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
    ord("\xa0"): " ",  # non-breaking space; `&nbsp` in HTML
    ord("\u2007"): " ",  # figure space, often used in tables
    ord("\u202f"): " ",  # narrow no-break space, common in SI
    ord("\u200a"): " ",  # hair space
    ord("^"): None,
    ord("•"): " ",
    ord("→"): " ",
    ord("×"): "x",
    ord("±"): "+/-",
    # Different “micro” symbols both become a simple “u”
    ord("µ"): "u",
    ord("μ"): "u",
}

GREEK_CHAR_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "Δ": "delta",
    "ε": "epsilon",
    "κ": "kappa",
    "Ω": "omega",
    "ω": "omega",
}

_RE_SPACES = re.compile(r"\s+")
_RE_NUM_TO_LETTER = re.compile(r"(?<=\d)(?=[a-z])")
_RE_LETTER_TO_NUM = re.compile(r"(?<=[a-z])(?=\d)")

# Break aliases on the connectors above so we can rebuild them with the flexible
# pattern e.g. "calcium-magnesium+citrate" -> ["calcium", "magnesium", "citrate"]
CONNECTOR_SPLIT = re.compile(r"[-_/+,;:\s]+")

CONNECTOR_CHAR_SET = set("-_/+,;:")
CONNECTOR_CHAR_SET.add(" ")

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


@lru_cache(maxsize=50_000)
def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for case-insensitive, accent-insensitive matching.

    Applies a comprehensive normalization pipeline to standardize text for
    matching operations. The process handles Unicode variations, case differences,
    accent marks, symbol replacements, and spacing to ensure consistent matches
    across different text representations.

    The normalization pipeline performs these steps in order:
    1. Remove specific problematic characters (™, ®, BOM)
    2. Apply NFKC normalization to standardize Unicode forms
    3. Convert to lowercase
    4. Replace various symbols (hyphens, Greek letters, etc.)
    5. Apply NFD normalization and remove accent marks
    6. Collapse repeated whitespace and trim loose edge punctuation
    7. Insert spaces between numbers and letters ("10mg" → "10 mg")

    Connector characters (hyphens, slashes, etc.) are preserved here and
    collapsed later in ``canonicalize_reports_for_matching``.

    Parameters
    ----------
    text : str
        Raw alias or surface-form text to normalize. Can be treatment names,
        drug aliases, or any text requiring standardization.

    Returns
    -------
    str
        Normalized representation suitable for case-insensitive, accent-insensitive
        matching. Returns empty string if input is empty or None.

    Examples
    --------
    >>> normalize_text_for_matching("Ibuprofen-200mg")
    'ibuprofen-200 mg'

    >>> normalize_text_for_matching("Naproxen/Naprosyn")
    'naproxen/naprosyn'

    >>> normalize_text_for_matching("α-blocker")
    'alpha-blocker'

    Notes
    -----
    This function is cached with LRU cache (maxsize=50,000) for performance,
    as the same aliases are often normalized repeatedly.

    Greek letters are replaced with their English names (α→"alpha", β→"beta", etc.)
    to ensure matches between Unicode and ASCII representations.

    Various hyphen-like characters (\\u2010-\\u2014, \\u2212, -) are normalized
    to ASCII hyphens. Connector collapsing happens in
    ``canonicalize_reports_for_matching``.

    """
    if not text:
        return ""

    text = text.translate(PRE_NFKC_TRANSLATION_TABLE)
    text = text.translate(POST_NFKC_TRANSLATION_TABLE)

    if text.isascii():
        text = text.casefold()
    else:
        text = unicodedata.normalize("NFKC", text).casefold()
        text = text.translate(POST_NFKC_TRANSLATION_TABLE)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = "".join(GREEK_CHAR_MAP.get(char, char) for char in text)
    text = _RE_SPACES.sub(" ", text).strip("".join(CONNECTOR_CHAR_SET))
    text = _RE_NUM_TO_LETTER.sub(" ", text)
    return _RE_LETTER_TO_NUM.sub(" ", text)


def iter_canonical_variations(text: str) -> list[str]:
    """
    Generate all canonical variations for a single treatment term.

    Creates multiple normalized versions of a treatment name by applying different
    text transformations. This generates variations to improve matching recall:
    - Full normalized version with connector characters collapsed
    - Version without parenthetical qualifiers (e.g., "(oral)", "(tablet)")

    These variations handle cases where text appears with or without qualifiers,
    or with different connector styles (hyphen vs. space).

    Parameters
    ----------
    text : str
        Raw treatment term, alias, or drug name to generate variations for.

    Returns
    -------
    list of str
        List of unique canonical variations, typically 1-2 variations. Returns
        empty list if input normalizes to empty string.

    Examples
    --------
    >>> iter_canonical_variations("Ibuprofen-200mg (oral tablet)")
    ['ibuprofen 200 mg oral tablet', 'ibuprofen 200 mg']

    >>> iter_canonical_variations("calcium-magnesium+citrate")
    ['calcium magnesium citrate']

    Notes
    -----
    This function encapsulates the logic for normalization, connector collapsing,
    and parenthetical stripping so it can be shared between the automaton builder
    and the treatment registry builder.

    """
    variations = set()
    normalised_text = normalize_text_for_matching(text)

    if not normalised_text:
        return []

    # Replace runs of connector punctuation with a single space so variants like
    # "sandoz-topiramate" and "Sandoz Topiramate" match the same alias
    collapsed_full = " ".join(
        token for token in CONNECTOR_SPLIT.split(normalised_text) if token
    )
    if collapsed_full:
        variations.add(collapsed_full)

    # Also keep a version without trailing qualifiers in parentheses, because posts
    # often drop the parenthetical note
    stripped_text = PAREN_STRIP.sub(" ", normalised_text)
    stripped_text = re.sub(r"\s+", " ", stripped_text).strip(" -_/+,;:")

    if stripped_text and stripped_text != normalised_text:
        collapsed_stripped = " ".join(
            token for token in CONNECTOR_SPLIT.split(stripped_text) if token
        )
        if collapsed_stripped:
            variations.add(collapsed_stripped)

    return list(variations)


def build_treatment_automaton(aliases: Sequence[str]) -> ahocorasick.Automaton:
    """
    Build an Aho-Corasick automaton for fast multi-pattern string matching.

    Creates a finite-state automaton that can find all occurrences of multiple
    treatment aliases in text in a single pass.

    Each alias is normalized and expanded into canonical variations (with and
    without parenthetical qualifiers). Short variations (≤3 characters) are
    filtered out to reduce false positives.

    Parameters
    ----------
    aliases : Sequence of str
        Raw treatment names and aliases including brand names, generic names,
        dosage forms, etc. Can contain special characters, Unicode, mixed case.

    Returns
    -------
    ahocorasick.Automaton
        Compiled automaton ready for matching. Call ``.iter(text)`` to find
        all alias occurrences in one pass over canonicalized text.

    Notes
    -----
    The automaton maps each pattern to its canonical form, not the original alias.
    This ensures consistent representation in match results.

    Very short canonical forms (3 characters or less) are excluded because they
    generate too many false positives (e.g., "mg", "ml", "a", "b").

    See Also
    --------
    iter_canonical_variations : Generates variations for each alias
    extract_mentions : Uses the automaton to find mentions in text

    """
    automaton = ahocorasick.Automaton()

    # Use the shared iterator to ensure consistency
    for alias in aliases:
        for canonical_alias in iter_canonical_variations(alias):
            if len(canonical_alias) <= 3:  # Drop short canonical forms
                continue

            # Map the pattern -> canonical form
            automaton.add_word(canonical_alias, canonical_alias)

    automaton.make_automaton()
    return automaton


def canonicalize_reports_for_matching(text: str) -> tuple[str, list[int]]:
    """
    Collapse connector characters to spaces and track original character indices.

    Transforms text to match the canonical form used in the automaton by replacing
    runs of connector characters (hyphens, slashes, spaces, etc.) with single spaces.
    Also inserts spaces at digit-letter boundaries ("10mg" → "10 mg").

    Maintains a mapping from each character in the output to its position in the
    original text, enabling recovery of exact substrings when matches are found.
    This is crucial for extracting the original text rather than the normalized form.

    Parameters
    ----------
    text : str
        Normalized post text to prepare for automaton matching. Should already be
        normalized via ``normalize_text_for_matching()``.

    Returns
    -------
    canonical_text : str
        Text where each run of connector characters (hyphen, slash, space, comma,
        semicolon, colon, underscore, plus) has been replaced by a single space.
        This mirrors how aliases are canonicalized when building the automaton.
    index_map : list of int
        Index mapping where ``index_map[i]`` gives the position in the original
        ``text`` corresponding to ``canonical_text[i]``. Used to recover exact
        substrings from match positions.

    Examples
    --------
    >>> text = "ibuprofen-200mg/5ml"
    >>> canonical, indices = canonicalize_reports_for_matching(text)
    >>> canonical
    'ibuprofen 200 mg 5 ml'
    >>> len(canonical) == len(indices)
    True

    Notes
    -----
    Connector characters are defined in ``CONNECTOR_CHAR_SET``: `-_/+,;:` and space.

    Consecutive connector characters are collapsed to a single space to prevent
    multiple spaces in the output, which would interfere with pattern matching.

    Spaces are automatically inserted at digit-letter boundaries (e.g., "10mg"
    becomes "10 mg") to ensure matches against patterns that include such spacing.

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
    """
    Extract all treatment mentions from text using an Aho-Corasick automaton.

    Canonicalizes the input text and finds all occurrences of treatment aliases
    using the provided automaton. Returns unique canonical forms of matched aliases
    in sorted order.

    This is the main entry point for finding treatment mentions in posts, comments,
    or other text documents.

    Parameters
    ----------
    text : str
        The text to extract mentions from. Can be raw text (doesn't need to be
        pre-normalized). Empty or None text returns empty list.
    automaton : ahocorasick.Automaton
        Pre-built automaton created by ``build_treatment_automaton()``. Must be
        finalized with ``.make_automaton()``.

    Returns
    -------
    list of str
        Sorted list of unique canonical treatment mentions found in the text.
        Each mention is in its canonical form (lowercase, normalized). Returns
        empty list if no matches found or if input text is empty.

    Examples
    --------
    >>> aliases = ["Ibuprofen", "Advil", "Naproxen"]
    >>> automaton = build_treatment_automaton(aliases)
    >>> text = "I take Advil-200mg for pain, sometimes Naproxen too."
    >>> mentions = extract_mentions(text, automaton)
    >>> mentions
    ['advil', 'naproxen']

    Notes
    -----
    The function automatically handles text canonicalization via
    ``canonicalize_reports_for_matching()``, so raw text can be passed directly.

    Duplicate mentions are automatically removed - if "ibuprofen" appears multiple
    times in the text, it appears only once in the output.

    The returned mentions are in canonical form (normalized), not the original text.
    This ensures consistent representation across different text variations.

    See Also
    --------
    build_treatment_automaton : Creates the automaton
    canonicalize_reports_for_matching : Text canonicalization function

    """
    canonical_text, _ = canonicalize_reports_for_matching(text)
    if not canonical_text:
        return []

    found_mentions: set[str] = set()

    # .iter() returns (end_index, value).
    # In build_treatment_automaton, we set 'value' to be the canonical_alias.
    for _, canonical_alias in automaton.iter(canonical_text):
        found_mentions.add(canonical_alias)

    return sorted(found_mentions)


def filter_by_date(
    adf: pd.DataFrame, cutoff_dt: pd.Timestamp, date_col: str
) -> pd.DataFrame:
    """
    Filter a DataFrame to include only rows before a specified date.

    Filters rows based on a date column, keeping only those with valid dates
    that occur strictly before the cutoff date. Rows with unparsable dates
    (NaT values) are automatically removed. This is commonly used to exclude
    data from after a trial enrollment cutoff or to filter posts by date.

    Parameters
    ----------
    adf : pandas.DataFrame
        The DataFrame to filter. Can be empty (returns empty DataFrame).
    cutoff_dt : pandas.Timestamp
        The cutoff timestamp. Only rows with dates strictly before this date
        will be kept (exclusive upper bound).
    date_col : str
        The name of the column in the DataFrame containing date information.
        Can contain strings, datetime objects, or other parsable date formats.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame with index reset (starting from 0). Contains only
        rows with valid dates before the cutoff. Returns empty DataFrame if
        input is empty or no rows satisfy the condition.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "created_utc": ["2020-01-01", "2020-06-01", "2021-01-01"],
    ...         "text": ["post1", "post2", "post3"],
    ...     }
    ... )
    >>> cutoff = pd.Timestamp("2020-07-01")
    >>> filtered = filter_by_date(df, cutoff, "created_utc")
    >>> len(filtered)
    2

    Notes
    -----
    Rows with unparsable dates (resulting in NaT after ``pd.to_datetime()``) are
    dropped from the result. A debug log message is emitted if any such rows exist.

    The comparison is strictly less than (``<``), not less than or equal (``<=``),
    so a row dated exactly at the cutoff time is excluded.

    The result has its index reset to start from 0, so the returned DataFrame
    has a clean sequential index regardless of the input index.

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
