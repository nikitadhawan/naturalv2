import logging
from functools import partial
from typing import Any, Generator, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from ahocorasick import Automaton

from naturalv2.sources.components import filter_by_date
from naturalv2.sources.components.helpers import (
    extract_mentions,
    normalize_text_for_matching,
)
from naturalv2.sources.reddit.processing.contextualize import (
    _make_partition_filter,
    _normalize_subreddits,
)


logger = logging.getLogger(__name__)


SENTINELS = pa.array(["[deleted]", "[removed]"], type=pa.string())


def apply_rule_based_filter(table: pa.Table, text_field: str) -> pa.ChunkedArray:
    """
    Apply rule-based filtering to a pyarrow Table of Reddit posts.

    Parameters
    ----------
    table : pa.Table
        A table containing Reddit posts. Expects the `text_field` and `author`
        columns to exist.
    text_field : str
        The name of the text field column to be filtered.

    Returns
    -------
    pa.ChunkedArray
        Boolean mask indicating valid rows.

    Notes
    -----
    The resulting mask enforces multiple heuristics:
    - Normalizes the text field by filling nulls, unescaping basic HTML entities,
      replacing control characters, and trimming whitespace.
    - Rejects empty strings and records that match known sentinel content such
      as deleted or removed posts.
    - Requires a permalink and removes rows authored by obvious bot accounts.
    - Ensures the first 2,048 code units include at least one space, signalling
      multi-token text.
    - Verifies that alphabetic characters (plus spaces) make up at least half of
      the trimmed text.
    """

    # Helper to cast to string and fill nulls
    def to_string_filled(arr: pa.ChunkedArray) -> pa.ChunkedArray:
        """Fill nulls with empty string."""
        return pc.fill_null(arr, pa.scalar("", type=pa.string()))

    # Normalize text
    text = to_string_filled(table[text_field])
    # Replace HTML entities (order matters: amp last)
    text = pc.replace_substring(text, "&gt;", ">")
    text = pc.replace_substring(text, "&lt;", "<")
    text = pc.replace_substring(text, "&amp;", "&")

    # Collapse runs of whitespace then trim
    text = pc.replace_substring_regex(text, r"[ \t\r\n]+", " ")
    text = pc.utf8_trim_whitespace(text)

    # Require non-empty and not sentinel strings
    has_text = pc.greater(pc.utf8_length(text), 0)
    is_deleted = pc.is_in(text, value_set=SENTINELS)
    valid_text_mask = pc.and_kleene(has_text, pc.invert(is_deleted))

    # Filter out bot-like authors
    author = to_string_filled(table["author"])
    author_lower = pc.utf8_lower(author)
    is_automod = pc.equal(author_lower, "automoderator")
    looks_like_bot = pc.match_substring_regex(author_lower, r"(?:^|[_-])bot\d*$")
    is_bot_author = pc.or_kleene(is_automod, looks_like_bot)
    valid_text_mask = pc.and_kleene(valid_text_mask, pc.invert(is_bot_author))

    # Drop URLs from the full text (keep markdown link labels)
    text_no_urls = pc.replace_substring_regex(
        text,
        r"\[([^\]]+)\]\(\s*(?:https?://|www\.)\S+\s*\)|https?://\S+|\bwww\.\S+",
        r"\1",  # Keep the first match (the markdown label), drop the rest
    )

    # Check for >=3-letter token within first 2048 code units
    preview = pc.utf8_slice_codeunits(text_no_urls, 0, stop=2048)
    # Use Unicode letter class to allow accented characters
    has_long_token = pc.match_substring_regex(preview, r"\p{L}{3,}")
    valid_text_mask = pc.and_kleene(valid_text_mask, has_long_token)

    # Ensure at least 25% of characters are letters
    total_len = pc.utf8_length(text_no_urls)
    letter_count = pc.count_substring_regex(text_no_urls, r"\p{L}")
    ratio_ok = pc.greater_equal(pc.multiply(letter_count, 4), total_len)

    return pc.and_kleene(valid_text_mask, ratio_ok)


def scan_subreddit(
    base_dir: str,
    subreddits: str | Sequence[str],
    content_type: str | None = None,
    columns: list[str] | None = None,
    batch_size: int = 128_000,
) -> Generator[pd.DataFrame, Any, None]:
    # Normalize to a de-duped, non-empty list
    subs = _normalize_subreddits(
        [subreddits] if isinstance(subreddits, str) else subreddits
    )
    filter_expr = _make_partition_filter(subs, content_type)

    dataset = ds.dataset(base_dir, format="parquet", partitioning="hive")
    scanner = dataset.scanner(
        filter=filter_expr, columns=columns, use_threads=True, batch_size=batch_size
    )
    for record_batch in scanner.to_batches():
        batch_df = record_batch.to_pandas(types_mapper=pd.ArrowDtype)
        if not batch_df.empty:
            yield batch_df


def get_study_relevant_posts(
    clean_data: Union[str, pd.DataFrame],
    treatment_automaton: Automaton,
    cutoff_dt: pd.Timestamp | None,
    date_column: str = "date_created",
) -> pd.DataFrame:
    """Select posts mentioning treatments before an optional cutoff date.

    Parameters
    ----------
    clean_data : str | pandas.DataFrame
        Either the path to a cleaned subreddit parquet file created by
        :func:`download_submissions_and_comments` or a pre-loaded DataFrame.
    treatment_automaton : ahocorasick.Automaton
        Compiled ahocorasick automaton for matching treatment aliases.
    cutoff_dt : pandas.Timestamp | None
        If provided, only posts with dates before this timestamp are
        considered.
    date_column : str, default="date_created"
        Column name containing the post/comment timestamp.

    Returns
    -------
    pandas.DataFrame
        DataFrame of posts mentioning any treatment term, with an additional
        ``treatments_mentioned`` column listing the matched terms.
    """
    if isinstance(clean_data, str):
        df = pd.read_parquet(clean_data)
        data_label = clean_data
    else:
        df = clean_data
        data_label = "provided DataFrame chunk"

    if df.empty:
        return pd.DataFrame()

    if cutoff_dt is not None:
        df = filter_by_date(df, cutoff_dt, date_column)
        if df.empty:
            return pd.DataFrame()

    text_cols = [
        col
        for col in ("report_text", "title", "initial_post", "subreddit")
        if col in df.columns
    ]
    if not text_cols:
        logger.warning(
            "No textual columns found in %s to evaluate treatment matches.",
            data_label,
        )
        return pd.DataFrame()

    reports = (
        df[text_cols]
        .fillna("")
        .astype("string")
        .agg(" ".join, axis=1)
        .map(normalize_text_for_matching)
    )

    mentions = reports.map(partial(extract_mentions, automaton=treatment_automaton))
    mask = mentions.str.len().gt(0)
    return df.loc[mask].assign(treatments_mentioned=mentions.loc[mask])
