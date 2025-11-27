"""Functions for filtering Reddit data."""

import logging
from collections.abc import Iterator

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


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
        r"(?i)\[([^\]]+)\]\(\s*(?:https?://|www\.)\S+\s*\)|https?://\S+|\bwww\.\S+",
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


def scan_reddit_dataset(
    file_paths: list[str] | str,
    columns: list[str] | None = None,
    target_subreddits: list[str] | None = None,
    batch_size: int = 128_000,
) -> Iterator[pl.DataFrame]:
    """Stream chunks of Reddit data from partitioned dataset."""

    # Define the filter expression
    filter_expr = None
    if target_subreddits:
        # Create a pyarrow expression: subreddit is in [list]
        filter_expr = ds.field("subreddit").isin(target_subreddits)

    # Initialize PyArrow Dataset
    dataset = ds.dataset(file_paths, format="parquet")

    # Filter columns to only those that exist in the actual dataset schema
    # This handles cases where requested columns don't exist in the parquet files
    available_columns = set(dataset.schema.names)

    filtered_columns: list[str] | None = None
    if columns:
        filtered_columns = [col for col in columns if col in available_columns]
        if not filtered_columns:
            return

    # Create a Scanner
    # This pushes the filter down to the I/O layer.
    # It will skip row groups where 'subreddit' stats don't match the list.
    scanner = dataset.scanner(
        columns=filtered_columns, filter=filter_expr, batch_size=batch_size
    )

    # Stream Batches
    # to_batches() returns an iterator that keeps its place.
    # It only decompresses the specific row groups that pass the filter.
    for batch in scanner.to_batches():
        if batch.num_rows > 0:
            # Zero-copy convert to Polars
            yield pl.from_arrow(batch)
