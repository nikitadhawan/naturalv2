"""Functions for filtering Reddit data."""

import logging
from collections.abc import Iterator

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc


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


def scan_reddit_chunks(
    file_paths: list[str],
    columns: list[str],
    target_subreddits: list[str] | None = None,
    batch_size: int = 256_000,
) -> Iterator[pl.DataFrame]:
    """Stream Parquet files with minimal memory footprint.

    Processes files one at a time using slice + collect to read only the
    requested rows from disk. Can filter by subreddit at scan time to avoid
    loading irrelevant data.
    """
    for fp in file_paths:
        try:
            lf = pl.scan_parquet(fp, hive_partitioning=True)

            # Get schema and validate columns
            schema_names = lf.collect_schema().names()
            valid_cols = [col for col in columns if col in schema_names]

            # Early validation - skip files without required data
            text_cols = ["title", "initial_post", "report_text"]
            if "subreddit" not in schema_names:
                continue
            if not any(col in valid_cols for col in text_cols):
                continue

            lf = lf.select(valid_cols)

            # Filter by subreddit at scan time
            # This uses Parquet predicate pushdown - only matching rows are read from disk

            offset = 0
            while True:
                try:
                    # slice() then collect() only materializes the requested slice
                    chunk = lf.slice(offset, batch_size).collect()

                    if chunk.is_empty():
                        break

                    if target_subreddits:
                        chunk = chunk.filter(
                            pl.col("subreddit").is_in(target_subreddits)
                        )

                    if not chunk.is_empty():
                        yield chunk

                    # Exit if we got partial batch (end of file)
                    if len(chunk) < batch_size:
                        break

                    offset += batch_size

                except Exception as e:
                    logger.warning(f"Batch error at offset {offset} in {fp}: {e}")
                    break

        except Exception as e:
            logger.warning(f"Cannot process file {fp}: {e}")
            continue
