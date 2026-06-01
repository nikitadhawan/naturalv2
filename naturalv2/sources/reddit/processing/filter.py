"""Functions for filtering Reddit data."""

import logging
from collections.abc import Iterator

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from naturalv2.sources.reddit.processing._utils import (
    bucket_from_subreddit,
    release_memory,
)


logger = logging.getLogger(__name__)


SENTINELS = pa.array(["[deleted]", "[removed]"], type=pa.string())


def apply_rule_based_filter(table: pa.Table, text_field: str) -> pa.ChunkedArray:
    """Apply rule-based filtering to a pyarrow Table of Reddit posts.

    This function filters out low-quality Reddit posts by checking multiple criteria:
    - Removes deleted/removed posts (text is "[deleted]" or "[removed]")
    - Removes posts by bot accounts (e.g., "AutoModerator", names ending in "bot")
    - Removes posts without meaningful text (after stripping URLs)
    - Requires at least one word with 3+ letters in the first 2048 characters
    - Requires at least 25% of characters to be letters

    The function also normalizes the text by:
    - Replacing HTML entities (&gt;, &lt;, &amp;)
    - Collapsing whitespace runs into single spaces
    - Trimming leading/trailing whitespace

    Parameters
    ----------
    table : pa.Table
        A pyarrow Table containing Reddit posts. Must have columns named
        `text_field` (specified by parameter) and `author`.
    text_field : str
        The name of the column containing the text content to be filtered
        (typically "body" for comments or "selftext" for submissions).

    Returns
    -------
    pa.ChunkedArray
        A boolean mask array where True indicates the row passes all filters
        and should be kept, False indicates it should be filtered out.

    Examples
    --------
    >>> import pyarrow as pa
    >>> table = pa.table(
    ...     {
    ...         "body": ["Hello world!", "[deleted]", "Check https://example.com"],
    ...         "author": ["user123", "deleted_user", "AutoModerator"],
    ...     }
    ... )
    >>> mask = apply_rule_based_filter(table, "body")
    >>> filtered_table = table.filter(mask)

    """

    # Helper to cast to string and fill nulls
    def to_string_filled(arr: pa.ChunkedArray) -> pa.ChunkedArray:
        """
        Convert null values in an array to empty strings.

        Parameters
        ----------
        arr : pa.ChunkedArray
            Input array that may contain null values.

        Returns
        -------
        pa.ChunkedArray
            Array with all null values replaced by empty strings.

        """
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
    data_source: str | list[str],
    schema: pa.Schema | None = None,
    partitioning: ds.Partitioning | str = "hive",
    columns: str | list[str] | None = None,
    subreddit: str | list[str] | None = None,
    batch_size: int = 65_536,
    use_threads: bool = True,
) -> Iterator[pl.DataFrame]:
    """Scan a Reddit parquet dataset and yield batches as Polars DataFrames.

    This function provides memory-efficient streaming access to a large Reddit
    partitioned parquet dataset by reading data in batches rather than loading
    everything into memory at once.
    It supports filtering by subreddit and selecting specific columns.

    The function is optimized for low memory usage by:
    - Using batched reading with configurable batch sizes
    - Disabling readahead to prevent loading data before it's needed
    - Explicitly releasing memory after processing each batch
    - Using buffered streams for efficient I/O

    Parameters
    ----------
    data_source : str or list of str
        Path(s) to the Parquet dataset. Can be a single directory, file,
        or a list of paths.
    schema : pa.Schema, optional, default=None
        PyArrow schema for the dataset. If ``None``, the schema will be inferred
        from the data files.
    partitioning : ds.Partitioning or str, default="hive"
        Partitioning scheme used in the dataset. "hive" means the data uses
        Hive-style partitioning (e.g., "subreddit=AskReddit/bucket=0/file.parquet").
    columns : str or list of str, optional
        Column name(s) to read from the dataset. If ``None``, all columns are read.
        Selecting specific columns reduces memory usage and improves performance.
    subreddit : str or list of str, optional
        Filter data to only include posts from these subreddit(s). If ``None``,
        all subreddits are included.
    batch_size : int, default=65536
        Number of rows to read in each batch. Larger batches are more I/O-efficient
        but use more memory.
    use_threads : bool, default=True
        Whether to use multiple threads for reading. Can speed up I/O but may
        increase memory usage.

    Yields
    ------
    pl.DataFrame
        Polars DataFrame containing one batch of data from the dataset.
        Each DataFrame has up to `batch_size` rows.

    Examples
    --------
    >>> # Read all data from a dataset
    >>> for batch in scan_reddit_dataset("/path/to/reddit/data"):
    ...     print(f"Processing {len(batch)} rows")
    ...     # Process the batch

    >>> # Read only specific columns from specific subreddits
    >>> for batch in scan_reddit_dataset(
    ...     "/path/to/reddit/data",
    ...     columns=["body", "author", "created_utc"],
    ...     subreddit=["AskReddit", "science"],
    ...     batch_size=10000,
    ... ):
    ...     # Process filtered data
    ...     pass

    Notes
    -----
    The function fails fast on fragment read errors to avoid silently returning
    incomplete results.
    Memory is aggressively released after each batch to prevent accumulation.

    """
    dataset = ds.dataset(
        data_source, schema=schema, format="parquet", partitioning=partitioning
    )

    fragment_filter_expr = None
    scanner_filter_expr = None
    if subreddit:
        if isinstance(subreddit, str):
            subreddit = [subreddit]

        scanner_filter_expr = ds.field("subreddit").isin(subreddit)
        if "bucket" in dataset.schema.names:
            buckets = bucket_from_subreddit(pa.array(subreddit)).to_pylist()
            # Only apply bucket pruning at fragment selection time; applying the full
            # (bucket + subreddit) filter again at scanner level can fail when
            # partition fields are not materialized as scan columns.
            fragment_filter_expr = ds.field("bucket").isin(buckets)
        else:
            fragment_filter_expr = scanner_filter_expr

    if isinstance(columns, str):
        columns = [columns]
    if columns:
        available_columns = set(dataset.schema.names)
        columns = [column for column in columns if column in available_columns]

    fragments = dataset.get_fragments(fragment_filter_expr)

    for fragment in fragments:
        try:
            # Create a localized scanner for this file only
            scanner = fragment.scanner(
                columns=columns,
                batch_size=batch_size,
                filter=scanner_filter_expr,
                batch_readahead=0,  # Don't load Batch 2 until Batch 1 is done
                fragment_readahead=0,  # Don't open File B until File A is done
                fragment_scan_options=ds.ParquetFragmentScanOptions(
                    use_buffered_stream=True,
                    buffer_size=16 << 20,  # 16 MiB
                    pre_buffer=False,
                ),
                cache_metadata=True,
                use_threads=use_threads,
            )

            for batch in scanner.to_reader():
                if batch.num_rows > 0:
                    yield pl.from_arrow(batch)

                    del batch

            del scanner

        except Exception:
            logger.exception("Error scanning fragment %s", fragment.path)
            # Fail fast so callers do not silently receive incomplete results.
            raise
        finally:
            release_memory()


def get_subreddit_filter_expr(subreddits: list[str]) -> pc.Expression:
    """Create a PyArrow filter expression for selecting specific subreddits.

    This function creates an optimized filter that checks both the bucket
    (a hash-based partition key) and subreddit name. Filtering by bucket
    first is more efficient because it allows skipping entire partition
    directories without opening the files.

    Parameters
    ----------
    subreddits : list of str
        List of subreddit names to filter for (e.g., ['AskReddit', 'science']).
        Names should match exactly as they appear in the dataset.

    Returns
    -------
    pc.Expression
        A PyArrow compute expression that evaluates to True for rows where
        the subreddit matches one of the provided names. This expression
        can be passed to dataset scanning functions.

    Examples
    --------
    >>> subreddits = ["AskReddit", "science"]
    >>> filter_expr = get_subreddit_filter_expr(subreddits)
    >>> # Use with dataset scanning

    Notes
    -----
    The function uses a two-level filter:
    1. bucket.isin(buckets) - Fast partition-level filtering
    2. subreddit.isin(subreddits) - Exact name matching

    This approach is much faster than filtering by subreddit alone when
    working with partitioned datasets.

    """
    buckets = bucket_from_subreddit(pa.array(subreddits)).to_pylist()
    return ds.field("bucket").isin(buckets) & ds.field("subreddit").isin(subreddits)
