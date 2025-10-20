import argparse
import logging
import os
import resource
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.json as paj
import re2
import zstandard as zstd
from tqdm import tqdm


logger = logging.getLogger(__name__)

RAW_RECORD_SCHEMA = pa.schema(
    [
        ("created_utc", pa.int64()),
        ("author", pa.string()),
        ("permalink", pa.string()),
        ("subreddit", pa.string()),
        ("score", pa.float64()),
        ("title", pa.string()),
        ("selftext", pa.string()),
        ("body", pa.string()),
    ]
)

CONTENT_TYPE_FIELD = pa.field("content_type", pa.string())
BUCKET_FIELD = pa.field("bucket", pa.string())

PROCESSED_RECORD_SCHEMA = RAW_RECORD_SCHEMA.append(CONTENT_TYPE_FIELD).append(
    BUCKET_FIELD
)

SENTINELS = pa.array(["[deleted]", "[removed]"], type=pa.string())

# This regex finds JSON-like key-value pairs where:
#   - The key is either "created_utc" or "score".
#   - The value is a quoted number (possibly negative, possibly decimal).
# It captures:
#   - The key (with colon and spaces)
#   - The numeric string value (without the quotes)
_DEQUOTE_BOTH = re2.compile(rb'((?:"created_utc"|"score")\s*:\s*)"(-?\d+(?:\.\d+)?)"')


def iter_bucketed_batches(
    zst_path: str, chunk_size: int = 256 << 20, *, progress_enabled: bool = True
) -> Iterator[tuple[str, pa.RecordBatch]]:
    # Verify that the file exists and is a .zst file
    if not zst_path.endswith(".zst"):
        raise ValueError(f"File {zst_path} is not a .zst file")
    if not os.path.exists(zst_path):
        raise FileNotFoundError(f"File {zst_path} does not exist")

    reader_pbar: tqdm | None = None
    if progress_enabled:
        file_size = os.path.getsize(zst_path)
        reader_pbar = tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Streaming {Path(zst_path).name}",
            leave=False,
            dynamic_ncols=True,
        )

    try:
        for chunk in iter_zst_ndjson_blocks(
            zst_path, chunk_size=chunk_size, tqdm_pbar=reader_pbar
        ):
            table = _parse_ndjson_bytes_to_table(chunk)
            if table is None:
                continue

            # Yield chunks by bucket
            for bucket in pc.unique(table["bucket"]).to_pylist():
                mask = pc.equal(table["bucket"], pa.scalar(bucket))
                for batch in table.filter(mask).to_batches(max_chunksize=256_000):
                    yield bucket, batch
    finally:
        if reader_pbar is not None:
            reader_pbar.close()


def iter_zst_ndjson_blocks(
    zst_path: str,
    *,
    chunk_size: int = 256 << 20,  # 256 MiB
    max_window_size: int = 2 << 30,  # 2 GiB
    read_across_frames: bool = True,
    tqdm_pbar: tqdm | None = None,
) -> Iterator[bytes | bytearray]:
    """Stream decompressed NDJSON chunks from a Zstandard archive.

    Parameters
    ----------
    zst_path : str
        Filesystem path to a `.zst` archive.
    chunk_size : int, optional, default=256<<20
        Number of decompressed bytes to request per read, defaulting to
        about 256 megabytes. Keeping this large amortises I/O and Python call
        overhead, while still capping the working-set size for downstream
        consumers.
    max_window_size : int, optional, default=2<<30
        Maximum dictionary window advertised to the decompressor. This guards
        against “decompression bomb” archives that claim enormous windows and
        would otherwise allocate unbounded memory.
    read_across_frames : bool, optional, default=True
        If True (default), the reader seamlessly continues into subsequent
        zstd frames so archives composed of multiple concatenated frames are
        handled without special casing.
    tqdm_pbar : tqdm.tqdm or None, optional
        Optional progress bar to update with the number of compressed bytes
        consumed. Passing ``None`` leaves progress reporting to the caller.

    Yields
    ------
    bytes | bytearray
        Consecutive NDJSON byte blocks, each containing only complete newline-
        terminated records suitable for direct parsing by fast JSON readers.

    Raises
    ------
    ValueError
        If ``path`` does not end with ``.zst``.
    FileNotFoundError
        If the target archive cannot be located.

    Notes
    -----
    Internally the decompressor reuses a single scratch buffer to keep memory churn
    low, but each emitted chunk is copied into its own bytes or bytearray. That
    means later iterations won’t overwrite earlier chunks and callers can keep or
    modify what they’ve already received without needing defensive copies.
    """

    # Verify that the file exists and is a .zst file
    if not zst_path.endswith(".zst"):
        raise ValueError(f"File {zst_path} is not a .zst file")
    if not os.path.exists(zst_path):
        raise FileNotFoundError(f"File {zst_path} does not exist")

    with open(zst_path, "rb") as file_handle:  # 'rb': read binary
        # Create a decompression context
        dctx = zstd.ZstdDecompressor(max_window_size=max_window_size)

        # Wrap the file handle in a stream reader so decompression is done in chunks
        # instead of reading the entire file into memory
        with dctx.stream_reader(
            file_handle, read_size=chunk_size, read_across_frames=read_across_frames
        ) as reader:
            # Get the current stream position
            last_tell = file_handle.tell()

            # Create a reusable buffer to read the chunks into
            read_buffer = bytearray(chunk_size)

            # Expose the buffer as a memoryview for efficient reading and slicing
            # We don't need to copy the buffer because the memoryview is a view of
            # the buffer and will be updated in place when the buffer is updated
            buffer_mem_view = memoryview(read_buffer)

            # Create a buffer to hold partial NDJSON lines that straddle chunk
            # boundaries so nothing is lost between reads
            carry = bytearray()

            while True:
                # Fill read_buffer[:num_bytes_read] in place (i.e. same memory address)
                # with the bytes read from the file
                num_bytes_read = reader.readinto(buffer_mem_view)
                if not num_bytes_read:  # End of file (num_bytes_read==0)
                    break

                # Update the progress bar if it is provided
                if tqdm_pbar is not None:
                    current_position = file_handle.tell()
                    delta = current_position - last_tell
                    if delta:
                        tqdm_pbar.update(delta)
                        last_tell = current_position

                # Find the index of the last newline character in the freshly-read
                # chunk; start=0, end=num_bytes_read
                last_newline_index = read_buffer.rfind(b"\n", 0, num_bytes_read)
                if last_newline_index == -1:
                    # No newline character found in the freshly-read chunk, which
                    # means the entire chunk is part of a split NDJSON line. So
                    # we add the entire chunk to the carry buffer and continue
                    # reading until a newline character is found.
                    carry.extend(buffer_mem_view[:num_bytes_read])
                    continue

                prefix_view = buffer_mem_view[: last_newline_index + 1]
                if carry:
                    # Concatenate the accumulated bytes in the carry buffer with the
                    # full line (including the newline character) from the current chunk
                    carry.extend(prefix_view)
                    emit = carry  # hand off buffer without copying (i.e. no `bytes()`)
                    carry = bytearray()  # start fresh buffer for partial lines
                else:
                    # We have a full NDJSON record, make a copy to yield
                    emit = bytes(prefix_view)
                yield emit

                # Seed the new carry buffer with remaining bytes after final newline
                # so the next iteration will prepend them to the new chunk
                carry.extend(buffer_mem_view[last_newline_index + 1 : num_bytes_read])

            if carry:
                # File doesn't end with a newline; Flush the final NDJSON line
                yield carry


def _parse_ndjson_bytes_to_table(
    chunk: bytes | bytearray,
    read_block_size: int = 8 << 20,  # 8 MiB
) -> pa.Table | None:
    """Parse raw NDJSON bytes or bytearray to ``pyarrow.Table``."""
    # Specify read and write options for the parser
    read_options = paj.ReadOptions(
        block_size=min(len(chunk), read_block_size), use_threads=True
    )
    parse_options = paj.ParseOptions(
        explicit_schema=RAW_RECORD_SCHEMA, unexpected_field_behavior="ignore"
    )

    try:
        table = paj.read_json(
            pa.py_buffer(chunk), read_options=read_options, parse_options=parse_options
        )
    except pa.ArrowInvalid as exc:
        msg = str(exc)
        if ("changed from number to string" not in msg) and (
            "changed from string to number" not in msg
        ):
            raise

        # Arrow detected mixed types (strings and number) in 'created_utc' or 'score'
        # column, so for rows with string data types, we remove the quotes around
        # the value so that arrow can parse it as a number
        sanitized = _dequote_numeric_fields(chunk)
        table = paj.read_json(
            pa.py_buffer(sanitized),
            read_options=read_options,
            parse_options=parse_options,
        )

    if table.num_rows == 0:
        logger.debug("Skipping chunk: parsed table has zero rows")
        return None

    content_type, keep_mask = _derive_content_type(table)
    if not pc.any(keep_mask).as_py():
        logger.debug("Skipping chunk: no rows remain after content filtering")
        return None

    # Apply the mask
    table = table.filter(keep_mask)
    content_type = pc.filter(content_type, keep_mask)

    # Add the derived content_type column to the table
    table = table.append_column("content_type", content_type)

    # Construct and add the 'bucket' column to the table
    subreddit_lower = pc.ascii_lower(table["subreddit"])
    first_letter = pc.utf8_slice_codeunits(subreddit_lower, 0, 1)
    bucket = pc.if_else(pc.is_valid(first_letter), first_letter, pa.scalar("_"))
    return table.append_column("bucket", bucket).select(PROCESSED_RECORD_SCHEMA.names)


def _dequote_numeric_fields(raw: bytes | bytearray | memoryview) -> bytes:
    """
    De-quote purely numeric scalars for known fields in a single pass with
    a single output allocation. If there are no matches, returns the input
    (as bytes) without copying.
    """
    payload = raw if isinstance(raw, (bytes, bytearray, memoryview)) else bytes(raw)

    # Skip regex work quickly when the known numeric fields are absent.
    if (b'"created_utc"' not in payload) and (b'"score"' not in payload):
        return payload if isinstance(payload, bytes) else bytes(payload)

    numeric_field_matches = list(_DEQUOTE_BOTH.finditer(payload))
    if not numeric_field_matches:
        return payload if isinstance(payload, bytes) else bytes(payload)

    # Allocate the de-quoted buffer once; every match removes exactly two quotes.
    output_length = len(payload) - 2 * len(numeric_field_matches)
    output_buffer = bytearray(output_length)

    write_index = 0  # Current write position in the output buffer.
    last_copied_position = 0  # Last byte in the source that has been copied.
    payload_view = memoryview(payload)  # Zero-copy slicing view over the payload.

    for match in numeric_field_matches:
        prefix_start, prefix_end = match.span(1)
        digits_start, digits_end = match.span(2)

        # Copy any untouched bytes before the prefix (still quoted data).
        untouched_span = prefix_start - last_copied_position
        output_buffer[write_index : write_index + untouched_span] = payload_view[
            last_copied_position:prefix_start
        ]
        write_index += untouched_span

        # Copy the prefix, which already excludes the opening quote.
        prefix_length = prefix_end - prefix_start
        output_buffer[write_index : write_index + prefix_length] = payload_view[
            prefix_start:prefix_end
        ]
        write_index += prefix_length

        # Copy the numeric digits to keep them unquoted.
        digits_length = digits_end - digits_start
        output_buffer[write_index : write_index + digits_length] = payload_view[
            digits_start:digits_end
        ]
        write_index += digits_length

        # Skip the closing quote by advancing past the full match.
        last_copied_position = match.end()

    # Append the remaining tail segment after the final match.
    output_buffer[write_index:] = payload_view[last_copied_position:]
    return bytes(output_buffer)


def _derive_content_type(table: pa.Table) -> tuple[pa.ChunkedArray, pa.ChunkedArray]:
    """Infer row-wise content type signals for a Reddit Arrow table.

    Parameters
    ----------
    table : pa.Table
        Arrow table expected to contain ``selftext`` and ``body`` columns that
        each encode post content from submissions or comments, respectively

    Returns
    -------
    tuple of pa.ChunkedArray
        - The first element is a UTF-8 chunked array with values
          ``"submissions"``, ``"comments"``, or ``"unknown"`` chosen per row.
        - The second element is a boolean chunked array marking rows that have
          at least one usable text field and should be retained for downstream
          processing.
    """
    # Use the filter rules to identify rows with valid submission and comment text.
    has_selftext = _apply_rule_based_filter(table, "selftext")
    has_body = _apply_rule_based_filter(table, "body")

    # Keep rows when either text field survives the quality filter.
    keep_mask = pc.or_(has_selftext, has_body)

    make_string = lambda value: pa.scalar(value, type=pa.string())
    # Prefer submission labeling when both fields exist; fall back to comments or unknown.
    content_type = pc.if_else(
        has_selftext,
        make_string("submissions"),
        pc.if_else(has_body, make_string("comments"), make_string("unknown")),
    )

    return content_type, keep_mask


def _apply_rule_based_filter(table: pa.Table, text_field: str) -> pa.ChunkedArray:
    """Construct a boolean mask that keeps rows containing meaningful human-written text.

    Parameters
    ----------
    table : pa.Table
        Arrow table containing Reddit submission or comment data.
    text_field : str
        Name of the UTF-8 column in ``table`` holding the free-text payload.

    Returns
    -------
    pa.ChunkedArray
        Chunked Boolean array indicating which rows satisfy all quality checks.

    Notes
    -----
    The resulting mask enforces multiple heuristics:
    - Normalizes the text field by filling nulls, unescaping basic HTML entities, replacing control characters, and trimming whitespace.
    - Rejects empty strings and records that match known sentinel content such as deleted or removed posts.
    - Requires a permalink and removes rows authored by obvious bot accounts.
    - Ensures the first 2,048 code units include at least one space, signalling multi-token text.
    - Verifies that alphabetic characters (plus spaces) make up at least half of the trimmed text.
    """
    empty_string = pa.scalar("", type=pa.string())

    # Normalize the text field:
    # fill nulls
    text_field_values = pc.fill_null(table[text_field], empty_string)

    # decode HTML entities
    normalized_text = pc.replace_substring(text_field_values, "&gt;", ">")
    normalized_text = pc.replace_substring(normalized_text, "&lt;", "<")
    normalized_text = pc.replace_substring(normalized_text, "&amp;", "&")

    # collapse whitespace
    normalized_text = pc.replace_substring(normalized_text, "\n", " ")
    normalized_text = pc.replace_substring(normalized_text, "\t", " ")
    trimmed_text = pc.utf8_trim_whitespace(normalized_text)

    # Require non-empty text that is not one of the known sentinel strings
    has_text = pc.greater(pc.utf8_length(trimmed_text), 0)
    matched_sentinel = pc.is_in(trimmed_text, value_set=SENTINELS)
    valid_text_mask = pc.and_(has_text, pc.invert(matched_sentinel))

    # Require a permalink to ensure the row points back to Reddit content
    permalink_values = pc.fill_null(table["permalink"], empty_string)
    permalink_trimmed = pc.utf8_trim_whitespace(permalink_values)
    has_permalink = pc.greater(pc.utf8_length(permalink_trimmed), 0)
    valid_text_mask = pc.and_(valid_text_mask, has_permalink)

    # Filter out obvious bot accounts based on the author name
    author_values = pc.fill_null(table["author"], empty_string)
    author_lower = pc.ascii_lower(author_values)
    bot_position = pc.find_substring(author_lower, "bot")
    is_bot_author = pc.not_equal(bot_position, pa.scalar(-1, type=bot_position.type))
    valid_text_mask = pc.and_(valid_text_mask, pc.invert(is_bot_author))

    # Require at least one space early in the text to catch multi-word content
    preview_window = pc.utf8_slice_codeunits(trimmed_text, 0, 2048)
    space_index = pc.find_substring(preview_window, " ")
    contains_space = pc.not_equal(space_index, pa.scalar(-1, type=space_index.type))
    valid_text_mask = pc.and_(valid_text_mask, contains_space)

    # Favour text that is mostly alphabetic characters or spaces
    total_length = pc.utf8_length(trimmed_text)
    alphabetic_or_space = pc.count_substring_regex(trimmed_text, r"[A-Za-z ]")
    at_least_half_alpha = pc.greater_equal(
        pc.multiply(alphabetic_or_space, 2), total_length
    )

    # Only evaluate the alphabetic heuristic for rows that already passed prior checks.
    return pc.if_else(pc.and_(valid_text_mask, has_text), at_least_half_alpha, False)


from collections.abc import Iterable


def _stream_record_batches(archive_dir_or_paths: str | Iterable[str]):
    """
    Yield record batches from one or more Reddit archive sources.

    Parameters
    ----------
    archive_dir_or_paths : str or iterable of str
        A single `.zst` path, a directory, or an iterable mixing files and directories.
        Directories are searched recursively.
    """
    normalized_inputs = _normalize_archive_inputs(archive_dir_or_paths)
    zst_files = list(_discover_zst_archives(normalized_inputs))
    if not zst_files:
        logger.error("No .zst archives found for input: %s", archive_dir_or_paths)
        return

    logger.info("Found %d .zst files to process", len(zst_files))
    for archive_path in zst_files:
        logger.info("Processing file: %s", archive_path.name)
        for _, batch in iter_bucketed_batches(str(archive_path)):
            if batch.num_rows:
                yield batch


def _normalize_archive_inputs(raw_inputs: str | Iterable[str]) -> list[Path]:
    if isinstance(raw_inputs, (str, os.PathLike)):
        return [Path(raw_inputs).expanduser()]
    try:
        return [Path(p).expanduser() for p in raw_inputs]
    except TypeError as exc:
        raise TypeError(
            "Expected a path string or an iterable of path strings/directories."
        ) from exc


def _discover_zst_archives(candidates: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield from _iter_zst_archives(resolved)


def _iter_zst_archives(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() == ".zst":
            yield path
        else:
            logger.warning("File %s is not a .zst archive; skipping.", path)
        return

    if not path.is_dir():
        logger.warning("Path %s is neither a file nor a directory; skipping.", path)
        return

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".zst"):
                yield Path(dirpath, filename).resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Reddit .zst archives into a partitioned Parquet dataset."
    )
    parser.add_argument(
        "archives",
        nargs="+",
        help="One or more .zst files or directories containing .zst files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the partitioned Parquet dataset.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    fmt = ds.ParquetFileFormat()
    write_opts = fmt.make_write_options(
        compression="zstd",
        use_dictionary=True,
        compression_level=5,
        write_statistics="minimal",
    )
    partitioning = ds.partitioning(
        pa.schema([("content_type", pa.string()), ("bucket", pa.string())]),
        flavor="hive",
    )

    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)

    ds.write_dataset(
        data=_stream_record_batches(args.archives),
        base_dir=args.output_dir,
        format=fmt,
        file_options=write_opts,
        partitioning=partitioning,
        schema=PROCESSED_RECORD_SCHEMA,
        basename_template="part-{i}.parquet",
        max_partitions=1024,
        min_rows_per_group=256_000,
        max_rows_per_group=1_000_000,
        max_rows_per_file=5_000_000,
        max_open_files=min(256, soft_limit),
        existing_data_behavior="overwrite_or_ignore",
    )


if __name__ == "__main__":
    main()
