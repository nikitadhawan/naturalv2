"""Utilities for downloading Reddit data from The Eye archive."""

import logging
import os
import ssl
from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable, Literal, Optional, TypeVar
from urllib import error, request

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as paj
import re2
import wget
import zstandard as zstd
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from naturalv2.sources.reddit.api import is_retryable_error
from naturalv2.sources.reddit.processing import apply_rule_based_filter


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

RAW_RECORD_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("link_id", pa.string()),
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

# The following regex finds JSON-like key-value pairs where:
#   - The key is either "created_utc" or "score".
#   - The value is a quoted number (possibly negative, possibly decimal).
# It captures:
#   - The key (with colon and spaces)
#   - The numeric string value (without the quotes)
_DEQUOTE_BOTH = re2.compile(rb'((?:"created_utc"|"score")\s*:\s*)"(-?\d+(?:\.\d+)?)"')


_T = TypeVar("_T")

# Insecure SSL context for downloading Reddit data when TLS verification fails
_INSECURE_SSL_CONTEXT = ssl.create_default_context()
_INSECURE_SSL_CONTEXT.check_hostname = False
_INSECURE_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _with_tls_fallback(
    url: str,
    fetch: Callable[[str, Optional[ssl.SSLContext]], _T],
    *,
    description: str,
) -> _T:
    """Execute ``fetch`` for ``url`` with an insecure TLS fallback if verification fails."""

    # First attempt uses system verification
    try:
        return fetch(url, None)
    except error.URLError as exc:
        ssl_error = getattr(exc, "reason", None)
        is_ssl_failure = isinstance(ssl_error, ssl.SSLCertVerificationError)
        if not (is_ssl_failure and url.startswith("https://")):
            raise

        logger.warning(
            "TLS verification failed while %s from %s; retrying without certificate verification.",
            description,
            url,
        )
        return fetch(url, _INSECURE_SSL_CONTEXT)


def download_subs_list(data_path: str) -> str:
    """Download the list of subreddits from The Eye archive.

    Parameters
    ----------
    data_path : str
        The path where the list of subreddits will be saved.

    Returns
    -------
    str
        The path to the file containing the list of subreddits.

    """
    filepath = os.path.join(data_path, "subs_list.txt")
    if not os.path.exists(filepath):
        logger.info("Downloading the list of subreddits from The Eye archive...")
        url = "https://the-eye.eu/redarcs/"

        def _fetch_html(target_url: str, context: Optional[ssl.SSLContext]) -> str:
            if context is None:
                with request.urlopen(target_url) as response:
                    return response.read().decode("utf-8")

            with request.urlopen(target_url, context=context) as response:
                return response.read().decode("utf-8")

        html: str = _with_tls_fallback(
            url, _fetch_html, description="fetching subreddit index"
        )

        # Extract subreddit names from links
        subs = []
        for line in html.split("\n"):
            if "href=" in line and ".zst" in line:
                sub = line.split("href=")[1].split("_")[0].split("/")[-1]
                if sub not in subs:
                    subs.append(sub)

        with open(filepath, "w") as f:
            f.write("\n".join(subs))

        logger.info(f"{len(subs)} subreddits listed.")

    return filepath


@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(is_retryable_error),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def download_sub_data(
    subreddit: str, content_type: Literal["submissions", "comments"], data_path: str
) -> None:
    """Download subreddit data from The Eye archive.

    Parameters
    ----------
    subreddit : str
        The name of the subreddit to download data for.
    content_type : Literal["submissions", "comments"]
        The type of data to download, either "submissions" or "comments".
    data_path : str
        The path where the data will be saved.

    Raises
    -------
    ValueError
        If `content_type` is not "submissions" or "comments".
    """
    if content_type not in ["submissions", "comments"]:
        raise ValueError(
            "Expected ``content_type`` to be 'submissions' or 'comments', "
            f"but got {content_type}"
        )

    os.makedirs(data_path, exist_ok=True)

    save_path = os.path.join(data_path, f"{subreddit}_{content_type}.parquet")
    if os.path.exists(save_path):
        logger.warning(
            f"File {save_path} already exists. "
            f"Skipping download for {subreddit} {content_type}."
        )
        return

    file_path = os.path.join(data_path, f"{subreddit}_{content_type}.zst")
    if not os.path.exists(file_path):
        # Go to TMPDIR if set, otherwise stay current working directory, since
        # wget doesn't respect TMPDIR
        tmpdir = os.environ.get("TMPDIR", os.getcwd())
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            url = f"https://the-eye.eu/redarcs/files/{subreddit}_{content_type}.zst"

            def _download(target_url: str, context: Optional[ssl.SSLContext]) -> str:
                if context is None:
                    return wget.download(target_url, out=data_path, bar=None)

                opener = request.build_opener(request.HTTPSHandler(context=context))
                previous_opener = request._opener  # type: ignore[attr-defined]
                try:
                    request.install_opener(opener)
                    return wget.download(target_url, out=data_path, bar=None)
                finally:
                    if previous_opener is None:
                        request._opener = None  # type: ignore[attr-defined]
                    else:
                        request.install_opener(previous_opener)

            _with_tls_fallback(
                url,
                _download,
                description=f"downloading {subreddit} {content_type} archive",
            )
        finally:
            os.chdir(original_cwd)

        # TODO: If download is successful, return the path, otherwise return None

    # file_lines = 0
    # bad_lines = 0
    # data = []

    # for line, _ in _read_lines_zst(file_path):
    #     try:
    #         obj = json.loads(line)
    #         data += [obj]
    #     except (KeyError, json.JSONDecodeError):
    #         bad_lines += 1
    #     file_lines += 1

    # df = pd.DataFrame(data, dtype="string")

    # # remove deleted posts or comments
    # df = (
    #     df[~df["selftext"].isin(["[deleted]", "[removed]"])]
    #     if data_type == "submissions"
    #     else df[~df["body"].isin(["[deleted]", "[removed]"])]
    # )

    # df.loc[:, "score"] = pd.to_numeric(df["score"], errors="coerce")

    # cols_to_keep = ["id", "created_utc", "author", "permalink", "subreddit", "score"]
    # if data_type == "comments":
    #     cols_to_keep.append("link_id")

    # # anonymize dataframe
    # if anonymizer_instance is not None:
    #     df = anonymizer_instance.anonymize_dataframe(
    #         df,
    #         cols_to_keep=cols_to_keep,
    #         cols_to_anonymize=["selftext", "title"]
    #         if data_type == "submissions"
    #         else ["body"],
    #         data_source_name=f"{subreddit}_{data_type}",
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #     )

    # df.convert_dtypes(dtype_backend="pyarrow").to_parquet(
    #     save_path, index=False, compression="snappy"
    # )
    # os.remove(file_path)
    # logger.info(
    #     f"Completed download of {subreddit} {data_type} data with: {file_lines:,} lines "
    #     f"({bad_lines:,} bad lines) and {len(df):,} valid records."
    # )


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

    logger.info("Processing %s", zst_path)

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
    zst_path: str,
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

    logger.info("%s: Before filtering - %d rows", zst_path, table.num_rows)
    if table.num_rows == 0:
        logger.warning("Skipping chunk: parsed table has zero rows")
        return None

    content_type, keep_mask = _derive_content_type(table)
    if not pc.any(keep_mask).as_py():
        logger.warning(
            "%s: Skipping chunk: no rows remain after content filtering", zst_path
        )
        return None

    # Apply the mask
    table = table.filter(keep_mask)
    content_type = pc.filter(content_type, keep_mask)
    logger.warning("%s: After filtering - %d rows", zst_path, table.num_rows)

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
    has_selftext = apply_rule_based_filter(table, "selftext")
    has_body = apply_rule_based_filter(table, "body")

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
