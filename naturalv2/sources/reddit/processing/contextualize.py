import contextlib
import json
import logging
import os
import resource
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    NamedTuple,
    Sequence,
    Union,
)

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from ahocorasick import Automaton
from tqdm import tqdm

from naturalv2.sources.components import filter_by_date
from naturalv2.sources.components.helpers import (
    extract_mentions,
    normalize_text_for_matching,
)
from naturalv2.sources.reddit.processing._utils import (
    _author_replies_column,
    _bucket_from_subreddit,
    _build_comment_permalink_series,
    _build_report_text_array,
    _build_submission_permalink_series,
    _comment_permalink_array,
    _comment_post_id_array,
    _constant_string_array,
    _empty_list_array,
    _ensure_int64_array,
    _ensure_string_array,
    _filter_array,
    _format_timestamp_array,
    _mask_has_true,
    _non_empty_mask,
    _post_bucket_array,
    _submission_permalink_array,
    _submission_post_id_array,
    _unique_strings,
)


logger = logging.getLogger(__name__)

CONTEXTUALIZED_RECORD_SCHEMA = pa.schema(
    [
        ("subreddit", pa.string()),
        ("title", pa.string()),
        ("initial_post", pa.string()),
        ("report_text", pa.string()),
        ("report_type", pa.string()),
        ("score", pa.int64()),
        ("date_created", pa.string()),
        ("permalink", pa.string()),
        ("author_replies", pa.list_(pa.string())),
        ("content_type", pa.string()),
        ("bucket", pa.string()),
    ]
)

PARTITIONING = ds.partitioning(
    pa.schema([("content_type", pa.string()), ("bucket", pa.string())]),
    flavor="hive",
)

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


def write_to_parquet_partitions(
    data_stream: Iterable[pa.RecordBatch],
    output_dir: str,
    schema: pa.Schema,
    parquet_compression_level: int = 5,
    write_parquet_stats: Literal["none", "minimal", "all"] = "minimal",
    use_dictionary: bool | dict[str, bool] = True,
    max_partitions: int = 1024,
    min_rows_per_group: int = 128_000,
    max_rows_per_group: int = 256_000,
    max_open_files: int = 512,
    existing_data_behavior: Literal[
        "error", "delete_matching", "overwrite_or_ignore"
    ] = "overwrite_or_ignore",
    run_tag: str | None = None,
) -> list[str]:
    # Validate Literals
    if write_parquet_stats not in ["none", "minimal", "all"]:
        raise ValueError(
            "``write_parquet_stats`` must be one of 'none', 'minimal', or 'all'"
        )
    if existing_data_behavior not in [
        "error",
        "delete_matching",
        "overwrite_or_ignore",
    ]:
        raise ValueError(
            "``existing_data_behavior`` must be one of "
            "'error', 'delete_matching', or 'overwrite_or_ignore'"
        )

    # `output_dir` must not be a file
    if os.path.isfile(output_dir):
        raise ValueError(
            f"Expected output_dir to be a directory, but found file: {output_dir}"
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # All integers must be positive
    for param_name, param_value in [
        ("parquet_compression_level", parquet_compression_level),
        ("max_partitions", max_partitions),
        ("min_rows_per_group", min_rows_per_group),
        ("max_rows_per_group", max_rows_per_group),
        ("max_open_files", max_open_files),
    ]:
        if param_value <= 0:
            raise ValueError(f"{param_name} must be a positive integer")

    # `parquet_compression_level` must be between 1 and 22
    if not (1 <= parquet_compression_level <= 22):
        raise ValueError("``parquet_compression_level`` must be between 1 and 22")

    fmt = ds.ParquetFileFormat()
    write_opts = fmt.make_write_options(
        compression="zstd",
        use_dictionary=use_dictionary,
        compression_level=parquet_compression_level,
        write_statistics=write_parquet_stats,
    )

    basename_template = (
        f"{run_tag}-part-{{i}}.parquet" if run_tag is not None else "part-{i}.parquet"
    )

    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)

    written_paths: list[str] = []

    def visitor(file: ds.WrittenFile) -> None:
        written_paths.append(os.path.join(output_dir, file.path))

    effective_max_rows = max(1, max_rows_per_group)
    effective_min_rows = min(min_rows_per_group, effective_max_rows)
    if effective_min_rows <= 0:
        effective_min_rows = effective_max_rows

    ds.write_dataset(
        data=data_stream,
        base_dir=output_dir,
        basename_template=basename_template,
        format=fmt,
        partitioning=PARTITIONING,
        schema=schema,
        file_options=write_opts,
        max_partitions=max_partitions,
        min_rows_per_group=effective_min_rows,
        max_rows_per_group=effective_max_rows,
        max_open_files=min(max_open_files, soft_limit),
        existing_data_behavior=existing_data_behavior,
        file_visitor=visitor,
    )

    if written_paths:
        logger.debug("Wrote new parquet files: %s", ", ".join(written_paths))

    return written_paths


def build_contextualized_dataset(
    source_dir: str | Path | Sequence[str | Path],
    dest_dir: str | Path,
    *,
    subreddits: Sequence[str] | None = None,
    batch_size: int = 256_000,
    run_tag: str | None = None,
    cleanup_source: bool = False,
) -> list[str]:
    """Arrow-native contextualization pipeline for hive-partitioned Reddit data."""

    if subreddits is None:
        raise ValueError(
            "`subreddits` must be provided when building contextualized datasets."
        )
    normalized_subreddits = _normalize_subreddits(subreddits)
    if not normalized_subreddits:
        raise ValueError("No valid subreddit names provided for contextualization.")

    dest_path = Path(dest_dir)
    os.makedirs(dest_path, exist_ok=True)

    effective_run_tag = run_tag or "ctx"
    manifest_path = dest_path / "_context_manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with contextlib.suppress(Exception):
            manifest = json.loads(manifest_path.read_text())

    manifest_run_tag = manifest.get("run_tag")
    source_roots: Sequence[str | Path]
    source_roots = (
        [source_dir]
        if isinstance(source_dir, (str, os.PathLike, Path))
        else list(source_dir)
    )
    if manifest_run_tag == effective_run_tag:
        if not all(Path(root).exists() for root in source_roots):
            raise FileExistsError(
                f"Contextualized data already exists for run_tag '{effective_run_tag}' in "
                f"{dest_path}, but one or more source dirs {source_roots} are missing. "
                "Choose a new run_tag (experiment_name) or clean the destination."
            )

        processed_subs: set[str] = set()
        for root in source_roots:
            processed_subs.update(_processed_subreddits(root))

        if not processed_subs or not processed_subs.issuperset(
            set(normalized_subreddits)
        ):
            raise FileExistsError(
                f"Contextualized data already exists for run_tag '{effective_run_tag}' in "
                f"{dest_path}, but the processed subreddits ({sorted(processed_subs)}) do not "
                f"cover requested subreddits ({normalized_subreddits}). "
                "Choose a new run_tag or clean the destination."
            )
    else:
        existing_for_tag = sorted(dest_path.glob(f"{effective_run_tag}-part-*.parquet"))
        if existing_for_tag:
            raise FileExistsError(
                f"Contextualized data already exists for run_tag '{effective_run_tag}' in "
                f"{dest_path} (found {len(existing_for_tag)} parquet files). "
                "Choose a new run_tag or remove the existing outputs for this tag."
            )

    dest_parquet_files = _find_parquet_files(dest_path)
    source_parquet_files = _find_parquet_files(source_dir)
    if not source_parquet_files and dest_parquet_files:
        logger.info(
            "Skipping contextualized dataset build: no source partitions in %s and "
            "destination %s already populated with %d parquet files.",
            source_dir,
            dest_path,
            len(dest_parquet_files),
        )
        return dest_parquet_files

    dataset = ds.dataset(source_dir, format="parquet", partitioning="hive")
    submission_filter = _make_partition_filter(normalized_subreddits, "submissions")
    comment_filter = _make_partition_filter(normalized_subreddits, "comments")

    reply_store = _AuthorReplyStore()
    written: list[str] = []
    try:
        _collect_author_replies(
            dataset=dataset,
            comment_filter=comment_filter,
            submission_filter=submission_filter,
            batch_size=batch_size,
            store=reply_store,
        )

        batch_stream = _get_contextualized_batches(
            dataset=dataset,
            comment_filter=comment_filter,
            submission_filter=submission_filter,
            reply_store=reply_store,
            batch_size=batch_size,
        )

        written = write_to_parquet_partitions(
            data_stream=batch_stream,
            output_dir=str(dest_dir),
            schema=CONTEXTUALIZED_RECORD_SCHEMA,
            parquet_compression_level=5,
            write_parquet_stats="minimal",
            use_dictionary={"initial_posts": False},
            max_partitions=1024,
            min_rows_per_group=64_000,
            max_rows_per_group=64_000,
            max_open_files=512,
            # existing_data_behavior="error", # TODO
            run_tag=effective_run_tag,
        )
        _write_manifest(
            manifest_path,
            {
                "run_tag": effective_run_tag,
                "subreddits": normalized_subreddits,
                "source_dir": [str(root) for root in source_roots],
                "ts": int(time.time()),
            },
        )
    finally:
        reply_store.close()
        if cleanup_source:
            _cleanup_source_dir(source_dir)

    return written


def _normalize_subreddits(values: Sequence[str]) -> list[str]:
    cleaned = []
    for value in values:
        if value and isinstance(value, str):
            candidate = value.strip()
            if candidate:
                cleaned.append(candidate)
    # dict.fromkeys preserves order while de-duping
    return list(dict.fromkeys(cleaned))


def _make_partition_filter(
    subreddits: Sequence[str], content_type: str | None
) -> ds.Expression:
    expr: ds.Expression = ds.field("subreddit").isin(
        pa.array(subreddits, type=pa.string())
    )
    buckets = sorted({(sub[:1].lower() if sub else "_") for sub in subreddits})
    expr = expr & ds.field("bucket").isin(pa.array(buckets or ["_"], type=pa.string()))

    if content_type:
        expr = expr & ds.field("content_type") == pa.scalar(content_type)
    return expr


class _AuthorReplyStore:
    """Spill author replies to disk per post bucket to keep memory bounded."""

    def __init__(self, tmp_root: str | Path | None = None) -> None:
        root = (
            Path(tmp_root)
            if tmp_root
            else Path(tempfile.mkdtemp(prefix="reddit-replies-"))
        )
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._files_by_bucket: defaultdict[str, list[Path]] = defaultdict(list)
        self._lock = threading.Lock()
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        shutil.rmtree(self._root, ignore_errors=True)
        self._closed = True

    def add_replies(
        self, *, post_ids: pa.Array, replies: pa.Array, created_utc: pa.Array
    ) -> None:
        if len(post_ids) == 0:
            return

        pid_arr = _ensure_string_array(post_ids)
        reply_arr = _ensure_string_array(replies)
        bucket_arr = _post_bucket_array(pid_arr)
        for bucket in _unique_strings(bucket_arr):
            mask = pc.equal(bucket_arr, pa.scalar(bucket, pa.string()))
            if not _mask_has_true(mask):
                continue
            subset = pa.table(
                {
                    "post_id": _filter_array(pid_arr, mask),
                    "reply": _filter_array(reply_arr, mask),
                    "created_utc": _filter_array(created_utc, mask),
                }
            )
            if subset.num_rows == 0:
                continue

            bucket_dir = self._root / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
            file_path = bucket_dir / f"{uuid.uuid4().hex}.ipc"
            with (
                pa.OSFile(str(file_path), "wb") as sink,
                pa.ipc.new_file(sink, subset.schema) as writer,
            ):
                writer.write_table(subset)
            with self._lock:
                self._files_by_bucket[bucket].append(file_path)

    def fetch(self, post_ids: pa.Array | Sequence[str]) -> dict[str, list[str]]:
        values = (
            post_ids.to_pylist()
            if isinstance(post_ids, (pa.Array, pa.ChunkedArray))
            else list(post_ids)
        )
        result: dict[str, list[str]] = defaultdict(list)
        if not values:
            return result

        ids_by_bucket: dict[str, set[str]] = defaultdict(set)
        for post_id in values:
            if not post_id:
                continue

            if not post_id:
                bucket = "_"
            else:
                first = post_id[0].lower()
                bucket = first if first else "_"
            ids_by_bucket[bucket].add(post_id)

        for bucket, ids in ids_by_bucket.items():
            if not ids:
                continue
            bucket_dir = self._root / bucket
            if not bucket_dir.is_dir():
                continue
            dataset = ds.dataset(str(bucket_dir), format="ipc")
            filter_expr = ds.field("post_id").isin(
                pa.array(sorted(ids), type=pa.string())
            )
            table = dataset.to_table(filter=filter_expr)
            table.sort_by([("post_id", "ascending"), ("created_utc", "ascending")])
            if table.num_rows == 0:
                continue
            for post_id, reply in zip(
                table.column("post_id").to_pylist(),
                table.column("reply").to_pylist(),
            ):
                result[post_id].append(reply)
        return dict(result)


class PreparedCommentBatch(NamedTuple):
    aligned: pa.RecordBatch
    matched_post_ids: pa.Array
    indices: pa.Array
    lookup_table: pa.Table


def _prepare_comment_batch(
    *, batch: pa.RecordBatch, lookup_fn: Callable[[pa.Array], pa.Table | None]
) -> PreparedCommentBatch | None:
    if batch.num_rows == 0:
        return None

    post_ids = _comment_post_id_array(batch.column("link_id"))
    # Fast-path: drop rows without a usable post_id before any joins
    valid_mask = _non_empty_mask(post_ids)
    if not _mask_has_true(valid_mask):
        return None

    filtered = batch.filter(valid_mask)
    post_ids = _filter_array(post_ids, valid_mask)

    lookup_table = lookup_fn(post_ids)
    if lookup_table is None or lookup_table.num_rows == 0:
        return None

    submission_ids = lookup_table.column("post_id")
    # Locate each comment's post_id within the lookup table
    indices = pc.index_in(post_ids, submission_ids)
    # Keep only rows whose post_ids are present in the lookup_table
    matched_mask = pc.not_equal(indices, -1)
    if not _mask_has_true(matched_mask):
        return None

    indices = _filter_array(indices, matched_mask)
    matched_post_ids = _filter_array(post_ids, matched_mask)
    aligned = filtered.filter(matched_mask)
    return PreparedCommentBatch(aligned, matched_post_ids, indices, lookup_table)


def _collect_author_replies(
    *,
    dataset: ds.Dataset,
    comment_filter: ds.Expression,
    submission_filter: ds.Expression,
    batch_size: int,
    store: "_AuthorReplyStore",
) -> None:
    scanner = dataset.scanner(
        filter=comment_filter,
        columns=["link_id", "author", "body", "created_utc"],
        batch_size=batch_size,
        use_threads=True,
    )
    for batch in tqdm(
        scanner.to_batches(),
        desc="Collecting author replies",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        prepared = _prepare_comment_batch(
            batch=batch,
            lookup_fn=lambda post_ids: _fetch_submission_author_table(
                dataset=dataset,
                submission_filter=submission_filter,
                post_ids=post_ids,
            ),
        )
        if prepared is None:
            continue

        aligned = prepared.aligned
        matched_post_ids = prepared.matched_post_ids
        indices = prepared.indices
        submission_authors = prepared.lookup_table.column("author")

        # Case-insensitive author equality on aligned rows
        comment_authors = pc.utf8_lower(_ensure_string_array(aligned.column("author")))
        target_authors = pc.take(submission_authors, indices, boundscheck=False)
        target_authors = pc.fill_null(target_authors, pa.scalar("", pa.string()))
        is_author_reply = pc.equal(comment_authors, target_authors)

        # Create a mask for author replies that are not null
        has_body = pc.is_valid(aligned.column("body"))
        # Combine author match + presence of a body to find candidate replies
        reply_mask = pc.and_(is_author_reply, has_body)
        if not _mask_has_true(reply_mask):
            continue

        # Get the post_ids, replies and timestamp
        reply_post_ids = _filter_array(matched_post_ids, reply_mask)
        if len(reply_post_ids) == 0:
            continue

        reply_bodies = _filter_array(
            _ensure_string_array(aligned.column("body")), reply_mask
        )
        reply_created_utc = _filter_array(aligned.column("created_utc"), reply_mask)

        # Add to store to spill to disk
        store.add_replies(
            post_ids=reply_post_ids, replies=reply_bodies, created_utc=reply_created_utc
        )


def _comment_record_batches(
    *,
    dataset: ds.Dataset,
    comment_filter: ds.Expression,
    submission_filter: ds.Expression,
    reply_store: "_AuthorReplyStore",
    batch_size: int,
) -> Generator[pa.RecordBatch, Any, None]:
    scanner = dataset.scanner(
        filter=comment_filter,
        columns=[
            "id",
            "link_id",
            "author",
            "body",
            "score",
            "created_utc",
            "permalink",
        ],
        batch_size=batch_size,
        use_threads=True,
    )
    for batch in tqdm(
        scanner.to_batches(),
        desc="Writing comments",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        if batch.num_rows == 0:
            continue
        record_batch = _build_comment_record_batch(
            batch=batch,
            dataset=dataset,
            submission_filter=submission_filter,
            reply_store=reply_store,
        )
        if record_batch is not None:
            yield record_batch


def _submission_record_batches(
    *,
    dataset: ds.Dataset,
    submission_filter: ds.Expression,
    reply_store: "_AuthorReplyStore",
    batch_size: int,
) -> Generator[pa.RecordBatch, Any, None]:
    scanner = dataset.scanner(
        filter=submission_filter,
        columns=[
            "id",
            "subreddit",
            "title",
            "selftext",
            "score",
            "created_utc",
            "permalink",
            "author",
        ],
        batch_size=batch_size,
        use_threads=True,
    )
    for batch in tqdm(
        scanner.to_batches(),
        desc="Writing submissions",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        if batch.num_rows == 0:
            continue
        record_batch = _build_submission_record_batch(
            batch=batch, reply_store=reply_store
        )
        if record_batch is not None:
            yield record_batch


def _build_comment_record_batch(
    *,
    batch: pa.RecordBatch,
    dataset: ds.Dataset,
    submission_filter: ds.Expression,
    reply_store: "_AuthorReplyStore",
) -> pa.RecordBatch | None:
    prepared = _prepare_comment_batch(
        batch=batch,
        lookup_fn=lambda post_ids: _fetch_submission_context_table(
            dataset=dataset,
            submission_filter=submission_filter,
            post_ids=post_ids,
            reply_store=reply_store,
        ),
    )
    if prepared is None:
        return None

    filtered = prepared.aligned
    post_ids = prepared.matched_post_ids
    indices = prepared.indices
    context_table = prepared.lookup_table

    target_authors = pc.take(
        context_table.column("author_normalized"), indices, boundscheck=False
    )
    comment_authors = pc.utf8_lower(_ensure_string_array(filtered.column("author")))
    is_author_reply = pc.equal(
        comment_authors, pc.fill_null(target_authors, pa.scalar("", pa.string()))
    )
    # Exclude author replies; we only want comments from other users
    non_author_mask = pc.invert(is_author_reply)
    if not _mask_has_true(non_author_mask):
        return None

    indices = _filter_array(indices, non_author_mask)
    filtered = filtered.filter(non_author_mask)
    post_ids = _filter_array(post_ids, non_author_mask)

    # Guard against any lingering invalid indices before using them in a take()
    valid_idx_mask = pc.and_(pc.is_valid(indices), pc.greater_equal(indices, 0))
    if not _mask_has_true(valid_idx_mask):
        return None

    indices = _filter_array(indices, valid_idx_mask)
    filtered = filtered.filter(valid_idx_mask)
    post_ids = _filter_array(post_ids, valid_idx_mask)

    indices = pc.cast(indices, pa.int64(), safe=False)
    # Align submission context to the filtered comment rows
    context_rows = context_table.take(indices)

    submission_subreddit = _ensure_string_array(context_rows.column("subreddit"))
    submission_title = _ensure_string_array(context_rows.column("title"))
    submission_report_text = _ensure_string_array(context_rows.column("report_text"))

    num_rows = filtered.num_rows
    if num_rows == 0:
        return None

    report_text = _ensure_string_array(filtered.column("body"))
    score = _ensure_int64_array(filtered.column("score"))
    date_created = _format_timestamp_array(filtered.column("created_utc"))
    permalink = _comment_permalink_array(
        existing=_ensure_string_array(filtered.column("permalink"), default=""),
        post_ids=post_ids,
        comment_ids=_ensure_string_array(filtered.column("id")),
    )
    report_type = _constant_string_array("comment", num_rows)
    author_replies = _empty_list_array(num_rows)
    content_type = _constant_string_array("comments", num_rows)
    bucket = _bucket_from_subreddit(submission_subreddit)

    batch_arrays = [
        submission_subreddit,
        submission_title,
        submission_report_text,
        report_text,
        report_type,
        score,
        date_created,
        permalink,
        author_replies,
        content_type,
        bucket,
    ]
    return pa.RecordBatch.from_arrays(batch_arrays, schema=CONTEXTUALIZED_RECORD_SCHEMA)


def _build_submission_record_batch(
    *,
    batch: pa.RecordBatch,
    reply_store: "_AuthorReplyStore",
) -> pa.RecordBatch | None:
    post_ids = _submission_post_id_array(batch.column("id"))
    valid_mask = _non_empty_mask(post_ids)
    if not _mask_has_true(valid_mask):
        return None

    filtered = batch.filter(valid_mask)
    post_ids = _filter_array(post_ids, valid_mask)

    if filtered.num_rows == 0:
        return None

    subreddit = _ensure_string_array(filtered.column("subreddit"))
    title = _ensure_string_array(filtered.column("title"))
    base_text = _ensure_string_array(filtered.column("selftext"))
    # Rehydrate replies for each submission so we can append them to the text
    reply_lookup = reply_store.fetch(post_ids)

    report_text = _build_report_text_array(
        base_text=base_text,
        post_ids=post_ids,
        reply_lookup=reply_lookup,
    )
    initial_post = _constant_string_array("", filtered.num_rows)
    report_type = _constant_string_array("submission", filtered.num_rows)
    score = _ensure_int64_array(filtered.column("score"))
    date_created = _format_timestamp_array(filtered.column("created_utc"))
    permalink = _submission_permalink_array(
        existing=_ensure_string_array(filtered.column("permalink"), default=""),
        subreddits=subreddit,
        post_ids=post_ids,
    )
    replies_column = _author_replies_column(post_ids, reply_lookup)
    content_type = _constant_string_array("submissions", filtered.num_rows)
    bucket = _bucket_from_subreddit(subreddit)

    batch_arrays = [
        subreddit,
        title,
        initial_post,
        report_text,
        report_type,
        score,
        date_created,
        permalink,
        replies_column,
        content_type,
        bucket,
    ]
    return pa.RecordBatch.from_arrays(batch_arrays, schema=CONTEXTUALIZED_RECORD_SCHEMA)


def _get_contextualized_batches(
    *,
    dataset: ds.Dataset,
    comment_filter: ds.Expression,
    submission_filter: ds.Expression,
    reply_store: "_AuthorReplyStore",
    batch_size: int,
) -> Generator[pa.RecordBatch, Any, None]:
    yield from _comment_record_batches(
        dataset=dataset,
        comment_filter=comment_filter,
        submission_filter=submission_filter,
        reply_store=reply_store,
        batch_size=batch_size,
    )
    yield from _submission_record_batches(
        dataset=dataset,
        submission_filter=submission_filter,
        reply_store=reply_store,
        batch_size=batch_size,
    )


def _fetch_submission_author_table(
    *,
    dataset: ds.Dataset,
    submission_filter: ds.Expression,
    post_ids: pa.Array,
) -> pa.Table | None:
    # Query only the distinct submission ids present in the current batch
    unique_ids = _unique_strings(post_ids)
    if not unique_ids:
        return None

    filter_expr = submission_filter & ds.field("id").isin(
        pa.array(unique_ids, type=pa.string())
    )
    table = dataset.to_table(filter=filter_expr, columns=["id", "author"])
    if table.num_rows == 0:
        return None
    post_id_arr = _submission_post_id_array(table.column("id"))
    author_arr = pc.utf8_lower(_ensure_string_array(table.column("author")))
    return pa.table({"post_id": post_id_arr, "author": author_arr})


def _fetch_submission_context_table(
    *,
    dataset: ds.Dataset,
    submission_filter: ds.Expression,
    post_ids: pa.Array,
    reply_store: "_AuthorReplyStore",
) -> pa.Table | None:
    unique_ids = _unique_strings(post_ids)
    if not unique_ids:
        return None
    filter_expr = submission_filter & ds.field("id").isin(
        pa.array(unique_ids, type=pa.string())
    )
    columns = [
        "id",
        "subreddit",
        "title",
        "selftext",
        "score",
        "created_utc",
        "permalink",
        "author",
    ]
    table = dataset.to_table(filter=filter_expr, columns=columns)
    if table.num_rows == 0:
        return None

    post_id_arr = _submission_post_id_array(table.column("id"))
    subreddit = _ensure_string_array(table.column("subreddit"))
    title = _ensure_string_array(table.column("title"))
    # Pull any previously collected author replies for these submissions
    reply_lookup = reply_store.fetch(post_id_arr)

    report_text = _build_report_text_array(
        base_text=_ensure_string_array(table.column("selftext")),
        post_ids=post_id_arr,
        reply_lookup=reply_lookup,
    )
    score = _ensure_int64_array(table.column("score"))
    created = table.column("created_utc")
    permalink = _submission_permalink_array(
        existing=_ensure_string_array(table.column("permalink"), default=""),
        subreddits=subreddit,
        post_ids=post_id_arr,
    )
    bucket = _bucket_from_subreddit(subreddit)
    author_norm = pc.utf8_lower(_ensure_string_array(table.column("author")))

    return pa.table(
        {
            "post_id": post_id_arr,
            "subreddit": subreddit,
            "title": title,
            "report_text": report_text,
            "score": score,
            "created_utc": created,
            "permalink": permalink,
            "bucket": bucket,
            "author_normalized": author_norm,
        }
    )


def _find_parquet_files(
    roots: str | Path | Sequence[str | Path],
) -> list[str]:
    """Return sorted parquet file paths under one or more roots."""
    root_list: Sequence[str | Path]
    root_list = [roots] if isinstance(roots, (str, Path, os.PathLike)) else list(roots)

    files: set[str] = set()
    for root in root_list:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("*.parquet"):
            if path.is_file():
                files.add(str(path))
    return sorted(files)


def _cleanup_source_dir(
    source_dir: str | Path, preserved: Sequence[str] = (".processed",)
) -> None:
    """Remove everything under ``source_dir`` except the preserved entries."""
    base = Path(source_dir)
    if not base.exists():
        return

    for entry in base.iterdir():
        if entry.name in preserved:
            continue
        with contextlib.suppress(Exception):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()


def _processed_subreddits(source_dir: str | Path) -> set[str]:
    """Return the set of subreddits marked processed in the source directory."""
    archives_dir = _proc_dir(source_dir) / "archives"
    if not archives_dir.exists():
        return set()

    subreddits: set[str] = set()
    for done_file in archives_dir.glob("*.done"):
        stem = done_file.stem  # "subreddit-content_type"
        if "-" in stem:
            subreddits.add(stem.split("-", 1)[0])
    return subreddits


def _proc_dir(root: str | Path) -> Path:
    path = Path(root) / ".processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _archive_done_path(root: str | Path, archive_id: str) -> Path:
    path = _proc_dir(root) / "archives"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{archive_id}.done"


def _compacted_done_path(root: str | Path, archive_id: str) -> Path:
    path = _proc_dir(root) / "compacted"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{archive_id}.done"


def is_archive_processed(root: str | Path, archive_id: str) -> bool:
    return _archive_done_path(root, archive_id).exists()


def is_archive_compacted(root: str | Path, archive_id: str) -> bool:
    return _compacted_done_path(root, archive_id).exists()


def _write_manifest(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, path)  # atomic on POSIX


def mark_archive_done(root: str | Path, archive_id: str, **extra) -> None:
    _write_manifest(
        _archive_done_path(root, archive_id),
        {"archive_id": archive_id, "ts": int(time.time()), **extra},
    )


def mark_archive_compacted(root: str | Path, archive_id: str, **extra) -> None:
    _write_manifest(
        _compacted_done_path(root, archive_id),
        {"archive_id": archive_id, "ts": int(time.time()), **extra},
    )


def scan_subreddit(
    base_dir: str,
    subreddits: str | Sequence[str],
    content_type: str | None = None,
    columns: list[str] | None = None,
    batch_size: int = 256_000,
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


def get_context_post_df(
    submissions: pd.DataFrame, comments: pd.DataFrame
) -> pd.DataFrame:
    """Join submissions and comments DataFrames to create a context post DataFrame.

    Parameters
    ----------
    submissions : pd.DataFrame
        DataFrame containing submission data with columns including:
        - id
        - subreddit
        - title
        - selftext
        - author
        - score
        - created_utc
        - permalink
    comments : pd.DataFrame
        DataFrame containing comment data with columns including:
        - id
        - link_id
        - body
        - author
        - score
        - created_utc
        - permalink

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the context posts with the following columns:
        - subreddit
        - title
        - initial_post
        - report
        - score
        - date_created
        - permalink
        - author_replies (list of replies from the author)
    """
    submissions = submissions.copy()

    submissions["post_id"] = (
        submissions.get("id", pd.Series(index=submissions.index, dtype="object"))
        .astype("string")
        .str.lower()
    )
    submissions = submissions[submissions["post_id"].notna()].copy()

    submissions["date_created"] = pd.to_datetime(
        submissions["created_utc"], unit="s", errors="coerce"
    ).dt.strftime("%B %d, %Y")

    # Normalize the submission permalink
    if "permalink" not in submissions.columns:
        submissions["permalink"] = pd.NA
    missing_submission_permalink = submissions["permalink"].isna() | (
        submissions["permalink"].astype("string").str.len() == 0
    )
    submissions.loc[missing_submission_permalink, "permalink"] = (
        _build_submission_permalink_series(
            submissions.loc[missing_submission_permalink].get("subreddit", ""),
            submissions.loc[missing_submission_permalink, "post_id"],
        )
    )
    submissions = submissions.rename(columns={"permalink": "submission_permalink"})

    comments = comments.copy()

    if "link_id" in comments.columns:
        comments["post_id"] = (
            comments["link_id"]
            .astype("string")
            .str.lower()
            .str.replace(r"^t3_", "", regex=True)
        )
    else:
        comments["post_id"] = pd.Series(pd.NA, index=comments.index, dtype="object")
    comments = comments[comments["post_id"].notna()].copy()

    comments["date_created"] = pd.to_datetime(
        comments["created_utc"], unit="s", errors="coerce"
    ).dt.strftime("%B %d, %Y")

    # Normalize the comment permalink
    if "permalink" not in comments.columns:
        comments["permalink"] = pd.NA

    need_comment_permalink = comments["permalink"].isna() | (
        comments["permalink"].astype("string").str.len() == 0
    )
    if "id" in comments.columns:
        has_ids = need_comment_permalink & comments["id"].notna()
        comments.loc[has_ids, "permalink"] = _build_comment_permalink_series(
            comments.loc[has_ids, "post_id"], comments.loc[has_ids, "id"]
        )

    # Map post_id to submission author
    author_map = submissions.set_index("post_id")["author"]
    comments["is_author_reply"] = comments["post_id"].map(author_map).fillna("").astype(
        "string"
    ) == comments.get("author", "").astype("string").fillna("")

    # Gather author's replies per post (as list of strings)
    author_replies = (
        comments.loc[comments["is_author_reply"], ["post_id", "body"]]
        .assign(body=lambda df: df["body"].astype("string"))
        .groupby("post_id", sort=False)["body"]
        .agg(list)
        .rename("author_replies")
    )

    # Merge author replies into submissions, filling non replies with empty list
    submissions = submissions.merge(
        author_replies, left_on="post_id", right_index=True, how="left"
    ).reset_index(drop=True)
    submissions["author_replies"] = submissions["author_replies"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Build 'report_text' column, appending author replies when present
    submissions["report_text"] = (
        submissions.get("selftext", "").astype("string").fillna("")
    )
    has_replies = submissions["author_replies"].str.len().gt(0)

    if has_replies.any():
        quoted_suffix = (
            submissions.loc[has_replies, "author_replies"]
            .apply(
                lambda replies: "\n\nThe original poster also replied with the following comments in the thread:"
                + "".join("\n> " + str(reply) for reply in replies)
            )
            .astype("string")
        )
        submissions.loc[has_replies, "report_text"] = submissions.loc[
            has_replies, "report_text"
        ].str.cat(quoted_suffix, na_rep="")

    # Build output rows for each submission
    output_submissions = pd.DataFrame(
        {
            "subreddit": submissions.get(
                "subreddit", pd.Series(index=submissions.index, dtype="object")
            ),
            "title": submissions.get(
                "title", pd.Series(index=submissions.index, dtype="object")
            ),
            "initial_post": "",
            "report_text": submissions["report_text"],
            "report_type": "submission",
            "score": pd.to_numeric(submissions.get("score", 0), errors="coerce")
            .fillna(0)
            .astype(int),
            "date_created": submissions["date_created"],
            "permalink": submissions["submission_permalink"],
            "author_replies": submissions["author_replies"],
        }
    )

    # Build output rows per non-author comment, joined to submission context
    non_author = comments.loc[~comments["is_author_reply"]].copy()
    ctx = submissions[["post_id", "subreddit", "title", "report_text"]]
    non_author = non_author.merge(ctx, on="post_id", how="left", suffixes=("", "_sub"))

    output_comments = pd.DataFrame(
        {
            "subreddit": non_author["subreddit"],
            "title": non_author["title"],
            "initial_post": non_author["report_text"],
            "report_text": non_author.get("body", "").astype("string"),
            "report_type": "comment",
            "score": pd.to_numeric(non_author.get("score", 0), errors="coerce")
            .fillna(0)
            .astype(int),
            "date_created": non_author["date_created"],
            "permalink": non_author["permalink"],
            "author_replies": [[] for _ in range(len(non_author))],
        }
    )

    return pd.concat([output_submissions, output_comments], ignore_index=True)
