"""Build contextualized datasets from Reddit parquet sources.

This module provides functionality to:
- Build contextualized datasets from Reddit submissions and comments
- Write parquet datasets with hive partitioning
- Track archive processing status
"""

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
from pathlib import Path
from typing import Any, Callable, Generator, Literal, NamedTuple, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from tqdm import tqdm

from naturalv2.sources.reddit.processing._utils import (
    BUCKET_COUNT,
    author_replies_column,
    bucket_from_subreddit,
    build_comment_permalink_array,
    build_report_text_array,
    build_submission_permalink_array,
    comment_post_id_array,
    constant_string_array,
    empty_list_array,
    ensure_int64_array,
    ensure_string_array,
    ensure_timestamp_array,
    filter_array,
    format_timestamp_array,
    mask_has_true,
    non_empty_mask,
    post_bucket_array,
    submission_post_id_array,
    unique_strings,
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
    """Write a stream of record batches to a hive-partitioned parquet dataset.

    Parameters
    ----------
    data_stream : Iterable[pa.RecordBatch]
        Record batches to persist.
    output_dir : str
        Destination directory for the dataset. Created if missing.
    schema : pa.Schema
        Schema to enforce on the written dataset.
    parquet_compression_level : int, default=5
        Compression level for zstd codec (1–22).
    write_parquet_stats : {'none', 'minimal', 'all'}, default='minimal'
        Parquet statistics granularity to emit.
    use_dictionary : bool or dict[str, bool], default=True
        Dictionary encoding toggle, either global or column-level.
    max_partitions : int, default=1024
        Maximum distinct partition directories.
    min_rows_per_group : int, default=128_000
        Minimum rows per row group (bounded by ``max_rows_per_group``).
    max_rows_per_group : int, default=256_000
        Maximum rows per row group.
    max_open_files : int, default=512
        Cap on concurrently open files during write.
    existing_data_behavior : {'error', 'delete_matching', 'overwrite_or_ignore'}, \
default='overwrite_or_ignore'
        Strategy when output already exists.
    run_tag : str or None, optional
        Optional prefix for generated parquet filenames.

    Returns
    -------
    list[str]
        Full paths of parquet files written.

    Raises
    ------
    ValueError
        If parameter values are invalid (e.g., non-positive ints, bad literal choices),
        or ``output_dir`` is a file.
    """
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
        """Collect the full path of each file written by the dataset writer."""
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
    """
    Build a contextualized hive-partitioned dataset from Reddit parquet sources.

    Parameters
    ----------
    source_dir : str or pathlib.Path or Sequence[str | pathlib.Path]
        Root path(s) containing submission/comment parquet partitions.
    dest_dir : str or pathlib.Path
        Destination directory for contextualized parquet outputs.
    subreddits : Sequence[str], optional
        Required list of subreddit names to include.
    batch_size : int, default=256_000
        Scanner batch size for reading source data.
    run_tag : str or None, optional
        Tag appended to output filenames to avoid collisions (default ``"ctx"``).
    cleanup_source : bool, default=False
        When True, remove source files after successful processing.

    Returns
    -------
    list[str]
        Paths to contextualized parquet files written.

    Raises
    ------
    ValueError
        If ``subreddits`` is missing/empty or params are invalid.
    FileExistsError
        If outputs for the given ``run_tag`` already exist and conflict with inputs.
    """
    from naturalv2.sources.reddit.pushshift_archive import (  # noqa: PLC0415
        _write_manifest,
    )

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

    skip_result = _validate_and_check_skip(
        manifest_run_tag=manifest_run_tag,
        effective_run_tag=effective_run_tag,
        source_roots=source_roots,
        normalized_subreddits=normalized_subreddits,
        dest_path=dest_path,
    )
    if skip_result is not None:
        return skip_result

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
            write_parquet_stats="all",
            use_dictionary={
                "title": False,
                "initial_post": False,
                "report_text": False,
                "author_replies": False,
                "score": False,
                "permalink": False,
                "date_created": False,
                "report_type": True,
                "content_type": True,
                "bucket": True,
                "subreddit": True,
            },
            max_partitions=1024,
            min_rows_per_group=64_000,
            max_rows_per_group=128_000,
            max_open_files=512,
            existing_data_behavior="overwrite_or_ignore",
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


# ============================================================================
# Dataset Building Core Functions
# ============================================================================


def _validate_and_check_skip(
    *,
    manifest_run_tag: str | None,
    effective_run_tag: str,
    source_roots: Sequence[str | Path],
    normalized_subreddits: list[str],
    dest_path: Path,
) -> list[str] | None:
    """
    Validate manifest and check if processing should be skipped.

    Returns
    -------
    list[str] | None
        If processing should be skipped, returns list of existing parquet files.
        Otherwise returns None.
    """
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

        # All validations passed - check if destination files exist and skip if they do
        dest_parquet_files = _find_parquet_files(dest_path)
        if dest_parquet_files:
            logger.info(
                "Skipping contextualized dataset build: data already exists for run_tag '%s' in "
                "%s with %d parquet files.",
                effective_run_tag,
                dest_path,
                len(dest_parquet_files),
            )
            return dest_parquet_files
    else:
        existing_for_tag = sorted(dest_path.glob(f"{effective_run_tag}-part-*.parquet"))
        if existing_for_tag:
            raise FileExistsError(
                f"Contextualized data already exists for run_tag '{effective_run_tag}' in "
                f"{dest_path} (found {len(existing_for_tag)} parquet files). "
                "Choose a new run_tag or remove the existing outputs for this tag."
            )

    return None


def _normalize_subreddits(values: Sequence[str]) -> list[str]:
    """Strip whitespace, drop empties, and de-duplicate subreddit names."""
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
    """Build a dataset filter matching subreddits, buckets, and optional content type."""
    fallback_bucket = str(0).zfill(len(str(BUCKET_COUNT - 1)))
    subreddit_arr = pa.array(subreddits, type=pa.string())
    expr: ds.Expression = ds.field("subreddit").isin(subreddit_arr)
    bucket_labels = unique_strings(bucket_from_subreddit(subreddit_arr))
    expr = expr & ds.field("bucket").isin(
        pa.array(bucket_labels or [fallback_bucket], type=pa.string())
    )

    if content_type:
        expr = expr & (ds.field("content_type") == pa.scalar(content_type))
    return expr


# ============================================================================
# Data Store Classes
# ============================================================================


class _AuthorReplyStore:
    """Spill author replies to disk per post bucket to keep memory bounded."""

    def __init__(self, tmp_root: str | Path | None = None) -> None:
        """Initialize a temp directory for spilling replies, optionally under tmp_root."""
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
        """Delete any spilled reply files and mark the store closed."""
        if self._closed:
            return
        shutil.rmtree(self._root, ignore_errors=True)
        self._closed = True

    def add_replies(
        self, *, post_ids: pa.Array, replies: pa.Array, created_utc: pa.Array
    ) -> None:
        """Persist replies grouped by first-character bucket to bounded directories."""
        if len(post_ids) == 0:
            return

        pid_arr = ensure_string_array(post_ids)
        reply_arr = ensure_string_array(replies)
        bucket_arr = post_bucket_array(pid_arr)
        for bucket in unique_strings(bucket_arr):
            mask = pc.equal(bucket_arr, pa.scalar(bucket, pa.string()))
            if not mask_has_true(mask):
                continue
            subset = pa.table(
                {
                    "post_id": filter_array(pid_arr, mask),
                    "reply": filter_array(reply_arr, mask),
                    "created_utc": filter_array(created_utc, mask),
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
        """Load replies for the provided post ids, ordered by creation time."""
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
    """Container for a prepared comment batch with aligned submission metadata."""

    aligned: pa.RecordBatch
    matched_post_ids: pa.Array
    indices: pa.Array
    lookup_table: pa.Table


# ============================================================================
# Batch Processing Functions
# ============================================================================


def _prepare_comment_batch(
    *, batch: pa.RecordBatch, lookup_fn: Callable[[pa.Array], pa.Table | None]
) -> PreparedCommentBatch | None:
    """Align a comment batch to submission metadata, filtering rows without matches."""
    if batch.num_rows == 0:
        return None

    post_ids = comment_post_id_array(batch.column("link_id"))
    # Fast-path: drop rows without a usable post_id before any joins
    valid_mask = non_empty_mask(post_ids)
    if not mask_has_true(valid_mask):
        return None

    filtered = batch.filter(valid_mask)
    post_ids = filter_array(post_ids, valid_mask)

    lookup_table = lookup_fn(post_ids)
    if lookup_table is None or lookup_table.num_rows == 0:
        return None

    submission_ids = lookup_table.column("post_id")
    # Locate each comment's post_id within the lookup table
    indices = pc.index_in(post_ids, submission_ids)
    # Keep only rows whose post_ids are present in the lookup_table
    matched_mask = pc.not_equal(indices, -1)
    if not mask_has_true(matched_mask):
        return None

    indices = filter_array(indices, matched_mask)
    matched_post_ids = filter_array(post_ids, matched_mask)
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
    """Collect author replies from comments and spill them to the reply store."""
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
        comment_authors = pc.utf8_lower(ensure_string_array(aligned.column("author")))
        target_authors = pc.take(submission_authors, indices, boundscheck=False)
        target_authors = pc.fill_null(target_authors, pa.scalar("", pa.string()))
        is_author_reply = pc.equal(comment_authors, target_authors)

        # Create a mask for author replies that are not null
        has_body = pc.is_valid(aligned.column("body"))
        # Combine author match + presence of a body to find candidate replies
        reply_mask = pc.and_(is_author_reply, has_body)
        if not mask_has_true(reply_mask):
            continue

        # Get the post_ids, replies and timestamp
        reply_post_ids = filter_array(matched_post_ids, reply_mask)
        if len(reply_post_ids) == 0:
            continue

        reply_bodies = filter_array(
            ensure_string_array(aligned.column("body")), reply_mask
        )
        reply_created_utc = filter_array(aligned.column("created_utc"), reply_mask)

        # Add to store to spill to disk
        store.add_replies(
            post_ids=reply_post_ids, replies=reply_bodies, created_utc=reply_created_utc
        )


def _get_contextualized_batches(
    *,
    dataset: ds.Dataset,
    comment_filter: ds.Expression,
    submission_filter: ds.Expression,
    reply_store: "_AuthorReplyStore",
    batch_size: int,
) -> Generator[pa.RecordBatch, Any, None]:
    """Chain together comment and submission batch generators."""
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


def _comment_record_batches(
    *,
    dataset: ds.Dataset,
    comment_filter: ds.Expression,
    submission_filter: ds.Expression,
    reply_store: "_AuthorReplyStore",
    batch_size: int,
) -> Generator[pa.RecordBatch, Any, None]:
    """Yield contextualized comment record batches, skipping author replies."""
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
    """Yield contextualized submission record batches."""
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
    """Construct a contextualized comment batch joined with submission context."""
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
    comment_authors = pc.utf8_lower(ensure_string_array(filtered.column("author")))
    is_author_reply = pc.equal(
        comment_authors, pc.fill_null(target_authors, pa.scalar("", pa.string()))
    )
    # Exclude author replies; we only want comments from other users
    non_author_mask = pc.invert(is_author_reply)
    if not mask_has_true(non_author_mask):
        return None

    indices = filter_array(indices, non_author_mask)
    filtered = filtered.filter(non_author_mask)
    post_ids = filter_array(post_ids, non_author_mask)

    # Guard against any lingering invalid indices before using them in a take()
    valid_idx_mask = pc.and_(pc.is_valid(indices), pc.greater_equal(indices, 0))
    if not mask_has_true(valid_idx_mask):
        return None

    indices = filter_array(indices, valid_idx_mask)
    filtered = filtered.filter(valid_idx_mask)
    post_ids = filter_array(post_ids, valid_idx_mask)

    indices = pc.cast(indices, pa.int64(), safe=False)
    # Align submission context to the filtered comment rows
    context_rows = context_table.take(indices)

    submission_subreddit = ensure_string_array(context_rows.column("subreddit"))
    submission_title = ensure_string_array(context_rows.column("title"))
    submission_report_text = ensure_string_array(context_rows.column("report_text"))

    num_rows = filtered.num_rows
    if num_rows == 0:
        return None
    created_ts = ensure_timestamp_array(filtered.column("created_utc"))

    report_text = ensure_string_array(filtered.column("body"))
    score = ensure_int64_array(filtered.column("score"))
    date_created = format_timestamp_array(created_ts)
    permalink = build_comment_permalink_array(
        existing=ensure_string_array(filtered.column("permalink"), default=""),
        post_ids=post_ids,
        comment_ids=ensure_string_array(filtered.column("id")),
    )
    report_type = constant_string_array("comment", num_rows)
    author_replies = empty_list_array(num_rows)
    content_type = constant_string_array("comments", num_rows)
    bucket = bucket_from_subreddit(submission_subreddit)

    arrays = {
        "subreddit": submission_subreddit,
        "title": submission_title,
        "initial_post": submission_report_text,
        "report_text": report_text,
        "report_type": report_type,
        "score": score,
        "date_created": date_created,
        "permalink": permalink,
        "author_replies": author_replies,
        "content_type": content_type,
        "bucket": bucket,
    }
    sorted_arrays = _sort_batch_arrays_by_subreddit_and_created(
        arrays, created_ts=created_ts
    )
    batch_arrays = [
        sorted_arrays["subreddit"],
        sorted_arrays["title"],
        sorted_arrays["initial_post"],
        sorted_arrays["report_text"],
        sorted_arrays["report_type"],
        sorted_arrays["score"],
        sorted_arrays["date_created"],
        sorted_arrays["permalink"],
        sorted_arrays["author_replies"],
        sorted_arrays["content_type"],
        sorted_arrays["bucket"],
    ]
    return pa.RecordBatch.from_arrays(batch_arrays, schema=CONTEXTUALIZED_RECORD_SCHEMA)


def _build_submission_record_batch(
    *,
    batch: pa.RecordBatch,
    reply_store: "_AuthorReplyStore",
) -> pa.RecordBatch | None:
    """Construct a contextualized submission batch with appended author replies."""
    post_ids = submission_post_id_array(batch.column("id"))
    valid_mask = non_empty_mask(post_ids)
    if not mask_has_true(valid_mask):
        return None

    filtered = batch.filter(valid_mask)
    post_ids = filter_array(post_ids, valid_mask)

    if filtered.num_rows == 0:
        return None

    subreddit = ensure_string_array(filtered.column("subreddit"))
    title = ensure_string_array(filtered.column("title"))
    base_text = ensure_string_array(filtered.column("selftext"))
    # Rehydrate replies for each submission so we can append them to the text
    reply_lookup = reply_store.fetch(post_ids)

    report_text = build_report_text_array(
        base_text=base_text,
        post_ids=post_ids,
        reply_lookup=reply_lookup,
    )
    initial_post = constant_string_array("", filtered.num_rows)
    report_type = constant_string_array("submission", filtered.num_rows)
    score = ensure_int64_array(filtered.column("score"))
    created_ts = ensure_timestamp_array(filtered.column("created_utc"))
    date_created = format_timestamp_array(created_ts)
    permalink = build_submission_permalink_array(
        existing=ensure_string_array(filtered.column("permalink"), default=""),
        subreddits=subreddit,
        post_ids=post_ids,
    )
    replies_column = author_replies_column(post_ids, reply_lookup)
    content_type = constant_string_array("submissions", filtered.num_rows)
    bucket = bucket_from_subreddit(subreddit)

    arrays = {
        "subreddit": subreddit,
        "title": title,
        "initial_post": initial_post,
        "report_text": report_text,
        "report_type": report_type,
        "score": score,
        "date_created": date_created,
        "permalink": permalink,
        "author_replies": replies_column,
        "content_type": content_type,
        "bucket": bucket,
    }
    sorted_arrays = _sort_batch_arrays_by_subreddit_and_created(
        arrays, created_ts=created_ts
    )
    batch_arrays = [
        sorted_arrays["subreddit"],
        sorted_arrays["title"],
        sorted_arrays["initial_post"],
        sorted_arrays["report_text"],
        sorted_arrays["report_type"],
        sorted_arrays["score"],
        sorted_arrays["date_created"],
        sorted_arrays["permalink"],
        sorted_arrays["author_replies"],
        sorted_arrays["content_type"],
        sorted_arrays["bucket"],
    ]
    return pa.RecordBatch.from_arrays(batch_arrays, schema=CONTEXTUALIZED_RECORD_SCHEMA)


# ============================================================================
# Submission/Context Fetching Functions
# ============================================================================


def _fetch_submission_author_table(
    *,
    dataset: ds.Dataset,
    submission_filter: ds.Expression,
    post_ids: pa.Array,
) -> pa.Table | None:
    """Fetch author ids for a set of submission ids needed for reply matching."""
    # Query only the distinct submission ids present in the current batch
    unique_ids = unique_strings(post_ids)
    if not unique_ids:
        return None

    filter_expr = submission_filter & ds.field("id").isin(
        pa.array(unique_ids, type=pa.string())
    )
    table = dataset.to_table(filter=filter_expr, columns=["id", "author"])
    if table.num_rows == 0:
        return None
    post_id_arr = submission_post_id_array(table.column("id"))
    author_arr = pc.utf8_lower(ensure_string_array(table.column("author")))
    return pa.table({"post_id": post_id_arr, "author": author_arr})


def _fetch_submission_context_table(
    *,
    dataset: ds.Dataset,
    submission_filter: ds.Expression,
    post_ids: pa.Array,
    reply_store: "_AuthorReplyStore",
) -> pa.Table | None:
    """Fetch submission context used to populate contextualized comment rows."""
    unique_ids = unique_strings(post_ids)
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

    post_id_arr = submission_post_id_array(table.column("id"))
    subreddit = ensure_string_array(table.column("subreddit"))
    title = ensure_string_array(table.column("title"))
    # Pull any previously collected author replies for these submissions
    reply_lookup = reply_store.fetch(post_id_arr)

    report_text = build_report_text_array(
        base_text=ensure_string_array(table.column("selftext")),
        post_ids=post_id_arr,
        reply_lookup=reply_lookup,
    )
    score = ensure_int64_array(table.column("score"))
    created = table.column("created_utc")
    permalink = build_submission_permalink_array(
        existing=ensure_string_array(table.column("permalink"), default=""),
        subreddits=subreddit,
        post_ids=post_id_arr,
    )
    bucket = bucket_from_subreddit(subreddit)
    author_norm = pc.utf8_lower(ensure_string_array(table.column("author")))

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


# ============================================================================
# Utility Functions
# ============================================================================


def _sort_batch_arrays_by_subreddit_and_created(
    arrays: dict[str, pa.Array], *, created_ts: pa.Array
) -> dict[str, pa.Array]:
    """
    Return arrays sorted by subreddit then creation time.

    Sorting here improves row-group stats and compression without changing the
    external schema. Sorting keys are kept narrow to avoid materializing large
    intermediate tables.
    """
    if len(created_ts) == 0:
        return arrays

    sort_table = pa.table({"subreddit": arrays["subreddit"], "_created_ts": created_ts})
    indices = pc.sort_indices(
        sort_table,
        sort_keys=[("subreddit", "ascending"), ("_created_ts", "ascending")],
    )
    return {
        name: pc.take(arr, indices, boundscheck=False) for name, arr in arrays.items()
    }


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
    from naturalv2.sources.reddit.pushshift_archive import _proc_dir  # noqa: PLC0415

    archives_dir = _proc_dir(source_dir) / "archives"
    if not archives_dir.exists():
        return set()

    subreddits: set[str] = set()
    for done_file in archives_dir.glob("*.done"):
        stem = done_file.stem  # "subreddit-content_type"
        if "-" in stem:
            subreddits.add(stem.split("-", 1)[0])
    return subreddits
