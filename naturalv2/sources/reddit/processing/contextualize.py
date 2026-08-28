"""Build contextualized datasets from Reddit parquet sources.

This module provides functionality to:
- Write parquet datasets with hive partitioning
- Build contextualized datasets from Reddit submissions and comments
"""

import hashlib
import logging
import os
import resource
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Sequence

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from tqdm import tqdm


logger = logging.getLogger(__name__)


_AUTHOR_KEY_NAMESPACE = "naturalv2:reddit:author:v1:"
_MISSING_AUTHORS = frozenset({"", "[deleted]", "[removed]"})


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
        ("author_key", pa.string()),
        ("author_replies", pa.list_(pa.string())),
        ("content_type", pa.string()),
        ("bucket", pa.string()),
    ]
)

PARTITIONING = ds.partitioning(
    pa.schema([("content_type", pa.string()), ("bucket", pa.string())]),
    flavor="hive",
)


def _pseudonymize_author(author: str | None) -> str | None:
    """Return a stable, source-scoped key for a Reddit author."""
    if author is None:
        return None

    normalized_author = author.strip().casefold()
    if normalized_author in _MISSING_AUTHORS:
        return None

    value = f"{_AUTHOR_KEY_NAMESPACE}{normalized_author}".encode()
    return hashlib.sha256(value).hexdigest()


def _author_key_expr() -> pl.Expr:
    """Build the pseudonymous author-key column."""
    return (
        pl.col("author")
        .map_elements(_pseudonymize_author, return_dtype=pl.String)
        .alias("author_key")
    )


def write_to_parquet_partitions(
    data_stream: Iterable[pa.RecordBatch],
    output_dir: str,
    schema: pa.Schema,
    parquet_compression_level: int = 5,
    write_parquet_stats: Literal["none", "minimal", "all"] = "minimal",
    use_dictionary: bool | dict[str, bool] = True,
    max_partitions: int = 1024,
    min_rows_per_group: int = 131_072,
    max_rows_per_group: int = 1024 * 1024,
    max_open_files: int = 1000,
    existing_data_behavior: Literal[
        "error", "delete_matching", "overwrite_or_ignore"
    ] = "overwrite_or_ignore",
    use_threads: bool = True,
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
    min_rows_per_group : int, default=131_072
        Minimum rows per row group (bounded by ``max_rows_per_group``).
    max_rows_per_group : int, default=1024*1024
        Maximum rows per row group.
    max_open_files : int, default=1000
        Cap on concurrently open files during write.
    existing_data_behavior : {'error', 'delete_matching', 'overwrite_or_ignore'}, default='overwrite_or_ignore'
        Strategy when output already exists.
    use_threads : bool, default=True
        If ``True``, use multiple threads when writing parquet partitions.
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
        use_threads=use_threads,
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
    run_tag: str = "ctx",
    cleanup_source: bool = False,
) -> list[str]:
    """Build contextualized datasets from Reddit parquet sources.

    Processes Reddit submission and comment parquet files organized by bucket,
    enriches them with contextual information (author replies, permalinks, etc.),
    and writes the results to a hive-partitioned parquet dataset.

    Parameters
    ----------
    source_dir : str or Path or Sequence[str or Path]
        Source directory or directories containing bucketed parquet files.
        Files should be organized with hive partitioning (content_type=* and bucket=*).
    dest_dir : str or Path
        Destination directory for the contextualized dataset. Will be created
        if it doesn't exist.
    run_tag : str, default='ctx'
        Prefix tag for generated parquet filenames.
    cleanup_source : bool, default=False
        If ``True``, delete source parquet files after successful processing.

    Returns
    -------
    list[str]
        List of full file paths to all parquet files written.

    Notes
    -----
    The function processes data bucket by bucket, joining submissions with
    their associated comments and author replies. Each bucket is written to
    separate parquet files partitioned by content_type (submissions/comments)
    and bucket ID.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    sources = [source_dir] if isinstance(source_dir, (str, Path)) else list(source_dir)
    bucket_files = _scan_and_group_bucketed_parquet_files(sources)

    files_written: list[str] = []
    for bucket_id, file_mapping in tqdm(
        bucket_files.items(),
        desc="Building contextualized dataset",
        unit="bucket",
        leave=False,
        dynamic_ncols=True,
    ):
        files_to_delete = []
        submission_file, comment_file = _process_bucket(
            bucket_id, file_mapping, dest_dir, run_tag
        )
        if submission_file:
            files_written.append(submission_file)
            files_to_delete.extend(file_mapping["submissions"])

        if comment_file:
            files_written.append(comment_file)
            files_to_delete.extend(file_mapping["comments"])

        if cleanup_source:
            for file_path in files_to_delete:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except OSError as exc:
                    logger.warning(
                        "Failed to cleanup source file %s: %s", file_path, exc
                    )

    return files_written


def _scan_and_group_bucketed_parquet_files(
    source_dirs: list[Path],
) -> dict[str, dict[str, list[str]]]:
    """Scan source directories and group parquet files by bucket and content type.

    Recursively searches for parquet files in the given directories and groups
    them by bucket ID and content type (submissions or comments) based on
    hive partition naming conventions in the file paths.

    Parameters
    ----------
    source_dirs : list[Path]
        List of root directories to scan for parquet files.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Nested dictionary mapping bucket IDs to content type dictionaries.
        Each content type dictionary maps 'submissions' or 'comments' to
        a list of file paths. Structure: {bucket_id: {'submissions': [...],
        'comments': [...]}}.

    Raises
    ------
    KeyError
        If a parquet file is found without required partition keys ('bucket'
        and 'content_type') in its path.

    Notes
    -----
    Expects parquet files to be organized with hive partitioning where the
    path contains segments like 'bucket=*' and 'content_type=*'.
    """
    bucket_files = defaultdict(lambda: {"submissions": [], "comments": []})

    for root_dir in source_dirs:
        for file_path in Path(root_dir).glob("**/*.parquet"):
            partitions = dict(
                part.split("=") for part in str(file_path).split("/") if "=" in part
            )
            bucket_id = partitions["bucket"]
            content_type = partitions["content_type"]

            bucket_files[bucket_id][content_type].append(str(file_path))

    return bucket_files


def _process_bucket(
    bucket_id: str, file_mapping: dict[str, list[str]], dest_dir: Path, run_tag: str
) -> tuple[str | None, str | None]:
    """Process a single bucket of Reddit data into contextualized format.

    Loads submission and comment parquet files for a given bucket, enriches
    them with contextual information (author replies, permalinks, formatted
    dates), and writes the results to hive-partitioned parquet files.

    Parameters
    ----------
    bucket_id : str
        Identifier for the bucket being processed.
    file_mapping : dict[str, list[str]]
        Dictionary mapping content types to file paths. Expected keys are
        'submissions' and 'comments', each mapping to a list of parquet
        file paths.
    dest_dir : Path
        Base destination directory for output files.
    run_tag : str
        Prefix tag for generated parquet filenames.

    Returns
    -------
    tuple[str | None, str | None]
        Tuple of (submissions_filepath, comments_filepath). Each element is
        the full path to the written parquet file, or ``None`` if processing
        failed or no data was available for that content type.

    Notes
    -----
    For submissions, this function:
    - Joins submissions with author replies (comments by the submission author)
    - Combines selftext with author replies into report_text
    - Generates permalinks and formatted dates

    For comments, this function:
    - Filters out comments by the submission author
    - Includes the original submission's title and selftext as context
    - Generates permalinks and formatted dates

    Both outputs are sorted by subreddit and timestamp before writing.
    """
    submissions_files = file_mapping["submissions"]
    comments_files = file_mapping["comments"]

    if not submissions_files and not comments_files:
        return None, None

    # Generate Unique ID for this specific batch of inputs
    batch_hash = _compute_input_hash(submissions_files + comments_files)
    target_filename = f"{run_tag}-{batch_hash}.parquet"

    # Check existence BEFORE processing (for idempotency)
    submissions_filepath = _get_hive_path(
        dest_dir, "submissions", bucket_id, target_filename
    )
    comments_filepath = _get_hive_path(dest_dir, "comments", bucket_id, target_filename)

    # If both exist, we assume this exact input batch was already processed.
    # We return the paths so the caller knows they are "done" (and can cleanup source).
    if submissions_filepath.exists() and comments_filepath.exists():
        return str(submissions_filepath), str(comments_filepath)

    # Scan inputs
    submissions = pl.scan_parquet(
        submissions_files,
        schema={
            "id": pl.String,
            "created_utc": pl.Int64,
            "subreddit": pl.String,
            "title": pl.String,
            "selftext": pl.String,
            "author": pl.String,
            "score": pl.Float64,
        },
        extra_columns="ignore",
        low_memory=True,
    ).with_columns(
        pl.col("id").alias("post_id"),
        pl.col("created_utc").cast(pl.Int64).alias("timestamp"),
        pl.lit(bucket_id).alias("bucket"),
    )
    comments = pl.scan_parquet(
        comments_files,
        schema={
            "id": pl.String,
            "link_id": pl.String,
            "created_utc": pl.Int64,
            "subreddit": pl.String,
            "body": pl.String,
            "author": pl.String,
            "score": pl.Float64,
        },
        extra_columns="ignore",
        low_memory=True,
    ).with_columns(
        pl.col("link_id").str.replace(r"^t3_", "").alias("post_id"),
        pl.col("created_utc").cast(pl.Int64).alias("timestamp"),
        pl.lit(bucket_id).alias("bucket"),
    )

    # Find author replies
    author_replies = (
        comments.join(
            submissions.select(["post_id", "author"]), on="post_id", how="inner"
        )
        .filter(
            (pl.col("author") == pl.col("author_right"))
            & pl.col("body").is_not_null()
            & ~pl.col("body").is_in(["[deleted]", "[removed]"])
        )
        .select(["post_id", "body", "timestamp"])
        .sort("timestamp")
        .group_by("post_id")
        .agg(pl.col("body").alias("replies"))
    )

    # Create the enriched text column
    submissions_enriched = submissions.join(
        author_replies, left_on="id", right_on="post_id", how="left"
    ).select(
        [
            pl.col("id").alias("post_id"),
            pl.col("author"),
            pl.col("subreddit"),
            pl.col("title"),
            pl.col("score"),
            pl.col("timestamp"),
            pl.col("bucket"),
            pl.col("selftext"),
            pl.col("replies"),
            pl.when(
                pl.col("replies").is_not_null().and_(pl.col("replies").list.len() > 0)
            )
            .then(
                pl.col("selftext").fill_null("")
                + pl.lit(
                    "\n\nThe original poster also replied with the following comments in the thread:"
                )
                + pl.col("replies")
                .list.eval(pl.lit("\n> ") + pl.element())
                .list.join("")
            )
            .otherwise(pl.col("selftext").fill_null(""))
            .alias("enriched_text"),
        ]
    )

    # Contextualize Submissions
    submissions_context = submissions_enriched.select(
        [
            pl.col("subreddit"),
            pl.col("title"),
            pl.lit("").alias("initial_post"),
            pl.col("enriched_text").alias("report_text"),
            pl.lit("submission").alias("report_type"),
            pl.col("score"),
            pl.from_epoch("timestamp", time_unit="s")
            .dt.strftime("%B %d, %Y")
            .alias("date_created"),
            (
                pl.lit("/r/")
                + pl.col("subreddit")
                + pl.lit("/comments/")
                + pl.col("post_id")
                + pl.lit("/")
                + pl.col("title").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", "_")
                + pl.lit("/")
            ).alias("permalink"),
            _author_key_expr(),
            pl.col("replies").fill_null([]).alias("author_replies"),
            pl.lit("submissions").alias("content_type"),
            pl.col("bucket"),
            pl.col("timestamp"),  # Keeping for sort later
        ]
    )

    # Contextualize Comments
    # Join against 'submissions_enriched'
    comments_context = (
        comments.join(submissions_enriched, on="post_id", how="left", suffix="_sub")
        .filter(
            pl.col("author_sub").is_null()
            | (
                pl.col("author").str.to_lowercase()
                != pl.col("author_sub").str.to_lowercase()
            )
        )
        .select(
            pl.col("subreddit"),
            pl.col("title"),
            # Use 'enriched_text' as initial_post
            pl.col("enriched_text").alias("initial_post"),
            pl.col("body").alias("report_text"),
            pl.lit("comment").alias("report_type"),
            pl.col("score"),
            pl.from_epoch("timestamp", time_unit="s")
            .dt.strftime("%B %d, %Y")
            .alias("date_created"),
            (
                pl.lit("/r/")
                + pl.col("subreddit")
                + pl.lit("/comments/")
                + pl.col("post_id")
                + pl.lit("/_/")
                + pl.col("id")
            ).alias("permalink"),
            _author_key_expr(),
            pl.lit([]).cast(pl.List(pl.String)).alias("author_replies"),
            pl.lit("comments").alias("content_type"),
            pl.col("bucket"),
            pl.col("timestamp"),  # Keeping for sort later
        )
    )

    # Union, sort and write to disk
    parquet_options = {
        "compression": "zstd",
        "compression_level": 3,
        "statistics": "full",
        "row_group_size": 131_072,
    }

    try:
        submissions_context.sink_parquet(submissions_filepath, **parquet_options)

        comments_context.sink_parquet(comments_filepath, **parquet_options)
    except Exception as exc:
        logger.error("Failed to process bucket %s: %s", bucket_id, exc, exc_info=True)

        return None, None

    return str(submissions_filepath), str(comments_filepath)


def _compute_input_hash(file_paths: list[str]) -> str:
    """Generate a deterministic short hash based on input file paths."""
    if not file_paths:
        return "empty"

    hasher = hashlib.md5()
    # Sort to ensure order of files doesn't change the hash
    for path in sorted(file_paths):
        hasher.update(str(path).encode("utf-8"))

    return hasher.hexdigest()[:12]


def _get_hive_path(
    base_dir: Path, content_type: str, bucket_id: str, filename: str
) -> Path:
    """Build a standardized Hive-partitioned file path.

    Constructs a directory structure following Hive partitioning conventions
    and returns the full path including the filename. Creates parent
    directories if they don't exist.

    Parameters
    ----------
    base_dir : Path
        Base directory for the partitioned structure.
    content_type : str
        Content type partition value (e.g., 'submissions' or 'comments').
    bucket_id : str
        Bucket partition value.
    filename : str
        Name of the file to place in the partitioned directory.

    Returns
    -------
    Path
        Full path to the file: base_dir/content_type={content_type}/bucket={bucket_id}/filename

    Notes
    -----
    The resulting directory structure follows Hive partitioning format:
    base_dir/content_type={value}/bucket={value}/filename.parquet
    """
    path = Path(base_dir) / f"content_type={content_type}" / f"bucket={bucket_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path / filename
