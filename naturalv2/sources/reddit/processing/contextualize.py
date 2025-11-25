"""Build contextualized datasets from Reddit parquet sources.

This module provides functionality to:
- Write parquet datasets with hive partitioning
- Build contextualized datasets from Reddit submissions and comments
"""

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
    run_tag: str = "ctx",
    cleanup_source: bool = False,
) -> list[str]:
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    sources = [source_dir] if isinstance(source_dir, (str, Path)) else list(source_dir)
    bucket_files = _scan_and_group_bucketed_parquet_files(sources)

    files_written = []
    for bucket_id, file_mapping in tqdm(
        bucket_files.items(),
        desc="Building contextualized dataset",
        unit="bucket",
        leave=False,
        dynamic_ncols=True,
    ):
        submission_file, comment_file = _process_bucket(
            bucket_id, file_mapping, dest_dir, run_tag
        )
        if submission_file:
            files_written.append(submission_file)

        if comment_file:
            files_written.append(comment_file)

        if cleanup_source:
            for file_path in file_mapping["submissions"] + file_mapping["comments"]:
                os.remove(file_path)

    return files_written


def _scan_and_group_bucketed_parquet_files(
    source_dirs: list[Path],
) -> dict[str, dict[str, list[str]]]:
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
    submissions_files = file_mapping["submissions"]
    comments_files = file_mapping["comments"]

    if not submissions_files and not comments_files:
        return None, None

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

    # Contextualize submissions
    submissions_context = submissions.join(
        author_replies, left_on="id", right_on="post_id", how="left"
    ).select(
        [
            pl.col("subreddit"),
            pl.col("title"),
            pl.lit("").alias("initial_post"),
            pl.when(pl.col("replies").is_not_null())
            .then(pl.col("selftext") + "\n\n" + pl.col("replies").list.join("\n"))
            .otherwise(pl.col("selftext"))
            .alias("report_text"),
            pl.lit("submission").alias("report_type"),
            pl.col("score"),
            pl.from_epoch("timestamp", time_unit="s")
            .dt.strftime("%B %d, %Y")
            .alias("date_created"),
            (
                pl.lit("/r/")
                + pl.col("subreddit")
                + pl.lit("/comments/")
                + pl.col("id")
                + pl.lit("/")
                + pl.col("title").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", "_")
                + pl.lit("/")
            ).alias("permalink"),
            pl.col("replies").fill_null([]).alias("author_replies"),
            pl.lit("submissions").alias("content_type"),
            pl.col("bucket"),
            pl.col("timestamp"),  # Keeping for sort later
        ]
    )

    # Contextualize comments
    comments_context = (
        comments.join(submissions, on="post_id", how="left", suffix="_sub")
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
            pl.col("selftext").alias("initial_post"),
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
            pl.lit([]).cast(pl.List(pl.String)).alias("author_replies"),
            pl.lit("comments").alias("content_type"),
            pl.col("bucket"),
            pl.col("timestamp"),  # Keeping for sort later
        )
    )

    # Union, sort and write to disk
    parquet_options = {
        "compression": "zstd",
        "compression_level": 5,
        "statistics": "full",
        "row_group_size": 64_000,
        "data_page_size": 10 << 20,  # 10 MiB
    }

    try:
        submissions_filepath = _get_hive_path(
            dest_dir, "submissions", bucket_id, filename=f"{run_tag}-part-0.parquet"
        )
        submissions_context.sort(["subreddit", "timestamp"]).sink_parquet(
            submissions_filepath, **parquet_options
        )

        comments_filepath = _get_hive_path(
            dest_dir, "comments", bucket_id, filename=f"{run_tag}-part-0.parquet"
        )
        comments_context.sort(["subreddit", "timestamp"]).sink_parquet(
            comments_filepath, **parquet_options
        )
    except Exception as exc:
        logger.error("Failed to process bucket %s: %s", bucket_id, exc, exc_info=True)
        return None, None

    return submissions_filepath, comments_filepath


def _get_hive_path(
    base_dir: Path, content_type: str, bucket_id: str, filename: str
) -> Path:
    """Standardized Hive Path Builder."""
    path = Path(base_dir) / f"content_type={content_type}" / f"bucket={bucket_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path / filename
