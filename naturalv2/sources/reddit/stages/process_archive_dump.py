"""Reddit archive dump processor."""

import logging
import os
import resource
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import psutil
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from naturalv2.sources.core import SourceStage
from naturalv2.sources.reddit.processing import (
    build_contextualized_dataset,
    write_to_parquet_partitions,
)
from naturalv2.sources.reddit.processing._utils import (
    BUCKET_COUNT,
    get_default_num_workers,
    get_tqdm_position,
    release_memory,
    worker_initializer,
)
from naturalv2.sources.reddit.pushshift_archive import (
    PROCESSED_RECORD_SCHEMA,
    is_archive_processed,
    iter_bucketed_batches,
    mark_archive_done,
)


if TYPE_CHECKING:
    from naturalv2.sources.core import CurationContext, StageState

logger = logging.getLogger(__name__)


class RedditDumpProcessor(SourceStage):
    """Process dump of Reddit archive from pushshift.

    This stage processes compressed Reddit archive files (.zst format) from the
    pushshift dataset. It performs the following operations:

    1. Discovers all .zst archive files in the specified directories
    2. Parses and decompresses the archives in parallel using multiple workers
    3. Applies filtering and cleaning to the Reddit posts/comments
    4. Partitions the data by subreddit and bucket for efficient querying
    5. Writes the processed data to a partitioned Parquet dataset, with one parquet
       writer per zst file processed.
    6. Builds a contextualized dataset, including adding author replies to posts.
       This step also compacts the data from step 5 into fewer Parquet files.

    This stage assumes that all the archive files have been downloaded and are
    available locally.

    Parameters
    ----------
    archive_dir_or_paths : str or Path or Sequence of str or Path
        Path or list of paths to archive file(s) or folder(s) containing ``.zst``
        files. Can be a single file, single directory, or list of mixed files
        and directories. Directories are searched recursively for .zst files.
    chunk_size : int, default=268435456 (256 MiB)
        The chunk size in bytes to use when parsing the ``.zst`` files. Larger
        chunks are more efficient but use more memory per worker.
    num_workers : int or None, optional, default=None
        The number of workers that will be used to process archive files in parallel.
        If ``None``, automatically calculated based on available CPU cores, memory
        constraints, and file handle limits. Set to 1 for single-threaded processing.
    num_threads_per_worker : int, default=4
        Number of PyArrow threads allocated to each worker process. Helps prevent
        thread oversubscription in multiprocessing scenarios.
    max_open_files : int, default=1000
        Maximum number of file handles that can be open simultaneously. This limit
        is respected by the system's soft limit (RLIMIT_NOFILE). Affects the number
        of concurrent Parquet partition files that can be written.
    name : str or None, optional, default=None
        Optional explicit stage name; defaults to the class name. Used for logging
        and identification in the pipeline.

    Attributes
    ----------
    archive_dir_or_paths : str or Path or Sequence
        Stored archive paths as provided during initialization.
    chunk_size : int
        Decompression chunk size in bytes.
    max_open_files : int
        Maximum number of simultaneously open file handles.
    num_threads_per_worker : int
        PyArrow thread count per worker.
    num_workers : int
        Number of parallel worker processes to use.

    Examples
    --------
    >>> # Process archives from a single directory
    >>> processor = RedditDumpProcessor(
    ...     archive_dir_or_paths="/data/reddit/archives",
    ...     num_workers=4,
    ...     chunk_size=256 << 20,
    ... )

    >>> # Process specific archive files
    >>> processor = RedditDumpProcessor(
    ...     archive_dir_or_paths=[
    ...         "/data/reddit/RS_2020-01.zst",
    ...         "/data/reddit/RS_2020-02.zst",
    ...     ]
    ... )

    Notes
    -----
    The processor automatically handles:
    - Duplicate file detection (same file via different paths)
    - Resumption of interrupted processing (skips already-processed archives)
    - Memory cleanup after processing each file
    - Load balancing by processing larger files first

    File handle limits are particularly important when writing partitioned data,
    as each unique bucket requires an open file.

    """

    def __init__(
        self,
        archive_dir_or_paths: str | Path | Sequence[str | Path],
        *,
        chunk_size: int = 256 << 20,  # 256 MiB
        num_workers: int | None = None,
        num_threads_per_worker: int = 4,
        max_open_files: int = 1000,
        name: str | None = None,
    ) -> None:
        """Initialize the stage."""
        super().__init__(name=name)

        self.archive_dir_or_paths = archive_dir_or_paths
        self.chunk_size = chunk_size
        self.max_open_files = max_open_files

        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if self.max_open_files and self.max_open_files > soft_limit:
            logger.warning(
                (
                    "Requested max_open_files (%d) exceeds system soft limit (%d); "
                    "capping to soft limit."
                ),
                self.max_open_files,
                soft_limit,
            )
            self.max_open_files = soft_limit

        cpu_count: int = psutil.cpu_count() or 1
        self.num_threads_per_worker = min(num_threads_per_worker, cpu_count)

        self.num_workers = num_workers or max(
            1,
            min(
                get_default_num_workers(
                    mem_gb_per_worker=2,  # ~2 GB per worker for chunk size of 256 MiB
                    threads_per_worker=self.num_threads_per_worker,
                ),
                self.max_open_files // BUCKET_COUNT,
            ),
        )

    async def run(
        self, context: "CurationContext", state: "StageState"
    ) -> "StageState":
        """
        Parse, filter, and clean Reddit archive files from pushshift.

        This method orchestrates the entire processing pipeline:
        1. Discovers all .zst archive files from the configured paths
        2. Checks which archives have already been processed (for resumption)
        3. Processes archives in parallel using a worker pool (or sequentially if single-worker)
        4. Writes filtered data to staged Parquet partitions
        5. Builds the final contextualized dataset
        6. Updates the pipeline state with output paths
        7. Persists metadata in the study dataset

        The processing includes:
        - Decompression of .zst archives
        - JSON parsing of Reddit posts/comments
        - Rule-based filtering (bots, deleted posts, low-quality content)
        - Bucketing by subreddit for partitioning
        - Writing to Parquet format with schema enforcement

        Parameters
        ----------
        context : CurationContext
            Pipeline context including experiment name, source name, and save
            directories. Used to determine output paths and metadata storage.
        state : StageState
            Mutable pipeline state. Will be updated with ``data_root`` (path to
            final processed data) and ``source_dir`` (base directory for this source).

        Returns
        -------
        StageState
            Updated state object with the following modifications:
            - ``payload``: Set to the final output directory path
            - ``data_root``: Path to the cleaned, contextualized dataset
            - ``source_dir``: Base directory for this source

        Raises
        ------
        RuntimeError
            If no .zst archive files are found in the specified paths.
        Exception
            If individual archive processing fails (logged but doesn't stop
            processing of remaining archives).

        Notes
        -----
        The method uses a staging directory for intermediate outputs and a final
        directory for the completed dataset. The staging directory structure is:
        ``{source_dir}/reddit_dump/staging/content_type=<name>/bucket=<id>/*.parquet``

        The final directory structure includes threading context:
        ``{source_dir}/reddit_dump/final/content_type=<name>/bucket=<id>/*.parquet``

        Progress is displayed using tqdm progress bars. When using multiple workers,
        each worker displays its own progress bar for the archive it's processing.

        Archives that have been successfully processed are marked as done to enable
        resumption if the process is interrupted. This is tracked in a ".done" file
        in the staging directory.

        """
        source_dir = self.source_dir(context)

        normalized_paths = _normalize_paths(self.archive_dir_or_paths)
        zst_files = list(_discover_zst_archives(normalized_paths))
        if not zst_files:
            raise RuntimeError(
                f"No .zst archives found in {self.archive_dir_or_paths}",
            )

        with logging_redirect_tqdm():
            done: list[str] = []
            staging_dir = os.path.join(source_dir, "reddit_dump", "staging")

            if self.num_workers > 1:
                with ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=worker_initializer,
                    initargs=(self.num_threads_per_worker,),
                ) as executor:
                    futures: dict[Future, Path] = {}
                    for path in zst_files:
                        if is_archive_processed(staging_dir, path.stem):
                            done.append(path.stem)
                            continue

                        futures[
                            executor.submit(
                                self._process_single_file,
                                path,
                                staging_dir,
                                context.experiment_name,
                            )
                        ] = path

                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Processing Archives [{self.num_workers} workers]",
                        unit="file",
                        position=0,
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        path = futures[future]
                        try:
                            files_written = future.result()
                            if files_written:
                                mark_archive_done(staging_dir, path.stem)
                        except Exception as exc:
                            logger.exception("Failed to process %s: %s", path.name, exc)
            else:
                for path in tqdm(
                    zst_files,
                    desc="Processing Archives",
                    unit="file",
                    position=0,
                    leave=False,
                    dynamic_ncols=True,
                ):
                    if is_archive_processed(staging_dir, path.stem):
                        done.append(path.stem)
                        continue

                    files_written = self._process_single_file(
                        path, staging_dir, context.experiment_name
                    )
                    if files_written:
                        mark_archive_done(staging_dir, path.stem)

            if done:
                logger.info(
                    "%s: Completed processing of %d subreddit archives",
                    self.stage_name,
                    len(done),
                )

            final_dir = os.path.join(source_dir, "reddit_dump", "final")
            _ = build_contextualized_dataset(
                source_dir=staging_dir,
                dest_dir=final_dir,
                run_tag=context.experiment_name,
                cleanup_source=True,
            )

        # Update state
        state.payload = final_dir
        state.update(data_root=final_dir, source_dir=source_dir)

        # Update and persist metadata in StudyDataset
        self.persist_dataset(
            context,
            namespace_paths={f"{context.source_name}_cleaned": final_dir},
        )
        return state

    def _process_single_file(
        self, zst_path: Path, output_dir: str, run_tag: str
    ) -> list[str]:
        """
        Process a single .zst archive file and write to Parquet partitions.

        This method handles the complete processing of one archive file:
        1. Creates an iterator that decompresses and parses the .zst file in chunks
        2. Applies filtering rules to each batch of Reddit posts/comments
        3. Buckets the data by subreddit for partitioning
        4. Writes filtered batches to Parquet files in partition directories
        5. Ensures memory cleanup after processing

        The method is designed to be executed in a worker process and handles
        its own progress reporting via tqdm.

        Parameters
        ----------
        zst_path : Path
            Path to the .zst archive file to process. Should be a valid compressed
            Reddit archive file in pushshift format.
        output_dir : str
            Directory where partitioned Parquet files will be written. The output
            structure will be: ``{output_dir}/subreddit=<name>/bucket=<id>/*.parquet``
        run_tag : str
            Unique identifier for this processing run, typically combining the
            experiment name and archive filename. Used for tracking and debugging.

        Returns
        -------
        list of str
            List of file paths that were written during processing. Each path
            corresponds to a Parquet partition file. An empty list indicates
            no data was written (e.g., all data was filtered out).

        Notes
        -----
        Memory is aggressively released after processing via ``release_memory()``
        to prevent accumulation in long-running worker processes. This is
        particularly important when processing many archives sequentially.

        The method uses ``iter_bucketed_batches()`` which yields data in batches,
        allowing for streaming processing of large archives without loading the
        entire file into memory.

        Progress bars are positioned using ``get_tqdm_position()`` to avoid
        overlapping displays when multiple workers are running in parallel.

        If threading is enabled (``use_threads=True``), PyArrow will use multiple
        threads for I/O operations within this single file processing. This is
        only set when ``num_workers == 1`` (single-process mode) to avoid
        over-subscription.

        """

        try:
            # Create the iterator for this specific file
            batch_iter = iter_bucketed_batches(
                str(zst_path),
                chunk_size=self.chunk_size,
                progress_enabled=True,
                tqdm_bar_position=get_tqdm_position(),
            )

            # Write directly to parquet
            files_written = write_to_parquet_partitions(
                data_stream=batch_iter,
                output_dir=output_dir,
                schema=PROCESSED_RECORD_SCHEMA,
                max_open_files=self.max_open_files,
                existing_data_behavior="overwrite_or_ignore",
                use_threads=self.num_workers == 1,
                run_tag=f"{run_tag}-{zst_path.stem}",
            )
        finally:
            release_memory()

        return files_written


def _normalize_paths(dir_or_file_path: str | Path | Sequence[str | Path]) -> list[Path]:
    """
    Convert various path inputs to a normalized list of Path objects.

    This utility function handles multiple input formats and converts them to
    a consistent format for downstream processing. It expands user home directory
    shortcuts (``~``) and ensures all paths are proper Path objects.

    Parameters
    ----------
    dir_or_file_path : str or Path or Sequence of str or Path
        Input path(s) to normalize. Can be:
        - A single string path (e.g., "/data/archives")
        - A single Path object
        - A list of string paths
        - A list of Path objects
        - A mixed list of strings and Path objects

    Returns
    -------
    list of Path
        List of normalized Path objects with expanded user directories.
        Order is preserved from the input sequence.

    Raises
    ------
    TypeError
        If the input is not a string, Path object, or sequence of such objects.

    Examples
    --------
    >>> _normalize_paths("/data/reddit")
    [PosixPath('/data/reddit')]

    >>> _normalize_paths(["~/archives", "/data/reddit"])
    [PosixPath('/home/user/archives'), PosixPath('/data/reddit')]

    >>> from pathlib import Path
    >>> _normalize_paths(Path("~/data"))
    [PosixPath('/home/user/data')]

    """
    if isinstance(dir_or_file_path, (str, Path)):
        return [Path(dir_or_file_path).expanduser()]
    try:
        return [Path(path).expanduser() for path in dir_or_file_path]
    except TypeError as exc:
        raise TypeError(
            "Expected a string, Path object or a list of string or Path objects, "
            f"but got {dir_or_file_path}"
        ) from exc


def _discover_zst_archives(candidates: list[Path]) -> list[Path]:
    """
    Discover and collect all .zst archive files from candidate paths.

    This function recursively searches directories and validates files to build
    a comprehensive list of .zst archives to process. It handles:
    - Direct file paths (if they have .zst extension)
    - Directory paths (searched recursively for .zst files)
    - Duplicate detection (same file via different paths, e.g., symlinks)
    - Sorting by file size for load balancing

    The function resolves all paths to their canonical form to detect duplicates
    that might arise from symlinks or relative path references.

    Parameters
    ----------
    candidates : list of Path
        List of candidate paths to search. Each path can be either:
        - A file path ending in .zst (will be included directly)
        - A directory path (will be searched recursively for *.zst files)

    Returns
    -------
    list of Path
        Sorted list of resolved .zst file paths. Files are sorted by size in
        descending order (largest first) to help with load balancing when
        processing in parallel. Duplicates are removed.

    Examples
    --------
    >>> from pathlib import Path
    >>> candidates = [
    ...     Path("/data/reddit/RS_2020-01.zst"),
    ...     Path("/data/reddit/archives/"),
    ... ]
    >>> archives = _discover_zst_archives(candidates)
    >>> len(archives)
    15
    >>> archives[0].suffix
    '.zst'

    Notes
    -----
    The function uses ``rglob("*.zst")`` for recursive directory searching,
    which follows symlinks by default. Combined with duplicate detection via
    resolved paths, this ensures each unique file is only processed once.

    """
    seen: set[Path] = set()
    file_paths: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        if candidate.is_file() and candidate.suffix.lower() == ".zst":
            file_paths.append(resolved)
        elif candidate.is_dir():
            for file_path in candidate.rglob("*.zst"):
                if not file_path.is_file():
                    continue

                resolved_file = file_path.resolve()
                # Dedupe files discovered via overlapping inputs (e.g., directory + explicit file path).
                if resolved_file in seen:
                    continue

                seen.add(resolved_file)
                file_paths.append(resolved_file)

    # sort files by size (largest first) to help with load balancing
    file_paths.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)

    return file_paths
