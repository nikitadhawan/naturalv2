"""Reddit archive fetch and clean stage."""

import concurrent.futures
import contextlib
import itertools
import logging
import os
from collections.abc import Iterator
from concurrent.futures._base import Future
from typing import TYPE_CHECKING, Literal

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from naturalv2.sources.core import SourceStage
from naturalv2.sources.reddit.api import get_sub_about_info
from naturalv2.sources.reddit.processing import write_to_parquet_partitions
from naturalv2.sources.reddit.processing._utils import (
    get_default_num_workers,
    get_tqdm_position,
)
from naturalv2.sources.reddit.processing.contextualize import (
    build_contextualized_dataset,
)
from naturalv2.sources.reddit.pushshift_archive import (
    PROCESSED_RECORD_SCHEMA,
    download_sub_data,
    is_archive_processed,
    iter_bucketed_batches,
    mark_archive_done,
)


if TYPE_CHECKING:
    from naturalv2.sources.core import CurationContext, StageState

logger = logging.getLogger(__name__)


class RedditDownloadAndClean(SourceStage):
    """Download and clean subreddit data from Pushshift archives.

    This stage orchestrates the complete download and processing pipeline for
    Reddit data from Pushshift archives. It performs the following operations:

    1. Identifies relevant subreddits based on condition-to-subreddit mappings
    2. Validates subreddit availability in Pushshift archives via Reddit API
    3. Downloads .zst archive files for both submissions and comments
    4. Decompresses and parses the archives in parallel
    5. Applies rule-based filtering to remove low-quality content
    6. Partitions data by subreddit and bucket for efficient querying
    7. Writes cleaned data to Parquet format
    8. Builds a contextualized dataset with threading information

    The stage uses a pipelined architecture with separate thread and process pools:
    - Thread pool for downloading archives (I/O-bound)
    - Process pool for decompressing and processing archives (CPU-bound)

    This design maximizes throughput by overlapping download and processing,
    and supports resumption by tracking completed archives.

    Parameters
    ----------
    reddit_rpm : int, default=10
        Rate limit for Reddit API requests (requests per minute). Used when
        fetching subreddit metadata to check availability. Lower values are
        safer but slower; higher values may trigger rate limiting.
    max_download_workers : int or None, optional, default=None
        Maximum number of parallel download and processing workers. Controls both
        the download thread pool size and the processing pool size. If None,
        automatically calculated based on available CPU cores and memory (~2GB
        per worker with 4 threads per worker).
    name : str or None, optional, default=None
        Optional explicit stage name; defaults to the class name. Used for logging
        and identification in the pipeline.

    Attributes
    ----------
    reddit_rpm : int
        Configured rate limit for Reddit API requests.
    max_download_workers : int
        Number of parallel workers for download and processing.

    Examples
    --------
    >>> # Use default settings
    >>> stage = RedditDownloadAndClean()

    >>> # Customize rate limiting and parallelism
    >>> stage = RedditDownloadAndClean(reddit_rpm=20, max_download_workers=8)

    Notes
    -----
    The stage requires a condition-to-subreddit mapping in the state metadata,
    which should be provided by the ``RedditConditionFilter`` stage running before
    this stage.

    Archives are downloaded to a temporary location and removed after successful
    processing to save disk space. The pipeline architecture ensures that
    downloads and processing can proceed concurrently for maximum efficiency.

    See Also
    --------
    RedditConditionFilter : Stage that generates the condition-to-subreddit mapping
    RedditDumpProcessor : Alternative stage for processing pre-downloaded archives

    """

    def __init__(
        self,
        *,
        reddit_rpm: int = 10,
        max_download_workers: int | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the stage."""
        super().__init__(name=name)

        self.reddit_rpm = reddit_rpm
        self.max_download_workers: int = (
            max_download_workers
            or get_default_num_workers(mem_gb_per_worker=2, threads_per_worker=4)
        )

    async def run(
        self, context: "CurationContext", state: "StageState"
    ) -> "StageState":
        """
        Download, process, and clean Reddit archive data for relevant subreddits.

        This method orchestrates the complete pipeline:
        1. Extracts condition-to-subreddit mapping from state metadata
        2. Determines relevant subreddits based on experiment conditions
        3. Validates subreddit availability via Reddit API and Pushshift
        4. Downloads and processes archives in parallel (submissions and comments)
        5. Writes filtered data to staged Parquet partitions
        6. Builds the final contextualized dataset
        7. Updates pipeline state and persists metadata

        Parameters
        ----------
        context : CurationContext
            Pipeline context containing experiments, experiment_name, source_name,
            and save directories for output.
        state : StageState
            Mutable pipeline state. Must contain ``condition_to_subreddit_map``
            in metadata. Will be updated with ``available_subreddits``,
            ``data_root``, ``source_dir``, and ``payload``.

        Returns
        -------
        StageState
            Updated state object with output paths and metadata.

        Raises
        ------
        ValueError
            If ``condition_to_subreddit_map`` is not found in state metadata.
        RuntimeError
            If no relevant subreddits are found for any conditions.

        Notes
        -----
        Archives are processed in pairs (submissions + comments) for each subreddit.
        Already-completed archives are skipped to support resumption.

        """
        condition_to_subreddit_map: dict[str, list[str]] = state.require_metadata(
            "condition_to_subreddit_map", stage=self.stage_name
        )
        if not condition_to_subreddit_map:
            raise ValueError(
                "No condition to subreddit mapping found in state metadata. "
                "This stage cannot proceed. "
                "Please ensure that the `RedditConditionFilter` stage has been "
                "run successfully before this stage."
            )

        relevant_subreddits = _get_relevant_subreddits(
            context, condition_to_subreddit_map
        )

        source_dir, subs_data_dir = self._get_subs_data_dir(context)

        if not relevant_subreddits:
            raise RuntimeError(
                f"{self.stage_name}: no relevant subreddits found for any conditions",
            )

        # Filter out subreddits that are not available in the Pushshift archives
        subs_about = await get_sub_about_info(source_dir, self.reddit_rpm)
        pushshift_subreddits = set(subs_about["subreddit"].to_list())
        available_subreddits = set(relevant_subreddits).intersection(
            pushshift_subreddits
        )
        logger.info(
            "%s: %d out of %d relevant subreddits are available in Pushshift",
            self.stage_name,
            len(available_subreddits),
            len(relevant_subreddits),
        )

        staging_dir = os.path.join(subs_data_dir, "staging")
        processed = self.download_and_clean_subreddit_archives(
            list(available_subreddits), staging_dir
        )
        if processed:
            logger.info(
                "%s: downloaded and cleaned %d subreddit archives for %d experiments",
                self.stage_name,
                len(processed),
                len(context.experiments),
            )

        final_dir = os.path.join(subs_data_dir, "final")
        _ = build_contextualized_dataset(
            source_dir=staging_dir,
            dest_dir=final_dir,
            run_tag=context.experiment_name,
            cleanup_source=True,
        )

        # Update state
        state.payload = final_dir
        state.update(
            available_subreddits=list(available_subreddits),
            data_root=final_dir,
            source_dir=source_dir,
        )

        # Update and persist metadata in StudyDataset
        self.persist_dataset(
            context,
            namespace_paths={f"{context.source_name}_cleaned": final_dir},
        )
        return state

    def download_and_clean_subreddit_archives(
        self, subreddits: list[str], output_dir: str
    ) -> list[str]:
        """
        Download and process Reddit archives using a pipelined architecture.

        This method implements a producer-consumer pipeline that maximizes throughput
        by overlapping I/O-bound downloads with CPU-bound processing. The pipeline
        maintains a buffer of concurrent operations up to max_download_workers.

        For each subreddit, both submissions and comments archives are processed.
        Already-completed archives are skipped to support resumption.

        Parameters
        ----------
        subreddits : list of str
            List of subreddit names to process (e.g., ['AskReddit', 'science']).
        output_dir : str
            Base directory to store cleaned subreddit datasets. Each archive gets
            a subdirectory: ``{output_dir}/{subreddit}-{submissions|comments}/``

        Returns
        -------
        list of str
            List of successfully processed archive IDs in the format
            ``{subreddit}-{submissions|comments}``.

        Notes
        -----
        The pipeline uses separate executors:
        - ThreadPoolExecutor for downloads (I/O-bound)
        - ProcessPoolExecutor for processing (CPU-bound)

        Downloaded .zst files are automatically deleted after successful processing.

        """

        done: list[str] = []
        futures_state: dict[str, dict[concurrent.futures.Future, tuple[str, str]]] = {
            "dl": {},
            "wr": {},
        }

        # Flatten input: (sub, type) tuples
        all_tasks = itertools.product(subreddits, ["submissions", "comments"])
        task_iter = iter(all_tasks)

        dl_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(2, self.max_download_workers),
            thread_name_prefix="archive_downloader",
        )
        wr_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_download_workers
        )

        try:
            with (
                logging_redirect_tqdm(),
                tqdm(
                    total=len(subreddits) * 2,
                    desc="Downloading and cleaning subreddit archives",
                    unit="archive",
                    position=0,
                    leave=False,
                    dynamic_ncols=True,
                ) as pbar,
            ):
                while True:
                    # Fill the Pipeline
                    has_more = self._fill_download_pipeline(
                        task_iter, dl_executor, futures_state, output_dir, done, pbar
                    )

                    # Check Exit Condition
                    dl_futures = futures_state["dl"]
                    wr_futures = futures_state["wr"]
                    if not has_more and not dl_futures and not wr_futures:
                        break

                    # Wait for Event
                    # Combine keys to wait on any active task
                    active = list(dl_futures.keys()) + list(wr_futures.keys())
                    if not active:
                        continue

                    done_futures, _ = concurrent.futures.wait(
                        active, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    # Dispatch Completion
                    for fut in done_futures:
                        if fut in dl_futures:
                            self._handle_download_completion(
                                fut, futures_state, wr_executor, pbar
                            )
                        elif fut in wr_futures:
                            self._handle_write_completion(
                                fut, futures_state, output_dir, done, pbar
                            )

        finally:
            dl_executor.shutdown(wait=True)
            wr_executor.shutdown(wait=True)

        return done

    def _fill_download_pipeline(
        self,
        task_iter: Iterator[tuple[str, Literal["submissions", "comments"]]],
        dl_executor: concurrent.futures.ThreadPoolExecutor,
        futures_state: dict,
        output_dir: str,
        done_list: list[str],
        pbar: tqdm,
    ) -> bool:
        """
        Fill the download pipeline with tasks up to the worker limit.

        Pulls tasks from the input iterator and submits them as download jobs
        until either the pipeline buffer is full or the input is exhausted.
        Handles skipping of already-processed archives.

        Parameters
        ----------
        task_iter : Iterator of tuple of (str, Literal["submissions", "comments"])
            Iterator yielding (subreddit, content_type) tuples for processing.
        dl_executor : concurrent.futures.ThreadPoolExecutor
            Thread pool executor for submitting download tasks.
        futures_state : dict
            Dictionary tracking active futures with keys 'dl' and 'wr'.
        output_dir : str
            Base output directory for checking if archives are already processed.
        done_list : list of str
            Accumulator for archive IDs that are already done.
        pbar : tqdm
            Progress bar to update when skipping already-processed archives.

        Returns
        -------
        bool
            True if there may be more items remaining, False if exhausted.

        """
        dl_futures = futures_state["dl"]
        wr_futures = futures_state["wr"]

        while len(dl_futures) + len(wr_futures) < self.max_download_workers:
            try:
                subreddit, content_type = next(task_iter)
            except StopIteration:
                return False  # Input exhausted

            archive_id = f"{subreddit}-{content_type}"
            if is_archive_processed(output_dir, archive_id):
                done_list.append(archive_id)
                pbar.update(1)
                continue

            output_path = os.path.join(output_dir, archive_id)
            os.makedirs(output_path, exist_ok=True)

            fut = dl_executor.submit(
                download_sub_data, subreddit, content_type, output_dir
            )
            dl_futures[fut] = (archive_id, output_path)

        return True  # Items may still remain

    def _handle_download_completion(
        self,
        future: concurrent.futures.Future,
        futures_state: dict,
        wr_executor: concurrent.futures.ProcessPoolExecutor,
        pbar: tqdm,
    ) -> None:
        """
        Handle completion of a download task and chain to processing.

        Retrieves the download result (path to .zst file), and if successful,
        submits the file for processing. This creates a pipeline where processing
        begins as soon as downloads complete.

        Parameters
        ----------
        future : concurrent.futures.Future
            The completed download future to handle.
        futures_state : dict
            Dictionary tracking active futures.
        wr_executor : concurrent.futures.ProcessPoolExecutor
            Process pool executor for submitting the processing task.
        pbar : tqdm
            Progress bar to update if the download was skipped or failed.

        Notes
        -----
        Threading is disabled in the writer worker when multiple processes are
        used to prevent thread oversubscription.

        """
        dl_futures = futures_state["dl"]
        wr_futures = futures_state["wr"]

        archive_id, archive_dir = dl_futures.pop(future)

        try:
            zst_path = future.result()
            if zst_path:
                # Chain to Processor
                wr_fut = wr_executor.submit(
                    _writer_worker,
                    zst_archive_path=zst_path,
                    archive_dataset_dir=archive_dir,
                    use_threads=wr_executor._max_workers < 2,
                )
                wr_futures[wr_fut] = (archive_id, zst_path)
            else:
                pbar.update(1)  # Skipped/Failed gracefully
        except Exception:
            logger.exception("Download failed: %s", archive_id)
            pbar.update(1)

    def _handle_write_completion(
        self,
        future: Future,
        futures_state: dict,
        output_dir: str,
        done_list: list[str],
        pbar: tqdm,
    ) -> None:
        """
        Handle completion of a processing task and mark archive as done.

        Checks if processing was successful and marks the archive as done to
        support resumption. The progress bar is always updated.

        Parameters
        ----------
        future : Future
            The completed processing future to handle.
        futures_state : dict
            Dictionary tracking active futures.
        output_dir : str
            Base output directory where the ".done" marker file will be created.
        done_list : list of str
            Accumulator for successfully processed archive IDs.
        pbar : tqdm
            Progress bar to update after handling the completion.

        """
        wr_futures = futures_state["wr"]
        archive_id, _ = wr_futures.pop(future)

        try:
            if future.result():
                mark_archive_done(output_dir, archive_id)
                done_list.append(archive_id)
        except Exception:
            logger.exception("Write failed: %s", archive_id)
        finally:
            pbar.update(1)

    def _get_subs_data_dir(self, context: "CurationContext") -> tuple[str, str]:
        """
        Get the source directory and ensure the subreddit data subdirectory exists.

        Determines the base source directory for Reddit data and creates a
        "subs_data" subdirectory for storing downloaded and processed subreddit
        archives.

        Parameters
        ----------
        context : CurationContext
            Pipeline context used to determine the source directory path.

        Returns
        -------
        tuple of (str, str)
            A tuple containing:
            - source_dir: Base directory for this Reddit source
            - subs_data_dir: Subdirectory path for subreddit data (created if needed)

        Notes
        -----
        The subs_data_dir will contain:
        - ``{subs_data_dir}/staging/``: Intermediate partitioned data per archive
        - ``{subs_data_dir}/final/``: Contextualized dataset

        """
        source_dir = self.source_dir(context)
        subs_data_dir = os.path.join(source_dir, "subs_data")
        os.makedirs(subs_data_dir, exist_ok=True)
        return source_dir, subs_data_dir


def _get_relevant_subreddits(
    context: "CurationContext",
    condition_to_subreddit_map: dict[str, list[str]],
) -> set[str]:
    """
    Collect all subreddits relevant to conditions across all experiments.

    Iterates through all experiments and extracts the subreddits associated
    with each condition. Returns the union of all relevant subreddits.

    Parameters
    ----------
    context : CurationContext
        Pipeline context containing the list of experiments with conditions.
    condition_to_subreddit_map : dict of str to list of str
        Mapping from condition keywords to lists of associated subreddit names.
        Generated by the RedditConditionFilter stage.

    Returns
    -------
    set of str
        Set of unique subreddit names relevant to at least one condition.
        Returns empty set if no experiments have conditions.

    Notes
    -----
    Safely handles:
    - Experiments with no conditions (None or empty)
    - Conditions not present in the mapping (silently skipped)
    - Duplicate subreddits across conditions (automatically deduplicated)

    """
    relevant_subreddits: set[str] = set()
    for experiment in context.experiments:
        for keyword in experiment.conditions or []:
            relevant_subreddits.update(condition_to_subreddit_map.get(keyword, []))
    return relevant_subreddits


def _writer_worker(
    zst_archive_path: str, archive_dataset_dir: str, use_threads: bool
) -> list[str]:
    """
    Process a .zst archive and write filtered data to Parquet partitions.

    Executed in a separate process to decompress, parse, filter, and partition
    a Reddit archive. Each archive is written to its own dataset directory,
    making it easy to track completion and enable resumption.

    Parameters
    ----------
    zst_archive_path : str
        Path to the compressed .zst archive file to process. Deleted after
        successful processing to save disk space.
    archive_dataset_dir : str
        Output directory for this archive's Parquet dataset. Structure:
        ``{archive_dataset_dir}/subreddit=<name>/bucket=<id>/*.parquet``
    use_threads : bool
        Whether to use PyArrow threading for I/O. Should be True in
        single-process mode, False when multiple processes are used.

    Returns
    -------
    list of str
        List of file paths that were written. Empty list if no data was written.

    Notes
    -----
    Uses ``iter_bucketed_batches()`` for memory-efficient streaming processing.
    Progress is reported via tqdm with position from ``get_tqdm_position()``.

    The .zst file is deleted only after successful processing. The parameter
    ``existing_data_behavior="delete_matching"`` ensures pre-existing Parquet
    files for matching partitions are replaced.

    """
    files_written = write_to_parquet_partitions(
        data_stream=iter_bucketed_batches(
            zst_archive_path, tqdm_bar_position=get_tqdm_position()
        ),
        output_dir=archive_dataset_dir,
        schema=PROCESSED_RECORD_SCHEMA,
        existing_data_behavior="delete_matching",
        use_threads=use_threads,
    )

    if files_written:
        with contextlib.suppress(FileNotFoundError):
            os.remove(zst_archive_path)

    return files_written
