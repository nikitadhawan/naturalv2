"""Reddit archive fetch and clean stage."""

import contextlib
import logging
import os
from concurrent.futures import as_completed
from concurrent.futures._base import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import TYPE_CHECKING

import psutil
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from naturalv2.sources.core import SourceStage
from naturalv2.sources.reddit.api import get_sub_about_info
from naturalv2.sources.reddit.processing import write_to_parquet_partitions
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
    """Download or synthesise subreddit data for selected candidates.

    This stage downloads missing subreddit data, optionally anonymizes text,
    performs rule-based cleaning, and writes consolidated parquet files.

    Parameters
    ----------
    reddit_rpm : int, default=10
        Rate limit for Reddit API requests (requests per minute).
    max_download_workers : int | None, optional
        Degree of parallelism for downloads/cleaning; defaults to min(CPU count,
        RAM-based limit), minimum 4. RAM limit reserves 30% for OS and assumes
        ~1 GiB per worker. Split evenly between download and write workers.
    name : str | None, optional
        Optional explicit stage name; defaults to the class name.
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

        cpu_count: int = psutil.cpu_count(logical=False) or 2
        if max_download_workers is not None:
            self.max_download_workers = min(max_download_workers, cpu_count)
        else:
            # Calculate RAM-based cap: each worker uses ~2 GiB:
            #   - Includes decompression window (up to 2 GiB limit, but typically much less)
            #   - 256 MiB chunk buffer + Arrow tables + parquet write buffers
            # Reserve 30% of RAM for OS/other processes
            available_ram_bytes = psutil.virtual_memory().available
            ram_reserved_bytes = int(available_ram_bytes * 0.3)
            ram_usable_bytes = available_ram_bytes - ram_reserved_bytes
            memory_per_worker_bytes = 2 << 30  # 2 GiB per worker
            max_workers_by_ram = max(1, ram_usable_bytes // memory_per_worker_bytes)

            self.max_download_workers = min(cpu_count, max_workers_by_ram)

    async def run(
        self, context: "CurationContext", state: "StageState"
    ) -> "StageState":
        """Download/clean subreddit data and update state with file paths.

        Parameters
        ----------
        context : CurationContext
            Pipeline context including experiment list and save directories.
        state : StageState
            Mutable pipeline state; will be updated with ``cleaned_paths`` and
            the ``source_dir`` used.

        Returns
        -------
        StageState
            Updated state containing cleaned subreddit file paths.
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
        num_dl = max(1, self.max_download_workers // 2)
        num_wr = max(1, self.max_download_workers - num_dl)

        def _writer_worker(
            zst_archive_path: str, archive_dataset_dir: str
        ) -> list[str]:
            """Write a partitioned parquet dataset per archive.

            This is done this way so that it's easy to tell which archive was
            downloaded and processed completely, so that we skip that in new runs.
            """
            files_written = write_to_parquet_partitions(
                data_stream=iter_bucketed_batches(zst_archive_path),
                output_dir=archive_dataset_dir,
                schema=PROCESSED_RECORD_SCHEMA,
                existing_data_behavior="delete_matching",
                use_threads=False,  # turn off pyarrow's internal threading
            )

            if files_written:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(zst_archive_path)

            return files_written

        done: list[str] = []

        download_futures: dict[Future, tuple[str, str]] = {}
        write_futures: dict[Future, str] = {}
        with (
            logging_redirect_tqdm(),  # Redirect logger output to tqdm.write
            ThreadPoolExecutor(
                max_workers=num_dl, thread_name_prefix="Archive-Download"
            ) as dl_executor,
            ThreadPoolExecutor(
                max_workers=num_wr, thread_name_prefix="Archive-Write"
            ) as writer_executor,
        ):
            for subreddit in subreddits:
                for content_type in ["submissions", "comments"]:
                    archive_id = f"{subreddit}-{content_type}"

                    if is_archive_processed(output_dir, archive_id):
                        # Skip subreddit-content_type that has already been
                        # downloaded and processed
                        done.append(archive_id)
                        continue

                    archive_dataset_dir = os.path.join(output_dir, archive_id)
                    os.makedirs(archive_dataset_dir, exist_ok=True)

                    download_futures[
                        dl_executor.submit(
                            download_sub_data, subreddit, content_type, output_dir
                        )
                    ] = (archive_id, archive_dataset_dir)

            for dl_fut in tqdm(
                as_completed(download_futures),
                total=len(download_futures),
                desc="Downloading and cleaning subreddit data",
                unit="archive",
                leave=False,
                position=0,
                dynamic_ncols=True,
            ):
                archive_id, archive_dataset_dir = download_futures[dl_fut]
                try:
                    zst_archive_path = dl_fut.result()
                except Exception:
                    logger.exception(
                        "%s: Download failed for %s", self.stage_name, archive_id
                    )
                    continue

                if zst_archive_path is None:
                    continue

                write_futures[
                    writer_executor.submit(
                        _writer_worker,
                        zst_archive_path,
                        archive_dataset_dir,
                    )
                ] = (archive_id, zst_archive_path)

            write_tasks = list(write_futures)
            for wr_fut in tqdm(
                as_completed(write_tasks),
                total=len(write_tasks),
                desc="Writing parquet partitions",
                unit="archive",
                leave=False,
                position=1,
                dynamic_ncols=True,
            ):
                archive_id, zst_archive_path = write_futures[wr_fut]

                try:
                    written_files = wr_fut.result()
                except Exception:
                    logging.error(
                        "%s: Write failed for %s",
                        self.stage_name,
                        archive_id,
                        exc_info=True,
                    )
                    written_files = None

                if written_files:
                    mark_archive_done(output_dir, archive_id)
                    done.append(archive_id)
                else:
                    logger.warning(
                        "%s: Failed to write any data for archive %s",
                        self.stage_name,
                        zst_archive_path,
                    )

        return done

    def _get_subs_data_dir(self, context: "CurationContext") -> tuple[str, str]:
        """Return the source directory and ensure the ``subs_data`` subdir exists.

        Returns
        -------
        tuple[str, str]
            A tuple of ``(source_dir, subs_data_dir)``.
        """
        source_dir = self.source_dir(context)
        subs_data_dir = os.path.join(source_dir, "subs_data")
        os.makedirs(subs_data_dir, exist_ok=True)
        return source_dir, subs_data_dir


def _get_relevant_subreddits(
    context: "CurationContext",
    condition_to_subreddit_map: dict[str, list[str]],
) -> set[str]:
    """Collect subreddits relevant to any condition in the experiments.

    Returns
    -------
    set[str]
        Set of subreddit names.
    """
    relevant_subreddits: set[str] = set()
    for experiment in context.experiments:
        for keyword in experiment.conditions or []:
            relevant_subreddits.update(condition_to_subreddit_map.get(keyword, []))
    return relevant_subreddits
