import concurrent
import logging
import os
import queue
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Sequence

import psutil
import pyarrow as pa
from tqdm import tqdm

from naturalv2.sources.core import CurationContext, SourceStage, StageState
from naturalv2.sources.reddit.processing import (
    build_contextualized_dataset,
    write_to_parquet_partitions,
)
from naturalv2.sources.reddit.pushshift_archive import (
    PROCESSED_RECORD_SCHEMA,
    iter_bucketed_batches,
)


logger = logging.getLogger(__name__)


class RedditDumpProcessor(SourceStage):
    def __init__(
        self,
        archive_dir_or_paths: str | Path | Sequence[str | Path],
        *,
        num_workers: int | None = None,
        chunk_size: int = 256 << 20,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.archive_dir_or_paths = archive_dir_or_paths

        cpu_count: int = psutil.cpu_count(logical=False) or 1
        self.num_workers = num_workers or cpu_count
        self.chunk_size = chunk_size

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        source_dir = self.source_dir(context)

        staging_dir = os.path.join(source_dir, "reddit_dump", "staging")
        write_to_parquet_partitions(
            data_stream=self._stream_record_batches(),
            output_dir=staging_dir,
            schema=PROCESSED_RECORD_SCHEMA,
            existing_data_behavior="delete_matching",
            run_tag=context.experiment_name,
        )

        final_dir = os.path.join(source_dir, "reddit_dump", "final")
        _ = build_contextualized_dataset(
            source_dir=staging_dir,
            dest_dir=final_dir,
            run_tag=context.experiment_name,
            cleanup_source=False,
        )

        # Update state
        state.payload = final_dir
        state.update(
            data_root=final_dir,
            source_dir=source_dir,
        )

        # Update and persist metadata in StudyDataset
        self.persist_dataset(
            context,
            namespace_paths={f"{context.source_name}_cleaned": final_dir},
        )
        return state

    def _stream_record_batches(self) -> Iterator[pa.RecordBatch]:
        """Yield record batches from one or more Reddit archive sources."""
        normalized_paths = _normalize_paths(self.archive_dir_or_paths)
        zst_files = list(_discover_zst_archives(normalized_paths))
        if not zst_files:
            logger.error(
                "No .zst archives found for input: %s", self.archive_dir_or_paths
            )
            return

        logger.info("Found %d .zst files to process", len(zst_files))

        # Simple sequential mode (avoid overhead if only 1 worker)
        if self.num_workers <= 1:
            for archive_path in tqdm(
                zst_files,
                desc="Processing .zst files",
                unit="file",
                position=0,
                leave=False,
                dynamic_ncols=True,
            ):
                for batch in iter_bucketed_batches(
                    str(archive_path), chunk_size=self.chunk_size, progress_enabled=True
                ):
                    if batch.num_rows:
                        yield batch
            return

        # Queue to hold batches from workers.
        # maxsize prevents loading too much data into RAM if the writer is slower than readers.
        batch_queue = queue.Queue(maxsize=self.num_workers * 2)
        sentinel = object()

        def _worker_produce_batches(path: Path):
            """Read a file and push batches to the queue."""
            try:
                for batch in iter_bucketed_batches(
                    str(path),
                    chunk_size=self.chunk_size,
                    progress_enabled=True,
                    use_threads_for_parsing=False,  # already using threads at file level
                ):
                    if batch.num_rows:
                        batch_queue.put(batch)
            except Exception as exc:
                logger.error("Worker failed on %s: %s", path.name, exc, exc_info=True)
            finally:
                # Signal that this specific worker is done
                batch_queue.put(sentinel)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            # Submit all files to the pool
            futures = [
                executor.submit(_worker_produce_batches, path) for path in zst_files
            ]

            active_workers = len(futures)

            # Main thread loop: consume from queue and yield to the writer
            while active_workers > 0:
                item = batch_queue.get()

                if item is sentinel:
                    active_workers -= 1
                else:
                    yield item

            # Check for any exceptions raised during execution
            for fut in futures:
                if fut.exception():
                    logger.error(
                        "A worker thread raised an exception: %s", fut.exception()
                    )


def _normalize_paths(dir_or_file_path: str | Path | Sequence[str | Path]) -> list[Path]:
    """Return a list of expanded ``Path`` objects."""
    if isinstance(dir_or_file_path, (str, Path)):
        return [Path(dir_or_file_path).expanduser()]
    try:
        return [Path(path).expanduser() for path in dir_or_file_path]
    except TypeError as exc:
        raise TypeError(
            "Expected a string, Path object or a list of string or Path objects, "
            f"but got {dir_or_file_path}"
        ) from exc


def _discover_zst_archives(candidates: list[Path]) -> Iterable[Path]:
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
            logger.warning("File %s is not a .zst archive; skipping...", path)
        return

    if not path.is_dir():
        logger.warning("Path %s is neither a file nor a directory; skipping...", path)
        return

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".zst"):
                yield Path(dirpath, filename).resolve()
