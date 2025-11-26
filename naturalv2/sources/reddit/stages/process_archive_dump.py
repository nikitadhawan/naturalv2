"""Reddit archive dump processor."""

import concurrent
import gc
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import psutil
import pyarrow as pa
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from naturalv2.sources.core import SourceStage
from naturalv2.sources.reddit.processing import (
    build_contextualized_dataset,
    write_to_parquet_partitions,
)
from naturalv2.sources.reddit.processing._utils import BUCKET_COUNT, get_max_open_files
from naturalv2.sources.reddit.pushshift_archive import (
    PROCESSED_RECORD_SCHEMA,
    iter_bucketed_batches,
)


if TYPE_CHECKING:
    from naturalv2.sources.core import CurationContext, StageState

logger = logging.getLogger(__name__)


class RedditDumpProcessor(SourceStage):
    """Process dump of Reddit archive from pushshift.

    This stage assumes that all the archive files have been downloaded and are
    available locally.

    Parameters
    ----------
    archive_dir_or_paths : str | Path | Sequence[str | Path]
        Path or list of paths to archive file(s) or folder(s) containing ``.zst``
        files.
    num_threads : int | None, optional, default=None
        The number of threads that will be used to process archive files in parallel.
        By default, this is set to a quarter of the number of CPUs available.
    chunk_size : int, default=128 << 20
        The chunk size to use when parsing the ``.zst`` files.
    name : str | None, optional, default=None
        Optional explicit stage name; defaults to the class name.
    """

    def __init__(
        self,
        archive_dir_or_paths: str | Path | Sequence[str | Path],
        *,
        num_threads: int | None = None,
        chunk_size: int = 256 << 20,
        name: str | None = None,
    ) -> None:
        """Initialize the stage."""
        super().__init__(name=name)
        self.archive_dir_or_paths = archive_dir_or_paths

        _cpu_count: int = psutil.cpu_count() or 1
        _max_open_files = get_max_open_files()
        self.num_threads = num_threads or max(
            1, min(_cpu_count // 4, _max_open_files // BUCKET_COUNT)
        )
        self.chunk_size = chunk_size

    async def run(
        self, context: "CurationContext", state: "StageState"
    ) -> "StageState":
        """Parse and clean ``.zst`` archive files.

        Parameters
        ----------
        context : CurationContext
            Pipeline context including experiment list and save directories.
        state : StageState
            Mutable pipeline state; will be updated with ``data_root`` and
            the ``source_dir`` used.

        Returns
        -------
        StageState
            Updated state containing path to cleaned data.
        """
        source_dir = self.source_dir(context)

        normalized_paths = _normalize_paths(self.archive_dir_or_paths)
        zst_files = list(_discover_zst_archives(normalized_paths))
        if not zst_files:
            raise RuntimeError(
                f"No .zst archives found in {self.archive_dir_or_paths}",
            )

        with logging_redirect_tqdm():
            staging_dir = os.path.join(source_dir, "reddit_dump", "staging")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_threads
            ) as executor:
                futures = {
                    executor.submit(
                        self._process_single_file,
                        path,
                        staging_dir,
                        context.experiment_name,
                    ): path
                    for path in zst_files
                }

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Processing Archives [{self.num_threads} workers]",
                    unit="file",
                    position=0,
                    leave=False,
                    dynamic_ncols=True,
                ):
                    path = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.exception("Failed to process %s: %s", path.name, exc)

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

    def _process_single_file(self, zst_path: Path, output_dir: str, run_tag: str):
        """Read, parse and filter ``.zst`` file, and write to parquet partitions."""
        try:
            # Create the iterator for this specific file
            batch_iter = iter_bucketed_batches(
                str(zst_path), chunk_size=self.chunk_size, progress_enabled=True
            )

            # Write directly to parquet
            write_to_parquet_partitions(
                data_stream=batch_iter,
                output_dir=output_dir,
                schema=PROCESSED_RECORD_SCHEMA,
                existing_data_behavior="overwrite_or_ignore",
                run_tag=f"{run_tag}-{zst_path.stem}",
            )
        finally:
            gc.collect()
            pa.default_memory_pool().release_unused()


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
