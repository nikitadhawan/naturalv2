"""Reddit Curation Stage.

Architecture:
1. Main Process: Scans for Parquet files, creates batches of file paths.
2. Workers: Initialize Registry once. Process batches. Write partial parquets to unique temp dirs.
3. Main Process: Consolidates partial parquets into final output.
"""

import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import ahocorasick
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from naturalv2.sources.components.helpers import (
    build_treatment_automaton,
    extract_mentions,
    iter_canonical_variations,
    normalize_text_for_matching,
)
from naturalv2.sources.core import SourceStage
from naturalv2.sources.reddit.processing.filter import scan_reddit_dataset
from naturalv2.utils import get_experiment_filepath


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.sources.core import CurationContext, StageState


logger = logging.getLogger(__name__)

_WORKER_REGISTRY: dict[str, "_SubredditContext"] = {}
_NUM_THREADS_PER_WORKER = 2


@dataclass
class _RegistryConfig:
    """Configuration for building the term registry in worker processes."""

    experiments_data: list[dict[str, Any]]
    condition_map: dict[str, list[str]]
    available_subreddits: list[str]
    filter_by_date: bool


@dataclass
class _SubredditContext:
    """Context for processing a single subreddit."""

    name: str
    automaton: ahocorasick.Automaton
    term_to_experiments: dict[str, set[str]]
    experiment_publication_dates: dict[str, datetime]
    global_max_date: datetime
    # Pre-calculated map of NCT_ID -> [List of valid terms]
    experiment_to_terms: dict[str, list[str]]


class RedditCurateStage(SourceStage):
    """Reddit curation stage.

    Parameters
    ----------
    num_workers : int | None, optional, default=None
        The number of workers to use to curate experiment data in parallel.
        If ``None``, a safe default will be set based on the available memory
        and CPUs.
    max_files_per_worker : int, default=20
        The maximum number of parquet files processed by one worker.
    batch_size : int, default=32_000
        The number of rows when scanning subreddits from parquet partitions.
    name : str | None, optional
        Optional explicit stage name; defaults to the class name.
    """

    def __init__(
        self,
        *,
        num_workers: int | None = None,
        max_files_per_worker: int = 20,
        batch_size: int = 32_000,
        name: str | None = None,
    ):
        super().__init__(name=name)
        cpu_count: int = psutil.cpu_count(logical=True) or 1

        # Auto-calculate worker count based on available memory
        total_mem_gb = psutil.virtual_memory().available / (1024**3)
        mem_workers_limit = int(  # Reserve 4GB; set 2GB per worker
            (total_mem_gb - 4) / 6
        )
        cpu_workers_limit = max(1, cpu_count // _NUM_THREADS_PER_WORKER)
        safe_workers = min(mem_workers_limit, cpu_workers_limit)
        self.num_workers = num_workers or max(1, safe_workers)
        self.max_files_per_worker = max_files_per_worker
        self.batch_size = batch_size

        self._curation_columns = [
            "subreddit",
            "title",
            "initial_post",
            "report_text",
            "report_type",
            "score",
            "date_created",
            "permalink",
            "author_replies",
        ]

    async def run(
        self, context: "CurationContext", state: "StageState"
    ) -> "StageState":
        """Execute parallelized curation."""
        root_dir = state.require_metadata("data_root", stage=self.stage_name)
        study_dir = self.condition_dir(context)

        temp_dir = os.path.join(study_dir, f"temp_curation_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)

        # Prepare configuration
        available_subreddits = state.metadata.get("available_subreddits", [])
        registry_config = self._prepare_registry_config(context, available_subreddits)
        registry = _build_worker_registry(registry_config)

        # Discover Parquet files
        # We do this manually, instead of letting polars use hive partitioning scheme,
        # so that we can distribute parquet files to different workers
        all_files = glob.glob(os.path.join(root_dir, "**", "*.parquet"), recursive=True)
        logger.info(f"Found {len(all_files)} Parquet files")

        # Create batches
        batch_size = max(
            1, min(len(all_files) // self.num_workers, self.max_files_per_worker)
        )
        batches = [
            all_files[i : i + batch_size] for i in range(0, len(all_files), batch_size)
        ]

        total_counts = defaultdict(int)

        # Process batches in parallel
        num_workers = min(len(batches), self.num_workers)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_initializer,
            initargs=(registry,),
        ) as pool:
            futures = [
                pool.submit(
                    _process_batch_task,
                    batch,
                    temp_dir,
                    batch_size=self.batch_size,
                    columns=self._curation_columns,
                    filter_by_date=context.filter_by_date,
                )
                for batch in batches
            ]

            with logging_redirect_tqdm():
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Curating experiment datasets [{num_workers} workers]",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    try:
                        result = future.result()
                        for nct_id, count in result.items():
                            total_counts[nct_id] += count
                    except Exception as exc:
                        logger.error(f"Curation of experiment data failed: {exc}")

        # Consolidate results
        curated_paths = {}

        for experiment in tqdm(
            context.experiments,
            desc="Consolidating experiment files",
            leave=False,
            dynamic_ncols=True,
        ):
            final_path, _ = self._get_experiment_save_path(
                context, experiment, study_dir
            )
            curated_paths[experiment.nct_id] = final_path

            # Update experiment paths
            if context.filter_by_date:
                experiment.source_paths[context.source_name] = final_path
            else:
                experiment.source_paths[f"{context.source_name}_no_date_filter"] = (
                    final_path
                )

            # Merge partial CSVs
            self._consolidate_parquet_chunks(temp_dir, experiment.nct_id, final_path)

            # Save updated experiment
            experiment.to_yaml(
                filename=get_experiment_filepath(context.save_dir, experiment.nct_id)
            )

        # Cleanup
        shutil.rmtree(temp_dir)

        state.update(curated_paths=curated_paths)
        logger.info(f"Curation complete. Record counts: {dict(total_counts)}")
        self.persist_dataset(
            context,
            per_experiment_paths=curated_paths,
            per_experiment_sizes=total_counts,
        )

        return state

    @staticmethod
    def _prepare_registry_config(
        context: "CurationContext", available_subreddits: list[str]
    ) -> _RegistryConfig:
        """Convert experiments to serializable configuration."""
        experiments_data = []

        for experiment in context.experiments:
            # Parse publication date
            publication_date = experiment.date

            if publication_date:
                if isinstance(publication_date, str):
                    try:
                        publication_date = datetime.fromisoformat(publication_date)
                    except ValueError:
                        publication_date = datetime.max

                if publication_date.tzinfo is None:
                    publication_date = publication_date.replace(tzinfo=timezone.utc)
            else:
                publication_date = datetime.max.replace(tzinfo=timezone.utc)

            treatment_names = set(experiment.treatment_names)
            common_names: set[str] = set()
            for aliases in experiment.treatment_common_names.get(
                context.source_name, {}
            ).values():
                for alias in aliases:
                    if len(alias) > 3:  # ignore short common names
                        common_names.add(alias)
            terms = list(treatment_names | common_names)

            experiments_data.append(
                {
                    "nct_id": experiment.nct_id,
                    "conditions": experiment.conditions,
                    "publication_date": publication_date,
                    "terms": terms,
                }
            )

        return _RegistryConfig(
            experiments_data=experiments_data,
            condition_map=context.study_dataset.sources.get(context.source_name, {}),
            available_subreddits=available_subreddits,
            filter_by_date=context.filter_by_date,
        )

    @staticmethod
    def _consolidate_parquet_chunks(temp_root: str, nct_id: str, target_path: str):
        """Merge all partial parquet files for an experiment."""
        partials = glob.glob(os.path.join(temp_root, "*", f"{nct_id}_*.parquet"))

        if not partials:
            return

        os.makedirs(target_path, exist_ok=True)

        for src_path in partials:
            filename = os.path.basename(src_path)
            dest_path = os.path.join(target_path, filename)
            try:
                shutil.move(src_path, dest_path)
            except Exception as exc:
                logger.error("Failed to move %s to %s: %s", src_path, dest_path, exc)

    @staticmethod
    def _get_experiment_save_path(
        context: "CurationContext", experiment: "Experiment", study_dir: str
    ) -> tuple[str, str]:
        """Generate save path for experiment results."""
        suffix = "" if context.filter_by_date else "_no_date_filter"
        path = os.path.join(
            study_dir, f"{context.source_name}_{experiment.nct_id}{suffix}"
        )
        return path, f"{context.source_name}{suffix}"


def _worker_initializer(registry: dict[str, _SubredditContext]) -> None:
    """Initialize worker process with registry.

    Called once per worker."""

    # Minimize CPU thrashing by setting only 2 threads per worker for polars
    # and pyarrow
    os.environ["POLARS_MAX_THREADS"] = str(_NUM_THREADS_PER_WORKER)
    os.environ["OMP_NUM_THREADS"] = str(_NUM_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(_NUM_THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(_NUM_THREADS_PER_WORKER)

    pa.set_cpu_count(_NUM_THREADS_PER_WORKER)
    pa.set_io_thread_count(_NUM_THREADS_PER_WORKER)

    _WORKER_REGISTRY.clear()
    _WORKER_REGISTRY.update(registry)


def _build_worker_registry(config: _RegistryConfig) -> dict[str, _SubredditContext]:
    """Build Aho-Corasick automaton registry for each subreddit."""
    subreddit_term_map = defaultdict(lambda: defaultdict(set))
    subreddit_publication_dates = defaultdict(dict)
    allowed_subreddits = (
        set(config.available_subreddits) if config.available_subreddits else None
    )

    # Build term mappings per subreddit
    for experiment_data in config.experiments_data:
        # Build the set of subreddit that are relevant to the experiment
        relevant_subreddits = set()
        for condition in experiment_data["conditions"]:
            relevant_subreddits.update(config.condition_map.get(condition, []))

        # Only keep the ones in the pushshift data
        if allowed_subreddits:
            relevant_subreddits &= allowed_subreddits
        if not relevant_subreddits:
            continue

        # Build the mapping: subreddit -> {treatment name -> [experiment IDs]}
        publication_datetime = experiment_data.get(
            "publication_date"
        ) or datetime.max.replace(tzinfo=timezone.utc)
        terms = experiment_data["terms"]
        nct_id = experiment_data["nct_id"]

        for subreddit in relevant_subreddits:
            subreddit_publication_dates[subreddit][nct_id] = publication_datetime
            for term in terms:
                for canonical_term in iter_canonical_variations(term):
                    subreddit_term_map[subreddit][canonical_term].add(nct_id)

    # Create automaton for each subreddit
    registry: dict[str, _SubredditContext] = {}
    for subreddit, term_map in subreddit_term_map.items():
        automaton = build_treatment_automaton(list(term_map.keys()))

        # Get the date of the most recent experiment for which the subreddit is relevant
        publication_dates = subreddit_publication_dates[subreddit]
        global_max_datetime = max(
            publication_dates.values(),
            default=datetime.max.replace(tzinfo=timezone.utc),
        )

        # Invert the term map to create (NCT -> Valid Terms)
        exp_to_terms = defaultdict(list)
        for term, nct_ids in term_map.items():
            for nct_id in nct_ids:
                exp_to_terms[nct_id].append(term)

        registry[subreddit] = _SubredditContext(
            name=subreddit,
            automaton=automaton,
            term_to_experiments=term_map,
            experiment_publication_dates=publication_dates,
            global_max_date=global_max_datetime,
            experiment_to_terms=exp_to_terms,
        )

    return registry


# -----------------------------------------------------------------------------
# Batch Processing
# -----------------------------------------------------------------------------


def _process_batch_task(
    file_paths: list[str],
    temp_output_dir: str,
    columns: list[str],
    batch_size: int,
    filter_by_date: bool,
) -> dict[str, int]:
    """Process a batch of files and write results to temp directory."""
    if not _WORKER_REGISTRY:
        return {}

    batch_id = str(uuid.uuid4())
    worker_out_dir = os.path.join(temp_output_dir, batch_id)
    os.makedirs(worker_out_dir, exist_ok=True)

    counts = defaultdict(int)

    # Keep open file handles for this batch
    # Dict[nct_id, pyarrow.parquet.ParquetWriter]
    writers: dict[str, pq.ParquetWriter] = {}

    # Pass target subreddits to iterator to enable predicate pushdown
    target_subreddits = list(_WORKER_REGISTRY.keys())
    chunk_iterator = scan_reddit_dataset(
        file_paths, columns, target_subreddits=target_subreddits, batch_size=batch_size
    )

    try:
        for df_chunk in chunk_iterator:
            try:
                batch_id = uuid.uuid4().hex[:8]
                for sub_name, sub_df in df_chunk.group_by("subreddit"):
                    normalized_sub_name = (
                        sub_name[0] if isinstance(sub_name, tuple) else sub_name
                    )

                    # Case-insensitive subreddit lookup
                    ctx = _WORKER_REGISTRY.get(
                        normalized_sub_name
                    ) or _WORKER_REGISTRY.get(normalized_sub_name.lower())

                    if ctx:
                        matches_df = _process_chunk(ctx, sub_df, filter_by_date)

                        if matches_df is not None and not matches_df.is_empty():
                            # Write immediately to the open file handle
                            _stream_to_disk(
                                matches_df, writers, worker_out_dir, counts, batch_id
                            )

                # Memory cleanup after each chunk
                del df_chunk

            except Exception as e:
                logger.error("Worker error in batch %s: %s", batch_id, e)
            finally:
                # Force garbage collection more frequently
                gc.collect()
    finally:
        # Close all open writers when batch finishes
        for writer in writers.values():
            writer.close()

    return dict(counts)


def _process_chunk(
    ctx: _SubredditContext, df: pl.DataFrame, filter_by_date: bool
) -> pl.DataFrame | None:
    """Process a subreddit chunk: match terms, filter, and write results."""
    # Prepare text for matching
    txt_cols = [
        col for col in ["title", "initial_post", "report_text"] if col in df.columns
    ]
    if not txt_cols:
        return None

    df_prep = df.with_columns(
        pl.concat_str(txt_cols, separator=" ", ignore_nulls=True)
        .map_elements(normalize_text_for_matching, return_dtype=pl.String)
        .alias("_normalized_text")
    )

    # Global date filtering
    # Remove any posts AFTER the most recent publication that one of the experiments
    # that are relevant to the subreddit being processed. This means less data
    # for the automaton to try to match on
    if filter_by_date:
        df_prep = _parse_date_column(df_prep, "date_created")
        df_prep = df_prep.filter(pl.col("_dt").is_not_null())
        df_prep = df_prep.filter(pl.col("_dt") <= ctx.global_max_date)

    if df_prep.is_empty():
        return None

    # Term matching with Aho-Corasick
    automaton = ctx.automaton

    def batch_matcher(text_series: pl.Series) -> pl.Series:
        results = [
            extract_mentions(text, automaton) if text is not None else []
            for text in text_series
        ]
        return pl.Series(results, dtype=pl.List(pl.String))

    df_matches = df_prep.with_columns(
        pl.col("_normalized_text").map_batches(batch_matcher).alias("_matches")
    ).filter(pl.col("_matches").list.len() > 0)

    del df_prep

    if df_matches.is_empty():
        return None

    df_matches = df_matches.drop("_normalized_text").with_columns(
        pl.col("_matches").alias("treatments_mentioned")
    )

    # Explode matches and join with experiments
    df_exploded = df_matches.explode("_matches")

    del df_matches

    lookup_data = [
        (term, nct_id)
        for term, nct_ids in ctx.term_to_experiments.items()
        for nct_id in nct_ids
    ]
    df_lookup = pl.DataFrame(lookup_data, schema=["_matches", "nct_id"], orient="row")

    df_final = df_exploded.join(df_lookup, on="_matches", how="inner")

    # Create a lightweight DataFrame defining valid terms per Experiment
    valid_terms_data = [(nct, terms) for nct, terms in ctx.experiment_to_terms.items()]
    df_valid_terms = pl.DataFrame(
        valid_terms_data,
        schema={"nct_id": pl.String, "valid_terms": pl.List(pl.String)},
        orient="row",
    )

    # Join valid terms to the main result
    # We join on 'nct_id' so every row knows what terms are allowed for its
    # assigned experiment
    df_final = df_final.join(df_valid_terms, on="nct_id", how="left")

    # Intersect the found terms with valid terms
    # "treatments_mentioned" = ["Advil", "Tylenol"]
    # "valid_terms" (for Exp A) = ["Advil", "Motrin"]
    # Result = ["Advil"]
    df_final = df_final.with_columns(
        pl.col("treatments_mentioned").list.set_intersection(pl.col("valid_terms"))
    ).drop("valid_terms")

    # Experiment-specific date filter
    # Now we apply the date filter on the experiment-level, using the publication
    # date of the trial
    if filter_by_date and "_dt" in df_final.columns:
        experiment_metadata = list(ctx.experiment_publication_dates.items())
        df_metadata = pl.DataFrame(
            experiment_metadata, schema=["nct_id", "publication_date"], orient="row"
        )

        df_final = (
            df_final.join(df_metadata, on="nct_id", how="left")
            .filter(pl.col("_dt") <= pl.col("publication_date"))
            .drop("publication_date")
        )

    # Deduplicate by experiment ID and permalink
    if "permalink" in df_final.columns:
        df_final = df_final.unique(subset=["nct_id", "permalink"])

    if df_final.is_empty():
        return None

    # Build markdown report
    report_expr = _build_report_expr(df_final.columns)
    df_final = df_final.with_columns(report_expr.alias("report"))

    # Serialize nested data structures for CSV
    # df_final = _serialize_nested_columns(df_final)

    # Drop temporary columns
    cols_to_drop = [c for c in ["_matches", "_dt"] if c in df_final.columns]
    if cols_to_drop:
        df_final = df_final.drop(cols_to_drop)

    return df_final


def _stream_to_disk(
    df: pl.DataFrame, writers: dict, out_dir: str, counts: dict, batch_id: str
) -> None:
    """Write a chunk to the appropriate ParquetWriter, creating it if needed."""

    # Group by NCT ID so we route data to the correct file
    for key, part_df in df.partition_by("nct_id", as_dict=True).items():
        nct_id = key[0] if isinstance(key, tuple) else key

        # Remove nct_id column (it's in the filename)
        payload_df = part_df.drop("nct_id") if "nct_id" in part_df.columns else part_df

        # Convert to Arrow Table
        table = payload_df.to_arrow()

        if nct_id not in writers:
            # Initialize Writer on first contact
            filename = f"{nct_id}_{batch_id}.parquet"
            save_path = os.path.join(out_dir, filename)

            # Use schema from the first chunk
            writers[nct_id] = pq.ParquetWriter(
                save_path, table.schema, compression="lz4"
            )

        # Check for schema mismatch (safe-guard)
        # If columns are missing/added, this would normally crash.
        # Since we use consistent input columns, this is usually fine.
        try:
            writers[nct_id].write_table(table)
            counts[nct_id] += len(payload_df)
        except Exception:
            # Fallback: Cast to existing schema if there's a slight mismatch (e.g. null vs string)
            # This is slower but safer
            safe_table = table.cast(writers[nct_id].schema)
            writers[nct_id].write_table(safe_table)
            counts[nct_id] += len(payload_df)


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------


def _build_report_expr(available_cols: list[str]) -> pl.Expr:
    """Generate markdown report column expression."""

    def safe_col(name: str) -> pl.Expr:
        """Return column or empty string if not available."""
        return pl.col(name).fill_null("") if name in available_cols else pl.lit("")

    # Format date column
    date_col = "date_created"
    if date_col in available_cols:
        date_expr = pl.col(date_col).fill_null("").cast(pl.String)
    else:
        date_expr = pl.lit("")

    # Submission format
    fmt_submission = pl.format(
        "**Subreddit**\nThis post was found on the subreddit r/{}.\n\n"
        "**Title**\nThis post was titled: {}\n\n"
        "**Date created**\nThis post was created on {}.\n\n"
        "**Post**\n{}",
        safe_col("subreddit"),
        safe_col("title"),
        date_expr,
        safe_col("report_text"),
    )

    # Comment format
    fmt_comment = pl.format(
        "**Subreddit**\nThis comment was found on the subreddit r/{}.\n\n"
        "**Initial Post**\nThis comment was in response to the following post:\n"
        "Title: {}\nPost content: {}\n\n"
        "**Date created**\nThis comment was created on {}.\n\n"
        "**Comment**\n{}",
        safe_col("subreddit"),
        safe_col("title"),
        safe_col("initial_post"),
        date_expr,
        safe_col("report_text"),
    )

    if "report_type" not in available_cols:
        return fmt_submission

    return (
        pl.when(pl.col("report_type") == "submission")
        .then(fmt_submission)
        .when(pl.col("report_type") == "comment")
        .then(fmt_comment)
        .otherwise(pl.lit(""))
    )


# -----------------------------------------------------------------------------
# Chunk Processing
# -----------------------------------------------------------------------------


def _parse_date_column(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Parse date column using multiple format strategies."""
    if date_col not in df.columns:
        return df

    date_expr = pl.coalesce(
        [
            # String formats
            pl.col(date_col).str.to_datetime("%B %d, %Y", strict=False),
            pl.col(date_col).str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False),
            pl.col(date_col).str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False),
            pl.col(date_col).str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False),
            pl.col(date_col).str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
            pl.col(date_col).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col(date_col).str.to_datetime("%Y-%m-%d", strict=False),
            # Unix timestamp (seconds to microseconds)
            (pl.col(date_col).cast(pl.Float64, strict=False) * 1_000_000)
            .cast(pl.Int64, strict=False)
            .cast(pl.Datetime("us"), strict=False),
        ]
    ).dt.replace_time_zone("UTC")

    return df.with_columns(date_expr.alias("_dt"))
