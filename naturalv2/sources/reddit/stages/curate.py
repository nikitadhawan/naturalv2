"""Reddit Curation Stage.

Architecture:
1. Main Process: Scans for Parquet files, creates batches of file paths.
2. Workers: Initialize Registry once. Process batches. Write partial parquets to unique temp dirs.
3. Main Process: Consolidates partial parquets into final output.
"""

import gc
import glob
import logging
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
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from naturalv2.sources.components.helpers import (
    CONNECTOR_CHAR_SET,
    CONNECTOR_SPLIT,
    POST_NFKC_TRANSLATION_TABLE,
    PRE_NFKC_TRANSLATION_TABLE,
    build_treatment_automaton,
    iter_canonical_variations,
)
from naturalv2.sources.core import SourceStage
from naturalv2.sources.reddit.processing._utils import (
    get_default_num_workers,
    get_tqdm_position,
    release_memory,
)
from naturalv2.sources.reddit.processing._utils import (
    worker_initializer as common_worker_initializer,
)
from naturalv2.sources.reddit.processing.contextualize import (
    CONTEXTUALIZED_RECORD_SCHEMA,
    PARTITIONING,
)
from naturalv2.sources.reddit.processing.filter import (
    get_subreddit_filter_expr,
    scan_reddit_dataset,
)
from naturalv2.utils import get_experiment_filepath


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.sources.core import CurationContext, StageState


logger = logging.getLogger(__name__)

_NUM_THREADS_PER_WORKER = 2
_WORKER_REGISTRY: dict[str, "_SubredditContext"] = {}


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
    batch_size : int, default=16_384
        The number of rows when scanning subreddits from parquet partitions.
    name : str | None, optional
        Optional explicit stage name; defaults to the class name.
    """

    def __init__(
        self,
        *,
        num_workers: int | None = None,
        batch_size: int = 16_384,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.num_workers = num_workers or get_default_num_workers(
            mem_gb_per_worker=8, threads_per_worker=_NUM_THREADS_PER_WORKER
        )
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

        # Prepare registry
        available_subreddits = state.metadata.get("available_subreddits", [])
        registry_config = self._prepare_registry_config(context, available_subreddits)
        registry = _build_registry(registry_config)

        # Create pyarrow dataset for automatic file discovery
        dataset = ds.dataset(
            root_dir,
            schema=CONTEXTUALIZED_RECORD_SCHEMA,
            format="parquet",
            partitioning=PARTITIONING,
        )

        total_counts = defaultdict(int)

        # Process files in parallel
        num_workers = min(len(registry), self.num_workers)
        with ProcessPoolExecutor(
            max_workers=max(num_workers, 1),
            initializer=_worker_initializer,
            initargs=(registry,),
        ) as pool:
            futures = []
            for subreddit in registry:
                filter_expr = get_subreddit_filter_expr([subreddit])

                fragments = dataset.get_fragments(filter_expr)
                file_paths = [frag.path for frag in fragments]

                if not file_paths:
                    continue

                for file_path in file_paths:
                    futures.append(
                        pool.submit(
                            _process_batch_task,
                            file_path,
                            subreddit,
                            temp_dir,
                            batch_size=self.batch_size,
                            columns=self._curation_columns,
                            filter_by_date=context.filter_by_date,
                        )
                    )

            with logging_redirect_tqdm():
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Curating experiment datasets [{num_workers} workers]",
                    leave=False,
                    position=0,
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

    This function is called once per worker process in the ProcessPoolExecutor
    to set up the shared registry and configure PyArrow thread settings.

    Parameters
    ----------
    registry : dict[str, _SubredditContext]
        Mapping of subreddit names to their processing contexts, including
        Aho-Corasick automatons and experiment metadata.

    Notes
    -----
    This initialization ensures that each worker has its own copy of the
    registry to avoid serialization overhead on every task submission.
    """

    common_worker_initializer(num_threads=_NUM_THREADS_PER_WORKER)

    _WORKER_REGISTRY.clear()
    _WORKER_REGISTRY.update(registry)


def _build_registry(config: _RegistryConfig) -> dict[str, _SubredditContext]:
    """Build Aho-Corasick automaton registry for each subreddit.

    Creates a mapping of subreddit names to processing contexts, where each
    context contains an Aho-Corasick automaton for efficient multi-pattern
    string matching of treatment terms.

    Parameters
    ----------
    config : _RegistryConfig
        Configuration containing experiments data, condition-to-subreddit mappings,
        available subreddits, and date filtering preferences.

    Returns
    -------
    dict[str, _SubredditContext]
        Mapping of subreddit names to their processing contexts. Each context
        includes:
        - Aho-Corasick automaton for treatment term matching
        - Term-to-experiment mappings
        - Publication dates for date filtering
        - Experiment-to-terms reverse mappings
    """
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
    file_path: str,
    subreddit: str,
    temp_output_dir: str,
    columns: list[str],
    batch_size: int,
    filter_by_date: bool,
) -> dict[str, int]:
    """Process a batch of files and write results to temp directory.

    Scans a Parquet file for Reddit posts/comments from a specific subreddit,
    matches treatment terms using the pre-built Aho-Corasick automaton, and
    streams matching records to temporary Parquet files grouped by experiment ID.

    Parameters
    ----------
    file_path : str
        Absolute path to the Parquet file to process.
    subreddit : str
        Name of the subreddit being processed.
    temp_output_dir : str
        Directory path where temporary Parquet files will be written.
    columns : list[str]
        List of column names to read from the Parquet file.
    batch_size : int
        Number of rows to read per batch when scanning the dataset.
    filter_by_date : bool
        Whether to filter posts by publication date of experiments.

    Returns
    -------
    dict[str, int]
        Mapping of experiment IDs (NCT IDs) to the count of matching records
        found in this batch.

    Notes
    -----
    This function maintains open ParquetWriter handles for each experiment
    to enable efficient streaming writes without loading all data into memory.
    """
    if not _WORKER_REGISTRY:
        return {}

    batch_id = str(uuid.uuid4())
    worker_out_dir = os.path.join(temp_output_dir, batch_id)
    os.makedirs(worker_out_dir, exist_ok=True)

    counts = defaultdict(int)

    # Keep open file handles for this batch
    # Dict[nct_id, pyarrow.parquet.ParquetWriter]
    writers: dict[str, pq.ParquetWriter] = {}

    try:
        ctx = _WORKER_REGISTRY[subreddit]
        for df in tqdm(
            scan_reddit_dataset(
                file_path,
                schema=CONTEXTUALIZED_RECORD_SCHEMA,
                partitioning=PARTITIONING,
                columns=columns,
                subreddit=ctx.name,
                batch_size=batch_size,
                use_threads=False,
            ),
            desc=f"Curating from {ctx.name}",
            leave=False,
            position=get_tqdm_position(),
        ):
            try:
                matches_df = _process_chunk(ctx, df, filter_by_date)

                if matches_df is not None and not matches_df.is_empty():
                    # Write immediately to the open file handle
                    _stream_to_disk(
                        matches_df, writers, worker_out_dir, counts, batch_id[:8]
                    )
                del df
                del matches_df
            finally:
                gc.collect()

    finally:
        # Close all open writers when batch finishes
        for writer in writers.values():
            writer.close()

        # Clean up worker memory
        writers.clear()
        del writers

        release_memory()

    return dict(counts)


def _process_chunk(
    ctx: _SubredditContext, df: pl.DataFrame, filter_by_date: bool
) -> pl.DataFrame | None:
    """Process a subreddit chunk: match terms, filter, and write results.

    Normalizes text from Reddit posts/comments, uses the Aho-Corasick automaton
    to find treatment mentions, joins with experiment metadata, and applies
    date filtering based on experiment publication dates.

    Parameters
    ----------
    ctx : _SubredditContext
        Subreddit processing context containing the Aho-Corasick automaton,
        term mappings, and publication dates.
    df : pl.DataFrame
        DataFrame containing Reddit posts/comments to process.
    filter_by_date : bool
        Whether to filter records by publication dates.

    Returns
    -------
    pl.DataFrame | None
        DataFrame containing matched records with columns:
        - All original columns from input DataFrame
        - nct_id: Experiment ID
        - treatments_mentioned: List of treatment terms found
        - report: Markdown-formatted report text
        Returns None if no matches are found.

    """
    # Prepare text for matching
    txt_cols = [
        col for col in ["title", "initial_post", "report_text"] if col in df.columns
    ]
    if not txt_cols:
        logger.warning(
            "No text columns available for matching in subreddit %s", ctx.name
        )
        return None

    df_prep = df

    # Global date filtering
    # Remove any posts AFTER the most recent publication that one of the experiments
    # that are relevant to the subreddit being processed. This means less data
    # for the automaton to try to match on
    if filter_by_date:
        df_prep = _parse_date_column(df_prep, "date_created")
        df_prep = df_prep.filter(pl.col("_dt").is_not_null())
        df_prep = df_prep.filter(pl.col("_dt") <= ctx.global_max_date)

        if df_prep.is_empty():
            logger.info(
                "All posts/comments in this chunk are after the publication date of "
                "relevant experiments for subreddit %s. Skipping automaton matching.",
                ctx.name,
            )
            return None

    df_prep = (
        df_prep.with_columns(
            pl.concat_str(txt_cols, separator=" ", ignore_nulls=True).alias("_raw_text")
        )
        .with_columns(_get_normalization_expr("_raw_text").alias("_normalized_text"))
        .drop("_raw_text")
    )

    def batch_automaton_matcher(text_series: pl.Series) -> pl.Series:
        results = []
        for text in text_series:
            if not text:
                results.append([])
                continue

            found = set()

            for _, canonical_alias in ctx.automaton.iter(text):
                found.add(canonical_alias)
            results.append(sorted(found))

        return pl.Series(results, dtype=pl.List(pl.String))

    df_matches = df_prep.with_columns(
        pl.col("_normalized_text")
        .map_batches(batch_automaton_matcher)
        .alias("_matches")
    ).filter(pl.col("_matches").list.len() > 0)

    del df_prep

    if df_matches.is_empty():
        logger.debug(
            "No treatment mentions found in this chunk for subreddit %s after automaton matching.",
            ctx.name,
        )
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
        logger.debug(
            "No treatment mentions remain in this chunk for subreddit %s after "
            "joining with experiments and applying date filters.",
            ctx.name,
        )
        return None

    # Build markdown report
    report_expr = _build_report_expr(df_final.columns)
    df_final = df_final.with_columns(report_expr.alias("report"))

    # Drop temporary columns
    cols_to_drop = [c for c in ["_matches", "_dt"] if c in df_final.columns]
    if cols_to_drop:
        df_final = df_final.drop(cols_to_drop)

    return df_final


def _stream_to_disk(
    df: pl.DataFrame, writers: dict, out_dir: str, counts: dict, batch_id: str
) -> None:
    """Write a chunk to the appropriate ParquetWriter, creating it if needed.

    Partitions the DataFrame by experiment ID (NCT ID) and streams each partition
    to its corresponding Parquet file. Creates new ParquetWriter instances on
    first contact with each experiment.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing matched records with an 'nct_id' column.
    writers : dict
        Dictionary mapping NCT IDs to open ParquetWriter instances.
        Modified in place to add new writers as needed.
    out_dir : str
        Directory path where Parquet files will be written.
    counts : dict
        Dictionary mapping NCT IDs to record counts.
        Modified in place to update counts.
    batch_id : str
        Unique identifier for this batch, used in output filenames.

    Notes
    -----
    This function maintains open file handles to avoid repeated file open/close
    operations, which significantly improves write performance. Schema mismatches
    are handled by casting to the existing schema if necessary.
    """

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
    """Generate markdown report column expression.

    Creates a Polars expression that formats Reddit post/comment data into
    structured markdown reports. The format differs for submissions vs comments.

    Parameters
    ----------
    available_cols : list[str]
        List of column names available in the DataFrame.

    Returns
    -------
    pl.Expr
        Polars expression that generates markdown-formatted report text.
        For submissions, includes: subreddit, title, date, post content.
        For comments, includes: subreddit, initial post, date, comment content.

    Raises
    ------
    ValueError
        If required columns (date_created, subreddit, title, report_text,
        report_type) are not available.

    """

    def safe_col(name: str) -> pl.Expr:
        """Return column or empty string if not available."""
        return pl.col(name).fill_null("") if name in available_cols else pl.lit("")

    date_col = "date_created"
    if not all(
        col in available_cols
        for col in [date_col, "subreddit", "title", "report_text", "report_type"]
    ):
        raise ValueError("Required columns for report generation are not available.")

    # Format date column
    date_expr = pl.col(date_col).fill_null("").cast(pl.String)

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
    """Parse date column using multiple format strategies.

    Attempts to parse dates using multiple formats including ISO 8601 strings,
    human-readable formats, and Unix timestamps. Creates a new UTC-aware
    datetime column '_dt'.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the date column to parse.
    date_col : str
        Name of the column containing date values to parse.

    Returns
    -------
    pl.DataFrame
        DataFrame with an additional '_dt' column containing parsed UTC datetime
        values. Returns the original DataFrame if the date column doesn't exist.

    Notes
    -----
    Supported date formats:
    - ISO 8601 with timezone (e.g., "2024-01-15T10:30:00Z")
    - ISO 8601 with microseconds
    - Date only (e.g., "2024-01-15")
    - Human-readable (e.g., "January 15, 2024")
    - Unix timestamps (seconds since epoch)

    The function uses `pl.coalesce` to try formats in sequence, using the
    first successful parse.
    """
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


def _get_normalization_expr(col_name: str) -> pl.Expr:
    """Create Polars expression for text normalization.

    Generates a Polars expression equivalent to `normalize_text_for_matching`
    from the helpers module, implementing the same 8-step normalization pipeline
    for consistent treatment term matching.

    Parameters
    ----------
    col_name : str
        Name of the column containing text to normalize.

    Returns
    -------
    pl.Expr
        Polars expression that performs text normalization with these steps:
        1. Pre-NFKC character translation
        2. NFKC Unicode normalization
        3. Post-NFKC character translation
        4. NFD normalization + accent removal
        5. Lowercase conversion
        6. Letter-number boundary spacing
        7. Connector collapsing
        8. Connector trimming

    See Also
    --------
    naturalv2.sources.components.helpers.normalize_text_for_matching :
        The original function this expression replicates.

    Notes
    -----
    This vectorized implementation is significantly faster than applying
    the Python function row-by-row using `map_batches`.
    """

    pre_nfkc_map = {
        chr(k): (v if v is not None else "")
        for k, v in PRE_NFKC_TRANSLATION_TABLE.items()
    }
    post_nfkc_map = {
        chr(k): (v if v is not None else "")
        for k, v in POST_NFKC_TRANSLATION_TABLE.items()
    }

    # \p{M} matches all Unicode combining marks (accents)
    strip_accents_pattern = r"\p{M}"

    # Connector Pattern: Matches _ / + , ; : space and hyphen
    # Hyphen must be last in the class to be a literal
    connector_pattern = CONNECTOR_SPLIT.pattern

    return (
        pl.col(col_name)
        .fill_null("")
        # Pre-NFKC Translation Table
        .str.replace_many(pre_nfkc_map, ascii_case_insensitive=False)
        # NFKC Normalization (Canonicalize unicode forms)
        .str.normalize("NFKC")
        .str.to_lowercase()
        # Apply Translation Table
        .str.replace_many(post_nfkc_map, ascii_case_insensitive=False)
        # NFD + Strip Accents (café -> cafe)
        .str.normalize("NFD")
        .str.replace_all(strip_accents_pattern, "")
        # Handle Letter-Number Boundaries (10mg -> 10 mg)
        # Use ${1} syntax for capture groups in Polars
        .str.replace_all(r"(\d)([a-z])", "${1} ${2}")
        .str.replace_all(r"([a-z])(\d)", "${1} ${2}")
        # Collapse Connectors
        # This turns "10mg-advil" -> "10 mg advil"
        .str.replace_all(connector_pattern, " ")
        # Remove leading/trailing connectors
        .str.strip_chars("".join(CONNECTOR_CHAR_SET))
    )
