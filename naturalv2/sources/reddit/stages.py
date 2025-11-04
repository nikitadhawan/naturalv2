"""Reddit curation stages compatible with the modern pipeline.

This module provides the following stages:

- ``RedditConditionFilter``: maps trial condition keywords to candidate
  subreddits using a mix of Reddit search and LLM filtering.
- ``RedditDownloadAndClean``: downloads, anonymizes (optional), cleans and
  consolidates subreddit data for the selected candidates.
- ``RedditCurateStage``: selects study-relevant posts per experiment and
  materializes curated CSVs.

"""

import ast
import asyncio
import contextlib
import json
import logging
import math
import os
from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING

import ahocorasick
import asyncpraw
import pandas as pd
import psutil
import pyarrow.parquet as pq
from aiolimiter import AsyncLimiter
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.concurrent import process_map

from naturalv2.models.lm import APIModel
from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.anonymizer import Anonymizer
from naturalv2.sources.components.helpers import build_treatment_automaton
from naturalv2.sources.components.llm_extraction import (
    ExtractType,
    extract_curation_info,
)
from naturalv2.sources.core import CurationContext, SourceStage, StageState
from naturalv2.sources.reddit.operations import (
    download_submissions_and_comments,
    get_study_relevant_posts,
    search_posts_in_subreddit,
    search_subreddits,
)
from naturalv2.sources.reddit.utils import get_sub_about_info
from naturalv2.utils import get_experiment_filepath


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment


logger = logging.getLogger(__name__)


class RedditConditionFilter(SourceStage):
    """Identify relevant subreddits for trial conditions via LLM assistance.

    This stage first collects candidate subreddits and post snippets using the
    Reddit API, then prompts an LLM to filter those candidates down to the most
    relevant subreddits per condition.

    Parameters
    ----------
    llm : APIModel
        Language model client used to filter candidate subreddits.
    llm_max_concurrency : int, default=10
        Maximum number of concurrent LLM requests.
    reddit_rpm : int, default=10
        Rate limit for Reddit API requests (requests per minute).
    subreddit_post_limit : int, default=5
        Maximum number of posts to fetch per subreddit during search.
    subreddit_post_char_limit : int, default=1000
        Maximum number of characters to include per post body snippet.
    name : str | None, optional
        Optional explicit stage name; defaults to the class name.
    """

    def __init__(
        self,
        *,
        llm: APIModel,
        llm_max_concurrency: int = 10,
        reddit_rpm: int = 10,
        subreddit_post_limit: int = 5,
        subreddit_post_char_limit: int = 1000,
        name: str | None = None,
    ) -> None:
        """Initialize the stage."""
        super().__init__(name=name)

        self.llm = llm
        self.reddit_rpm = reddit_rpm
        self.llm_max_concurrency = llm_max_concurrency
        self.subreddit_post_limit = subreddit_post_limit
        self.subreddit_post_char_limit = subreddit_post_char_limit

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Execute subreddit discovery and LLM filtering for conditions.

        Parameters
        ----------
        context : CurationContext
            Pipeline context with experiments, source name and save directories.
        state : StageState
            Mutable pipeline state; updated with condition-to-subreddit mapping
            and summary metadata.

        Returns
        -------
        StageState
            Updated state containing the mapping and counts.
        """
        trial_conditions: list[str] = []
        for experiment in context.experiments:
            if experiment.conditions:
                trial_conditions.extend(experiment.conditions)
        trial_conditions = sorted(dict.fromkeys(trial_conditions))

        # Get existing mapping if available
        condition_to_subreddit_map = context.study_dataset.sources.get(
            context.source_name, {}
        )
        # Flatten existing subreddits list for metadata
        relevant_subreddits_list = [
            sub
            for subs in condition_to_subreddit_map.values()
            for sub in subs
            if isinstance(subs, list)
        ]

        # Update state with existing mapping
        state.payload = condition_to_subreddit_map
        state.update(
            condition_to_subreddit_map=condition_to_subreddit_map,
            num_unique_subreddits=len(set(relevant_subreddits_list)),
        )

        # Add prompt template to metadata for logging
        prompt_id = f"{ExtractType.CONDITION.value}_{context.source_name}"
        template = load_prompt(
            base_dir="naturalv2/prompts/templates",
            prompt_type=prompt_id,
            return_format="prompt",
        )
        state.metadata.setdefault("prompt_templates", {})[prompt_id] = template

        # Skip keywords that have already been processed
        trial_conditions = [
            cond for cond in trial_conditions if cond not in condition_to_subreddit_map
        ]
        if not trial_conditions:
            logger.warning("%s: no trial conditions to process", self.stage_name)
            return state

        # Search Reddit for candidate subreddits and posts
        candidate_subs_and_posts = await self._collect_candidate_subs_and_posts(
            keywords=trial_conditions
        )

        # Collect results for DataFrame
        keyword_queries: list[dict[str, str | list[str]]] = []
        for keyword in trial_conditions:
            if keyword in candidate_subs_and_posts:
                result = candidate_subs_and_posts[keyword]
                if not (result["subreddit_posts"] or result["candidate_subs"]):
                    continue

                # Stringify subreddit posts for LLM input
                llm_input = json.dumps(result["subreddit_posts"], indent=4)
                keyword_queries.append(
                    {
                        "keyword": keyword,
                        "candidate_subs": result["candidate_subs"],
                        "input_data": llm_input,
                    }
                )

        logger.info(
            "%s: found candidate subreddits for %d out of %d keywords",
            self.stage_name,
            len(keyword_queries),
            len(trial_conditions),
        )

        if not keyword_queries:
            logger.warning(
                "%s: no candidate subreddits found for any keywords", self.stage_name
            )
            return state

        df = pd.DataFrame(keyword_queries)

        save_dir = self.results_dir(context)
        file_path = os.path.join(
            save_dir,
            f"{context.source_name}_condition_queries_{context.experiment_name}.csv",
        )

        output_df = await extract_curation_info(
            df,
            stage_name=self.stage_name,
            source_name=context.source_name,
            extract_type=ExtractType.CONDITION,
            llm=self.llm,
            file_path=file_path,
            token_tracker=context._token_tracker,
            max_concurrent_requests=self.llm_max_concurrency,
        )

        output_df["llm_output"] = output_df["llm_output"].fillna("[]")

        for keyword, output in zip(output_df["keyword"], output_df["llm_output"]):
            llm_filtered_subreddits: list[str] = ast.literal_eval(output)
            condition_to_subreddit_map[keyword] = llm_filtered_subreddits
            relevant_subreddits_list.extend(llm_filtered_subreddits)

        num_unique_subreddits = len(list(set(relevant_subreddits_list)))

        # Update state with new mapping
        state.payload = condition_to_subreddit_map
        state.update(
            condition_to_subreddit_map=condition_to_subreddit_map,
            num_unique_subreddits=num_unique_subreddits,
        )

        # Update and persist metadata in StudyDataset
        context.study_dataset.sources[context.source_name] = condition_to_subreddit_map
        context.study_dataset.to_yaml(context.extras["study_dataset_path"])

        logger.info(
            "%s: mapped %d trial conditions to %d unique subreddits",
            self.stage_name,
            len(condition_to_subreddit_map),
            num_unique_subreddits,
        )
        context._token_tracker.log_table()

        return state

    async def _collect_candidate_subs_and_posts(
        self, keywords: list[str]
    ) -> dict[str, dict[str, list[str] | dict[str, str | list[str]]]]:
        """Search Reddit for candidate subreddits and fetch post snippets.

        Parameters
        ----------
        keywords : list[str]
            Trial condition keywords to search for.

        Returns
        -------
        dict
            Mapping from keyword to a dict with keys ``candidate_subs`` and
            ``subreddit_posts`` (a list of subreddit → posts).
        """
        logger.info(
            "Getting candidate subreddits and posts for %d keywords.", len(keywords)
        )

        async with asyncpraw.Reddit(
            client_id=os.environ.get("PRAW_CLIENT_ID"),
            client_secret=os.environ.get("PRAW_CLIENT_SECRET"),
            password=os.environ.get("PRAW_PWD"),
            username=os.environ.get("PRAW_USERNAME"),
            user_agent=os.environ.get("PRAW_AGENT"),
        ) as reddit_client:
            reddit_rate_limiter = AsyncLimiter(self.reddit_rpm)

            async def _search(
                keyword: str,
            ) -> tuple[str, list[str] | Exception]:
                """Search for subreddits with a keyword."""
                try:
                    subs = await search_subreddits(
                        keyword, reddit_client, reddit_rate_limiter
                    )
                    return keyword, subs
                except Exception as exc:  # noqa: BLE001
                    return keyword, exc

            tasks = [asyncio.create_task(_search(keyword)) for keyword in keywords]

            candidate_subs_per_keyword: dict[str, list[str]] = {}
            for fut in tqdm_asyncio.as_completed(
                tasks,
                desc="Searching subreddits",
                total=len(tasks),
                leave=False,
                dynamic_ncols=True,
            ):
                keyword, result = await fut
                if isinstance(result, Exception):
                    logger.error(
                        "Searching for subreddits with keyword '%s' failed with error: %s",
                        keyword,
                        result,
                    )
                else:
                    candidate_subs_per_keyword[keyword] = result
                    logger.debug(
                        "Found %d candidate subreddits for keyword '%s'",
                        len(result),
                        keyword,
                    )

            async def _fetch_posts(
                keyword: str, subreddit: str
            ) -> tuple[str, str, list[str] | Exception]:
                """Fetch posts for a keyword/subreddit pair and surface errors."""
                try:
                    posts = await search_posts_in_subreddit(
                        subreddit,
                        keyword,
                        reddit_client,
                        reddit_rate_limiter,
                        limit=self.subreddit_post_limit,
                        char_limit=self.subreddit_post_char_limit,
                    )
                    return keyword, subreddit, posts
                except Exception as exc:  # noqa: BLE001
                    return keyword, subreddit, exc

            post_search_tasks: list[
                asyncio.Task[tuple[str, str, list[str] | Exception]]
            ] = []
            for keyword, candidate_subs in candidate_subs_per_keyword.items():
                for subreddit in candidate_subs:
                    post_search_tasks.append(
                        asyncio.create_task(_fetch_posts(keyword, subreddit))
                    )

            results_by_keyword: dict[
                str, dict[str, list[str] | dict[str, str | list[str]]]
            ] = {
                keyword: {
                    "candidate_subs": candidate_subs,
                    "subreddit_posts": [],
                }
                for keyword, candidate_subs in candidate_subs_per_keyword.items()
            }

            if not post_search_tasks:
                return results_by_keyword

            for fut in tqdm_asyncio.as_completed(
                post_search_tasks,
                desc="Searching posts",
                total=len(post_search_tasks),
                leave=False,
                dynamic_ncols=True,
            ):
                keyword, subreddit, posts = await fut
                if isinstance(posts, Exception):
                    logger.error(
                        "Could not fetch posts for subreddit '%s' and keyword '%s': %s",
                        subreddit,
                        keyword,
                        posts,
                    )
                else:
                    results_by_keyword[keyword]["subreddit_posts"].append(
                        {"Subreddit": subreddit, "Example Posts": posts}
                    )

        return results_by_keyword


class RedditDownloadAndClean(SourceStage):
    """Download or synthesise subreddit data for selected candidates.

    This stage downloads missing subreddit data, optionally anonymizes text,
    performs rule-based cleaning, and writes consolidated parquet files.

    Parameters
    ----------
    reddit_rpm : int, default=10
        Rate limit for Reddit API requests (requests per minute).
    max_download_workers : int | None, optional
        Degree of parallelism for downloads/cleaning; defaults to half the
        physical cores, minimum 1.
    anonymize : bool, default=True
        Whether to anonymize text during download/clean.
    anonymizer_score_threshold : float, default=0.85
        Threshold for anonymizer detection.
    anonymizer_batch_size : int, default=1
        Batch size for anonymizer processing.
    name : str | None, optional
        Optional explicit stage name; defaults to the class name.
    """

    def __init__(
        self,
        *,
        reddit_rpm: int = 10,
        max_download_workers: int | None = None,
        anonymize: bool = True,
        anonymizer_score_threshold: float = 0.85,
        anonymizer_batch_size: int = 1,
        name: str | None = None,
    ) -> None:
        """Initialize the stage."""
        super().__init__(name=name)
        self.reddit_rpm = reddit_rpm
        self.max_download_workers = max_download_workers or max(
            1, (psutil.cpu_count(logical=False) or 1) // 2
        )
        self.anonymize = anonymize
        self.anonymizer_score_threshold = anonymizer_score_threshold
        self.anonymizer_batch_size = anonymizer_batch_size

    async def run(self, context: CurationContext, state: StageState) -> StageState:
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

        relevant_subreddits = self._get_relevant_subreddits(
            context, condition_to_subreddit_map
        )

        if not relevant_subreddits:
            logger.error(
                "%s: no relevant subreddits found for any conditions", self.stage_name
            )
            state.update(cleaned_paths={})
            return state

        source_dir, subs_data_dir = self._get_subs_data_dir(context)

        # Filter out subreddits that are not available in Pushshift
        subs_about = await get_sub_about_info(source_dir, self.reddit_rpm)
        pushshift_subreddits = set(subs_about["subreddit"].to_list())
        available_subs = set(relevant_subreddits).intersection(pushshift_subreddits)
        logger.info(
            "%s: %d out of %d relevant subreddits are available in Pushshift",
            self.stage_name,
            len(available_subs),
            len(relevant_subreddits),
        )

        # Filter out subreddits that are already cleaned/downloaded
        subreddit_cleaned_path_map, subs_to_filter = self._partition_cleaned(
            available_subs, subs_data_dir
        )

        n_downloaded = len(available_subs) - len(subs_to_filter)
        logger.info(
            "%s: %d subreddits already downloaded and cleaned, "
            "%d subreddits to download and clean",
            self.stage_name,
            n_downloaded,
            len(subs_to_filter),
        )

        anonymizer: Anonymizer | None = None
        if self.anonymize:
            anonymizer = Anonymizer(score_threshold=self.anonymizer_score_threshold)

        if self.max_download_workers > 1:
            worker = partial(
                download_submissions_and_comments,
                data_path=subs_data_dir,
                anonymizer=anonymizer,
                batch_size=self.anonymizer_batch_size,
            )
            results = process_map(
                worker,
                subs_to_filter,
                max_workers=self.max_download_workers,
                desc=f"Downloading Reddit data [{self.max_download_workers} workers]",
                chunksize=1,
                position=0,
                leave=True,
                dynamic_ncols=True,
                disable=len(subs_to_filter) == 0,
            )
            for clean_sub_path, sub in results:
                if clean_sub_path is not None:
                    subreddit_cleaned_path_map[sub] = clean_sub_path
        else:
            for sub in subs_to_filter:
                clean_sub_path, _ = download_submissions_and_comments(
                    sub,
                    subs_data_dir,
                    anonymizer=anonymizer,
                    batch_size=self.anonymizer_batch_size,
                )
                if clean_sub_path is not None:
                    subreddit_cleaned_path_map[sub] = clean_sub_path

        # Update state
        state.payload = subreddit_cleaned_path_map
        state.update(cleaned_paths=subreddit_cleaned_path_map, source_dir=source_dir)

        # Update and persist metadata in StudyDataset
        self.persist_dataset(
            context,
            namespace_paths={
                f"{context.source_name}_cleaned": list(
                    subreddit_cleaned_path_map.values()
                )
            },
        )

        logger.info(
            "%s: downloaded and cleaned %d subreddits for %d experiments",
            self.stage_name,
            len(subreddit_cleaned_path_map),
            len(context.experiments),
        )
        return state

    def _get_relevant_subreddits(
        self,
        context: CurationContext,
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

    def _get_subs_data_dir(self, context: CurationContext) -> tuple[str, str]:
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

    def _partition_cleaned(
        self, available_subs: set[str], subs_data_dir: str
    ) -> tuple[dict[str, str], list[str]]:
        """Split available subreddits by presence of a cleaned parquet file.

        Parameters
        ----------
        available_subs : set[str]
            Subreddits available for processing.
        subs_data_dir : str
            Directory that should contain ``*_cleaned.parquet`` files.

        Returns
        -------
        tuple[dict[str, str], list[str]]
            Mapping of already-cleaned subreddits to file paths, and a list of
            subreddits that still need to be downloaded/cleaned.
        """
        subreddit_cleaned_path_map: dict[str, str] = {}
        subs_to_filter: list[str] = []
        for sub in available_subs:
            clean_sub_path = os.path.join(subs_data_dir, f"{sub}_cleaned.parquet")
            if os.path.exists(clean_sub_path):
                subreddit_cleaned_path_map[sub] = clean_sub_path
            else:
                subs_to_filter.append(sub)
        return subreddit_cleaned_path_map, subs_to_filter


class RedditCurateStage(SourceStage):
    """Curate subreddit data for each experiment.

    This stage builds treatment term patterns per experiment, filters cleaned
    subreddit data by date (optional), selects relevant posts, and writes a
    curated CSV per experiment.

    Parameters
    ----------
    name : str | None, optional
        Optional explicit stage name; defaults to the class name.
    """

    def __init__(self, *, name: str | None = None) -> None:
        """Initialize the stage."""
        super().__init__(name=name)
        self._max_chunk_bytes = 256 << 20

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Produce curated CSVs of Reddit posts per experiment."""
        subreddit_cleaned_path_map: dict[str, str] = state.require_metadata(
            "cleaned_paths", stage=self.stage_name
        )
        if not subreddit_cleaned_path_map:
            raise ValueError(
                "No cleaned subreddit data paths found in state metadata. "
                "This stage cannot proceed. "
                "Please ensure that the `RedditDownloadAndClean` stage has been "
                "run successfully before this stage."
            )

        study_dir = self.condition_dir(context)
        condition_to_subreddit_map = context.study_dataset.sources.get(
            context.source_name, {}
        )

        curated_paths: dict[str, str] = {}
        curated_data_sizes: dict[str, int] = {}
        num_bad_dates = 0

        for experiment in context.experiments:
            save_path, path_key = self._experiment_save_path(
                context, experiment, study_dir
            )
            if self._already_curated(save_path, experiment.nct_id):
                continue

            clean_data_paths = self._collect_clean_paths_for_experiment(
                experiment, condition_to_subreddit_map, subreddit_cleaned_path_map
            )
            if not clean_data_paths:
                logger.warning(
                    "No clean data paths found for experiment with NCT ID: %s",
                    experiment.nct_id,
                )
                curated_data_sizes[experiment.nct_id] = 0
                continue

            treatment_names = experiment.get_all_treatment_names_for_source(
                context.source_name
            )
            treatment_pattern = build_treatment_automaton(treatment_names)
            cutoff_dt, bad_date = self._parse_cutoff_date(context, experiment)
            num_bad_dates += int(bad_date)

            rows_written, header_written = self._curate_experiment_files(
                clean_data_paths=clean_data_paths,
                treatment_pattern=treatment_pattern,
                cutoff_dt=cutoff_dt,
                save_path=save_path,
                experiment_id=experiment.nct_id,
            )
            if rows_written == 0:
                self._handle_empty_result(save_path, experiment.nct_id, header_written)
                curated_data_sizes[experiment.nct_id] = 0
                continue

            curated_paths[experiment.nct_id] = save_path
            curated_data_sizes[experiment.nct_id] = rows_written
            self._persist_experiment_metadata(
                context=context,
                experiment=experiment,
                save_path=save_path,
                path_key=path_key,
            )

            logger.info(
                "%s: curated %d relevant Reddit posts for experiment %s",
                self.stage_name,
                rows_written,
                experiment.nct_id,
            )

        state.payload = curated_paths
        state.update(curated_paths=curated_paths, num_bad_dates=num_bad_dates)
        logger.info(
            "%s: curated Reddit datasets for %d experiments (bad dates: %d)",
            self.stage_name,
            len(curated_paths),
            num_bad_dates,
        )
        self.persist_dataset(
            context,
            per_experiment_paths=curated_paths,
            per_experiment_sizes=curated_data_sizes,
        )
        return state

    def _experiment_save_path(
        self, context: CurationContext, experiment, study_dir: str
    ) -> tuple[str, str]:
        """Return the output path and experiment source key for an experiment."""
        if context.filter_by_date:
            file_name = f"{context.source_name}_{experiment.nct_id}.csv"
            path_key = context.source_name
        else:
            file_name = f"{context.source_name}_{experiment.nct_id}_no_date_filter.csv"
            path_key = f"{context.source_name}_no_date_filter"
        return os.path.join(study_dir, file_name), path_key

    def _already_curated(self, save_path: str, experiment_id: str) -> bool:
        """Return True when the curated file already exists on disk."""
        if not os.path.exists(save_path):
            return False
        logger.info(
            "Skipping experiment %s as curated data already exists at %s",
            experiment_id,
            save_path,
        )
        return True

    def _collect_clean_paths_for_experiment(
        self,
        experiment,
        condition_to_subreddit_map: dict[str, list[str]],
        cleaned_path_map: dict[str, str],
    ) -> list[str]:
        """Return cleaned parquet paths relevant to the given experiment."""
        trial_relevant_subs = {
            sub
            for keyword in experiment.conditions or []
            for sub in condition_to_subreddit_map.get(keyword, [])
            if sub in cleaned_path_map
        }
        return [cleaned_path_map[sub] for sub in trial_relevant_subs]

    def _parse_cutoff_date(
        self, context: CurationContext, experiment
    ) -> tuple[pd.Timestamp | None, bool]:
        """Parse the experiment cutoff date and report parsing issues."""
        if not (context.filter_by_date and experiment.date):
            return None, False
        try:
            return pd.to_datetime(experiment.date), False
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Failed to parse date '%s' for experiment %s: %s. No date filter will "
                "be applied.",
                experiment.date,
                experiment.nct_id,
                exc,
            )
            return None, True

    def _curate_experiment_files(
        self,
        *,
        clean_data_paths: list[str],
        treatment_pattern: ahocorasick.Automaton,
        cutoff_dt: pd.Timestamp | None,
        save_path: str,
        experiment_id: str,
    ) -> tuple[int, bool]:
        """Process parquet files for a single experiment and return rows written."""
        header_written = False
        rows_written = 0
        seen_report_hashes: set[bytes] = set()
        with tqdm(
            total=len(clean_data_paths),
            desc=f"Curating Reddit data for {experiment_id}",
            unit="file",
            leave=False,
            dynamic_ncols=True,
            position=1,
            disable=len(clean_data_paths) == 0,
        ) as file_pbar:
            for path in clean_data_paths:
                parquet_file = self._safe_open_parquet(path)
                if parquet_file is None:
                    file_pbar.update(1)
                    continue
                header_written, added_rows = self._process_parquet_file(
                    parquet_file=parquet_file,
                    path=path,
                    save_path=save_path,
                    treatment_pattern=treatment_pattern,
                    cutoff_dt=cutoff_dt,
                    seen_report_hashes=seen_report_hashes,
                    header_written=header_written,
                )
                rows_written += added_rows
                file_pbar.update(1)
        return rows_written, header_written

    def _safe_open_parquet(self, path: str) -> pq.ParquetFile | None:
        """Open a parquet file while handling IO errors."""
        try:
            return pq.ParquetFile(path)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s: failed to open parquet %s: %s",
                self.stage_name,
                path,
                exc,
            )
            return None

    def _process_parquet_file(
        self,
        *,
        parquet_file: pq.ParquetFile,
        path: str,
        save_path: str,
        treatment_pattern: ahocorasick.Automaton,
        cutoff_dt: pd.Timestamp | None,
        seen_report_hashes: set[bytes],
        header_written: bool,
    ) -> tuple[bool, int]:
        """Process a single parquet file and append curated rows to disk."""
        added_rows = 0
        estimated_chunks = self._estimate_chunk_total(parquet_file)
        chunk_desc = f"Processing {os.path.basename(path)}"
        with tqdm(
            total=estimated_chunks or None,
            desc=chunk_desc,
            unit="chunk",
            leave=False,
            dynamic_ncols=True,
            position=2,
            disable=False,
        ) as chunk_pbar:
            for chunk_df in self._iter_parquet_chunks(parquet_file):
                relevant_posts_df = get_study_relevant_posts(
                    chunk_df, treatment_pattern, cutoff_dt
                )
                formatted_chunk = self._format_curated_chunk(
                    relevant_posts_df, seen_report_hashes
                )
                if not formatted_chunk.empty:
                    header_written = self._append_curated_chunk(
                        formatted_chunk,
                        save_path,
                        header_written=header_written,
                    )
                    added_rows += len(formatted_chunk)
                del formatted_chunk
                del relevant_posts_df
                del chunk_df
                chunk_pbar.update(1)
                if chunk_pbar.total is not None and chunk_pbar.n > chunk_pbar.total:
                    chunk_pbar.total = chunk_pbar.n
                    chunk_pbar.refresh()
        return header_written, added_rows

    def _handle_empty_result(
        self, save_path: str, experiment_id: str, header_written: bool
    ) -> None:
        """Cleanup partially written files when no curated rows were produced."""
        logger.warning("No valid matches found for experiment %s", experiment_id)
        if header_written and os.path.exists(save_path):
            with contextlib.suppress(OSError):
                os.remove(save_path)

    def _persist_experiment_metadata(
        self,
        *,
        context: CurationContext,
        experiment: "Experiment",
        save_path: str,
        path_key: str,
    ) -> None:
        """Persist experiment output path to the experiment YAML."""
        experiment.source_paths[path_key] = save_path
        experiment.to_yaml(
            filename=get_experiment_filepath(context.save_dir, experiment.nct_id)
        )

    def _estimate_chunk_total(self, parquet_file: pq.ParquetFile) -> int:
        """Estimate chunk count for progress reporting."""
        metadata = parquet_file.metadata
        total_bytes = 0
        if metadata is not None:
            try:
                total_bytes = sum(
                    max(int(metadata.row_group(i).total_byte_size), 0)
                    for i in range(metadata.num_row_groups)
                )
            except Exception:
                total_bytes = 0
        if total_bytes > 0:
            return max(1, math.ceil(total_bytes / self._max_chunk_bytes))
        if metadata is not None:
            return max(1, metadata.num_row_groups)
        return 0

    def _iter_parquet_chunks(
        self, parquet_file: pq.ParquetFile
    ) -> Iterator[pd.DataFrame]:
        """Yield DataFrame chunks from a parquet file bounded by ``max_chunk_bytes``."""
        metadata = parquet_file.metadata
        total_rows = 0
        total_bytes = 0
        if metadata is not None:
            try:
                for idx in range(metadata.num_row_groups):
                    row_group = metadata.row_group(idx)
                    total_rows += row_group.num_rows
                    total_bytes += max(int(row_group.total_byte_size), 0)
            except Exception:
                total_rows = 0
                total_bytes = 0

        if total_rows <= 0 and metadata is not None:
            try:
                total_rows = metadata.num_rows
            except Exception:
                total_rows = 0

        avg_row_bytes = (total_bytes / total_rows) if total_rows > 0 else None
        if not avg_row_bytes or avg_row_bytes <= 0:
            # Fallback to 1 KiB per row when metadata is missing
            avg_row_bytes = 1024

        rows_per_chunk = max(1, int(self._max_chunk_bytes / avg_row_bytes))

        for batch in parquet_file.iter_batches(
            batch_size=rows_per_chunk, use_threads=True
        ):
            yield batch.to_pandas()

    def _format_curated_chunk(
        self, curated_df: pd.DataFrame, seen_hashes: set[bytes]
    ) -> pd.DataFrame:
        """Format curated data chunk and drop duplicates using hashed reports."""
        if curated_df.empty:
            return pd.DataFrame()

        formatted_df = curated_df.copy()
        if "report_type" in formatted_df.columns:
            post_mask = formatted_df["report_type"] == "submission"
            if post_mask.any():
                formatted_df.loc[post_mask, "report"] = (
                    "**Subreddit**\nThis post was found on the subreddit r/"
                    + formatted_df.loc[post_mask, "subreddit"].astype(str)
                    + ".\n\n"
                    + "**Title**\nThis post was titled: "
                    + formatted_df.loc[post_mask, "title"].astype(str)
                    + "\n\n"
                    + "**Date created**\nThis post was created on "
                    + formatted_df.loc[post_mask, "date_created"].astype(str)
                    + ".\n\n"
                    + "**Post**\n"
                    + formatted_df.loc[post_mask, "report_text"].astype(str)
                )

            comment_mask = formatted_df["report_type"] == "comment"
            if comment_mask.any():
                formatted_df.loc[comment_mask, "report"] = (
                    "**Subreddit**\nThis comment was found on the subreddit r/"
                    + formatted_df.loc[comment_mask, "subreddit"].astype(str)
                    + ".\n\n"
                    + "**Initial Post**\nThis comment was in response to the following post: "
                    + "\nTitle: "
                    + formatted_df.loc[comment_mask, "title"].astype(str)
                    + "\nPost content: "
                    + formatted_df.loc[comment_mask, "initial_post"].astype(str)
                    + "\n\n"
                    + "**Date created**\nThis comment was created on "
                    + formatted_df.loc[comment_mask, "date_created"].astype(str)
                    + ".\n\n"
                    + "**Comment**\n"
                    + formatted_df.loc[comment_mask, "report_text"].astype(str)
                )

        if "report" not in formatted_df.columns:
            return pd.DataFrame()

        formatted_df["__report_hash"] = pd.util.hash_pandas_object(
            formatted_df["report"].fillna("").astype(str), index=False
        )
        dedup_mask = ~formatted_df["__report_hash"].isin(seen_hashes)
        if not dedup_mask.any():
            return pd.DataFrame()

        deduped_df = formatted_df.loc[dedup_mask].copy()
        seen_hashes.update(deduped_df["__report_hash"])
        return deduped_df.drop(columns="__report_hash")

    def _append_curated_chunk(
        self, formatted_chunk: pd.DataFrame, save_path: str, *, header_written: bool
    ) -> bool:
        """Append a formatted chunk to the experiment CSV."""
        if formatted_chunk.empty:
            return header_written
        if not header_written:
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        formatted_chunk.to_csv(
            save_path,
            mode="a",
            header=not header_written,
            index=False,
        )
        return True
