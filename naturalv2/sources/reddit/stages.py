"""Reddit curation stages compatible with the modern pipeline."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import json
import logging
import os
from functools import partial

import asyncpraw
import pandas as pd
import psutil
from aiolimiter import AsyncLimiter
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.concurrent import process_map

from naturalv2.models.lm import APIModel
from naturalv2.sources.anonymizer import Anonymizer
from naturalv2.sources.components.llm_extraction import extract_curation_info
from naturalv2.sources.components.text import build_term_pattern
from naturalv2.sources.curation import CurationContext, SourceStage, StageState
from naturalv2.sources.reddit.operations import (
    download_submissions_and_comments,
    get_study_relevant_posts,
    search_posts_in_subreddit,
    search_subreddits,
)
from naturalv2.sources.reddit.utils import get_sub_about_info


logger = logging.getLogger(__name__)


class RedditConditionFilter(SourceStage):
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
        super().__init__(name=name)

        self.llm = llm
        self.reddit_rpm = reddit_rpm
        self.llm_max_concurrency = llm_max_concurrency
        self.subreddit_post_limit = subreddit_post_limit
        self.subreddit_post_char_limit = subreddit_post_char_limit

    async def run(self, context: CurationContext, state: StageState) -> StageState:
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
        state.metadata["condition_to_subreddit_map"] = condition_to_subreddit_map
        state.metadata["num_unique_subreddits"] = len(set(relevant_subreddits_list))

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
            input_df=df,
            stage_name=self.stage_name,
            source_name=context.source_name,
            extract_type="condition",
            llm=self.llm,
            file_path=file_path,
            token_tracker=context._token_tracker,
            max_concurrent_requests=self.llm_max_concurrency,
        )

        for keyword, output in zip(output_df["keyword"], output_df["llm_output"]):
            llm_filtered_subreddits: list[str] = ast.literal_eval(output)
            condition_to_subreddit_map[keyword] = llm_filtered_subreddits
            relevant_subreddits_list.extend(llm_filtered_subreddits)

        num_unique_subreddits = len(list(set(relevant_subreddits_list)))

        # Update state with new mapping
        state.payload = condition_to_subreddit_map
        state.metadata["condition_metadata"] = condition_to_subreddit_map
        state.metadata["num_unique_subreddits"] = num_unique_subreddits

        logger.info(
            "%s: mapped %d trial conditions to %d unique subreddits",
            self.stage_name,
            len(condition_to_subreddit_map),
            num_unique_subreddits,
        )
        return state

    async def _collect_candidate_subs_and_posts(
        self, keywords: list[str]
    ) -> dict[str, dict[str, list[str] | dict[str, str | list[str]]]]:
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

            subreddit_tasks: list[asyncio.Task[list[str]]] = []
            task_to_keyword: dict[asyncio.Task[list[str]], str] = {}
            for keyword in keywords:
                task = asyncio.create_task(
                    search_subreddits(keyword, reddit_client, reddit_rate_limiter)
                )
                task_to_keyword[task] = keyword
                subreddit_tasks.append(task)

            candidate_subs_per_keyword: dict[str, list[str]] = {}
            for task in tqdm_asyncio.as_completed(
                subreddit_tasks,
                desc="Searching subreddits",
                total=len(subreddit_tasks),
                leave=False,
                dynamic_ncols=True,
            ):
                keyword = task_to_keyword[task]
                try:
                    result = await task
                    candidate_subs_per_keyword[keyword] = result
                    logger.info(
                        "Found %d candidate subreddits for keyword '%s'",
                        len(result),
                        keyword,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Searching for subreddits with keyword '%s' failed with error: %s",
                        keyword,
                        exc,
                    )

            post_search_tasks: list[asyncio.Task[list[str]]] = []
            task_metadata: list[tuple[str, str]] = []
            for keyword, candidate_subs in candidate_subs_per_keyword.items():
                for subreddit in candidate_subs:
                    task = asyncio.create_task(
                        search_posts_in_subreddit(
                            subreddit,
                            keyword,
                            reddit_client,
                            reddit_rate_limiter,
                            limit=self.subreddit_post_limit,
                            char_limit=self.subreddit_post_char_limit,
                        )
                    )
                    post_search_tasks.append(task)
                    task_metadata.append((keyword, subreddit))

            post_search_results: list[list[str] | Exception]
            post_search_results = await tqdm_asyncio.gather(
                *post_search_tasks,
                desc="Searching posts",
                total=len(post_search_tasks),
                leave=False,
                dynamic_ncols=True,
                return_exceptions=True,
            )

            results_by_keyword: dict[
                str, dict[str, list[str] | dict[str, str | list[str]]]
            ] = {}

            for (keyword, subreddit), posts in zip(task_metadata, post_search_results):
                if isinstance(posts, Exception):
                    logger.error(
                        "Could not fetch posts for subreddit '%s' and keyword '%s': %s",
                        subreddit,
                        keyword,
                        posts,
                    )
                else:
                    results_by_keyword.setdefault(
                        keyword,
                        {
                            "candidate_subs": candidate_subs_per_keyword.get(
                                keyword, []
                            ),
                            "subreddit_posts": [],
                        },
                    )["subreddit_posts"].append(
                        {"subreddit": subreddit, "posts": posts}
                    )

        return results_by_keyword


class RedditDownloadAndClean(SourceStage):
    """Download or synthesise subreddit data for selected candidates."""

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
        super().__init__(name=name)
        self.reddit_rpm = reddit_rpm
        self.max_download_workers = max_download_workers or max(
            1, (psutil.cpu_count(logical=False) or 1) // 2
        )
        self.anonymize = anonymize
        self.anonymizer_score_threshold = anonymizer_score_threshold
        self.anonymizer_batch_size = anonymizer_batch_size

    async def run(self, context: CurationContext, state: StageState) -> StageState:
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

        relevant_subreddits: set[str] = set()
        for experiment in context.experiments:
            for keyword in experiment.conditions or []:
                relevant_subreddits.update(condition_to_subreddit_map.get(keyword, []))

        if not relevant_subreddits:
            logger.error(
                "%s: no relevant subreddits found for any conditions", self.stage_name
            )
            return state

        source_dir = self.source_dir(context)
        subs_data_dir = os.path.join(source_dir, "subreddits")
        os.makedirs(subs_data_dir, exist_ok=True)

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

        # Filter out subreddits that have already been downloaded and cleaned
        subreddit_cleaned_path_map: dict[str, str] = {}
        subs_to_filter = []
        for sub in available_subs:
            clean_sub_path = os.path.join(subs_data_dir, f"{sub}_cleaned.parquet")
            if os.path.exists(clean_sub_path):
                subreddit_cleaned_path_map[sub] = clean_sub_path
            else:
                subs_to_filter.append(sub)

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

        state.payload = subreddit_cleaned_path_map
        state.update(cleaned_paths=subreddit_cleaned_path_map, source_dir=source_dir)
        logger.info(
            "%s: downloaded and cleaned %d subreddits for %d experiments",
            self.stage_name,
            len(subreddit_cleaned_path_map),
            len(context.experiments),
        )
        return state


class RedditCurateStage(SourceStage):
    """Curate subreddit data for each experiment."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.max_workers = max_workers or max(1, (psutil.cpu_count(logical=False) or 1))

    async def run(self, context: CurationContext, state: StageState) -> StageState:
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
        total_rows = 0
        for experiment in context.experiments:
            save_path = os.path.join(study_dir, f"reddit_{experiment.nct_id}.csv")
            if os.path.exists(save_path):
                # TODO: log
                continue

            # Get subreddits relevant to this experiment
            trial_relevant_subs = {
                sub
                for keyword in experiment.conditions or []
                for sub in condition_to_subreddit_map.get(keyword, [])
                if sub in subreddit_cleaned_path_map
            }

            # Get cleaned data paths for relevant subreddits, or all if none found
            clean_data_paths = [
                subreddit_cleaned_path_map[sub] for sub in trial_relevant_subs
            ] or list(subreddit_cleaned_path_map.values())

            treatment_names = experiment.get_all_treatment_names_for_source(
                context.source_name
            )
            treatment_pattern = build_term_pattern(treatment_names)

            cutoff_dt = None
            if context.filter_by_date and experiment.date:
                try:
                    cutoff_dt = pd.to_datetime(experiment.date)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse date '{experiment.date}' for experiment "
                        f"{experiment.nct_id}: {e}. No date filter will be applied."
                    )

            curated_experiment_data = []
            with (
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.max_workers
                ) as executor,
                tqdm(
                    total=len(clean_data_paths),
                    desc="Curating Reddit data for experiment",
                    unit="file",
                    leave=False,
                    dynamic_ncols=True,
                    position=1,
                    disable=len(clean_data_paths) == 0,  # disable if no futures
                ) as pbar,
            ):
                futures = []
                for path in clean_data_paths:
                    future = executor.submit(
                        get_study_relevant_posts,
                        path,
                        treatment_pattern,
                        cutoff_dt,
                    )

                    # Add callback to update progress bar when future completes
                    future.add_done_callback(lambda f: pbar.update(1))
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    relevant_posts_df: pd.DataFrame = future.result()
                    if not relevant_posts_df.empty:
                        curated_experiment_data.append(relevant_posts_df)

            if not curated_experiment_data:
                # Save an empty DataFrame so that we don't try to process later
                # columns = pq.ParquetFile(clean_data_paths[0]).schema.names
                # empty_df = pd.DataFrame(
                #     columns=columns + ["treatments_mentioned", "outcome_words"]
                # )
                # empty_df.to_csv(save_path, index=False)
                logger.warning(
                    f"No valid matches found for experiment {experiment.nct_id}"
                )
                continue

            # Concatenate all DataFrames, format reports, and save to CSV
            final_df = pd.concat(curated_experiment_data, ignore_index=True)
            post_mask = final_df["report_type"] == "submission"
            final_df.loc[post_mask, "report"] = (
                "**Subreddit**\nThis post was found on the subreddit r/"
                + final_df.loc[post_mask, "subreddit"].astype(str)
                + ".\n\n"
                + "**Title**\nThis post was titled: "
                + final_df.loc[post_mask, "title"].astype(str)
                + "\n\n"
                + "**Date created**\nThis post was created on "
                + final_df.loc[post_mask, "date_created"].astype(str)
                + ".\n\n"
                + "**Post**\n"
                + final_df.loc[post_mask, "report_text"].astype(str)
            )
            comment_mask = final_df["report_type"] == "comment"
            final_df.loc[comment_mask, "report"] = (
                "**Subreddit**\nThis comment was found on the subreddit r/"
                + final_df.loc[comment_mask, "subreddit"].astype(str)
                + ".\n\n"
                + "**Initial Post**\nThis comment was in response to the following post: "
                + "\nTitle: "
                + final_df.loc[comment_mask, "title"].astype(str)
                + "\nPost content: "
                + final_df.loc[comment_mask, "initial_post"].astype(str)
                + "\n\n"
                + "**Date created**\nThis comment was created on "
                + final_df.loc[comment_mask, "date_created"].astype(str)
                + ".\n\n"
                + "**Comment**\n"
                + final_df.loc[comment_mask, "report_text"].astype(str)
            )
            final_df = final_df.drop_duplicates("report")
            final_df.to_csv(save_path, index=False)

            curated_paths[experiment.nct_id] = save_path
            curated_data_sizes[experiment.nct_id] = len(final_df)
            total_rows += len(final_df)
            logger.info(
                "%s: curated %d relevant Reddit posts for experiment %s",
                self.stage_name,
                len(final_df),
                experiment.nct_id,
            )

        state.payload = curated_paths
        state.update(curated_paths=curated_paths)
        logger.info(
            "%s: curated Reddit datasets for %d experiments",
            self.stage_name,
            len(curated_paths),
        )
        self.persist_dataset(
            context,
            per_experiment_paths=curated_paths,
            per_experiment_sizes=curated_data_sizes,
        )
        return state
