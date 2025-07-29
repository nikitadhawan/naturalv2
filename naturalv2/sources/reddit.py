"""Module for downloading and processing Reddit data."""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
from functools import partial

import asyncpraw
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from aiolimiter import AsyncLimiter
from omegaconf import DictConfig
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    wait_random_exponential,
)
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from naturalv2.evals.experiment import Experiment
from naturalv2.models.lm import LM, build_lm_instance_from_cfg, extract_list_response
from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.anonymizer import Anonymizer
from naturalv2.sources.reddit_utils import (
    download_sub_data,
    filter_by_date,
    get_context_post_df,
    get_sub_about_info,
    is_retryable_error,
    rule_based_filter,
)
from naturalv2.study import StudyDataset
from naturalv2.utils import ListResponse, concurrency_limited, sanitize_filename


logger = logging.getLogger(__name__)


class RedditSource:
    """Reddit data source.

    Parameters
    ----------
    data_path : str
        Root directory for storing Reddit data.
    lm_cfg : DictConfig
        Configuration for the language model.
    max_download_workers : int | None, default=None
        Maximum number of workers for downloading or curating Reddit data.
        If ``None``, defaults to half the number of physical CPU cores.
    reddit_api_qpm : int, default=10
        Maximum queries per minute for Reddit API.
    max_llm_concurrency : int, default=10
        Maximum concurrency for LLM calls.
    anonymize : bool, default=True
        Whether to anonymize the data.
    anonymizer_score_threshold : float, default=0.85
        Threshold for the anonymizer to determine if detected entities should be
        anonymized.
    anonymizer_batch_size : int, default=1
        Batch size for the anonymizer when processing data.

    """

    def __init__(
        self,
        data_path: str,
        lm_cfg: DictConfig,
        max_download_workers: int | None = None,
        reddit_api_qpm: int = 10,
        max_llm_concurrency: int = 10,
        anonymize: bool = True,
        anonymizer_score_threshold: float = 0.85,
        anonymizer_batch_size: int = 1,
    ) -> None:
        """Initialize an instance of the class."""
        self.data_path = data_path
        self.lm_cfg = lm_cfg
        self.max_download_workers = max_download_workers or max(
            1, (psutil.cpu_count(logical=False) or 1) // 2
        )
        self.reddit_api_qpm = reddit_api_qpm
        self.max_llm_concurrency = max_llm_concurrency
        self.anonymize = anonymize
        self.anonymizer_score_threshold = anonymizer_score_threshold
        self.anonymizer_batch_size = anonymizer_batch_size

        self._subs_data_dir = os.path.join(self.data_path, "subs_data")
        os.makedirs(self._subs_data_dir, exist_ok=True)

    async def condition_filter(
        self, keywords: list[str], study_dataset: StudyDataset, study_dataset_file: str
    ) -> dict[str, list[str]]:
        """Filter subreddits based on keywords in their description.

        This method searches for subreddits that match the given keywords and
        retrieves posts from those subreddits. It uses the Reddit API to search
        for subreddits and posts, and then applies a language model to determine
        which subreddits are relevant based on the posts found.

        Parameters
        ----------
        keywords : list[str]
            List of keywords to search for in subreddit descriptions.
        study_dataset : StudyDataset
            The study dataset object where the results will be stored.
        study_dataset_file : str
            Path to the YAML file where the study dataset is saved.

        Returns
        -------
        dict[str, list[str]]
            A dictionary mapping each keyword to a list of relevant subreddits.
        """
        subs_about = await get_sub_about_info(self.data_path, self.reddit_api_qpm)
        pushshift_subreddits = set(subs_about["subreddit"].to_list())

        source_metadata = study_dataset.sources["reddit"]

        lm = build_lm_instance_from_cfg(self.lm_cfg)

        llm_semaphore = asyncio.Semaphore(self.max_llm_concurrency)
        reddit_rate_limiter = AsyncLimiter(self.reddit_api_qpm)
        keywords_semaphore = asyncio.Semaphore(
            min(self.max_llm_concurrency * 2, len(keywords))
        )  # Process keywords concurrently, but limit to this many at a time
        lock = asyncio.Lock()

        async with asyncpraw.Reddit(
            client_id=os.environ.get("PRAW_CLIENT_ID"),
            client_secret=os.environ.get("PRAW_CLIENT_SECRET"),
            password=os.environ.get("PRAW_PWD"),
            username=os.environ.get("PRAW_USERNAME"),
            user_agent=os.environ.get("PRAW_AGENT"),
        ) as reddit_client:
            logger.info(f"Getting relevant subreddits for {len(keywords)} keywords.")

            async def process_keyword(word: str) -> None:
                if word not in source_metadata:
                    async with keywords_semaphore:
                        try:
                            subreddits = await self._search_subreddits(
                                word, reddit_client, reddit_rate_limiter
                            )
                            candidate_subs = set(subreddits).intersection(
                                pushshift_subreddits
                            )

                            # Concurrently search for posts in candidate subreddits
                            post_search_tasks = [
                                asyncio.create_task(
                                    self._search_posts_in_subreddit(
                                        subreddit,
                                        word,
                                        reddit_client,
                                        reddit_rate_limiter,
                                    )
                                )
                                for subreddit in candidate_subs
                            ]
                            posts_results = await asyncio.gather(
                                *post_search_tasks, return_exceptions=True
                            )

                            llm_input = []
                            for subreddit, posts in zip(candidate_subs, posts_results):
                                if not isinstance(posts, Exception):
                                    llm_input.append(
                                        {"subreddit": subreddit, "example_posts": posts}
                                    )

                            matched_subreddits = []

                            # Don't make expensive LLM calls if no posts found
                            if llm_input:
                                llm_input_dict = {
                                    "condition": word,
                                    "input": json.dumps(llm_input, indent=4),
                                }
                                matched_subreddits = await concurrency_limited(
                                    self._get_subreddits_from_llm(lm, llm_input_dict),
                                    llm_semaphore,
                                )

                            logger.info(
                                f"{len(matched_subreddits)} relevant subreddits "
                                f"found for keyword: {word}."
                            )
                            source_metadata[word] = matched_subreddits

                            # Write to YAML immediately (protected by a lock)
                            async with lock:
                                study_dataset.sources["reddit"] = source_metadata
                                study_dataset.to_yaml(study_dataset_file)
                        except Exception as e:
                            logger.error(
                                f"Error processing keyword {word}: {e}", exc_info=True
                            )

            # Launch all tasks concurrently, but limited by the semaphore
            await asyncio.gather(*(process_keyword(word) for word in keywords))

            self.relevant_subreddits = {
                sub for keyword in source_metadata for sub in source_metadata[keyword]
            }
            logger.info(f"{len(self.relevant_subreddits)} relevant subreddits found!")
        return source_metadata

    def clean_data(self) -> list[str]:
        """Download and clean data for relevant subreddits.

        This method checks if the cleaned data for each relevant subreddit already
        exists. If it does, it returns the path to the cleaned data. If not, it
        downloads the submissions and comments for the subreddit, cleans the data,
        and saves it to a parquet file. It uses multiprocessing to speed up the
        download and cleaning process. If the `anonymize` flag is set, it also
        anonymizes the data using the specified anonymizer settings.

        Returns
        -------
        list[str]
            List of paths to the cleaned data files for each relevant subreddit.
        """
        if not self.relevant_subreddits:
            logger.info("No relevant subreddits found to clean.")
            return []

        clean_paths = []
        subs_to_filter = []
        for sub in self.relevant_subreddits:
            clean_sub_path = os.path.join(self._subs_data_dir, f"{sub}_cleaned.parquet")
            if os.path.exists(clean_sub_path):
                clean_paths.append(clean_sub_path)
            else:
                subs_to_filter.append(sub)

        if not subs_to_filter:
            # all data have been downloaded
            logger.info(
                "All relevant subreddit data has already been downloaded and cleaned."
            )
            return clean_paths

        n_downloaded = len(self.relevant_subreddits) - len(subs_to_filter)
        logger.info(
            f"{n_downloaded} relevant subreddits already downloaded, "
            f"{len(subs_to_filter)} remaining to download and clean."
        )

        anonymizer: Anonymizer | None = None
        if self.anonymize:
            anonymizer = Anonymizer(score_threshold=self.anonymizer_score_threshold)

        if self.max_download_workers > 1:
            results = process_map(
                partial(
                    _download_submissions_and_comments,
                    data_path=self._subs_data_dir,
                    anonymizer=anonymizer,
                    batch_size=self.anonymizer_batch_size,
                ),
                subs_to_filter,
                max_workers=self.max_download_workers,
                desc=f"Downloading Reddit data [{self.max_download_workers} workers]",
                chunksize=1,
                position=0,
                leave=True,
                dynamic_ncols=True,
                disable=len(subs_to_filter) == 0,  # disable if no subs to download
            )
            for clean_sub_path in results:
                if clean_sub_path is not None:
                    clean_paths.append(clean_sub_path)
        else:  # single-threaded download; avoids multiprocessing overhead
            for sub in subs_to_filter:
                clean_sub_path = _download_submissions_and_comments(
                    sub,
                    self._subs_data_dir,
                    anonymizer,
                    self.anonymizer_batch_size,
                )
                if clean_sub_path is not None:
                    clean_paths.append(clean_sub_path)

        return clean_paths

    def curate_experiment_data(
        self,
        experiment: Experiment,
        study_name: str,
        apply_date_filter: bool,
        clean_data_paths: list[str],
    ) -> tuple[str, int]:
        """Curate Reddit data for a specific experiment.

        This method filters the cleaned Reddit data for posts that mention both
        treatments and outcomes specified in the experiment. It saves the curated
        data to a CSV file and returns the path along with the number of valid posts.

        Parameters
        ----------
        experiment : Experiment
            The experiment object containing treatment and outcome information.
        study_name : str
            Name of the study for which to curate Reddit data.
        apply_date_filter : bool
            Whether to apply a date filter based on the experiment's date.
        clean_data_paths : list[str]
            List of paths to the cleaned Reddit data.

        Returns
        -------
        tuple[str, int]
            Path to the curated experiment data file and the number of valid posts
            that mention both treatments and outcomes.

        """
        assert len(clean_data_paths) > 0
        study_dir = os.path.join(self.data_path, sanitize_filename(study_name.lower()))
        os.makedirs(study_dir, exist_ok=True)
        save_path = os.path.join(study_dir, f"reddit_{experiment.nct_id}.csv")
        if os.path.exists(save_path):
            exp_df = pd.read_csv(save_path, index_col=0)
            return save_path, len(exp_df)

        drugbank_names = [
            item for sublist in experiment.drugbank_names.values() for item in sublist
        ]
        treatment_names = {
            name.lower()
            for name in list(experiment.treatment_common_names["reddit"].keys())
            + [
                item
                for sublist in experiment.treatment_common_names["reddit"].values()
                for item in sublist
            ]
            + drugbank_names
        }

        outcome_names = {
            name.lower()
            for name in list(experiment.outcome_common_names["reddit"].keys())
            + [
                item
                for sublist in experiment.outcome_common_names["reddit"].values()
                for item in sublist
            ]
        }

        # Compile regex pattern once and reuse it for all subreddits
        treatment_pattern = _compile_search_pattern(treatment_names)
        outcome_pattern = _compile_search_pattern(outcome_names)

        cutoff_dt = None
        if apply_date_filter and experiment.date:
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
                max_workers=self.max_download_workers
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
                    _get_study_relevant_posts,
                    path,
                    treatment_pattern,
                    outcome_pattern,
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
            columns = pq.ParquetFile(clean_data_paths[0]).schema.names
            empty_df = pd.DataFrame(
                columns=columns + ["treatments_mentioned", "outcome_words"]
            )
            empty_df.to_csv(save_path, index=False)
            logger.warning(f"No valid matches found for experiment {experiment.nct_id}")
            return save_path, 0

        # Concatenate all DataFrames and save to CSV
        final_df = pd.concat(curated_experiment_data, ignore_index=True)
        final_df.to_csv(save_path, index=False)

        return save_path, len(final_df)

    @retry(
        retry=retry_if_exception(is_retryable_error),
        wait=wait_random_exponential(multiplier=1, max=60),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def _search_subreddits(
        self, keyword: str, reddit_client: asyncpraw.Reddit, limiter: AsyncLimiter
    ) -> list[str]:
        """Search for subreddits matching a keyword."""
        async with limiter:
            return [
                subreddit.display_name
                async for subreddit in reddit_client.subreddits.search(keyword)
            ]

    @retry(
        retry=retry_if_exception(is_retryable_error),
        wait=wait_random_exponential(multiplier=1, max=60),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def _search_posts_in_subreddit(
        self,
        subreddit: str,
        keyword: str,
        reddit_client: asyncpraw.Reddit,
        limiter: AsyncLimiter,
    ) -> list[str]:
        """Search for posts in a subreddit matching a keyword."""
        async with limiter:
            posts = []
            subreddit_instance = await reddit_client.subreddit(subreddit)

            async for submission in subreddit_instance.search(keyword, limit=5):
                posts.append(
                    "**Title**: "
                    + submission.title
                    + "\n\n"
                    + "**Post content**: "
                    + (submission.selftext or "")[:1000]
                )

            return posts

    async def _get_subreddits_from_llm(
        self, lm: LM, llm_input: dict[str, str]
    ) -> list[str]:
        """Get subreddits from the language model based on the condition."""
        messages: list[dict[str, str]] = load_prompt(
            base_dir="naturalv2/prompts/templates",
            prompt_type="condition_subreddits",
            return_format="messages",
            **llm_input,
        )

        response = await lm(messages=messages, response_format=ListResponse)
        subreddits = extract_list_response(response)
        if not subreddits:
            logger.debug(
                "No subreddits found for the given condition. Returning an empty list."
            )
            return []
        return subreddits[0]


def _download_submissions_and_comments(
    sub: str, data_path: str, anonymizer: Anonymizer | None, batch_size: int
) -> str | None:
    """Download submissions and comments for a given subreddit, then clean."""
    clean_sub_path = os.path.join(data_path, f"{sub}_cleaned.parquet")
    if os.path.exists(clean_sub_path):
        return clean_sub_path

    try:
        submissions_path = os.path.join(data_path, f"{sub}_submissions.parquet")
        comments_path = os.path.join(data_path, f"{sub}_comments.parquet")
        if not os.path.exists(submissions_path):
            download_sub_data(
                sub,
                "submissions",
                data_path,
                anonymizer_instance=anonymizer,
                batch_size=batch_size,
            )
        if not os.path.exists(comments_path):
            download_sub_data(
                sub,
                "comments",
                data_path,
                anonymizer_instance=anonymizer,
                batch_size=batch_size,
            )

        rule_filtered_df = _clean_sub_data(data_path, sub)
        rule_filtered_df = rule_filtered_df.drop_duplicates("report")
        rule_filtered_df.to_parquet(clean_sub_path, index=False, compression="snappy")
        # Delete the submissions and comments files after cleaning
        os.remove(submissions_path)
        os.remove(comments_path)
    except Exception as e:
        logger.error(f"Error processing subreddit {sub}: {e}")
        return None

    return clean_sub_path


def _clean_sub_data(data_path: str, sub: str) -> pd.DataFrame:
    """Clean submissions and comments for a given subreddit."""
    submissions = pd.read_parquet(os.path.join(data_path, f"{sub}_submissions.parquet"))
    submissions = rule_based_filter(submissions, "selftext")

    comments = pd.read_parquet(os.path.join(data_path, f"{sub}_comments.parquet"))
    comments = rule_based_filter(comments, "body")

    return get_context_post_df(submissions, comments)


def _get_study_relevant_posts(
    clean_data_path: str,
    treatment_pattern: re.Pattern,
    outcome_pattern: re.Pattern,
    cutoff_dt: pd.Timestamp | None,
) -> pd.DataFrame:
    """Get posts from cleaned Reddit data that mention both treatments and outcomes."""
    df = pd.read_parquet(clean_data_path)
    if not df.empty and cutoff_dt is not None:
        df = filter_by_date(df, cutoff_dt, "date_created")

    if df.empty:
        return pd.DataFrame()

    text_cols = ["subreddit", "title", "report", "initial_post"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    # Find mentions of treatments and outcomes in each text column
    treatment_finds = [
        df[col].str.lower().str.findall(treatment_pattern) for col in text_cols
    ]
    outcome_finds = [
        df[col].str.lower().str.findall(outcome_pattern) for col in text_cols
    ]

    # Check if ANY column had a match for each row.
    has_treatment_mask = np.any([s.str.len() > 0 for s in treatment_finds], axis=0)
    has_outcome_mask = np.any([s.str.len() > 0 for s in outcome_finds], axis=0)

    valid_mask = has_treatment_mask & has_outcome_mask
    result: pd.DataFrame = df[valid_mask].copy()
    if result.empty:
        return result

    # Aggregate a unique list of words from the pre-computed finds.
    valid_treatment_finds = [s[valid_mask] for s in treatment_finds]
    valid_outcome_finds = [s[valid_mask] for s in outcome_finds]

    # Create a new DataFrame with unique mentions
    result["treatments_mentioned"] = [
        list({item for sublist in row for item in sublist})
        for row in zip(*valid_treatment_finds)
    ]
    result["outcome_words"] = [
        list({item for sublist in row for item in sublist})
        for row in zip(*valid_outcome_finds)
    ]

    return result.reset_index(drop=True)


def _compile_search_pattern(terms: set[str]) -> re.Pattern:
    """Compile a re pattern for searching terms in text."""
    if not terms:
        return re.compile(r"(?!)")  # Always false pattern

    # Sort terms by length (longest first) to allow the regex engine to skip smaller
    # terms that are substrings of larger terms
    terms_sorted = sorted(terms, key=len, reverse=True)

    # Escape each name to treat special characters (like '+', '.', '*') literally.
    escaped_terms = [re.escape(term) for term in terms_sorted]

    # Join the escaped names with the '|' (OR) operator and Wrap with `\b` to ensure
    # whole-word matching.
    return re.compile(
        r"\b(?:{})\b".format("|".join(escaped_terms)), flags=re.IGNORECASE
    )
