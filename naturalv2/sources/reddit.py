"""Module for downloading and processing Reddit data."""

import asyncio
import datetime
import json
import logging
import os
from functools import partial
from typing import Optional

import asyncpraw
import pandas as pd
import psutil
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map

from naturalv2.evals.experiment import Experiment
from naturalv2.models.lm import LM, build_lm_instance_from_cfg, extract_list_response
from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.anonymizer import Anonymizer
from naturalv2.sources.reddit_utils import (
    _get_context_post_df,
    download_sub_data,
    get_sub_about_info,
    rule_based_filter,
)
from naturalv2.utils import ListResponse


logger = logging.getLogger(__name__)


class RedditSource:
    """RedditSource class for downloading and processing Reddit data.

    Parameters
    ----------
    data_path : str
        Path to store downloaded Reddit data.
    lm_cfg : DictConfig
        Configuration for the language model.
    max_download_workers : Optional[int], default=None
        Maximum number of workers for downloading data. If ``None``, defaults to half
        the number of physical CPU cores.
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
        max_download_workers: Optional[int] = None,
        anonymize: bool = True,
        anonymizer_score_threshold: float = 0.85,
        anonymizer_batch_size: int = 1,
    ) -> None:
        """Initialize the RedditSource instance."""
        self.data_path = data_path
        self.lm_cfg = lm_cfg
        self.anonymize = anonymize
        self.anonymizer_score_threshold = anonymizer_score_threshold
        self.anonymizer_batch_size = anonymizer_batch_size

        if max_download_workers is None:
            max_download_workers = max(1, (psutil.cpu_count(logical=False) or 1) // 2)

        self.max_download_workers = max_download_workers

        self._subs_data_dir = os.path.join(self.data_path, "subs_data")
        os.makedirs(self._subs_data_dir, exist_ok=True)

        self._anonymizer = None
        if anonymize:
            self._anonymizer = Anonymizer(score_threshold=anonymizer_score_threshold)

    async def _get_subreddits_from_llm(
        self, lm: LM, llm_input: dict[str, str]
    ) -> list[str]:
        messages: list[dict[str, str]] = load_prompt(
            base_dir="naturalv2/prompts/templates",
            prompt_type="condition_subreddits",
            return_format="messages",
            **llm_input,
        )

        response = await lm(messages=messages, response_format=ListResponse)
        subreddits = extract_list_response(response)
        if subreddits is None:
            logger.warning(
                "No subreddits found for the given condition. Returning an empty list."
            )
            return []
        return subreddits[0]

    async def condition_filter(
        self,
        keywords: list[str],
        study_dataset,
        study_dataset_file: str,
        semaphore_limit: int = 50,
    ) -> list[str]:
        """Filter subreddits based on keywords in their description."""
        self.subs_about = get_sub_about_info(self.data_path)

        source_metadata = study_dataset.sources["reddit"]
        pushshift_subreddits = self.subs_about["sub"].to_list()

        lm = build_lm_instance_from_cfg(self.lm_cfg)
        relevant_subs: set[str] = set()
        async with asyncpraw.Reddit(
            client_id=os.environ.get("PRAW_CLIENT_ID"),
            client_secret=os.environ.get("PRAW_CLIENT_SECRET"),
            password=os.environ.get("PRAW_PWD"),
            username=os.environ.get("PRAW_USERNAME"),
            user_agent=os.environ.get("PRAW_AGENT"),
        ) as reddit_client:

            async def _get_relevant_subs_and_posts(word: str) -> list[str]:
                nonlocal relevant_subs  # Use the outer scope variable
                # Search for subreddits matching the word
                subreddits = [
                    subreddit.display_name
                    async for subreddit in reddit_client.subreddits.search(word)
                ]
                relevant_subs = set(subreddits).intersection(pushshift_subreddits)
                llm_input = []
                for subreddit in relevant_subs:
                    posts = []
                    subreddit_instance = await reddit_client.subreddit(subreddit)
                    async for submission in subreddit_instance.search(word, limit=5):
                        posts.append(
                            "**Title**: "
                            + submission.title
                            + "\n\n"
                            + "**Post content**: "
                            + (submission.selftext or "")[:1000]
                        )
                    llm_input.append({"subreddit": subreddit, "example_posts": posts})
                llm_input_dict = {
                    "condition": word,
                    "input": json.dumps(llm_input, indent=4),
                }
                return await self._get_subreddits_from_llm(lm, llm_input_dict)

            logger.info(f"Getting relevant subreddits for {len(keywords)} keywords.")
            semaphore = asyncio.Semaphore(semaphore_limit)
            lock = asyncio.Lock()

            async def process_keyword(word: str) -> None:
                nonlocal relevant_subs
                if word not in source_metadata:
                    async with semaphore:
                        try:
                            relevant_subs = await _get_relevant_subs_and_posts(word)
                            logger.info(
                                f"{len(relevant_subs)} relevant subreddits found for keyword: {word}."
                            )
                            relevant_subs.update(relevant_subs)
                            source_metadata[word] = relevant_subs

                            # Write to YAML immediately (protected by a lock)
                            async with lock:
                                study_dataset.sources["reddit"] = source_metadata
                                study_dataset.to_yaml(study_dataset_file)
                        except Exception as e:
                            logger.error(f"Error processing keyword {word}: {e}")
                else:
                    relevant_subs.update(source_metadata[word])

            # Launch all tasks concurrently, but limited by the semaphore
            await asyncio.gather(*(process_keyword(word) for word in keywords))

            self.relevant_subs = list(relevant_subs)
            logger.info(f"{len(self.relevant_subs)} relevant subreddits found!")
        return source_metadata

    def clean_data(self):
        clean_paths = []
        subs_to_filter = []
        for sub in self.relevant_subs:
            clean_sub_path = os.path.join(self._subs_data_dir, f"{sub}_cleaned.csv")
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

        n_downloaded = len(self.relevant_subs) - len(subs_to_filter)
        logger.info(
            f"{n_downloaded} relevant subreddits already downloaded, "
            f"{len(subs_to_filter)} remaining to download and clean."
        )

        if self.max_download_workers > 1:
            results = process_map(
                partial(
                    _download_submissions_and_comments,
                    data_path=self._subs_data_dir,
                    anonymizer=self._anonymizer,
                    batch_size=self.anonymizer_batch_size,
                ),
                subs_to_filter,
                max_workers=self.max_download_workers,
                desc=f"Downloading Reddit data [{self.max_download_workers} workers]",
                chunksize=1,
                position=0,
                leave=True,
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
                    self._anonymizer,
                    self.anonymizer_batch_size,
                )
                if clean_sub_path is not None:
                    clean_paths.append(clean_sub_path)

        return clean_paths

    def curate_experiment_data(
        self,
        experiment: Experiment,
        study_name: str,
        filter_by_date: bool,
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
        filter_by_date : bool
            Whether to filter the data by the experiment's date.
        clean_data_paths : list[str]
            List of paths to the cleaned Reddit data.

        Returns
        -------
        tuple[str, int]
            Path to the curated experiment data file and the number of valid posts
            that mention both treatments and outcomes.

        """
        assert len(clean_data_paths) > 0
        study_dir = os.path.join(self.data_path, study_name.lower().replace(" ", "_"))
        os.makedirs(study_dir, exist_ok=True)
        save_path = os.path.join(
            study_dir,
            f"reddit_{experiment.nct_id}.csv",
        )
        if os.path.exists(save_path):
            exp_df = pd.read_csv(save_path, index_col=0)
            return save_path, len(exp_df)

        drugbank_names = [
            item for sublist in experiment.drugbank_names.values() for item in sublist
        ]
        treatment_names = [
            name.lower()
            for name in experiment.treatment_common_names["reddit"] + drugbank_names
        ]
        outcome_names = [
            name.lower() for name in experiment.outcome_common_names["reddit"]
        ]

        # Prepare date filter
        date_cutoff = None
        if filter_by_date and experiment.date:
            try:
                date_obj = datetime.datetime.strptime(experiment.date, "%Y-%m-%d")
            except ValueError:
                date_obj = datetime.datetime.strptime(experiment.date, "%Y-%m")

            # Convert to UTC timestamp
            date_cutoff = int(
                date_obj.replace(tzinfo=datetime.timezone.utc).timestamp()
            )

        chunk_size = 5000
        valid_count = 0
        row_index = 0
        first_chunk = True

        for clean_data_path in clean_data_paths:
            file_size = os.path.getsize(clean_data_path)
            if file_size == 0:
                logger.warning(
                    f"Skipping empty file: {clean_data_path} (size: {file_size} bytes)"
                )
                continue
            for chunk in pd.read_csv(
                clean_data_path, index_col=0, chunksize=chunk_size
            ):
                if chunk.empty:
                    logger.warning(f"Skipping empty chunk from file: {clean_data_path}")
                    continue
                processed_chunk = self._process_and_filter_chunk(
                    chunk, treatment_names, outcome_names, date_cutoff
                )

                if not processed_chunk.empty:
                    processed_chunk.index = range(
                        row_index, row_index + len(processed_chunk)
                    )
                    row_index += len(processed_chunk)

                    # Write header only for first chunk
                    processed_chunk.to_csv(
                        save_path, mode="w" if first_chunk else "a", header=first_chunk
                    )
                    valid_count += len(processed_chunk)
                    first_chunk = False

        if valid_count == 0:
            columns = pd.read_csv(clean_data_path, nrows=0).columns.tolist()
            empty_df = pd.DataFrame(
                columns=columns + ["treatments_mentioned", "outcome_words"]
            )
            empty_df.to_csv(save_path)
            logger.warning(f"No valid matches found for experiment {experiment.nct_id}")

        return save_path, valid_count

    def _process_and_filter_chunk(
        self,
        chunk: pd.DataFrame,
        treatment_names: list,
        outcome_names: list,
        date_cutoff: Optional[int] = None,
    ) -> pd.DataFrame:
        """Process chunk and apply all filters in one pass."""
        chunk = chunk.fillna("")

        # Date filter first
        if date_cutoff:

            def parse_date(date_str):
                try:
                    dt = datetime.datetime.strptime(date_str, "%B %d, %Y")
                    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
                except:
                    return float("inf")

            timestamps = chunk["date_created"].map(parse_date)
            date_mask = timestamps <= date_cutoff
            chunk = chunk[date_mask]

            if chunk.empty:
                return pd.DataFrame()

        # Vectorized text matching
        combined_text = (
            chunk["subreddit"].astype(str)
            + " "
            + chunk["title"].astype(str)
            + " "
            + chunk["report"].astype(str)
            + " "
            + chunk["initial_post"].astype(str)
        ).str.lower()

        # Create boolean masks for treatments and outcomes
        treatment_masks = [
            combined_text.str.contains(name, regex=False, na=False)
            for name in treatment_names
        ]
        outcome_masks = [
            combined_text.str.contains(name, regex=False, na=False)
            for name in outcome_names
        ]

        # Combine masks
        has_treatment = (
            pd.concat(treatment_masks, axis=1).any(axis=1)
            if treatment_masks
            else pd.Series([False] * len(chunk))
        )
        has_outcome = (
            pd.concat(outcome_masks, axis=1).any(axis=1)
            if outcome_masks
            else pd.Series([False] * len(chunk))
        )

        valid_mask = has_treatment & has_outcome

        if not valid_mask.any():
            return pd.DataFrame()

        # Get valid rows and find specific matches
        result = chunk[valid_mask].copy()
        valid_combined_text = combined_text[valid_mask]

        treatments_list = []
        outcomes_list = []

        for text in valid_combined_text:
            found_treatments = [name for name in treatment_names if name in text]
            found_outcomes = [name for name in outcome_names if name in text]
            treatments_list.append(found_treatments)
            outcomes_list.append(found_outcomes)

        result["treatments_mentioned"] = treatments_list
        result["outcome_words"] = outcomes_list

        return result.reset_index(drop=True)


def _download_submissions_and_comments(
    sub: str, data_path: str, anonymizer: Optional[Anonymizer], batch_size: int
) -> tuple[str, str]:
    """Download submissions and comments for a given subreddit, then clean."""
    clean_sub_path = os.path.join(data_path, f"{sub}_cleaned.csv")
    if os.path.exists(clean_sub_path):
        return clean_sub_path

    try:
        submissions_path = os.path.join(data_path, f"{sub}_submissions.csv")
        comments_path = os.path.join(data_path, f"{sub}_comments.csv")
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
        rule_filtered_df.to_csv(clean_sub_path)
        # Delete the submissions and comments files after cleaning
        os.remove(submissions_path)
        os.remove(comments_path)
    except Exception as e:
        clean_sub_path = None
        logger.error(f"Error processing subreddit {sub}: {e}")

    return clean_sub_path


def _clean_sub_data(data_path: str, sub: str) -> pd.DataFrame:
    """Clean submissions and comments for a given subreddit."""
    submissions = pd.read_csv(
        os.path.join(data_path, f"{sub}_submissions.csv"),
        index_col=0,
        low_memory=False,
    )
    submissions = rule_based_filter(submissions, "selftext")

    comments = pd.read_csv(
        os.path.join(data_path, f"{sub}_comments.csv"),
        index_col=0,
        low_memory=False,
    )
    comments = rule_based_filter(comments, "body")

    return _get_context_post_df(submissions, comments)
