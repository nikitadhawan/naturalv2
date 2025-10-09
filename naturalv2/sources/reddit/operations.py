"""Reddit data operations used by curation stages and adapters.

This module contains asynchronous search utilities powered by PRAW as well as
batch-oriented helpers to download, clean, and select study-relevant Reddit
posts.
"""

from __future__ import annotations

import logging
import os
import re

import asyncpraw
import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    wait_random_exponential,
)

from naturalv2.sources.anonymizer import Anonymizer
from naturalv2.sources.components import filter_by_date
from naturalv2.sources.reddit.utils import (
    download_sub_data,
    get_context_post_df,
    is_retryable_error,
    rule_based_filter,
)


logger = logging.getLogger(__name__)


@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_random_exponential(multiplier=1, max=60),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
async def search_subreddits(
    keyword: str, reddit_client: asyncpraw.Reddit, limiter: AsyncLimiter
) -> list[str]:
    """Search for subreddits matching a keyword.

    Parameters
    ----------
    keyword : str
        Keyword to search for.
    reddit_client : asyncpraw.Reddit
        An authenticated async PRAW client.
    limiter : AsyncLimiter
        Async rate limiter to bound Reddit API calls.

    Returns
    -------
    list[str]
        Display names of matching subreddits.
    """

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
async def search_posts_in_subreddit(
    subreddit: str,
    keyword: str,
    reddit_client: asyncpraw.Reddit,
    limiter: AsyncLimiter,
    *,
    limit: int = 5,
    char_limit: int = 1000,
) -> list[str]:
    """Return formatted post snippets for a subreddit-keyword query.

    Parameters
    ----------
    subreddit : str
        Subreddit name (without the ``r/`` prefix).
    keyword : str
        Search query to filter submissions.
    reddit_client : asyncpraw.Reddit
        An authenticated async PRAW client.
    limiter : AsyncLimiter
        Async rate limiter to bound Reddit API calls.
    limit : int, default=5
        Maximum number of submissions to fetch.
    char_limit : int, default=1000
        Maximum number of characters from the selftext to include in the
        formatted snippet.

    Returns
    -------
    list[str]
        A list of human-readable snippets combining title and truncated body.
    """

    async with limiter:
        posts: list[str] = []
        subreddit_instance = await reddit_client.subreddit(subreddit)
        async for submission in subreddit_instance.search(keyword, limit=limit):
            posts.append(
                "**Title**: "
                + submission.title
                + "\n"
                + "**Post content**: "
                + (submission.selftext or "")[:char_limit]
            )
        return posts


def clean_subreddit_data(data_path: str, subreddit: str) -> pd.DataFrame:
    """Load, rule-filter, and join submissions/comments for a subreddit.

    Parameters
    ----------
    data_path : str
        Directory containing ``*_submissions.parquet`` and
        ``*_comments.parquet`` files for the subreddit.
    subreddit : str
        Subreddit name used to derive file names.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row represents a submission or comment with
        additional context fields, filtered using simple rule-based heuristics.
    """

    submissions = pd.read_parquet(
        os.path.join(data_path, f"{subreddit}_submissions.parquet")
    )
    submissions = rule_based_filter(submissions, "selftext")

    comments = pd.read_parquet(os.path.join(data_path, f"{subreddit}_comments.parquet"))
    comments = rule_based_filter(comments, "body")

    return get_context_post_df(submissions, comments)


def download_submissions_and_comments(
    subreddit: str,
    data_path: str,
    *,
    anonymizer: Anonymizer | None,
    batch_size: int,
) -> tuple[str | None, str]:
    """Download raw data for a subreddit and return the cleaned parquet path.

    Downloads submissions and comments if missing, applies rule-based cleaning
    and anonymization (if configured), then writes a consolidated cleaned
    parquet file. Original raw parquet files are removed upon success.

    Parameters
    ----------
    subreddit : str
        Subreddit name to download.
    data_path : str
        Directory to read/write subreddit parquet files.
    anonymizer : Anonymizer | None
        Optional anonymizer to scrub PII from text fields.
    batch_size : int
        Batch size used by the anonymizer when processing text.

    Returns
    -------
    tuple[str | None, str]
        A tuple ``(cleaned_parquet_path, subreddit)``. The path is ``None``
        if an error occurred.
    """

    clean_sub_path = os.path.join(data_path, f"{subreddit}_cleaned.parquet")
    if os.path.exists(clean_sub_path):
        return clean_sub_path, subreddit

    try:
        submissions_path = os.path.join(data_path, f"{subreddit}_submissions.parquet")
        comments_path = os.path.join(data_path, f"{subreddit}_comments.parquet")
        if not os.path.exists(submissions_path):
            download_sub_data(
                subreddit,
                "submissions",
                data_path,
                anonymizer_instance=anonymizer,
                batch_size=batch_size,
            )
        if not os.path.exists(comments_path):
            download_sub_data(
                subreddit,
                "comments",
                data_path,
                anonymizer_instance=anonymizer,
                batch_size=batch_size,
            )

        filtered_df = clean_subreddit_data(data_path, subreddit)
        filtered_df.to_parquet(clean_sub_path, index=False, compression="snappy")
        os.remove(submissions_path)
        os.remove(comments_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error processing subreddit %s: %s", subreddit, exc)
        return None, subreddit

    return clean_sub_path, subreddit


def get_study_relevant_posts(
    clean_data_path: str,
    treatment_pattern: re.Pattern,
    cutoff_dt: pd.Timestamp | None,
    *,
    date_column: str = "date_created",
) -> pd.DataFrame:
    """Select posts mentioning treatments before an optional cutoff date.

    Parameters
    ----------
    clean_data_path : str
        Path to a cleaned subreddit parquet file created by
        :func:`download_submissions_and_comments`.
    treatment_pattern : re.Pattern
        Compiled regex matching any treatment term of interest.
    cutoff_dt : pandas.Timestamp | None
        If provided, only posts with dates before this timestamp are
        considered.
    date_column : str, default="date_created"
        Column name containing the post/comment timestamp.

    Returns
    -------
    pandas.DataFrame
        DataFrame of posts mentioning any treatment term, with an additional
        ``treatments_mentioned`` column listing the matched terms.
    """

    df = pd.read_parquet(clean_data_path)
    if df.empty:
        return pd.DataFrame()

    if cutoff_dt is not None:
        df = filter_by_date(df, cutoff_dt, date_column)
        if df.empty:
            return pd.DataFrame()

    text_cols = ["subreddit", "title", "report_text", "initial_post"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    treatment_finds = [
        df[col].str.lower().str.findall(treatment_pattern) for col in text_cols
    ]
    has_treatment_mask = np.any(
        [series.str.len() > 0 for series in treatment_finds], axis=0
    )

    result = df[has_treatment_mask].copy()
    if result.empty:
        return result

    valid_finds = [series[has_treatment_mask] for series in treatment_finds]
    result["treatments_mentioned"] = [
        list({item for sublist in row for item in sublist}) for row in zip(*valid_finds)
    ]

    return result.reset_index(drop=True)
