"""Async Reddit API helpers with retry-aware search utilities.

This module offers thin wrappers around PRAW that add sensible defaults:
- Retry configuration tuned for rate/availability errors.
- Async rate limiting via ``AsyncLimiter``.
- Convenience functions to search for subreddits, fetch posts, and download
  posts/comments into dataframes for downstream processing.
"""

import asyncio
import glob
import json
import logging
import os
import ssl
from http.client import RemoteDisconnected
from urllib import error

import asyncpraw
import asyncprawcore
import pandas as pd
import tenacity
from aiolimiter import AsyncLimiter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm


logger = logging.getLogger(__name__)

_RETRYABLE_HTTP_ERRORS = {429, 500, 502, 503, 504}


def is_retryable_error(exception: BaseException) -> bool:
    """Return whether an exception is retryable for Reddit requests.

    Parameters
    ----------
    exception : BaseException
        The exception raised by a request.

    Returns
    -------
    bool
        ``True`` if the exception corresponds to a transient HTTP error that
        should be retried, ``False`` otherwise.
    """
    return (
        (
            isinstance(exception, asyncprawcore.exceptions.ResponseException)
            and exception.response.status in _RETRYABLE_HTTP_ERRORS
        )
        or (
            isinstance(exception, error.HTTPError)
            and exception.code in _RETRYABLE_HTTP_ERRORS
        )
        or isinstance(exception, (error.URLError, ssl.SSLError, RemoteDisconnected))
    )


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


@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(is_retryable_error),
    retry_error_callback=lambda retry_state: {
        "subreddit": "error",
        "description": f"Retry limit exceeded for {retry_state.args[0]}",
        "public_description": None,
    },
    before_sleep=before_sleep_log(logger, logging.INFO),
)
async def _fetch_sub_about(
    subreddit_name: str,
    reddit_client: asyncpraw.Reddit,
    save_dir: str,
    rate_limiter: AsyncLimiter,
) -> dict[str, str | None]:
    """Fetch subreddit about information from Reddit API or local JSON file.

    Parameters
    ----------
    subreddit_name : str
        The subreddit to fetch.
    reddit_client : asyncpraw.Reddit
        An authenticated async PRAW client.
    save_dir : str
        Directory to write a ``{subreddit_name}_about.json`` record.
    rate_limiter : AsyncLimiter
        Async limiter to bound API call rate.

    Returns
    -------
    dict[str, str | None]
        A dictionary with keys ``subreddit``, ``description`` and
        ``public_description``. On failure, ``subreddit`` is set to ``"error"``
        and an explanatory ``description`` is provided.
    """
    about_info = {"subreddit": "error", "description": None, "public_description": None}
    async with rate_limiter:
        try:
            subreddit = await reddit_client.subreddit(subreddit_name, fetch=True)
            if not subreddit.over18:
                about_info = {
                    "subreddit": subreddit.display_name,
                    "description": subreddit.description,
                    "public_description": subreddit.public_description,
                }
            else:
                about_info["description"] = (
                    "This subreddit is NSFW (not safe for work)."
                )
        except asyncprawcore.exceptions.ResponseException as e:
            if e.response.status == 429:  # catch and retry on rate limit errors
                raise error.HTTPError(
                    f"https://www.reddit.com/r/{subreddit_name}/about.json",
                    e.response.status,
                    e.response.reason,
                    e.response.headers,
                    None,
                ) from e
            logger.debug(
                f"Error fetching about info for subreddit {subreddit_name}: {e}"
            )
            about_info["description"] = e.response.reason
        except tenacity.RetryError as e:
            logger.error(
                f"Retry limit exceeded for subreddit {subreddit_name}: "
                f"{e.last_attempt.exception()}"
            )
            about_info["description"] = "retry limit exceeded"
        except asyncprawcore.exceptions.AsyncPrawcoreException as e:
            logger.error(
                f"Error fetching about info for subreddit {subreddit_name}: {e}"
            )
            about_info["description"] = str(e)
        finally:
            # Save the about info to a JSON file
            about_file = os.path.join(save_dir, f"{subreddit_name}_about.json")
            with open(about_file, "w") as f:
                json.dump(about_info, f)

    return about_info


async def get_sub_about_info(data_path: str, api_rate_limit: int = 10) -> pd.DataFrame:
    """Fetch subreddit about information and create a DataFrame.

    This function checks if the list of subreddits exists, downloads it if not,
    and then fetches the about information for each subreddit. It saves the
    information in a CSV file and returns a DataFrame containing the subreddit
    names, descriptions, and public descriptions.

    Parameters
    ----------
    data_path : str
        The path where the subreddit about information will be saved.
    api_rate_limit : int, default=10
        The number of requests allowed per minute to the Reddit API.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the subreddit names, descriptions, and public
        descriptions.

    Raises
    ------
    ValueError
        If required PRAW environment variables are missing. Required variables:
        PRAW_CLIENT_ID, PRAW_CLIENT_SECRET, PRAW_PWD, PRAW_USERNAME, PRAW_AGENT.

    """
    # Load the CSV file if it exists and return as DataFrame
    about_csv_path = os.path.join(data_path, "subs_about.csv")
    if os.path.exists(about_csv_path):
        return pd.read_csv(about_csv_path, index_col=0)

    from naturalv2.sources.reddit.pushshift_archive import (  # noqa: PLC0415
        download_subs_list,  # Import here to avoid circular imports
    )

    subs_list_path = download_subs_list(data_path)
    with open(subs_list_path, "r") as f:
        all_subs = f.read().splitlines()

    about_jsons_dir = os.path.join(data_path, "subs_about")
    os.makedirs(about_jsons_dir, exist_ok=True)
    all_json_files = glob.glob(os.path.join(about_jsons_dir, "*.json"))

    # Load already downloaded JSON files
    # NOTE: A json file is always saved, regardless of whether there was an error
    # while fetching the subreddit about info or not. But, subreddits that
    # had an error during downloading will not be added to the CSV file.
    subreddit_info_rows = []
    downloaded_subs = set()
    for json_file in all_json_files:
        sub_name_from_file = os.path.splitext(os.path.basename(json_file))[0].replace(
            "_about", ""
        )
        downloaded_subs.add(sub_name_from_file)
        with open(json_file, "r") as f:
            data = json.load(f)
            if data["subreddit"] != "error":
                subreddit_info_rows.append(data)

    # Resume downloading about info for subreddits not yet processed
    # by checking the difference between all_subs and downloaded_subs
    remaining_subs = list(set(all_subs) - downloaded_subs)

    # Fetch about info for remaining subreddits and combine with existing data
    if len(remaining_subs) > 0:
        # Validate required PRAW environment variables
        required_vars = [
            "PRAW_CLIENT_ID",
            "PRAW_CLIENT_SECRET",
            "PRAW_PWD",
            "PRAW_USERNAME",
            "PRAW_AGENT",
        ]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required PRAW environment variables: {', '.join(missing_vars)}. "
                f"Please set all of: {', '.join(required_vars)}"
            )

        async with asyncpraw.Reddit(
            client_id=os.environ.get("PRAW_CLIENT_ID"),
            client_secret=os.environ.get("PRAW_CLIENT_SECRET"),
            password=os.environ.get("PRAW_PWD"),
            username=os.environ.get("PRAW_USERNAME"),
            user_agent=os.environ.get("PRAW_AGENT"),
        ) as reddit_client:
            rate_limiter = AsyncLimiter(api_rate_limit, 60)
            subreddit_fetch_tasks = [
                asyncio.create_task(
                    _fetch_sub_about(
                        subreddit_name, reddit_client, about_jsons_dir, rate_limiter
                    )
                )
                for subreddit_name in remaining_subs
            ]

            for task in tqdm(
                asyncio.as_completed(subreddit_fetch_tasks),
                total=len(subreddit_fetch_tasks),
                desc="Fetching subreddit about info",
                unit="subreddit",
                leave=False,
                dynamic_ncols=True,
            ):
                result = await task
                if result["subreddit"] != "error":
                    subreddit_info_rows.append(result)

    cols = ["subreddit", "description", "public_description"]
    if not subreddit_info_rows:
        logger.warning(
            "No subreddit about information was fetched. Returning an empty DataFrame."
        )
        return pd.DataFrame(columns=cols)

    # Create DataFrame from the collected subreddit info
    about_df = (
        pd.DataFrame(subreddit_info_rows, columns=cols)
        .drop_duplicates(subset=["subreddit"])
        .sort_values(by="subreddit")
        .reset_index(drop=True)
    )
    about_df.to_csv(about_csv_path)

    return about_df
