"""Utilities for downloading and processing Reddit data from The Eye archive."""

import asyncio
import datetime
import glob
import json
import logging
import os
import warnings
from typing import Generator, Literal, Optional
from urllib import error, request

import asyncpraw
import asyncprawcore
import pandas as pd
import tenacity
import wget
import zstandard
from aiolimiter import AsyncLimiter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm

from naturalv2.sources.anonymizer import Anonymizer


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

logger = logging.getLogger(__name__)


def is_retryable_error(exception: BaseException) -> bool:
    """Check if the exception is retryable."""
    return (
        isinstance(exception, asyncprawcore.exceptions.ResponseException)
        and exception.response.status in [429, 500, 502, 503, 504]
    ) or (
        isinstance(exception, error.HTTPError)
        and exception.code in [429, 500, 502, 503, 504]
    )


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

    """
    # Load the CSV file if it exists and return as DataFrame
    about_csv_path = os.path.join(data_path, "subs_about.csv")
    if os.path.exists(about_csv_path):
        return pd.read_csv(about_csv_path, index_col=0)

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


def download_subs_list(data_path: str) -> str:
    """Download the list of subreddits from The Eye archive.

    Parameters
    ----------
    data_path : str
        The path where the list of subreddits will be saved.

    Returns
    -------
    str
        The path to the file containing the list of subreddits.

    """
    filepath = os.path.join(data_path, "subs_list.txt")
    if not os.path.exists(filepath):
        logger.info("Downloading the list of subreddits from The Eye archive...")
        url = "https://the-eye.eu/redarcs/"
        response = request.urlopen(url)
        html: str = response.read().decode("utf-8")

        # Extract subreddit names from links
        subs = []
        for line in html.split("\n"):
            if "href=" in line and ".zst" in line:
                sub = line.split("href=")[1].split("_")[0].split("/")[-1]
                if sub not in subs:
                    subs.append(sub)

        with open(filepath, "w") as f:
            f.write("\n".join(subs))

        logger.info(f"{len(subs)} subreddits listed.")

    return filepath


@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(is_retryable_error),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def download_sub_data(
    subreddit: str,
    data_type: Literal["submissions", "comments"],
    data_path: str,
    anonymizer_instance: Optional["Anonymizer"] = None,
    batch_size: int = 1,
    num_workers: int = 1,
) -> None:
    """Download subreddit data from The Eye archive.

    Parameters
    ----------
    subreddit : str
        The name of the subreddit to download data for.
    data_type : Literal["submissions", "comments"]
        The type of data to download, either "submissions" or "comments".
    data_path : str
        The path where the data will be saved.
    anonymizer_instance : Optional[Anonymizer], optional
        An instance of Anonymizer to anonymize the data, by default None.
    batch_size : int, optional
        The batch size for anonymization, by default 1.
    num_workers : int, optional
        The number of workers for anonymization, by default 1.

    Raises
    -------
    ValueError
        If `data_type` is not "submissions" or "comments".
    """
    if data_type not in ["submissions", "comments"]:
        raise ValueError(
            f"Expected data_type to be 'submissions' or 'comments', but got {data_type}"
        )

    os.makedirs(data_path, exist_ok=True)

    save_path = os.path.join(data_path, f"{subreddit}_{data_type}.parquet")
    if os.path.exists(save_path):
        logger.warning(
            f"File {save_path} already exists. Skipping download for {subreddit} {data_type}."
        )
        return

    file_path = os.path.join(data_path, f"{subreddit}_{data_type}.zst")
    if not os.path.exists(file_path):
        # Go to TMPDIR if set, otherwise stay current working directory, since
        # wget doesn't respect TMPDIR
        tmpdir = os.environ.get("TMPDIR", os.getcwd())
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            _ = wget.download(
                f"https://the-eye.eu/redarcs/files/{subreddit}_{data_type}.zst",
                out=data_path,
                bar=None,
            )
        finally:
            os.chdir(original_cwd)

    file_lines = 0
    bad_lines = 0
    data = []

    for line, _ in _read_lines_zst(file_path):
        try:
            obj = json.loads(line)
            data += [obj]
        except (KeyError, json.JSONDecodeError):
            bad_lines += 1
        file_lines += 1

    df = pd.DataFrame(data, dtype="string")

    # remove deleted posts or comments
    df = (
        df[df["selftext"] not in ["[deleted]", "[removed]"]]
        if data_type == "submissions"
        else df[df["body"] not in ["[deleted]", "[removed]"]]
    )

    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    cols_to_keep = ["created_utc", "author", "permalink", "subreddit", "score"]
    # anonymize dataframe
    if anonymizer_instance is not None:
        df = anonymizer_instance.anonymize_dataframe(
            df,
            cols_to_keep=cols_to_keep,
            cols_to_anonymize=["selftext", "title"]
            if data_type == "submissions"
            else ["body"],
            data_source_name=f"{subreddit}_{data_type}",
            batch_size=batch_size,
            num_workers=num_workers,
        )

    df.convert_dtypes(dtype_backend="pyarrow").to_parquet(
        save_path, index=False, compression="snappy"
    )
    os.remove(file_path)
    logger.info(
        f"Completed download of {subreddit} {data_type} data with: {file_lines:,} lines "
        f"({bad_lines:,} bad lines) and {len(df):,} valid records."
    )


def rule_based_filter(post_df: pd.DataFrame, text_field: str) -> pd.DataFrame:
    """Apply rule-based filtering to a DataFrame of Reddit posts.

    This function filters out posts based on several criteria:
    - Ensures the text field is of type str.
    - Ensures the permalink is of type str.
    - Removes posts without a score.
    - Removes posts that are deleted or removed.
    - Removes very short comments (less than 10 words).
    - Removes posts with "bot" in the author's name.
    - Cleans the text field by unescaping HTML tags and removing leading/trailing whitespace.
    - Ensures the text field has a space within the first 2048 characters.
    - Ensures the text field has at least 50% alphabetic characters (including spaces).

    Parameters
    ----------
    post_df : pd.DataFrame
        The DataFrame containing Reddit posts.
    text_field : str
        The name of the text field in the DataFrame to be filtered.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only valid posts.

    """
    # remove rows where the text field is not of type str
    idx = post_df[text_field].apply(lambda x: isinstance(x, str))
    post_df = post_df.loc[idx]

    # remove rows where the permalink is not of type str
    idx = post_df["permalink"].apply(lambda x: isinstance(x, str))
    post_df = post_df.loc[idx]

    # remove rows without a score
    post_df = post_df.loc[post_df["score"].notna()]

    # remove rows where the submission is deleted or removed
    post_df = post_df.loc[post_df[text_field] not in ["[deleted]", "[removed]"]]

    # remove very short comments
    if text_field == "body":
        idx = post_df[text_field].apply(lambda x: len(x.split()) >= 10)
        post_df = post_df.loc[idx]

    # remove posts with "bot" in the author's name
    idx = post_df["author"].apply(lambda x: "bot" not in x.lower())
    post_df = post_df.loc[idx]

    for i, row in post_df.iterrows():
        body: str = row[text_field]

        # unescape some common html tags
        body = body.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        body = body.replace("\n", " ").replace("\t", " ")
        body = body.strip()

        # drop if there is no space in first 2048 characters
        try:
            _ = body[: body.rindex(" ", 0, 2048)]
        except ValueError:
            post_df = post_df.drop([i])
            continue

        # drop everything with less than 50% alphabetic characters; space counts
        length_characters = float(len(body))
        filtered = [c for c in body if c.isalpha()]
        if float(len(filtered)) / length_characters < 0.5:
            post_df = post_df.drop([i])
            continue

    return post_df


def get_context_post_df(
    submissions: pd.DataFrame, comments: pd.DataFrame
) -> pd.DataFrame:
    """Join submissions and comments DataFrames to create a context post DataFrame.

    Parameters
    ----------
    submissions : pd.DataFrame
        DataFrame containing submission data with columns:
        - subreddit
        - title
        - selftext
        - score
        - created_utc
        - permalink
    comments : pd.DataFrame
        DataFrame containing comment data with columns:
        - permalink
        - author
        - body
        - score
        - created_utc

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the context posts with the following columns:
        - subreddit
        - title
        - initial_post
        - report
        - score
        - date_created
        - permalink
        - author_replies (list of replies from the author)
    """
    submissions = submissions.copy()
    submissions["date_created"] = submissions["created_utc"].astype(int).map(_get_date)
    submissions["submission_permalink"] = submissions["permalink"].map(
        _get_submission_permalink
    )

    comments = comments.copy()
    comments["permalink_processed"] = comments["permalink"].map(_get_comment_permalink)
    comments["date_created"] = comments["created_utc"].astype(int).map(_get_date)

    comments_grouped = comments.groupby("permalink_processed")

    all_results = []
    for _, submission in submissions.iterrows():
        submission_permalink = submission["submission_permalink"]

        if submission_permalink in comments_grouped.groups:
            submission_comments = comments_grouped.get_group(submission_permalink)

            # Separate author replies from other comments
            author_mask = submission_comments["author"] == submission["author"]
            author_comments = submission_comments[author_mask]
            other_comments = submission_comments[~author_mask]

            # Build submission text with author replies
            submission_text = str(submission["selftext"])
            author_replies_list = author_comments["body"].tolist()

            if author_replies_list:
                submission_text += (
                    "\n\nThe author also replied with the following in the thread:"
                )
                for reply in author_replies_list:
                    submission_text += "\n> " + str(reply)
        else:
            other_comments = pd.DataFrame()
            author_replies_list = []
            submission_text = str(submission["selftext"])

        submission_row = {
            "subreddit": submission["subreddit"],
            "title": submission["title"],
            "initial_post": "",
            "report": submission_text,
            "score": int(submission["score"]),
            "date_created": submission["date_created"],
            "permalink": submission_permalink,
            "author_replies": author_replies_list,
        }
        all_results.append(submission_row)

        if not other_comments.empty:
            comment_rows = {
                "subreddit": [submission["subreddit"]] * len(other_comments),
                "title": [submission["title"]] * len(other_comments),
                "initial_post": [submission_text] * len(other_comments),
                "report": other_comments["body"].astype(str).tolist(),
                "score": other_comments["score"].astype(int).tolist(),
                "date_created": other_comments["date_created"].tolist(),
                "permalink": other_comments["permalink"].tolist(),
                "author_replies": [[]] * len(other_comments),
            }

            # Convert to list of dicts for consistency
            for i in range(len(other_comments)):
                all_results.append(
                    {
                        "subreddit": comment_rows["subreddit"][i],
                        "title": comment_rows["title"][i],
                        "initial_post": comment_rows["initial_post"][i],
                        "report": comment_rows["report"][i],
                        "score": comment_rows["score"][i],
                        "date_created": comment_rows["date_created"][i],
                        "permalink": comment_rows["permalink"][i],
                        "author_replies": comment_rows["author_replies"][i],
                    }
                )

    if all_results:
        return pd.DataFrame(all_results)

    return pd.DataFrame(
        columns=[
            "subreddit",
            "title",
            "initial_post",
            "report",
            "score",
            "date_created",
            "permalink",
            "treatments_mentioned",
            "outcome_words",
            "author_replies",
        ]
    )


def filter_by_date(
    adf: pd.DataFrame, cutoff_dt: pd.Timestamp, date_col: str
) -> pd.DataFrame:
    """Filter a DataFrame by a date cutoff.

    Parameters
    ----------
    adf : pd.DataFrame
        The DataFrame to filter.
    cutoff_dt : pd.Timestamp
        The cutoff timestamp. Only rows with dates on or before this date will
        be kept.
    date_col : str
        The name of the column in the DataFrame containing date information.

    Returns
    -------
    pd.DataFrame
        A DataFrame filtered to include only rows with dates on or before the cutoff date.
    """
    if adf.empty:
        return pd.DataFrame()

    # Parse date column all at once, try inference and coerce errors
    date_series: pd.Series = pd.to_datetime(adf[date_col], errors="coerce")

    # Filter rows which have a datetime and are on or before cutoff
    mask = (date_series.notna()) & (date_series <= cutoff_dt)
    return adf.loc[mask].reset_index(drop=True)


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
    """Fetch subreddit about information from Reddit API or local JSON file."""
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


def _read_lines_zst(file_name: str) -> Generator[tuple[str, int], None, None]:
    """Read lines from a zstandard compressed file."""
    with open(file_name, "rb") as file_handle:
        buffer = ""
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(
            file_handle
        )

        while True:
            chunk = _read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

        for line in lines[:-1]:
            yield line, file_handle.tell()

        buffer = lines[-1]
    reader.close()


def _read_and_decode(
    reader: zstandard.ZstdDecompressionReader,
    chunk_size: int,
    max_window_size: int,
    previous_chunk: Optional[bytes] = None,
    bytes_read: int = 0,
) -> str:
    """Read and decode a chunk from the zstandard stream."""
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size

    if previous_chunk is not None:
        chunk = previous_chunk + chunk

    try:
        return chunk.decode()
    except UnicodeDecodeError as err:
        if bytes_read > max_window_size:
            raise UnicodeError(
                f"Unable to decode frame after reading {bytes_read:,} bytes"
            ) from err
        logger.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return _read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def _get_submission_permalink(permalink: str) -> str:
    """Extracts the submission permalink from a full Reddit permalink."""
    return "/" + permalink.split("/")[-2] + "/"


def _get_comment_permalink(permalink: str) -> str:
    """Extracts the comment permalink from a full Reddit permalink."""
    return "/" + permalink.split("/")[-3] + "/"


def _get_date(utc_timestamp: float) -> str:
    """Convert a UTC timestamp to a formatted date string."""
    dt = datetime.datetime.fromtimestamp(utc_timestamp, tz=datetime.timezone.utc)
    return dt.strftime("%B %d, %Y")
