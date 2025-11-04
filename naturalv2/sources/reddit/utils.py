"""Utilities for downloading and processing Reddit data from The Eye archive."""

import asyncio
import glob
import json
import logging
import os
import ssl
import warnings
from typing import TYPE_CHECKING, Callable, Generator, Literal, Optional, TypeVar
from urllib import error, request

import asyncpraw
import asyncprawcore
import numpy as np
import pandas as pd
import regex as re
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


if TYPE_CHECKING:
    from naturalv2.sources.anonymizer import Anonymizer


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Insecure SSL context for downloading Reddit data when TLS verification fails
_INSECURE_SSL_CONTEXT = ssl.create_default_context()
_INSECURE_SSL_CONTEXT.check_hostname = False
_INSECURE_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _with_tls_fallback(
    url: str,
    fetch: Callable[[str, Optional[ssl.SSLContext]], _T],
    *,
    description: str,
) -> _T:
    """Execute ``fetch`` for ``url`` with an insecure TLS fallback if verification fails."""

    # First attempt uses system verification
    try:
        return fetch(url, None)
    except error.URLError as exc:
        ssl_error = getattr(exc, "reason", None)
        is_ssl_failure = isinstance(ssl_error, ssl.SSLCertVerificationError)
        if not (is_ssl_failure and url.startswith("https://")):
            raise

        logger.warning(
            "TLS verification failed while %s from %s; retrying without certificate verification.",
            description,
            url,
        )
        return fetch(url, _INSECURE_SSL_CONTEXT)


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

        def _fetch_html(target_url: str, context: Optional[ssl.SSLContext]) -> str:
            if context is None:
                with request.urlopen(target_url) as response:
                    return response.read().decode("utf-8")

            with request.urlopen(target_url, context=context) as response:
                return response.read().decode("utf-8")

        html: str = _with_tls_fallback(
            url, _fetch_html, description="fetching subreddit index"
        )

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
            url = f"https://the-eye.eu/redarcs/files/{subreddit}_{data_type}.zst"

            def _download(target_url: str, context: Optional[ssl.SSLContext]) -> str:
                if context is None:
                    return wget.download(target_url, out=data_path, bar=None)

                opener = request.build_opener(request.HTTPSHandler(context=context))
                previous_opener = request._opener  # type: ignore[attr-defined]
                try:
                    request.install_opener(opener)
                    return wget.download(target_url, out=data_path, bar=None)
                finally:
                    if previous_opener is None:
                        request._opener = None  # type: ignore[attr-defined]
                    else:
                        request.install_opener(previous_opener)

            _with_tls_fallback(
                url,
                _download,
                description=f"downloading {subreddit} {data_type} archive",
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
        df[~df["selftext"].isin(["[deleted]", "[removed]"])]
        if data_type == "submissions"
        else df[~df["body"].isin(["[deleted]", "[removed]"])]
    )

    df.loc[:, "score"] = pd.to_numeric(df["score"], errors="coerce")

    cols_to_keep = ["id", "created_utc", "author", "permalink", "subreddit", "score"]
    if data_type == "comments":
        cols_to_keep.append("link_id")

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


def apply_rule_based_filter(reddit_data: pd.DataFrame, text_field: str) -> pd.Series:
    """Apply rule-based filtering to a DataFrame of Reddit posts.

    Parameters
    ----------
    reddit_data : pd.DataFrame
        A DataFrame containing Reddit posts. This function expects the ``'text_field'``
        and ``author`` columns to exist in the dataframe.
    text_field : str
        The name of the text field in the DataFrame to be filtered.

    Returns
    -------
    valid_text_mask : pd.Series
        A boolean Series indicating which rows in the DataFrame are valid.

    """
    # Stringify the 'text_field' column
    text_field_values = reddit_data[text_field].astype("string").fillna("")

    # Replace HTML entities with their corresponding characters
    normalized_text = (
        text_field_values.str.replace("&gt;", ">", regex=False)
        .str.replace("&lt;", "<", regex=False)
        .str.replace("&amp;", "&", regex=False)
    )

    # Replace runs of whitespace with a single space
    normalized_text = normalized_text.str.replace(r"[ \t\r\n]+", " ", regex=True)

    # Strip leading and trailing whitespace
    trimmed_text = normalized_text.str.strip()

    # Require non-empty text that is not one of the sentinel strings
    has_text = trimmed_text.str.len() > 0
    is_deleted = trimmed_text.isin(["[deleted]", "[removed]"])
    valid_text_mask = has_text & ~is_deleted

    # Filter out bot-like author names
    author_lower = reddit_data["author"].astype(str).str.lower()
    is_automod = author_lower.eq("automoderator")
    looks_like_bot = author_lower.str.contains(
        r"(?:^|[_-])bot\d*$", regex=True, na=False
    )
    is_bot_author = is_automod | looks_like_bot
    valid_text_mask &= ~is_bot_author

    # Drop URLs (markdown links, bare http(s), bare www) from the full text
    cleaned_text = trimmed_text.str.replace(
        pat=r"""
        \[([^\]]+)\]\(\s*(?:https?://|www\.)\S+\s*\)   # [label](url) -> keep label
        | https?://\S+                                  # bare http/https -> drop
        | \bwww\.\S+                                    # bare www.*      -> drop
        """,
        repl=r"\1",  # Keep the first match (the markdown label), drop the rest
        regex=True,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    # Look at the first 2048 characters of the cleaned text for a >=3-letter token
    preview = cleaned_text.str.slice(0, 2048)
    has_long_token = preview.str.contains(
        r"[^\W\d_]{3,}", regex=True, na=False, flags=re.UNICODE
    )
    valid_text_mask &= has_long_token

    # Keep posts/comments where at least 25% of the characters are word characters
    total_length = cleaned_text.str.len().fillna(0)
    letter_counts = cleaned_text.str.count(r"[^\W\d_]", flags=re.UNICODE).fillna(0)
    ratio_ok = (letter_counts * 4) >= total_length  # >=25% word characters

    return valid_text_mask & ratio_ok


def get_context_post_df(
    submissions: pd.DataFrame, comments: pd.DataFrame
) -> pd.DataFrame:
    """Join submissions and comments DataFrames to create a context post DataFrame.

    Parameters
    ----------
    submissions : pd.DataFrame
        DataFrame containing submission data with columns including:
        - id
        - subreddit
        - title
        - selftext
        - author
        - score
        - created_utc
        - permalink
    comments : pd.DataFrame
        DataFrame containing comment data with columns including:
        - id
        - link_id
        - body
        - author
        - score
        - created_utc
        - permalink

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

    submissions["post_id"] = (
        submissions.get("id", pd.Series(index=submissions.index, dtype="object"))
        .astype("string")
        .str.lower()
    )
    submissions = submissions[submissions["post_id"].notna()].copy()

    submissions["date_created"] = pd.to_datetime(
        submissions["created_utc"], unit="s", errors="coerce"
    ).dt.strftime("%B %d, %Y")

    # Normalize the submission permalink
    if "permalink" not in submissions.columns:
        submissions["permalink"] = pd.NA
    missing_submission_permalink = submissions["permalink"].isna() | (
        submissions["permalink"].astype("string").str.len() == 0
    )
    submissions.loc[missing_submission_permalink, "permalink"] = (
        _build_submission_permalink_series(
            submissions.loc[missing_submission_permalink].get("subreddit", ""),
            submissions.loc[missing_submission_permalink, "post_id"],
        )
    )
    submissions = submissions.rename(columns={"permalink": "submission_permalink"})

    comments = comments.copy()

    if "link_id" in comments.columns:
        comments["post_id"] = (
            comments["link_id"]
            .astype("string")
            .str.lower()
            .str.replace(r"^t3_", "", regex=True)
        )
    else:
        comments["post_id"] = pd.Series(pd.NA, index=comments.index, dtype="object")
    comments = comments[comments["post_id"].notna()].copy()

    comments["date_created"] = pd.to_datetime(
        comments["created_utc"], unit="s", errors="coerce"
    ).dt.strftime("%B %d, %Y")

    # Normalize the comment permalink
    if "permalink" not in comments.columns:
        comments["permalink"] = pd.NA

    need_comment_permalink = comments["permalink"].isna() | (
        comments["permalink"].astype("string").str.len() == 0
    )
    if "id" in comments.columns:
        has_ids = need_comment_permalink & comments["id"].notna()
        comments.loc[has_ids, "permalink"] = _build_comment_permalink_series(
            comments.loc[has_ids, "post_id"], comments.loc[has_ids, "id"]
        )

    # Map post_id to submission author
    author_map = submissions.set_index("post_id")["author"]
    comments["is_author_reply"] = comments["post_id"].map(author_map).fillna("").astype(
        "string"
    ) == comments.get("author", "").astype("string").fillna("")

    # Gather author's replies per post (as list of strings)
    author_replies = (
        comments.loc[comments["is_author_reply"], ["post_id", "body"]]
        .assign(body=lambda df: df["body"].astype("string"))
        .groupby("post_id", sort=False)["body"]
        .agg(list)
        .rename("author_replies")
    )

    # Merge author replies into submissions, filling non replies with empty list
    submissions = submissions.merge(
        author_replies, left_on="post_id", right_index=True, how="left"
    ).reset_index(drop=True)
    submissions["author_replies"] = submissions["author_replies"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Build 'report_text' column, appending author replies when present
    submissions["report_text"] = (
        submissions.get("selftext", "").astype("string").fillna("")
    )
    has_replies = submissions["author_replies"].str.len().gt(0)

    if has_replies.any():
        quoted_suffix = (
            submissions.loc[has_replies, "author_replies"]
            .apply(
                lambda replies: "\n\nThe original poster also replied with the following comments in the thread:"
                + "".join("\n> " + str(reply) for reply in replies)
            )
            .astype("string")
        )
        submissions.loc[has_replies, "report_text"] = submissions.loc[
            has_replies, "report_text"
        ].str.cat(quoted_suffix, na_rep="")

    # Build output rows for each submission
    output_submissions = pd.DataFrame(
        {
            "subreddit": submissions.get(
                "subreddit", pd.Series(index=submissions.index, dtype="object")
            ),
            "title": submissions.get(
                "title", pd.Series(index=submissions.index, dtype="object")
            ),
            "initial_post": "",
            "report_text": submissions["report_text"],
            "report_type": "submission",
            "score": pd.to_numeric(submissions.get("score", 0), errors="coerce")
            .fillna(0)
            .astype(int),
            "date_created": submissions["date_created"],
            "permalink": submissions["submission_permalink"],
            "author_replies": submissions["author_replies"],
        }
    )

    # Build output rows per non-author comment, joined to submission context
    non_author = comments.loc[~comments["is_author_reply"]].copy()
    ctx = submissions[["post_id", "subreddit", "title", "report_text"]]
    non_author = non_author.merge(ctx, on="post_id", how="left", suffixes=("", "_sub"))

    output_comments = pd.DataFrame(
        {
            "subreddit": non_author["subreddit"],
            "title": non_author["title"],
            "initial_post": non_author["report_text"],
            "report_text": non_author.get("body", "").astype("string"),
            "report_type": "comment",
            "score": pd.to_numeric(non_author.get("score", 0), errors="coerce")
            .fillna(0)
            .astype(int),
            "date_created": non_author["date_created"],
            "permalink": non_author["permalink"],
            "author_replies": [[] for _ in range(len(non_author))],
        }
    )

    return pd.concat([output_submissions, output_comments], ignore_index=True)


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


def _read_lines_zst(file_name: str) -> Generator[tuple[str, int], None, None]:
    """Yield lines from a zstandard compressed file.

    Parameters
    ----------
    file_name : str
        Path to a ``.zst`` file.

    Yields
    ------
    tuple[str, int]
        A tuple of the decoded line and the current file position.
    """
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
    """Read and decode a chunk from the zstandard stream.

    Parameters
    ----------
    reader : zstandard.ZstdDecompressionReader
        Open decompression reader.
    chunk_size : int
        Number of bytes to read in each chunk.
    max_window_size : int
        Maximum total bytes to attempt to decode before giving up.
    previous_chunk : bytes | None, optional
        Previous undecoded bytes to prepend to the next chunk.
    bytes_read : int, default=0
        Running total of bytes read so far, used for error messages.

    Returns
    -------
    str
        Decoded text for the current chunk or, on partial failures, for the
        concatenation of previous and current chunks.

    Raises
    ------
    UnicodeError
        If decoding fails after reading more than ``max_window_size`` bytes.
    """
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


def _build_submission_permalink_series(
    subreddit: pd.Series, post_id: pd.Series
) -> pd.Series:
    """Vectorized builder for submission permalinks.

    Subreddit may be empty.
    """
    sub = subreddit.astype("string").fillna("")
    pid = post_id.astype("string")
    with_sub = "/r/" + sub + "/comments/" + pid + "/"
    without_sub = "/comments/" + pid + "/"
    return pd.Series(
        np.where(sub.ne(""), with_sub, without_sub), index=post_id.index, dtype="string"
    )


def _build_comment_permalink_series(
    post_id: pd.Series, comment_id: pd.Series
) -> pd.Series:
    """Vectorized builder for absolute comment URLs.

    No title slug needed.
    """
    return (
        "https://www.reddit.com/comments/"
        + post_id.astype("string")
        + "/_/"
        + comment_id.astype("string")
    ).astype("string")
