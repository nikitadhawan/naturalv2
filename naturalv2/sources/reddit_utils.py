"""Utilities for downloading and processing Reddit data from The Eye archive."""

import datetime
import glob
import json
import logging
import os
import time
import warnings
from functools import partial
from typing import Any, Generator, Literal, Optional
from urllib import error, request

import pandas as pd
import tenacity
import wget
import zstandard
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.contrib.concurrent import process_map

from naturalv2.sources.anonymizer import Anonymizer


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

logger = logging.getLogger(__name__)


def _get_sub_desc(data: dict[str, Any]) -> Optional[dict[str, str]]:
    """Extract subreddit description and public description from the data."""
    if "data" in data:
        descr = data["data"].get("description", "")
        public_descr = data["data"].get("public_description", "")
        sub = data["data"].get("display_name", "")
        return {
            "sub": sub,
            "description": descr,
            "public_description": public_descr,
        }
    return None


def _is_rate_limit_error(exception: Exception) -> bool:
    """Check if the exception is a rate limit error (HTTP 429)."""
    return isinstance(exception, error.HTTPError) and exception.code == 429


def _fallback_return(retry_state: "tenacity.RetryCallState") -> dict[str, Any]:
    """Returns a dictionary with an error message and the URL that caused the error."""
    return {"error": "retry limit exceeded", "url": retry_state.args[0]}


@retry(
    wait=wait_exponential(multiplier=2.4, min=60, max=120),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_is_rate_limit_error),
    retry_error_callback=_fallback_return,
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def _download_from_url(url_str: str) -> dict[str, Any]:
    """Download JSON data from a URL with error handling and rate limiting.

    Parameters
    ----------
    url_str : str
        The URL to download the JSON data from.

    Returns
    -------
    dict[str, Any]
        The JSON data as a dictionary, or an error message if the download fails.

    Raises
    ------
    tenacity.RetryError
        If the download fails after the maximum number of retries.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = request.Request(url_str, headers=headers)
        data = json.load(request.urlopen(req))
        time.sleep(2)
        return data
    except error.HTTPError as e:
        if e.code == 429:
            raise  # tenacity will handle rate limit errors

        logger.error(f"HTTP error {e.code} for {url_str}: {e.reason}")
        return {"error": f"HTTP error {e.code}", "url": url_str}
    except Exception as e:
        logger.error(f"Unexpected error fetching {url_str}: {e}")
        return {"error": str(e), "url": url_str}


def _fetch_sub_about(data_path: str, sub: str) -> Optional[dict[str, str]]:
    """Fetch subreddit about information from Reddit API or local JSON file."""
    about_file = os.path.join(data_path, "subs_about", f"{sub}_about.json")
    if os.path.exists(about_file):
        with open(about_file, "r") as f:
            data = json.load(f)
    else:
        about_url = f"https://www.reddit.com/r/{sub}/about.json"
        data = _download_from_url(about_url)

        with open(about_file, "w") as f:
            json.dump(data, f)

    return _get_sub_desc(data)


def download_subs_list(data_path: str) -> None:
    """Download the list of subreddits from The Eye archive.

    Parameters
    ----------
    data_path : str
        The path where the list of subreddits will be saved.

    """
    filepath = os.path.join(data_path, "subs_list.txt")
    if not os.path.exists(filepath):
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


def get_sub_about_info(data_path: str) -> pd.DataFrame:
    """Fetch subreddit about information and create a DataFrame.

    This function checks if the list of subreddits exists, downloads it if not,
    and then fetches the about information for each subreddit. It saves the
    information in a CSV file and returns a DataFrame containing the subreddit
    names, descriptions, and public descriptions.

    Parameters
    ----------
    data_path : str
        The path where the subreddit about information will be saved.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the subreddit names, descriptions, and public
        descriptions.

    """
    subs_list_path = os.path.join(data_path, "subs_list.txt")
    if not os.path.exists(subs_list_path):
        logger.info("Subreddit list not found. Downloading the list of subreddits.")
        download_subs_list(data_path)

    with open(subs_list_path, "r") as f:
        subs_list = f.read().splitlines()

    about_jsons_dir = os.path.join(data_path, "subs_about")
    os.makedirs(about_jsons_dir, exist_ok=True)

    about_csv_path = os.path.join(data_path, "subs_about.csv")

    # construct the about_df from the existing JSON files
    def _create_about_csv_from_json_files():
        all_json_files = glob.glob(os.path.join(about_jsons_dir, "*.json"))
        rows = []
        for json_file in all_json_files:
            with open(json_file, "r") as f:
                data: dict[str, Any] = json.load(f)
                row_data = _get_sub_desc(data)
                if row_data is not None:
                    rows.append(row_data)

        df = pd.DataFrame(
            rows, columns=["sub", "description", "public_description"], copy=False
        ).convert_dtypes(dtype_backend="pyarrow")
        df.drop_duplicates(subset=["sub"], inplace=True)
        df.sort_values(by="sub", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # filter out already downloaded subreddits from subs_list
    # NOTE: A json file is always saved, regardless of whether there was an error
    # while fetching the subreddit about info or not. But, subreddits that
    # had an error during downloading will not be added to the CSV file.
    downloaded_subs = [
        os.path.splitext(file)[0].split("_")[0] for file in os.listdir(about_jsons_dir)
    ]
    subs_list = list(set(subs_list) - set(downloaded_subs))

    if len(subs_list) == 0:
        if os.path.exists(about_csv_path):
            logger.info(
                "All subreddits' about info already downloaded. Loading existing about CSV file."
            )
            return pd.read_csv(about_csv_path, index_col=0)

        logger.warning(
            "All subreddits' about info already downloaded, but no CSV file found. "
            "Creating a new CSV file from the existing JSON files."
        )
        df = _create_about_csv_from_json_files()
        df.to_csv(about_csv_path)
        return df

    partial_df: Optional[pd.DataFrame] = None
    if os.path.exists(about_csv_path):
        partial_df = pd.read_csv(about_csv_path, index_col=0)
    elif len(downloaded_subs) > 0:  # download was interrupted before CSV creation
        partial_df = _create_about_csv_from_json_files()

    rows = []
    results = process_map(
        partial(_fetch_sub_about, data_path),
        subs_list,
        chunksize=1,
        desc="Fetching subreddit about info",
    )
    for row in results:
        if row is not None:
            rows.append(row)

    if partial_df is None:
        about_df = pd.DataFrame(
            rows, columns=["sub", "description", "public_description"], copy=False
        ).convert_dtypes(dtype_backend="pyarrow")
        about_df.to_csv(about_csv_path)
    else:  # append new rows to the existing DataFrame
        new_df = pd.DataFrame(
            rows, columns=["sub", "description", "public_description"], copy=False
        )
        about_df = pd.concat([partial_df, new_df], ignore_index=True)
        about_df = about_df.drop_duplicates(subset=["sub"])
        about_df.sort_values(by="sub", inplace=True)
        about_df.reset_index(drop=True, inplace=True)
        about_df.to_csv(about_csv_path)

    return about_df


def _read_and_decode(
    reader: zstandard.ZstdDecompressionReader,
    chunk_size: int,
    max_window_size: int,
    previous_chunk: Optional[bytes] = None,
    bytes_read: int = 0,
) -> str:
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


def _read_lines_zst(file_name: str) -> Generator[tuple[str, int], None, None]:
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
            "Expected data_type to be 'submissions' or 'comments', but got {}".format(
                data_type
            )
        )

    os.makedirs(data_path, exist_ok=True)

    save_path = os.path.join(data_path, "{}_{}.csv".format(subreddit, data_type))
    if os.path.exists(save_path):
        logger.warning(
            f"File {save_path} already exists. Skipping download for {subreddit} {data_type}."
        )
        return

    file_path = os.path.join(data_path, "{}_{}.zst".format(subreddit, data_type))
    if not os.path.exists(file_path):
        # Go to TMPDIR if set, otherwise stay current working directory, since wget doesn't respect TMPDIR
        tmpdir = os.environ.get("TMPDIR", os.getcwd())
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            _ = wget.download(
                "https://the-eye.eu/redarcs/files/{}_{}.zst".format(
                    subreddit, data_type
                ),
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

    df = pd.DataFrame(data, copy=False)

    # remove deleted posts or comments
    df = (
        df[df["selftext"] != "[deleted]"]
        if data_type == "submissions"
        else df[df["body"] != "[deleted]"]
    )

    # anonymize dataframe
    if anonymizer_instance is not None:
        df = anonymizer_instance.anonymize_dataframe(
            df,
            cols_to_keep=["created_utc", "author", "permalink", "subreddit", "score"],
            cols_to_anonymize=["selftext", "title"]
            if data_type == "submissions"
            else ["body"],
            data_source_name=f"{subreddit}_{data_type}",
            batch_size=batch_size,
            num_workers=num_workers,
        )

    df.to_csv(save_path)
    os.remove(file_path)
    logger.info(
        f"Completed download of {subreddit} {data_type} data with: {file_lines:,} lines "
        f"({bad_lines:,} bad lines)"
    )


def _get_submission_permalink(permalink: str) -> str:
    """Extracts the submission permalink from a full Reddit permalink."""
    return "/" + permalink.split("/")[-2] + "/"


def _get_comment_permalink(permalink: str) -> str:
    """Extracts the comment permalink from a full Reddit permalink."""
    return "/" + permalink.split("/")[-3] + "/"


def _get_date(utc_timestamp: float) -> str:
    dt = datetime.datetime.fromtimestamp(utc_timestamp, tz=datetime.timezone.utc)
    return dt.strftime("%B %d, %Y")


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
    post_df = post_df.loc[post_df["score"] != None]

    # remove rows where the submission is deleted or removed
    post_df = post_df.loc[post_df[text_field] != "[deleted]"]
    post_df = post_df.loc[post_df[text_field] != "[removed]"]

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


def _get_context_post_df(
    submissions: pd.DataFrame, comments: pd.DataFrame
) -> pd.DataFrame:
    """Join submissions and comments DataFrames to create a context post DataFrame."""
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
                "score": other_comments["score"].astype(str).tolist(),
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
