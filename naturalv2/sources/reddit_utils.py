import asyncio
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

from naturalv2.models.lm import LM, get_message_content
from naturalv2.sources.anonymizer import Anonymizer


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

logger = logging.getLogger(__name__)


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
) -> None:
    if data_type not in ["submissions", "comments"]:
        raise ValueError(
            "Expected data_type to be 'submissions' or 'comments', but got {}".format(
                data_type
            )
        )

    os.makedirs(data_path, exist_ok=True)

    _ = wget.download(
        "https://the-eye.eu/redarcs/files/{}_{}.zst".format(subreddit, data_type),
        out=data_path,
    )

    file_path = os.path.join(data_path, "{}_{}.zst".format(subreddit, data_type))
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

    save_path = os.path.join(data_path, "{}_{}.csv".format(subreddit, data_type))
    df = pd.DataFrame(data, copy=False).convert_dtypes(dtype_backend="pyarrow")

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
            cols=[
                "author_flair_text",
                "author_flair_richtext",
                "author_fullname",
                "author_patreon_flair",
                "awarders",
                "banned_by",
                "link_flair_text",
                "link_flair_richtext",
                "removal_reason",
                "removed_by",
                "selftext",
                "title",
            ]
            if data_type == "submissions"
            else [
                "author_flair_text",
                "author_flair_richtext",
                "author_fullname",
                "author_patreon_flair",
                "body",
            ],
        )

    df.to_csv(save_path)
    os.remove(file_path)
    logger.info(
        f"Completed download of {subreddit} {data_type} data with: {file_lines:,} lines "
        f"({bad_lines:,} bad lines)"
    )


def download_subs_list(data_path: str) -> None:
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


def is_rate_limit_error(exception):
    return isinstance(exception, error.HTTPError) and exception.code == 429


def fallback_return(retry_state):
    """Returns a dictionary with an error message and the URL that caused the error."""
    return {"error": "retry limit exceeded", "url": retry_state.args[0]}


@retry(
    wait=wait_exponential(multiplier=2.4, min=60, max=120),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(is_rate_limit_error),
    retry_error_callback=fallback_return,
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def download_from_url(url_str: str) -> dict[str, Any]:
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


def _get_sub_desc(data: dict[str, Any]) -> Optional[dict[str, str]]:
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


def _fetch_sub_about(data_path: str, sub: str) -> Optional[dict[str, str]]:
    about_file = os.path.join(data_path, "subs_about", f"{sub}_about.json")
    if os.path.exists(about_file):
        with open(about_file, "r") as f:
            data = json.load(f)
    else:
        about_url = f"https://www.reddit.com/r/{sub}/about.json"
        data = download_from_url(about_url)

        with open(about_file, "w") as f:
            json.dump(data, f)

    return _get_sub_desc(data)


def get_sub_about_info(data_path: str) -> pd.DataFrame:
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

        return pd.DataFrame(
            rows, columns=["sub", "description", "public_description"], copy=False
        ).convert_dtypes(dtype_backend="pyarrow")

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
                "All subreddits already downloaded. Loading existing about CSV file."
            )
            return pd.read_csv(about_csv_path, index_col=0)

        logger.warning(
            "All subreddits already downloaded, but no CSV file found. "
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
        about_df.to_csv(about_csv_path)

    return about_df


def get_submission_permalink(permalink: str) -> str:
    return "/" + permalink.split("/")[-2] + "/"


def get_comment_permalink(permalink: str) -> str:
    return "/" + permalink.split("/")[-3] + "/"


def date_filter(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    utc_date_cutoff = int(date_obj.replace(tzinfo=datetime.timezone.utc).timestamp())

    idx = df["date_created"].apply(lambda x: get_utc_timestamp(x) <= utc_date_cutoff)
    return df[idx]


def get_date(utc_timestamp: float) -> str:
    dt = datetime.datetime.fromtimestamp(utc_timestamp, tz=datetime.timezone.utc)
    return dt.strftime("%B %d, %Y")


def get_utc_timestamp(date_str: str) -> int:
    dt = datetime.datetime.strptime(date_str, "%B %d, %Y")
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp())


def rule_based_filter(post_df: pd.DataFrame, text_field: str) -> pd.DataFrame:
    # remove rows where the text field is not of type str
    idx = post_df[text_field].apply(lambda x: isinstance(x, str))
    post_df = post_df.loc[idx]
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
        body = row[text_field]
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


# def check_treatment_mention(lst, treatment_names):
#     filtered_lst = []
#     for elem in lst:
#         matches = [x for x in treatment_names if x in elem["subreddit"].lower()]
#         matches += [x for x in treatment_names if x in elem["title"].lower()]
#         matches += [x for x in treatment_names if x in elem["post"].lower()]
#         if "initial_post" in list(elem.keys()):
#             matches += [x for x in treatment_names if x in elem["initial_post"].lower()]
#         matches = set(matches)
#         if len(matches) > 0:
#             elem["treatments"] = list(matches)
#             filtered_lst.append(elem)
#     return filtered_lst


# def check_outcome_mention(lst, outcome_words):
#     filtered_lst = []
#     for elem in lst:
#         matches = [x for x in outcome_words if x in elem["subreddit"].lower()]
#         matches += [x for x in outcome_words if x in elem["title"].lower()]
#         matches += [x for x in outcome_words if x in elem["post"].lower()]
#         if "initial_post" in list(elem.keys()):
#             matches += [x for x in outcome_words if x in elem["initial_post"].lower()]
#         matches = set(matches)
#         if len(matches) > 0:
#             elem["outcome_words"] = list(matches)
#             filtered_lst.append(elem)
#     return filtered_lst


def get_context_post_df(
    submissions: pd.DataFrame, comments: pd.DataFrame
) -> pd.DataFrame:
    merged_df = pd.DataFrame(
        columns=[
            "subreddit",
            "title",
            "initial_post",
            "post",
            "score",
            "date_created",
            "permalink",
            "treatments",
            "outcome_words",
            "author_replies",
        ]
    )
    comments["permalink_processed"] = comments["permalink"].map(
        lambda x: get_comment_permalink(x)
    )
    for _, submission in submissions.iterrows():
        subreddit = submission["subreddit"]
        title = submission["title"]
        submission_text = submission["selftext"]
        score = int(submission["score"])
        created_utc = int(submission["created_utc"])
        date_created = get_date(created_utc)
        submission_permalink = get_submission_permalink(submission["permalink"])
        submission_comments = comments[
            comments["permalink_processed"] == submission_permalink
        ]
        submission_author_comments = submission_comments[
            submission_comments["author"] == submission["author"]
        ]
        submission_comments = submission_comments.drop(submission_author_comments.index)
        if len(submission_author_comments["body"].to_list()) > 0:
            submission_text += (
                "\n\nThe author also replied with the following in the thread:"
            )
            for reply in submission_author_comments["body"].to_list():
                submission_text += "\n> " + reply
        to_append = [
            {
                "subreddit": subreddit,
                "title": title,
                "initial_post": "",
                "post": submission_text,
                "score": score,
                "date_created": date_created,
                "permalink": submission_permalink,
                "author_replies": submission_author_comments["body"].to_list(),
            }
        ]
        to_append += [
            {
                "subreddit": subreddit,
                "title": title,
                "initial_post": submission_text,
                "post": str(submission_comments.iloc[j]["body"]),
                "score": str(submission_comments.iloc[j]["score"]),
                "date_created": get_date(
                    int(submission_comments.iloc[j]["created_utc"])
                ),
                "permalink": submission_comments.iloc[j]["permalink"],
                "author_replies": [],
            }
            for j in range(len(submission_comments))
        ]
        # to_append = check_treatment_mention(to_append, treatment_names)
        # to_append = check_outcome_mention(to_append, outcome_words)
        if len(to_append) > 0:
            df_to_append = pd.DataFrame.from_dict(to_append)
            merged_df = pd.concat([merged_df, df_to_append], ignore_index=True)
    return merged_df


def get_reddit_synonyms(keywords: str, lm: LM) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical assistant who can translate medical terminology into common terms.",
        },
        {
            "role": "user",
            "content": f"""
            What are common brand names or terms that people specifically use when discussing any of {str(keywords)},
            especially on platforms like Reddit? Return only a Python list of at most 10 individual words, without any other text or formatting.
            """,
        },
    ]
    response = asyncio.run(lm(messages=messages))
    llm_keywords = get_message_content(response)
    all_keywords = []

    # process outputs into a list of individual words
    for keyword in llm_keywords:
        processed_keyword = (
            keyword[keyword.find("[") + 1 : keyword.find("]")]
            .replace('"', "")
            .replace("'", "")
        )
        keyword_list = [k.strip() for k in processed_keyword.split(",")]
        all_keywords.extend(keyword_list)

    logger.info(f"Reddit keywords: {all_keywords}")
    return [k.lower() for k in all_keywords]


def subreddit_relevance_llm(desc: str, keywords: str, lm: LM) -> str:
    system_prompt = "You are an expert in analyzing online forums for relevant discussions about clinical conditions."
    system_prompt += "Your task is to determine if a subreddit is likely to contain personal experiences related to a set of related conditions based on the subreddit description."

    user_prompt = "Evaluate the subreddit relevance to the conditions listed. Consider if the subreddit likely contains personal experiences with the condition. Answer Yes if relevant, No if not."
    user_prompt += (
        f"\n\n**Conditions:** {keywords}\n\n**Subreddit Description:** {desc}"
    )
    user_prompt += "Is the subreddit likely to contain personal experiences relevant to the listed conditions? Answer with Yes or No."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = asyncio.run(lm(messages=messages))
    return get_message_content(response)[0]
