import logging
import os
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from ahocorasick import Automaton

from naturalv2.sources.components import filter_by_date
from naturalv2.sources.components.helpers import (
    extract_mentions,
    normalize_text_for_matching,
)


logger = logging.getLogger(__name__)

SENTINELS = pa.array(["[deleted]", "[removed]"], type=pa.string())


def apply_rule_based_filter(table: pa.Table, text_field: str) -> pa.ChunkedArray:
    """Construct a boolean mask that keeps rows containing meaningful human-written text.

    Parameters
    ----------
    table : pa.Table
        Arrow table containing Reddit submission or comment data.
    text_field : str
        Name of the UTF-8 column in ``table`` holding the free-text payload.

    Returns
    -------
    pa.ChunkedArray
        Chunked Boolean array indicating which rows satisfy all quality checks.

    Notes
    -----
    The resulting mask enforces multiple heuristics:
    - Normalizes the text field by filling nulls, unescaping basic HTML entities,
      replacing control characters, and trimming whitespace.
    - Rejects empty strings and records that match known sentinel content such
      as deleted or removed posts.
    - Requires a permalink and removes rows authored by obvious bot accounts.
    - Ensures the first 2,048 code units include at least one space, signalling
      multi-token text.
    - Verifies that alphabetic characters (plus spaces) make up at least half of
      the trimmed text.
    """
    empty_string = pa.scalar("", type=pa.string())

    # Normalize the text field:
    # fill nulls
    text_field_values = pc.fill_null(table[text_field], empty_string)

    # decode HTML entities
    normalized_text = pc.replace_substring(text_field_values, "&gt;", ">")
    normalized_text = pc.replace_substring(normalized_text, "&lt;", "<")
    normalized_text = pc.replace_substring(normalized_text, "&amp;", "&")

    # collapse whitespace
    normalized_text = pc.replace_substring(normalized_text, "\n", " ")
    normalized_text = pc.replace_substring(normalized_text, "\t", " ")
    trimmed_text = pc.utf8_trim_whitespace(normalized_text)

    # Require non-empty text that is not one of the known sentinel strings
    has_text = pc.greater(pc.utf8_length(trimmed_text), 0)
    matched_sentinel = pc.is_in(trimmed_text, value_set=SENTINELS)
    valid_text_mask = pc.and_(has_text, pc.invert(matched_sentinel))

    # id: present + non-empty
    id_filled = pc.fill_null(table["id"], empty_string)
    id_trimmed = pc.utf8_trim_whitespace(id_filled)
    has_id = pc.greater(pc.utf8_length(id_trimmed), 0)
    valid_text_mask = pc.and_(valid_text_mask, has_id)

    if pc.count(table["link_id"]) != pa.scalar(0):  # comments
        # link_id: starts with t3_ and has a non-empty tail
        link_filled = pc.fill_null(table["link_id"], empty_string)
        has_t3_prefix = pc.starts_with(link_filled, "t3_")
        link_tail = pc.utf8_slice_codeunits(link_filled, 3)  # drop 't3_'
        tail_nonempty = pc.greater(pc.utf8_length(link_tail), 0)
        has_t3 = pc.and_(has_t3_prefix, tail_nonempty)
        valid_text_mask = pc.and_(valid_text_mask, has_t3)

    # Filter out bot accounts based on the author name
    author_values = pc.fill_null(table["author"], pa.scalar("", pa.string()))
    author_lower = pc.ascii_lower(author_values)
    is_automod = pc.equal(author_lower, "automoderator")
    looks_like_bot = pc.match_substring_regex(author_lower, r"(^|[_-])bot\d*$")
    is_bot_author = pc.or_(is_automod, looks_like_bot)
    valid_text_mask = pc.and_(valid_text_mask, pc.invert(is_bot_author))

    # Require at least one space early in the text to catch multi-word content
    preview = pc.utf8_slice_codeunits(trimmed_text, 0, 2048)
    has_two_words = pc.greater(
        pc.count_substring_regex(preview, r"[A-Za-z]{2,}\s+[A-Za-z]{2,}"),
        0,
    )
    valid_text_mask = pc.and_(valid_text_mask, has_two_words)

    # Favour mostly readable english
    total_length = pc.utf8_length(trimmed_text)
    allowed = pc.count_substring_regex(
        trimmed_text, r"[A-Za-z0-9\s.,;:!?@#%&'\"()\-_/\\\[\]{}<>~$^|*+=]"
    )
    ratio_ok = pc.greater_equal(pc.multiply(allowed, 4), total_length)  # ≥25% allowed
    return pc.if_else(pc.and_(valid_text_mask, has_text), ratio_ok, False)


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
    submissions_mask = apply_rule_based_filter(submissions, "selftext")
    submissions = submissions[submissions_mask]
    logger.info(
        "%s: %d submissions left after rule-based filtering",
        subreddit,
        len(submissions),
    )

    comments = pd.read_parquet(os.path.join(data_path, f"{subreddit}_comments.parquet"))
    comments_mask = apply_rule_based_filter(comments, "body")
    comments = comments[comments_mask]
    logger.info(
        "%s: %d comments left after rule-based filtering",
        subreddit,
        len(comments),
    )

    return get_context_post_df(submissions, comments)


def get_study_relevant_posts(
    clean_data: Union[str, pd.DataFrame],
    treatment_automaton: Automaton,
    cutoff_dt: pd.Timestamp | None,
    date_column: str = "date_created",
) -> pd.DataFrame:
    """Select posts mentioning treatments before an optional cutoff date.

    Parameters
    ----------
    clean_data : str | pandas.DataFrame
        Either the path to a cleaned subreddit parquet file created by
        :func:`download_submissions_and_comments` or a pre-loaded DataFrame.
    treatment_automaton : ahocorasick.Automaton
        Compiled ahocorasick automaton for matching treatment aliases.
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
    if isinstance(clean_data, str):
        df = pd.read_parquet(clean_data)
        data_label = clean_data
    else:
        df = clean_data
        data_label = "provided DataFrame chunk"

    if df.empty:
        return pd.DataFrame()

    if cutoff_dt is not None:
        df = filter_by_date(df, cutoff_dt, date_column)
        if df.empty:
            return pd.DataFrame()

    text_cols = [
        col
        for col in ("report_text", "title", "initial_post", "subreddit")
        if col in df.columns
    ]
    if not text_cols:
        logger.warning(
            "No textual columns found in %s to evaluate treatment matches.",
            data_label,
        )
        return pd.DataFrame()

    reports = (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(normalize_text_for_matching)
    )

    mentions = reports.map(partial(extract_mentions, automaton=treatment_automaton))
    mask = mentions.str.len().gt(0)
    return df.loc[mask].assign(treatments_mentioned=mentions.loc[mask])


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
