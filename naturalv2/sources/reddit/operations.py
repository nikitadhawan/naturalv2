"""Reddit data operations used by curation stages and adapters.

This module contains asynchronous search utilities powered by PRAW as well as
batch-oriented helpers to download, clean, and select study-relevant Reddit
posts.
"""

import logging
import os

from naturalv2.sources.anonymizer import Anonymizer
from naturalv2.sources.reddit.processing import clean_subreddit_data
from naturalv2.sources.reddit.utils import download_sub_data


logger = logging.getLogger(__name__)


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
        logger.error("Error processing subreddit %s: %s", subreddit, exc, exc_info=True)
        return None, subreddit

    return clean_sub_path, subreddit
