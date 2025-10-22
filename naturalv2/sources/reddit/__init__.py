"""Reddit source package.

This package contains stages and utilities for curating Reddit data:

- Stages: high-level pipeline components that discover relevant subreddits,
  download/clean data, and curate per-experiment CSVs.
- Operations: focused helpers for Reddit search, download and cleaning.
- Utils: archive access, rule-based filtering and subreddit metadata fetching.
"""

from .operations import (
    clean_subreddit_data,
    download_submissions_and_comments,
    get_study_relevant_posts,
    search_posts_in_subreddit,
    search_subreddits,
)
from .stages import RedditConditionFilter, RedditCurateStage, RedditDownloadAndClean
from .utils import apply_rule_based_filter, download_subs_list, get_sub_about_info
