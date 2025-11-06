"""Reddit source package.

This package contains stages and utilities for curating Reddit data:

- Stages: high-level pipeline components that discover relevant subreddits,
  download/clean data, and curate per-experiment CSVs.
- Operations: focused helpers for Reddit search, download and cleaning.
- Utils: archive access, rule-based filtering and subreddit metadata fetching.
"""

from .stages import RedditConditionFilter, RedditCurateStage, RedditDownloadAndClean
