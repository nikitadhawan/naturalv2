"""Reddit source integration for the Natural pipeline.

This package exposes:
- Pipeline stages that discover candidate subreddits, download/clean data, and
  curate study-specific datasets.
- Async Reddit API helpers for searching subreddits and posts with retries.
- Pushshift archive ingestion utilities plus Arrow-based processing/filtering
  used to contextualize records and export curated parquet/CSV outputs.
"""

from .stages import RedditConditionFilter, RedditCurateStage, RedditDownloadAndClean
