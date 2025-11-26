"""Condition filter stage."""

import ast
import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

import asyncpraw
import pandas as pd
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.logging import logging_redirect_tqdm

from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.components.llm_extraction import (
    ExtractType,
    extract_curation_info,
)
from naturalv2.sources.core import SourceStage
from naturalv2.sources.reddit.api import search_posts_in_subreddit, search_subreddits


if TYPE_CHECKING:
    from naturalv2.models.lm import APIModel
    from naturalv2.sources.core import CurationContext, StageState


logger = logging.getLogger(__name__)


class RedditConditionFilter(SourceStage):
    """Identify relevant subreddits for trial conditions via LLM assistance.

    This stage first collects candidate subreddits and post snippets using the
    Reddit API, then prompts an LLM to filter those candidates down to the most
    relevant subreddits per condition.

    Parameters
    ----------
    llm : APIModel
        Language model client used to filter candidate subreddits.
    llm_max_concurrency : int, default=10
        Maximum number of concurrent LLM requests.
    reddit_rpm : int, default=10
        Rate limit for Reddit API requests (requests per minute).
    subreddit_post_limit : int, default=5
        Maximum number of posts to fetch per subreddit during search.
    subreddit_post_char_limit : int, default=1000
        Maximum number of characters to include per post body snippet.
    name : str | None, optional
        Optional explicit stage name; defaults to the class name.
    """

    def __init__(
        self,
        *,
        llm: "APIModel",
        llm_max_concurrency: int = 10,
        reddit_rpm: int = 10,
        subreddit_post_limit: int = 5,
        subreddit_post_char_limit: int = 1000,
        name: str | None = None,
    ) -> None:
        """Initialize the stage."""
        super().__init__(name=name)

        self.llm = llm
        self.reddit_rpm = reddit_rpm
        self.llm_max_concurrency = llm_max_concurrency
        self.subreddit_post_limit = subreddit_post_limit
        self.subreddit_post_char_limit = subreddit_post_char_limit

    async def run(
        self, context: "CurationContext", state: "StageState"
    ) -> "StageState":
        """Execute subreddit discovery and LLM filtering for conditions.

        Parameters
        ----------
        context : CurationContext
            Pipeline context with experiments, source name and save directories.
        state : StageState
            Mutable pipeline state; updated with condition-to-subreddit mapping
            and summary metadata.

        Returns
        -------
        StageState
            Updated state containing the mapping and counts.
        """
        trial_conditions: list[str] = []
        for experiment in context.experiments:
            if experiment.conditions:
                trial_conditions.extend(experiment.conditions)
        trial_conditions = sorted(dict.fromkeys(trial_conditions))

        # Get existing mapping if available
        condition_to_subreddit_map = context.study_dataset.sources.get(
            context.source_name, {}
        )
        # Flatten existing subreddits list for metadata
        relevant_subreddits_list = [
            sub
            for subs in condition_to_subreddit_map.values()
            for sub in subs
            if isinstance(subs, list)
        ]

        # Update state with existing mapping
        state.payload = condition_to_subreddit_map
        state.update(
            condition_to_subreddit_map=condition_to_subreddit_map,
            num_unique_subreddits=len(set(relevant_subreddits_list)),
        )

        # Add prompt template to metadata for logging
        prompt_id = f"{ExtractType.CONDITION.value}_{context.source_name}"
        template = load_prompt(
            base_dir="naturalv2/prompts/templates",
            prompt_type=prompt_id,
            return_format="prompt",
        )
        state.metadata.setdefault("prompt_templates", {})[prompt_id] = template

        # Skip keywords that have already been processed
        trial_conditions = [
            cond for cond in trial_conditions if cond not in condition_to_subreddit_map
        ]
        if not trial_conditions:
            logger.warning("%s: no trial conditions to process", self.stage_name)
            return state

        # Search Reddit for candidate subreddits and posts
        with logging_redirect_tqdm():
            candidate_subs_and_posts = await self._collect_candidate_subs_and_posts(
                keywords=trial_conditions
            )

        # Collect results for DataFrame
        keyword_queries: list[dict[str, str | list[str]]] = []
        for keyword in trial_conditions:
            if keyword in candidate_subs_and_posts:
                result = candidate_subs_and_posts[keyword]
                if not (result["subreddit_posts"] or result["candidate_subs"]):
                    continue

                # Stringify subreddit posts for LLM input
                llm_input = json.dumps(result["subreddit_posts"], indent=4)
                keyword_queries.append(
                    {
                        "keyword": keyword,
                        "candidate_subs": result["candidate_subs"],
                        "input_data": llm_input,
                    }
                )

        logger.info(
            "%s: found candidate subreddits for %d out of %d keywords",
            self.stage_name,
            len(keyword_queries),
            len(trial_conditions),
        )

        if not keyword_queries:
            logger.warning(
                "%s: no candidate subreddits found for any keywords", self.stage_name
            )
            return state

        df = pd.DataFrame(keyword_queries)

        save_dir = self.results_dir(context)
        file_path = os.path.join(
            save_dir,
            f"{context.source_name}_condition_queries_{context.experiment_name}.csv",
        )

        with logging_redirect_tqdm():
            output_df = await extract_curation_info(
                df,
                stage_name=self.stage_name,
                source_name=context.source_name,
                extract_type=ExtractType.CONDITION,
                llm=self.llm,
                file_path=file_path,
                token_tracker=context._token_tracker,
                max_concurrent_requests=self.llm_max_concurrency,
            )

        output_df["llm_output"] = output_df["llm_output"].fillna("[]")

        for keyword, output in zip(output_df["keyword"], output_df["llm_output"]):
            llm_filtered_subreddits: list[str] = ast.literal_eval(output)
            condition_to_subreddit_map[keyword] = llm_filtered_subreddits
            relevant_subreddits_list.extend(llm_filtered_subreddits)

        num_unique_subreddits = len(list(set(relevant_subreddits_list)))

        # Update state with new mapping
        state.payload = condition_to_subreddit_map
        state.update(
            condition_to_subreddit_map=condition_to_subreddit_map,
            num_unique_subreddits=num_unique_subreddits,
        )

        # Update and persist metadata in StudyDataset
        context.study_dataset.sources[context.source_name] = condition_to_subreddit_map
        context.study_dataset.to_yaml(context.extras["study_dataset_path"])

        logger.info(
            "%s: mapped %d trial conditions to %d unique subreddits",
            self.stage_name,
            len(condition_to_subreddit_map),
            num_unique_subreddits,
        )
        context._token_tracker.log_table()

        return state

    async def _collect_candidate_subs_and_posts(
        self, keywords: list[str]
    ) -> dict[str, dict[str, list[str] | dict[str, str | list[str]]]]:
        """Search Reddit for candidate subreddits and fetch post snippets.

        Parameters
        ----------
        keywords : list[str]
            Trial condition keywords to search for.

        Returns
        -------
        dict
            Mapping from keyword to a dict with keys ``candidate_subs`` and
            ``subreddit_posts`` (a list of subreddit → posts).
        """
        logger.info(
            "Getting candidate subreddits and posts for %d keywords.", len(keywords)
        )

        async with asyncpraw.Reddit(
            client_id=os.environ.get("PRAW_CLIENT_ID"),
            client_secret=os.environ.get("PRAW_CLIENT_SECRET"),
            password=os.environ.get("PRAW_PWD"),
            username=os.environ.get("PRAW_USERNAME"),
            user_agent=os.environ.get("PRAW_AGENT"),
        ) as reddit_client:
            reddit_rate_limiter = AsyncLimiter(self.reddit_rpm)

            async def _search(
                keyword: str,
            ) -> tuple[str, list[str] | Exception]:
                """Search for subreddits with a keyword."""
                try:
                    subs = await search_subreddits(
                        keyword, reddit_client, reddit_rate_limiter
                    )
                    return keyword, subs
                except Exception as exc:  # noqa: BLE001
                    return keyword, exc

            tasks = [asyncio.create_task(_search(keyword)) for keyword in keywords]

            candidate_subs_per_keyword: dict[str, list[str]] = {}
            for fut in tqdm_asyncio.as_completed(
                tasks,
                desc="Searching subreddits",
                total=len(tasks),
                leave=False,
                dynamic_ncols=True,
            ):
                keyword, result = await fut
                if isinstance(result, Exception):
                    logger.error(
                        "Searching for subreddits with keyword '%s' failed with error: %s",
                        keyword,
                        result,
                    )
                else:
                    candidate_subs_per_keyword[keyword] = result
                    logger.debug(
                        "Found %d candidate subreddits for keyword '%s'",
                        len(result),
                        keyword,
                    )

            async def _fetch_posts(
                keyword: str, subreddit: str
            ) -> tuple[str, str, list[str] | Exception]:
                """Fetch posts for a keyword/subreddit pair and surface errors."""
                try:
                    posts = await search_posts_in_subreddit(
                        subreddit,
                        keyword,
                        reddit_client,
                        reddit_rate_limiter,
                        limit=self.subreddit_post_limit,
                        char_limit=self.subreddit_post_char_limit,
                    )
                    return keyword, subreddit, posts
                except Exception as exc:  # noqa: BLE001
                    return keyword, subreddit, exc

            post_search_tasks: list[
                asyncio.Task[tuple[str, str, list[str] | Exception]]
            ] = []
            for keyword, candidate_subs in candidate_subs_per_keyword.items():
                for subreddit in candidate_subs:
                    post_search_tasks.append(
                        asyncio.create_task(_fetch_posts(keyword, subreddit))
                    )

            results_by_keyword: dict[
                str, dict[str, list[str] | dict[str, str | list[str]]]
            ] = {
                keyword: {
                    "candidate_subs": candidate_subs,
                    "subreddit_posts": [],
                }
                for keyword, candidate_subs in candidate_subs_per_keyword.items()
            }

            if not post_search_tasks:
                return results_by_keyword

            for fut in tqdm_asyncio.as_completed(
                post_search_tasks,
                desc="Searching posts",
                total=len(post_search_tasks),
                leave=False,
                dynamic_ncols=True,
            ):
                keyword, subreddit, posts = await fut
                if isinstance(posts, Exception):
                    logger.error(
                        "Could not fetch posts for subreddit '%s' and keyword '%s': %s",
                        subreddit,
                        keyword,
                        posts,
                    )
                else:
                    results_by_keyword[keyword]["subreddit_posts"].append(
                        {"Subreddit": subreddit, "Example Posts": posts}
                    )

        return results_by_keyword
