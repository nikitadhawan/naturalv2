import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dateutil.parser import parse
from tqdm.asyncio import tqdm as tqdm_asyncio

from naturalv2.evals.experiment import Experiment
from naturalv2.utils import concurrency_limited, load_prompt, sanitize_filename

from .pubmed_utils import fetch_articles, search_pubmed


logger = logging.getLogger(__name__)


class PubMedSet:
    def __init__(self, data_path: str, api_key: str | None = None) -> None:
        self.data_path = data_path
        self.api_key = api_key

    async def condition_filter(self, keywords: list[str], metadata):
        self.data_files = []
        queries = []
        for keyword in keywords:
            queries.append(self._get_search_query(keyword))
            self.data_files.append(
                os.path.join(
                    self.data_path,
                    sanitize_filename(f"{keyword}_case_reports.csv".lower()),
                )
            )

        executor = ThreadPoolExecutor()
        try:
            semaphore = asyncio.Semaphore(10)
            coros = [
                concurrency_limited(
                    self._get_case_reports_for_keyword(
                        query, keyword, keyword_data_path, executor
                    ),
                    semaphore,
                )
                for query, keyword, keyword_data_path in zip(
                    queries, keywords, self.data_files
                )
                if not os.path.exists(keyword_data_path)
            ]

            if coros:
                for coro in tqdm_asyncio.as_completed(
                    coros,
                    total=len(coros),
                    desc="Fetching case reports for keywords",
                    unit="keyword",
                    position=0,
                    leave=False,
                    dynamic_ncols=True,
                ):
                    result = await coro
                    if result is None:
                        metadata[keyword] = 0
                        continue

                    keyword, num_reports = result
                    metadata[keyword] = num_reports
                    logging.info(
                        f"Fetched {num_reports} case reports for keyword: {keyword}"
                    )
        finally:
            executor.shutdown(wait=True)

        return self.data_files, metadata

    def curate_experiment_data(
        self,
        experiment: Experiment,
        study_name: str,
        filter_by_date: bool,
        clean_data_path: list[str],
    ) -> tuple[str, int]:
        save_path = os.path.join(
            self.data_path,
            f"{study_name.lower().replace(' ', '_')}/pubmed_{experiment.nct_id}.csv",
        )

        if os.path.exists(save_path):
            exp_df = pd.read_csv(save_path, index_col=0)
            return save_path, len(exp_df)

        treatment_names = [
            name.lower() for name in experiment.treatment_common_names["pubmed"]
        ]
        outcome_names = [
            name.lower() for name in experiment.outcome_common_names["pubmed"]
        ]

        valid_count = 0
        for path in clean_data_path:
            df = pd.read_csv(path, index_col=0)
            if df.empty:
                logger.warning(f"Skipping empty DataFrame at {path}")
                continue

            if filter_by_date and experiment.date:
                _ = _filter_by_date(df, experiment.date, "publication_date")

            # If row in full_text column in nan, replace with combination of title and abstract
            df["full_text"] = df["full_text"].fillna(
                df["title"] + "\n\n" + df["abstract"]
            )

            # Rename 'full_text' to 'report'
            df.rename(columns={"full_text": "report"}, inplace=True)

            reports = df["report"].astype(str).str.lower()

            # Create boolean masks for treatments and outcomes
            treatment_masks = [
                reports.str.contains(name, regex=False, na=False)
                for name in treatment_names
            ]
            outcome_masks = [
                reports.str.contains(name, regex=False, na=False)
                for name in outcome_names
            ]

            # Combine masks
            has_treatment = (
                pd.concat(treatment_masks, axis=1).any(axis=1)
                if treatment_masks
                else pd.Series([False] * len(df))
            )
            has_outcome = (
                pd.concat(outcome_masks, axis=1).any(axis=1)
                if outcome_masks
                else pd.Series([False] * len(df))
            )

            valid_mask = has_treatment & has_outcome

            if not valid_mask.any():
                return save_path, 0

            # Get valid rows and find specific matches
            result = df[valid_mask].copy()
            valid_combined_text = reports[valid_mask]

            treatments_list = []
            outcomes_list = []

            for text in valid_combined_text:
                found_treatments = [name for name in treatment_names if name in text]
                found_outcomes = [name for name in outcome_names if name in text]
                treatments_list.append(found_treatments)
                outcomes_list.append(found_outcomes)

            result["treatments_mentioned"] = treatments_list
            result["outcome_words"] = outcomes_list

            valid_count += len(result)

        return save_path, valid_count

    @staticmethod
    def get_common_name_prompts() -> dict[str, list[dict[str, str]]]:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
        )
        t_prompt = load_prompt(
            base_dir, "common_name_treatment", return_format="messages", source="PubMed"
        )
        o_prompt = load_prompt(
            base_dir, "common_name_outcome", return_format="messages", source="PubMed"
        )
        return {"treatment": t_prompt, "outcome": o_prompt}

    def _get_search_query(self, keyword: str) -> str:
        return (
            f'("{keyword}"[All Fields]) AND '
            '"english"[Language] AND '
            '"case reports"[Publication Type] AND '
            "hasabstract[Filter] AND "
            '"humans"[MeSH Terms]'
        )

    async def _get_case_reports_for_keyword(
        self,
        query: str,
        keyword: str,
        keyword_data_path: str,
        executor: ThreadPoolExecutor,
    ) -> tuple[str, int] | None:
        webenv, query_key = await search_pubmed(query, self.api_key, executor)
        case_reports = await fetch_articles(
            webenv, query_key, self.data_path, self.api_key, executor
        )

        os.makedirs(os.path.dirname(keyword_data_path), exist_ok=True)

        if not case_reports:
            logger.debug(f"No case reports found for keyword: {keyword}")
            # Save empty DataFrame to the CSV file
            pd.DataFrame().to_csv(keyword_data_path)
            return None

        # Save the case reports to a CSV file
        case_reports_df = pd.DataFrame(case_reports)
        case_reports_df.to_csv(keyword_data_path)

        return keyword, len(case_reports_df)


def _filter_by_date(adf: pd.DataFrame, date: str, date_col: str) -> pd.DataFrame:
    try:
        date_obj = parse(date)
    except Exception as e:
        logger.error(f"Error parsing date cutoff '{date}': {e}")
        return pd.DataFrame()

    cutoff_ts = pd.Timestamp(date_obj).timestamp()
    print(f"Cutoff datetime: {date_obj} (timestamp {cutoff_ts})")
    print(f"Date column stats before filtering:\n{adf[date_col].describe()}")

    # Parse date column all at once, try inference and coerce errors
    date_series = pd.to_datetime(
        adf[date_col], errors="coerce", infer_datetime_format=True
    )

    # Filter rows which have a datetime and are on or before cutoff
    mask = (date_series.notna()) & (date_series <= date_obj)

    filtered_df = adf.loc[mask].reset_index(drop=True)

    if adf.empty:
        print("No articles found after filtering by date.")
        return pd.DataFrame()

    return filtered_df
