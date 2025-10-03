"""PubMed curation pipeline stages."""

import asyncio
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm.asyncio import tqdm as tqdm_asyncio

from naturalv2.experiment import Experiment
from naturalv2.sources.curation import CurationContext, CurationStage, StageState
from naturalv2.sources.pubmed.utils import fetch_articles, search_pubmed
from naturalv2.sources.reddit.utils import filter_by_date
from naturalv2.utils import concurrency_limited, sanitize_filename


logger = logging.getLogger(__name__)


_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b")


def _tokenize_casefold(text: str) -> set[str]:
    """Return a casefolded token set for exact token matching."""

    if not text:
        return set()
    return set(_TOKEN_PATTERN.findall(text.casefold()))


class PubMedConditionFilter(CurationStage):
    """Stage to construct PubMed queries for each experiment condition.

    Parameters
    ----------
    name : str | None, optional, default=None
        Custom stage name used for logging.
    """

    def __init__(self, *, name: str | None = None):
        super().__init__(name=name)

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Construct PubMed queries for each experiment and update state.

        Parameters
        ----------
        context : CurationContext
            The pipeline context.
        state : StageState
            The current stage state.

        Returns
        -------
        StageState
            Updated state with constructed queries.
        """
        trial_query_map: dict[str, list[str]] = defaultdict(list)

        for experiment in context.experiments:
            for condition in experiment.conditions:
                query = self._build_search_query(condition, experiment, context)
                if query not in trial_query_map[experiment.nct_id]:
                    trial_query_map[experiment.nct_id].append(query)

        # Create pandas dataframe and save as CSV for inspection/debugging
        df = pd.DataFrame(
            [
                {"nct_id": nct_id, "query": query}
                for nct_id, queries in trial_query_map.items()
                for query in queries
            ]
        )
        query_log_path = os.path.join(
            context.save_dir, f"{context.source_name}_data", "queries.csv"
        )
        df.to_csv(query_log_path, index=False)

        state.payload = trial_query_map
        state.update(trial_query_map=trial_query_map)
        logger.info(
            "%s: constructed PubMed queries for %d experiments",
            self.stage_name,
            len(trial_query_map),
        )
        logger.info("%s: saved PubMed queries to %s", self.stage_name, query_log_path)
        return state

    @staticmethod
    def _build_search_query(
        condition: str, experiment: Experiment, context: CurationContext
    ) -> str:
        """Construct a PubMed search query for the given condition.

        Parameters
        ----------
        condition : str
            Condition keyword for the current experiment.
        experiment : Experiment
            Experiment metadata containing MeSH annotations.
        context : CurationContext
            The pipeline context.

        Returns
        -------
        str
            A PubMed query string that limits results to English case reports
            involving humans.
        """

        treatment_terms = " OR ".join(
            [
                f"{treatment}"
                for treatment in experiment.get_all_treatment_names_for_source(
                    context.source_name
                )
            ]
        )

        return (
            f"{condition} AND ({treatment_terms}) "
            f"AND ((fha[Filter]) AND (casereports[Filter]) "
            f"AND (humans[Filter]) AND (english[Filter]))"
        )


class PubMedFetchAndClean(CurationStage):
    """Download and normalize PubMed case reports for experiments.

    This stage issues PubMed queries for each experiment condition, writes the
    resulting case reports to disk as CSV files, and records the successfully
    materialised file paths in the stage state.

    Parameters
    ----------
    api_key : str, optional
        PubMed API key.  If omitted, the environment variable ``PUBMED_API_KEY``
        will be attempted.  If neither is provided, requests will be unauthenticated
        and might be subject to stricter rate limits.
    max_concurrent_requests : int, default=10
        Limit on the number of concurrent network requests issued to PubMed.
    name : str, optional, default=None
        Custom stage name used for logging.

    Raises
    ------
    TypeError
        If ``api_key`` is not ``None`` or ``str``.
    ValueError
        If ``max_concurrent_requests`` is not a positive integer.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_concurrent_requests: int = 10,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not isinstance(api_key, (str, type(None))):
            raise TypeError("`api_key` must be of type str or None")
        if not isinstance(max_concurrent_requests, int) or max_concurrent_requests < 1:
            raise ValueError("`max_concurrent_requests` must be a positive integer")

        self.api_key = api_key
        self.max_concurrent_requests = max_concurrent_requests

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Fetch and persist PubMed case reports.

        Parameters
        ----------
        context : CurationContext
            Pipeline context describing experiments, storage location, and
            additional metadata.
        state : StageState
            Mutable stage state.  The ``payload`` attribute will be replaced
            with a list of file paths containing the retrieved case reports.

        Returns
        -------
        StageState
            Updated state containing the list of file paths written during the
            stage execution.

        Raises
        ------
        ValueError
            If `trial_query_map` is missing from `state.metadata`.
        """
        trial_query_map: dict[str, list[str]] = state.metadata.get(
            "trial_query_map", {}
        )
        if not trial_query_map:
            raise ValueError(
                f"{self.stage_name}: missing trial_query_map; "
                "ensure that PubMedConditionFilter has been run previously."
            )

        source_dir = os.path.join(context.save_dir, f"{context.source_name}_data")
        os.makedirs(source_dir, exist_ok=True)

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        with ThreadPoolExecutor() as executor:
            (
                nctid_clean_path_map,
                num_case_reports_fetched,
                num_case_reports_cleaned,
                trials_with_no_case_reports,
            ) = await self._fetch_and_clean_case_reports(
                context,
                trial_query_map,
                source_dir,
                semaphore,
                executor,
            )

        state.payload = nctid_clean_path_map
        state.update(
            cleaned_paths=nctid_clean_path_map,
            source_dir=source_dir,
            num_case_reports_fetched=num_case_reports_fetched,
            num_case_reports_cleaned=num_case_reports_cleaned,
            trials_with_no_case_reports=list(trials_with_no_case_reports),
        )

        if trials_with_no_case_reports:
            logger.info(
                "%s: no case reports were found for the following %d experiments: %s",
                self.stage_name,
                len(trials_with_no_case_reports),
                ", ".join(sorted(trials_with_no_case_reports)),
            )

        context.study_dataset.data_paths[f"{context.source_name}_cleaned"] = list(
            nctid_clean_path_map.values()
        )
        context.study_dataset.to_yaml(context.extras["study_dataset_path"])
        return state

    def clean(
        self,
        adf: pd.DataFrame,
        *,
        experiment: Experiment,
        apply_date_filter: bool = True,
    ) -> pd.DataFrame | None:
        """Clean and normalize the fetched case reports DataFrame.

        Parameters
        ----------
        adf : pd.DataFrame
            Raw DataFrame of fetched case reports.
        experiment : Experiment
            The experiment metadata.
        apply_date_filter : bool, default=True
            Whether to filter by experiment date.

        Returns
        -------
        pd.DataFrame or None
            Cleaned DataFrame, or None if no valid rows remain.
        """
        if adf.empty:
            return None

        # If "full_text" column is missing, add it as all-NA
        if "full_text" not in adf.columns:
            adf["full_text"] = pd.NA

        # Check for the existence of key columns
        required_columns = {
            "pmid",
            "title",
            "authors",
            "abstract",
            "full_text",
            "publication_date",
        }
        missing_columns = required_columns - set(adf.columns)
        if missing_columns:
            logger.warning(
                "Missing required columns in fetched case reports: %s", missing_columns
            )
            return None

        # Normalise duplicated quotes ahead of text field usage
        adf["title"] = (
            adf["title"]
            .fillna("")
            .astype(str)
            .str.replace(r'"{2,}', '"', regex=True)
            .str.strip()
        )
        adf["title"] = adf["title"].replace("", pd.NA)

        adf["authors"] = (
            adf["authors"]
            .fillna("")
            .astype(str)
            .str.replace(r'"{2,}', '"', regex=True)
            .str.replace(r"\s{2,}", " ", regex=True)
            .str.strip()
        )
        adf["authors"] = adf["authors"].replace("", pd.NA)

        if "pmc_id" in adf.columns:
            # Treat empty and placeholder PMC identifiers as missing
            pmc_series = adf["pmc_id"].fillna("").astype(str).str.strip().str.upper()
            pmc_series = pmc_series.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
            adf["pmc_id"] = pmc_series

        # Drop rows where both `full_text` and a meaningful abstract are missing
        abstract_normalized = (
            adf["abstract"].fillna("").astype(str).str.strip().str.casefold()
        )
        missing_text_mask = adf["full_text"].isna() & (
            (abstract_normalized == "")
            | (abstract_normalized == "no abstract available")
        )
        df = adf.loc[~missing_text_mask].copy()
        if df.empty:
            logger.warning(
                "All rows dropped after removing entries with missing text for experiment %s",
                experiment.nct_id,
            )
            return None

        # If 'full_text' is missing fallback to title+abstract, if available
        title_fallback = df["title"].fillna("").astype(str).str.strip()
        abstract_fallback = df["abstract"].fillna("").astype(str).str.strip()
        fallback_text = title_fallback.str.cat(
            abstract_fallback, sep="\n\n"
        ).str.strip()

        full_text_series = df["full_text"].fillna("").astype(str).str.strip()
        missing_full_text = full_text_series == ""
        df.loc[missing_full_text, "full_text"] = fallback_text[missing_full_text]

        # Rename 'full_text' column to 'report'
        df.rename(columns={"full_text": "report"}, inplace=True)

        if apply_date_filter and experiment.date:
            df = filter_by_date(df, experiment.date, "publication_date")
            if df.empty:
                logger.warning(
                    "All rows dropped after date filtering for experiment %s",
                    experiment.nct_id,
                )
                return None

        return df

    async def _fetch_and_clean_case_reports(
        self,
        context: CurationContext,
        trial_query_map: dict[str, list[str]],
        source_dir: str,
        semaphore: asyncio.Semaphore,
        executor: ThreadPoolExecutor,
    ) -> tuple[dict[str, str], int, int, set[str]]:
        """Fetch and clean PubMed case reports for each experiment."""
        trials_with_no_case_reports: set[str] = set(trial_query_map.keys())

        def _unmark_trials_with_case_reports(nct_id: str) -> None:
            trials_with_no_case_reports.discard(nct_id)

        existing_nctid_clean_path_map: dict[str, str] = {}
        query_trial_map: dict[str, set[str]] = {}

        fetch_tasks = []
        for experiment in context.experiments:
            for query in trial_query_map.get(experiment.nct_id, []):
                if query in query_trial_map:
                    query_trial_map[query].add(experiment.nct_id)
                    continue

                query_trial_map.setdefault(query, set()).add(experiment.nct_id)

                filename = os.path.join(
                    source_dir, f"{experiment.nct_id}_case_reports.csv"
                )
                if os.path.exists(filename):  # Query processed in previous run
                    existing_nctid_clean_path_map[experiment.nct_id] = filename
                    _unmark_trials_with_case_reports(experiment.nct_id)
                    continue

                fetch_tasks.append(
                    concurrency_limited(
                        self._download_case_reports(
                            query,
                            source_dir,
                            experiment=experiment,
                            executor=executor,
                        ),
                        semaphore,
                    )
                )

        num_case_reports_fetched = 0
        num_case_reports_cleaned = 0
        new_nctid_clean_path_map: dict[str, str] = {}
        for fut in tqdm_asyncio.as_completed(
            fetch_tasks,
            total=len(fetch_tasks),
            desc="Fetching PubMed case reports",
            unit="query",
            position=0,
            leave=False,
            dynamic_ncols=True,
            disable=(len(fetch_tasks) == 0),
        ):
            case_reports, (query, experiment) = await fut
            if not case_reports:
                logger.warning(
                    "%s: No case reports fetched for query '%s' (experiment %s)",
                    self.stage_name,
                    query,
                    experiment.nct_id,
                )
                continue

            fetched_case_reports = pd.DataFrame(case_reports)

            # Track number of case reports downloaded
            num_case_reports_fetched += len(fetched_case_reports)

            # Clean dataframe
            cleaned_case_reports = self.clean(
                fetched_case_reports,
                experiment=experiment,
                apply_date_filter=context.filter_by_date,
            )
            if cleaned_case_reports is None:
                continue

            # Track number of case reports after cleaning
            num_case_reports_cleaned += len(cleaned_case_reports)

            # Save dataframe(s)
            for nct_id in query_trial_map[query]:
                file_path = os.path.join(source_dir, f"{nct_id}_case_reports.csv")
                if os.path.exists(file_path):
                    existing_df = pd.read_csv(file_path).drop(
                        columns=["Unnamed: 0"], errors="ignore"
                    )
                    combined_df = (
                        pd.concat([existing_df, cleaned_case_reports])
                        .drop_duplicates(subset=["pmid"])
                        .reset_index(drop=True)
                    )
                    combined_df.to_csv(file_path, index=False)
                else:
                    cleaned_case_reports.to_csv(file_path, index=False)

                new_nctid_clean_path_map[nct_id] = file_path
                _unmark_trials_with_case_reports(nct_id)

        logger.info(
            "%s: fetched %d case reports from PubMed (%d after cleaning) across %s experiments",
            self.stage_name,
            num_case_reports_fetched,
            num_case_reports_cleaned,
            len(existing_nctid_clean_path_map),
        )

        nctid_clean_path_map = {
            **existing_nctid_clean_path_map,
            **new_nctid_clean_path_map,
        }
        return (
            nctid_clean_path_map,
            num_case_reports_fetched,
            num_case_reports_cleaned,
            trials_with_no_case_reports,
        )

    async def _download_case_reports(
        self,
        query: str,
        save_dir: str,
        *,
        experiment: Experiment,
        executor: ThreadPoolExecutor,
    ) -> list[dict[str, str]]:
        """Execute a PubMed query and collect case reports.

        Parameters
        ----------
        query : str
            Fully constructed PubMed search query.
        save_dir : str
            Directory used by the lower-level fetch utilities for temporary
            storage (caching).
        experiment : Experiment
            The Experiment object associated with the query.
        executor : ThreadPoolExecutor
            Executor used for XML parsing and file I/O.

        Returns
        -------
        list[dict[str, str]], tuple[str, Experiment]
            The retrieved case reports (if any) and the original query and experiment
            object.
        """

        webenv, query_key = await search_pubmed(query, self.api_key, executor)
        if query_key == "-1":
            logger.warning(
                "No valid `<QueryKey>` was returned from esearch for query '%s'", query
            )
            case_reports = []
        else:
            case_reports = await fetch_articles(
                webenv, query_key, save_dir, self.api_key, executor
            )
        return case_reports, (query, experiment)


class PubMedCurateStage(CurationStage):
    """Curate PubMed articles per experiment.

    Parameters
    ----------
    overwrite_exisiting : bool, default=False
        If `True` overwrites existing cleaned data.
    name : str | None, optional, default=None
        Custom stage name used for logging.
    """

    def __init__(
        self,
        *,
        overwrite_existing: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.overwrite_existing = overwrite_existing

    async def run(self, context: CurationContext, state: StageState) -> StageState:  # noqa: PLR0915
        """Curate and filter PubMed case reports for each experiment.

        Parameters
        ----------
        context : CurationContext
            The pipeline context.
        state : StageState
            The current stage state.

        Returns
        -------
        StageState
            Updated state with curated file paths.
        """
        cleaned_paths_map: dict[str, str] = state.metadata.get(
            "nctid_clean_path_map", {}
        )
        if not cleaned_paths_map:
            raise ValueError(
                f"{self.stage_name}: No cleaned case report paths found in state."
            )

        condition_segment = sanitize_filename(context.condition.lower())
        study_dir = os.path.join(
            state.metadata.get("source_dir", context.save_dir), condition_segment
        )
        os.makedirs(study_dir, exist_ok=True)

        curated_paths: dict[str, str] = {}
        curated_data_sizes: dict[str, int] = {}
        total_rows = 0

        for experiment in context.experiments:
            file_path = cleaned_paths_map.get(experiment.nct_id)
            if not os.path.exists(file_path):
                logger.warning(
                    "%s: Cleaned case report file missing for experiment %s at %s",
                    self.stage_name,
                    experiment.nct_id,
                    file_path,
                )
                continue

            df = pd.read_csv(file_path).drop(columns=["Unnamed: 0"], errors="ignore")
            if df.empty:
                logger.warning(
                    "%s: Cleaned case report file is empty for experiment %s at %s",
                    self.stage_name,
                    experiment.nct_id,
                    file_path,
                )
                continue

            save_path = os.path.join(study_dir, f"pubmed_{experiment.nct_id}.csv")

            if os.path.exists(save_path) and not self.overwrite_existing:
                logger.debug(
                    "%s: reusing existing curated file for experiment %s at %s",
                    self.stage_name,
                    experiment.nct_id,
                    save_path,
                )
                curated_paths[experiment.nct_id] = save_path
                continue

            treatment_names = experiment.get_all_treatment_names_for_source(
                context.source_name
            )
            alias_token_map: dict[str, set[str]] = {}
            for alias in sorted(treatment_names):
                if not alias:
                    continue
                tokens = _tokenize_casefold(alias)
                if tokens:
                    alias_token_map[alias] = tokens

            if not alias_token_map:
                logger.warning(
                    "%s: no treatment aliases available for experiment %s",
                    self.stage_name,
                    experiment.nct_id,
                )
                continue

            reports = df["report"].fillna("").astype(str)
            report_tokens = reports.apply(_tokenize_casefold)

            def match_treatments(tokens, alias_token_map=alias_token_map):
                return [
                    alias
                    for alias, alias_tokens in alias_token_map.items()
                    if alias_tokens <= tokens
                ]

            matched_treatments = report_tokens.apply(match_treatments)

            has_treatment = matched_treatments.apply(bool)
            if not has_treatment.any():
                logger.info(
                    "%s: No treatments matched in any report for experiment %s",
                    self.stage_name,
                    experiment.nct_id,
                )
                continue

            result = df.loc[has_treatment].copy()
            result["treatments_mentioned"] = matched_treatments[has_treatment]

            result.to_csv(save_path, index=False)
            curated_paths[experiment.nct_id] = save_path
            curated_data_sizes[experiment.nct_id] = len(result)
            total_rows += len(result)

        state.payload = curated_paths
        state.update(curated_paths=curated_paths)
        logger.info(
            "%s: curated PubMed datasets for %d experiments (%d rows)",
            self.stage_name,
            len(curated_paths),
            total_rows,
        )
        context.study_dataset.data_paths.update(curated_paths)
        context.study_dataset.data_sizes.update(curated_data_sizes)
        context.study_dataset.to_yaml(context.extras["study_dataset_file"])
        return state
