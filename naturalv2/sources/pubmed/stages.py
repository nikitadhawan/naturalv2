"""Config-driven PubMed stages without legacy dependencies."""

import asyncio
import csv
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


def _normalize_condition_key(condition: str | None) -> str:
    """Normalise condition names for consistent lookups."""

    return sanitize_filename((condition or "").strip()).lower()


def _tokenize_casefold(text: str) -> set[str]:
    """Return a casefolded token set for exact token matching."""

    if not text:
        return set()
    return set(_TOKEN_PATTERN.findall(text.casefold()))


class PubMedCaseReportFetcher(CurationStage):
    """Download PubMed case reports that match experiment conditions.

    This stage issues PubMed queries for each experiment condition, writes the
    resulting case reports to disk as CSV files, and records the successfully
    materialised file paths in the stage state.

    Parameters
    ----------
    name : str, optional
        Custom stage name used for logging.
    api_key : str, optional
        PubMed API key.  If omitted, the environment variable ``PUBMED_API_KEY``
        will be attempted.  If neither is provided, requests will be unauthenticated
        and might be subject to stricter rate limits.
    max_concurrent_requests : int, default=10
        Limit on the number of concurrent network requests issued to PubMed.

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
        name: str | None = None,
        api_key: str | None = None,
        max_concurrent_requests: int = 10,
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
        """
        source_dir = os.path.join(context.save_dir, f"{context.source_name}_data")
        os.makedirs(source_dir, exist_ok=True)

        fetched_data_paths: list[str] = []
        fetched_paths_by_condition: defaultdict[str, list[str]] = defaultdict(list)
        path_conditions: defaultdict[str, set[str]] = defaultdict(set)
        seen_pairs: set[tuple[str, str]] = set()
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        total_case_reports = 0
        with ThreadPoolExecutor() as executor:
            fetch_tasks = []
            for experiment in context.experiments:
                for condition in experiment.conditions:
                    query = self._build_search_query(condition, experiment)
                    pair = (condition, query)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    normalized_condition = _normalize_condition_key(condition)
                    filename = sanitize_filename(
                        f"{normalized_condition or 'condition'}_case_reports.csv"
                    ).lower()
                    data_path = os.path.join(source_dir, filename)
                    path_conditions[data_path].add(condition)

                    if os.path.exists(data_path):
                        fetched_data_paths.append(data_path)
                        if (
                            data_path
                            not in fetched_paths_by_condition[normalized_condition]
                        ):
                            fetched_paths_by_condition[normalized_condition].append(
                                data_path
                            )

                        # add to total count; avoid loading csv into memory, if possible
                        try:
                            with open(data_path, "r", encoding="utf-8") as f:
                                reader = csv.reader(f)
                                total_case_reports += sum(1 for _ in reader) - 1
                        except Exception as e:
                            logger.warning(
                                "Failed to read existing case report file %s: %s",
                                data_path,
                                str(e),
                            )
                        continue

                    fetch_tasks.append(
                        concurrency_limited(
                            self._download_case_reports(
                                query,
                                condition,
                                source_dir,
                                data_path,
                                executor=executor,
                            ),
                            semaphore,
                        )
                    )

            no_case_report_conditions: set[str] = set()
            if fetch_tasks:
                for coro in tqdm_asyncio.as_completed(
                    fetch_tasks,
                    total=len(fetch_tasks),
                    desc="Fetching PubMed case reports per condition",
                    unit="condition",
                    position=0,
                    leave=False,
                    dynamic_ncols=True,
                ):
                    case_reports, condition, file_path = await coro
                    if not case_reports:
                        logger.debug(
                            "No case reports found for condition %s", condition
                        )
                        no_case_report_conditions.add(condition)
                        continue

                    case_reports_df = pd.DataFrame(case_reports)
                    total_case_reports += len(case_reports_df)

                    if os.path.exists(file_path):
                        existing_df = pd.read_csv(file_path).drop(
                            columns=["Unnamed: 0"], errors="ignore"
                        )
                        case_reports_df = (
                            pd.concat([existing_df, case_reports_df])
                            .drop_duplicates(subset=["pmid"])
                            .reset_index(drop=True)
                        )

                    case_reports_df.to_csv(file_path, index=False)
                    fetched_data_paths.append(file_path)
                    associated_conditions = path_conditions.get(file_path, {condition})
                    for cond in associated_conditions:
                        norm_cond = _normalize_condition_key(cond)
                        if file_path not in fetched_paths_by_condition[norm_cond]:
                            fetched_paths_by_condition[norm_cond].append(file_path)

        state.payload = fetched_data_paths
        state.update(
            fetched_data_paths=fetched_data_paths,
            fetched_paths_by_condition={
                key: list(paths) for key, paths in fetched_paths_by_condition.items()
            },
            source_dir=source_dir,
            total_case_reports=total_case_reports,
            no_case_report_conditions=list(no_case_report_conditions),
        )
        logger.info(
            "%s: fetched %d PubMed case reports across %d files",
            self.stage_name,
            total_case_reports,
            len(fetched_data_paths),
        )

        if no_case_report_conditions:
            logger.info(
                "%s: no case reports found for conditions: %s",
                self.stage_name,
                ", ".join(sorted(no_case_report_conditions)),
            )
        return state

    async def _download_case_reports(
        self,
        query: str,
        condition: str,
        save_dir: str,
        file_path: str,
        *,
        executor: ThreadPoolExecutor,
    ) -> tuple[list[dict[str, str]], str, str]:
        """Execute a PubMed query and collect case reports.

        Parameters
        ----------
        query : str
            Fully constructed PubMed search query.
        condition : str
            Condition label associated with the query.
        save_dir : str
            Directory used by the lower-level fetch utilities for temporary
            storage (caching).
        file_path : str
            Target CSV path for the results.
        executor : ThreadPoolExecutor
            Executor used for XML parsing and file I/O.

        Returns
        -------
        list[dict[str, str]], str, str
            Tuple containing the retrieved case reports (if any), the
            originating condition, and the destination file path.
        """

        webenv, query_key = await search_pubmed(query, self.api_key, executor)
        case_reports = await fetch_articles(
            webenv, query_key, save_dir, self.api_key, executor
        )

        return case_reports, condition, file_path

    @staticmethod
    def _build_search_query(condition: str, experiment: Experiment) -> str:
        """Construct a PubMed search query for the given condition.

        Parameters
        ----------
        condition : str
            Condition keyword for the current experiment.
        experiment : Experiment
            Experiment metadata containing MeSH annotations.

        Returns
        -------
        str
            A PubMed query string that limits results to English case reports
            involving humans.
        """

        mesh_terms = " OR ".join(
            [f'"{mesh}"[MeSH Terms]' for mesh in experiment.trial_disease_mesh]
        )

        return (
            f'(("{condition}") AND ({mesh_terms})) '
            f"AND ((fha[Filter]) AND (casereports[Filter]) "
            f"AND (humans[Filter]) AND (english[Filter]))"
        )


class PubMedCleanStage(CurationStage):
    """Clean PubMed case reports."""

    def __init__(
        self, *, name: str | None = None, overwrite_existing: bool = True
    ) -> None:
        super().__init__(name=name)
        self.overwrite_existing = overwrite_existing

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        download_paths: list[str] = state.metadata.get(
            "fetched_data_paths", state.payload or []
        )
        fetched_by_condition: dict[str, list[str]] = state.metadata.get(
            "fetched_paths_by_condition", {}
        )
        cleaned_paths: list[str] = []
        cleaned_by_condition: defaultdict[str, list[str]] = defaultdict(list)

        for path in download_paths:
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
            if df.empty:
                logger.warning(f"Skipping empty DataFrame at {path}")
                continue

            # if "full_text" column is missing, add it as all-NA
            if "full_text" not in df.columns:
                df["full_text"] = pd.NA

            # check for the existence of key columns
            required_columns = {
                "pmid",
                "title",
                "authors",
                "abstract",
                "full_text",
                "publication_date",
            }
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                logger.warning(
                    f"Skipping DataFrame at {path} missing columns: {missing_columns}"
                )
                continue

            # normalise duplicated quotes ahead of text field usage
            df["title"] = (
                df["title"]
                .fillna("")
                .astype(str)
                .str.replace(r'"{2,}', '"', regex=True)
                .str.strip()
            )
            df["title"] = df["title"].replace("", pd.NA)

            df["authors"] = (
                df["authors"]
                .fillna("")
                .astype(str)
                .str.replace(r'"{2,}', '"', regex=True)
                .str.replace(r"\s{2,}", " ", regex=True)
                .str.strip()
            )
            df["authors"] = df["authors"].replace("", pd.NA)

            if "pmc_id" in df.columns:
                # treat empty and placeholder PMC identifiers as missing
                pmc_series = df["pmc_id"].fillna("").astype(str).str.strip().str.upper()
                pmc_series = pmc_series.replace(
                    {"": pd.NA, "NAN": pd.NA, "NONE": pd.NA}
                )
                df["pmc_id"] = pmc_series

            # drop rows where both `full_text` and a meaningful abstract are missing
            abstract_normalized = (
                df["abstract"].fillna("").astype(str).str.strip().str.casefold()
            )
            missing_text_mask = df["full_text"].isna() & (
                (abstract_normalized == "")
                | (abstract_normalized == "no abstract available")
            )
            df = df.loc[~missing_text_mask]
            if df.empty:
                logger.warning(f"Skipping DataFrame with no valid text at {path}")
                continue

            base, ext = os.path.splitext(os.path.basename(path))
            suffix = "" if self.overwrite_existing else "_cleaned"
            output_filepath = os.path.join(
                os.path.dirname(path), f"{base}{suffix}{ext}"
            )

            df.to_csv(output_filepath, index=False)
            cleaned_paths.append(output_filepath)

            associated_conditions = [
                condition
                for condition, sources in fetched_by_condition.items()
                if path in sources
            ]
            if not associated_conditions:
                base_name = os.path.basename(path)
                inferred_condition = base_name.rsplit("_case_reports", 1)[0]
                associated_conditions = [inferred_condition]

            for condition in associated_conditions:
                normalized_key = _normalize_condition_key(condition)
                condition_paths = cleaned_by_condition[normalized_key]
                if output_filepath not in condition_paths:
                    condition_paths.append(output_filepath)

        state.payload = cleaned_paths
        state.update(
            cleaned_paths=cleaned_paths,
            cleaned_paths_by_condition={
                key: list(paths) for key, paths in cleaned_by_condition.items()
            },
        )
        logger.info(
            "%s: cleaned PubMed datasets for %d files",
            self.stage_name,
            len(cleaned_paths),
        )
        return state


class PubMedCurateStage(CurationStage):
    """Curate PubMed articles per experiment."""

    def __init__(
        self,
        *,
        overwrite_existing: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.overwrite_existing = overwrite_existing

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        cleaned_paths = state.metadata.get("cleaned_paths", state.payload or [])
        cleaned_paths = [p for p in cleaned_paths if os.path.exists(p)]
        cleaned_by_condition: dict[str, list[str]] = state.metadata.get(
            "cleaned_paths_by_condition", {}
        )

        condition_segment = _normalize_condition_key(context.condition) or "pubmed"
        experiment_dir = os.path.join(
            state.metadata.get("source_dir", context.save_dir), condition_segment
        )
        os.makedirs(experiment_dir, exist_ok=True)

        curated_paths: dict[str, str] = {}
        total_rows = 0

        for experiment in context.experiments:
            candidate_paths: list[str] = []
            for key in experiment.conditions:
                normalized_key = _normalize_condition_key(key)
                candidate_paths.extend(cleaned_by_condition.get(normalized_key, []))

            if not candidate_paths:  # fallback to basename matching
                candidate_paths = [
                    path
                    for path in cleaned_paths
                    if any(
                        _normalize_condition_key(key) in os.path.basename(path).lower()
                        for key in experiment.conditions
                    )
                ]

            candidate_paths = list(dict.fromkeys(candidate_paths))
            if not candidate_paths:
                logger.warning(
                    "No cleaned PubMed data files found for experiment %s (conditions=%s)",
                    experiment.nct_id,
                    experiment.conditions,
                )
                continue

            filename = sanitize_filename(f"pubmed_{experiment.nct_id}.csv").lower()
            save_path = os.path.join(experiment_dir, filename)

            if os.path.exists(save_path) and not self.overwrite_existing:
                logger.info(
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

            curated_frames: list[pd.DataFrame] = []

            for path in candidate_paths:
                df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
                if df.empty:
                    logger.warning("Skipping empty DataFrame at %s", path)
                    continue

                if context.filter_by_date and experiment.date:
                    df = filter_by_date(df, experiment.date, "publication_date")
                    if df.empty:
                        logger.warning(
                            "Skipping DataFrame with no valid dates at %s", path
                        )
                        continue

                title_fallback = df["title"].fillna("").astype(str).str.strip()
                abstract_fallback = df["abstract"].fillna("").astype(str).str.strip()
                fallback_text = title_fallback.str.cat(
                    abstract_fallback, sep="\n\n"
                ).str.strip()

                full_text_series = df["full_text"].fillna("").astype(str).str.strip()
                missing_full_text = full_text_series == ""
                df.loc[missing_full_text, "full_text"] = fallback_text[
                    missing_full_text
                ]

                full_text_series = df["full_text"].fillna("").astype(str).str.strip()
                df["full_text"] = full_text_series.replace("", pd.NA)
                if df["full_text"].isna().all():
                    logger.debug("Skipping file %s due to missing report text", path)
                    continue

                df = df.copy()
                df.rename(columns={"full_text": "report"}, inplace=True)

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
                    continue

                result = df.loc[has_treatment].copy()
                result["treatments_mentioned"] = matched_treatments[has_treatment]
                result["source_path"] = path
                curated_frames.append(result)

            if not curated_frames:
                logger.info(
                    "%s: no matching PubMed reports found for experiment %s",
                    self.stage_name,
                    experiment.nct_id,
                )
                continue

            curated_df = pd.concat(curated_frames, ignore_index=True)
            if "pmid" in curated_df.columns:
                curated_df = curated_df.drop_duplicates(subset=["pmid"])

            curated_df.to_csv(save_path, index=False)
            curated_paths[experiment.nct_id] = save_path
            total_rows += len(curated_df)

        state.payload = curated_paths
        state.update(curated_paths=curated_paths)
        logger.info(
            "%s: curated PubMed datasets for %d experiments (%d rows)",
            self.stage_name,
            len(curated_paths),
            total_rows,
        )
        return state
