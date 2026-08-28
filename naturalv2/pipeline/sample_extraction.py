"""Sample extraction stages of the NATURAL pipeline."""

import asyncio
import importlib.resources
import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig
from pydantic import BaseModel, FiniteFloat
from tqdm.asyncio import tqdm

from naturalv2.models.utils import TokenTracker
from naturalv2.outcome_metadata import OutcomeBounds
from naturalv2.pipeline.constants import (
    INCLUSION_COL_NAME,
    OUTCOME_COL_NAME,
    TREATMENT_COL_NAME,
    SampleValidationConfig,
    rejection_log_level,
)
from naturalv2.pipeline.natural import PipelineContext, PipelineStage
from naturalv2.pipeline.utils import _create_progress_bar, _csv_writer
from naturalv2.utils import create_response_format, get_save_path


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.models.lm import APIModel
    from naturalv2.models.types import ModelResponse

logger = logging.getLogger(__name__)


class ExtractType(str, Enum):
    """Enumeration for covariate extraction types."""

    RELEVANCE = "relevance"
    TY_FILTER = "ty_filter"
    KNOWNS = "knowns"
    IMPUTATIONS = "imputations"
    SAMPLE_TY = "sample_ty"
    SAMPLE_INCLUSION = "sample_inclusion"


class SampleExtractionStage(PipelineStage):
    """Base class for stages that extract information from reports.

    This class provides a common interface for stages that process reports
    to extract information about relevance, treatment, outcome, known covariates,
    and imputation of missing information. Each subclass should implement
    the ``process`` method to define how the data is processed.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    name : str, optional, default=None
        Optional name for the stage. If not provided, the class name will be used.
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after extraction.
    llm : APIModel
        Lazy-loaded language model instance used for extraction.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.

    """

    def __init__(
        self,
        model_cfg: DictConfig,
        name: str | None = None,
        max_concurrent_workers: int | None = None,
    ) -> None:
        """Initialize the class."""
        super().__init__(model_cfg, name=name)
        self.max_concurrent_workers = max_concurrent_workers
        self.data: pd.DataFrame | None = None
        self.extract_type: str | None = None

    def prompt_template(self) -> dict[str, Any]:
        prompt_data: dict[str, Any] = {}
        if self.extract_type:
            prompt_filepath = (
                importlib.resources.files("naturalv2.prompts.templates")
                / f"{self.extract_type}.yaml"
            )
            if not prompt_filepath.is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt_filepath}")

            with open(prompt_filepath, "r") as stream:
                prompt_data = yaml.safe_load(stream)
        return prompt_data

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Process the input data and return transformed data."""
        raise NotImplementedError("Subclasses must implement the process method.")


class RelevanceFilterStage(SampleExtractionStage):
    """Stage for filtering relevant reports.

    At this stage, an LLM is asked to determine if a report is relevant to a
    given condition, treatment, outcome or other covariates of interest.
    This stage processes the input data to filter out reports that are not relevant
    based on the LLM's response.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    name : str, optional, default=None
        Optional name for the stage. If not provided, the class name will be used.
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after relevance filtering.
    llm : APIModel
        Lazy-loaded language model instance used for relevance filtering.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        name: str | None = None,
        max_concurrent_workers: int | None = None,
    ) -> None:
        super().__init__(model_cfg, name, max_concurrent_workers)
        self.extract_type = ExtractType.RELEVANCE.value

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Process the input data to filter relevant reports.

        This method uses an LLM to determine if each report is relevant to the
        specified condition, treatment, outcome, or other covariate of interest.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing reports to be filtered.
        context : PipelineContext
            Context for the pipeline execution, containing experiment and
            configuration details.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the relevant reports after filtering.

        Raises
        ------
        Exception
            If there is an error during the extraction process.
        """
        response_format = create_response_format(
            "RelevanceResponse", ["is_relevant"], {"is_relevant": Literal["Yes", "No"]}
        )
        filtered_data = await extract_covariates(
            input_df=data,
            pipeline_context=context,
            pipeline_stage_name=self.stage_name,
            extract_type=ExtractType.RELEVANCE,
            llm=self.llm,
            model_name=self._model_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        self.data = filtered_data[filtered_data["is_relevant"].str.lower() == "yes"]
        logger.info(f"After relevance filter: {len(self.data)} reports.")
        return self.data


class TreatmentOutcomeFilterStage(SampleExtractionStage):
    """Stage for filtering reports with treatment and outcome information.

    In this stage, an LLM is used to determine the treatment taken and whether
    the outcome is mentioned in each report. The stage processes the input data
    to filter out reports that do not provide sufficient information about the
    treatment and outcome.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    name : str, optional, default=None
        Optional name for the stage. If not provided, the class name will be used.
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after treatment-outcome filtering.
    llm : APIModel
        Lazy-loaded language model instance used for treatment-outcome filtering.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        name: str | None = None,
        max_concurrent_workers: int | None = None,
    ) -> None:
        super().__init__(model_cfg, name, max_concurrent_workers)
        self.extract_type = ExtractType.TY_FILTER.value

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Process the input data to filter reports based on treatment and outcome.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing reports to be filtered.
        context : PipelineContext
            Context for the pipeline execution, containing experiment and
            configuration details.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the reports that provide information
            about the treatment taken and whether the outcome is mentioned.

        Raises
        ------
        Exception
            If there is an error during the extraction process.
        """
        treatment_options = context.experiment.treatment_names + ["Unknown"]
        response_format = create_response_format(
            "TYFilterResponse",
            [TREATMENT_COL_NAME, OUTCOME_COL_NAME],
            types={
                TREATMENT_COL_NAME: Literal[*treatment_options],
                OUTCOME_COL_NAME: Literal["Yes", "No", "Unknown"],
            },
        )
        ty_samples = await extract_covariates(
            input_df=data,
            pipeline_context=context,
            pipeline_stage_name=self.stage_name,
            extract_type=ExtractType.TY_FILTER,
            llm=self.llm,
            model_name=self._model_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        self.data = context.experiment.hard_filter_ty(ty_samples)
        logger.info(f"After treatment-outcome filter: {len(self.data)} reports.")
        return self.data


class KnownsStage(SampleExtractionStage):
    """Stage for extracting known covariates.

    In this stage, an LLM is used to extract known covariates from the reports.
    The LLM is allowed to return 'Unknown' for any covariate that it cannot
    determine.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    name : str, optional, default=None
        Optional name for the stage. If not provided, the class name will be used.
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after extracting known covariates.
    llm : APIModel
        Lazy-loaded language model instance used for covariate extraction.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.

    """

    def __init__(
        self,
        model_cfg: DictConfig,
        name: str | None = None,
        max_concurrent_workers: int | None = None,
    ) -> None:
        super().__init__(model_cfg, name, max_concurrent_workers)
        self.extract_type = ExtractType.KNOWNS.value

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Process the input data to extract known covariates.

        This method uses an LLM to extract known covariates from the reports.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing reports to be processed.
        context : PipelineContext
            Context for the pipeline execution, containing experiment and
            configuration details.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted known covariates from the reports.

        Raises
        ------
        Exception
            If there is an error during the extraction process.
        """
        response_format = create_response_format(
            "KnownsResponse",
            keys=context.experiment.covariate_names + [INCLUSION_COL_NAME],
            types={
                "Country": str | Literal["Unknown"],
                "Duration": str | Literal["Unknown"],
                INCLUSION_COL_NAME: Literal["Yes", "No", "Unknown"],
            },
        )
        self.data = await extract_covariates(
            input_df=data,
            pipeline_context=context,
            pipeline_stage_name=self.stage_name,
            extract_type=ExtractType.KNOWNS,
            llm=self.llm,
            model_name=self._model_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        self.data = context.experiment.hard_filter_inclusion(self.data)
        logger.info(f"After inclusion filter: {len(self.data)} reports.")
        return self.data


class ImputationsStage(SampleExtractionStage):
    """Stage for imputing missing covariates.

    In this stage, an LLM is used to impute missing covariates in the reports.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    name : str, optional, default=None
        Optional name for the stage. If not provided, the class name will be used.
    max_concurrent_workers : int | None, optional, default=None
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after imputing missing covariates.
    llm : APIModel
        Lazy-loaded language model instance used for covariate extraction.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        name: str | None = None,
        max_concurrent_workers: int | None = None,
    ) -> None:
        super().__init__(model_cfg, name, max_concurrent_workers)
        self.extract_type = ExtractType.IMPUTATIONS.value

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        response_format = create_response_format(
            "ImputationsResponse",
            keys=context.experiment.covariate_names,
            types={"Country": str, "Duration": int},
        )
        self.data = await extract_covariates(
            input_df=data,
            pipeline_context=context,
            pipeline_stage_name=self.stage_name,
            extract_type=ExtractType.IMPUTATIONS,
            llm=self.llm,
            model_name=self._model_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        self.data = context.experiment.discretize(self.data)
        logger.info(f"Final: {len(self.data)} reports after imputation.")
        return self.data


def _create_sample_ty_response_format(
    experiment: "Experiment", outcome: str
) -> type[BaseModel]:
    field_types = {
        TREATMENT_COL_NAME: Literal[*experiment.options[TREATMENT_COL_NAME]],
        OUTCOME_COL_NAME: (
            Literal["No", "Yes"]
            if experiment.is_binary_outcome(outcome)
            else FiniteFloat
        ),
    }
    return create_response_format(
        "SampleTYResponse", list(field_types), types=field_types
    )


def _filter_invalid_sampled_outcomes(
    extractions: pd.DataFrame,
    *,
    nct_id: str,
    outcome: str,
    bounds: OutcomeBounds | None,
    sample_validation: SampleValidationConfig,
) -> pd.DataFrame:
    """Remove invalid continuous outcomes and enforce one rejection policy."""
    if OUTCOME_COL_NAME not in extractions.columns:
        raise ValueError(
            f"Sampled extraction artifact is missing column: {OUTCOME_COL_NAME}"
        )

    numeric_outcomes = pd.to_numeric(extractions[OUTCOME_COL_NAME], errors="coerce")
    minimum = bounds.minimum if bounds is not None else -np.inf
    maximum = bounds.maximum if bounds is not None else np.inf
    # to_numeric always yields a numeric dtype, so this mask is plain bool with
    # no NaN to fill, and np.isfinite on a Series already returns one.
    finite_mask = np.isfinite(numeric_outcomes)
    below_minimum = finite_mask & numeric_outcomes.lt(minimum)
    above_maximum = finite_mask & numeric_outcomes.gt(maximum)
    valid_mask = finite_mask & ~below_minimum & ~above_maximum

    n_sampled = len(extractions)
    n_rejected = int((~valid_mask).sum())
    if n_rejected:

        unparsed = numeric_outcomes.isna()
        rejection_reasons = {
            "unparsed": int(unparsed.sum()),
            "infinite": int((~finite_mask & ~unparsed).sum()),
            "below_minimum": int(below_minimum.sum()),
            "above_maximum": int(above_maximum.sum()),
        }
        rejection_rate = n_rejected / n_sampled
        all_rejected = n_rejected == n_sampled
        high_rejection_rate = rejection_rate >= sample_validation.high_rejection_rate

        blocks_estimation = all_rejected or (
            high_rejection_rate and not sample_validation.allow_high_rejection_rate
        )
        log = rejection_log_level(
            logger,
            n_rejected,
            rejection_rate,
            high_rejection_rate=sample_validation.high_rejection_rate,
        )
        log(
            "Rejected %d/%d sampled outcomes for %s / %r (%.2f%%)",
            n_rejected,
            n_sampled,
            nct_id,
            outcome,
            rejection_rate * 100,
            extra={
                "phase": "sample_ty_artifact_validation",
                "schema_id": "sample_ty_outcome_validation.v3",
                "status": "blocked" if blocks_estimation else "rejected",
                "nct_id": nct_id,
                "outcome": outcome,
                "minimum": bounds.minimum if bounds is not None else None,
                "maximum": bounds.maximum if bounds is not None else None,
                "n_sampled": n_sampled,
                "n_rejected": n_rejected,
                "rejection_rate": rejection_rate,
                "rejection_reasons": rejection_reasons,
            },
        )

        if all_rejected:
            raise ValueError(
                f"No valid sampled outcomes remain for {nct_id!r} / {outcome!r}."
            )
        if blocks_estimation:
            raise ValueError(
                f"Estimation stopped for {nct_id!r} / {outcome!r}: invalid outcome "
                f"rate {rejection_rate:.2%} met the high-rejection threshold "
                f"of {sample_validation.high_rejection_rate:.2%}. Set "
                "`sample_validation.allow_high_rejection_rate=true` to continue."
            )

    validated = extractions.loc[valid_mask].copy()
    validated[OUTCOME_COL_NAME] = numeric_outcomes.loc[valid_mask]
    return validated


class SampleTYStage(SampleExtractionStage):
    """Stage for sampling treatment and outcome given covariates.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    name : str, optional, default=None
        Optional name for the stage. If not provided, the class name will be used.
    max_concurrent_workers : int | None, optional, default=None
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after imputing missing covariates.
    llm : APIModel
        Lazy-loaded language model instance used for covariate extraction.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        name: str | None = None,
        max_concurrent_workers: int | None = None,
    ) -> None:
        super().__init__(model_cfg, name, max_concurrent_workers)
        self.extract_type = ExtractType.SAMPLE_TY.value

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        if context.sample_validation is None:
            raise ValueError("SampleTYStage requires a sample-validation policy.")

        response_format = _create_sample_ty_response_format(
            context.experiment, context.outcome
        )
        self.data = await extract_covariates(
            input_df=data,
            pipeline_context=context,
            pipeline_stage_name=self.stage_name,
            extract_type=ExtractType.SAMPLE_TY,
            llm=self.llm,
            model_name=self._model_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        if not context.experiment.is_binary_outcome(context.outcome):
            self.data = _filter_invalid_sampled_outcomes(
                self.data,
                nct_id=context.experiment.nct_id,
                outcome=context.outcome,
                bounds=context.experiment.outcome_bounds.get(context.outcome),
                sample_validation=context.sample_validation,
            )

        self.data = context.experiment.discretize_ty(self.data, context.outcome)
        logger.info(f"Final: {len(self.data)} reports after sampling TY.")
        return self.data


class SampledInclusionProbStage(SampleExtractionStage):
    """API-model alternative to ``InclusionProbStage``.

    API models cannot return prompt log probabilities, so instead of scoring
    "A: No" / "A: Yes" prompt variants, this stage asks the same inclusion
    question ``num_samples`` times and uses the Yes-vote fraction as
    P(X ∈ Inclusion | report). It emits the same ``inclusion_probs``
    ([P(No), P(Yes)]) and ``meets_inclusion_criteria_sampled`` columns.

    Each report is replicated ``num_samples`` times before the standard
    ``extract_covariates`` pass, so per-vote extractions are saved (and
    resumed) like any other stage; votes are aggregated afterwards.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the (API) language model used in this stage.
    name : str, optional, default=None
        Optional name for the stage. If not provided, the class name will be used.
    num_samples : int, optional, default=25
        Number of Yes/No samples per report.
    call_kwargs : dict | None, optional, default=None
        Sampling kwargs sent with every LLM call (e.g. temperature); set as
        defaults in the stage config.
    seed : int | None, optional, default=None
        Seed for drawing the ``meets_inclusion_criteria_sampled`` label.
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent LLM requests. If None, defaults to 10.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        name: str | None = None,
        num_samples: int = 25,
        call_kwargs: dict[str, Any] | DictConfig | None = None,
        seed: int | None = None,
        max_concurrent_workers: int | None = None,
    ) -> None:
        super().__init__(model_cfg, name, max_concurrent_workers)
        self.extract_type = ExtractType.SAMPLE_INCLUSION.value
        self.num_samples = num_samples
        self.call_kwargs = dict(call_kwargs) if call_kwargs else {}
        self.seed = seed

    def get_language_model(self) -> "APIModel":
        """Instantiate the model with the configured sampling kwargs.

        ``Model.kwargs`` are merged into every request by ``LiteLLMModel``.
        """
        lm = super().get_language_model()
        lm.kwargs = {**lm.kwargs, **self.call_kwargs}
        return lm

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Estimate inclusion probabilities by sampling Yes/No votes."""
        response_format = create_response_format(
            "SampledInclusionResponse",
            [INCLUSION_COL_NAME],
            {INCLUSION_COL_NAME: Literal["No", "Yes"]},
        )

        # Replicate each report once per vote. The vote key is
        # "<report index>:<vote number>" rather than a running position, because
        # extract_covariates resumes on the index -- a positional key silently
        # remaps cached votes onto different reports when num_samples or the
        # input set changes between runs.
        replicated = data.loc[data.index.repeat(self.num_samples)].copy()
        replicated["source_index"] = replicated.index
        vote_number = replicated.groupby(level=0, sort=False).cumcount()
        replicated.index = pd.Index(
            [f"{src}:{k}" for src, k in zip(replicated["source_index"], vote_number)]
        )

        extracted = await extract_covariates(
            input_df=replicated,
            pipeline_context=context,
            pipeline_stage_name=self.stage_name,
            extract_type=ExtractType.SAMPLE_INCLUSION,
            llm=self.llm,
            model_name=self._model_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        self.data = self._aggregate(
            extracted, data, context.experiment.options[INCLUSION_COL_NAME]
        )
        logger.info(f"Sampled inclusion probabilities for {len(self.data)} reports.")
        return self.data

    def _aggregate(
        self, extracted: pd.DataFrame, data: pd.DataFrame, options: list[str]
    ) -> pd.DataFrame:
        """Collapse per-vote extractions into one row per report."""
        rng = np.random.default_rng(self.seed)
        records = []
        for idx, row in data.iterrows():
            votes = extracted.loc[extracted["source_index"] == idx, INCLUSION_COL_NAME]
            if votes.empty:
                logger.warning(
                    f"Dropping report at index {idx}: all {self.num_samples} "
                    "inclusion votes failed."
                )
                continue
            p_yes = float((votes.astype(str).str.lower() == "yes").mean())
            probs = [1.0 - p_yes, p_yes]

            record = row.to_dict()
            record[f"{INCLUSION_COL_NAME}_sampled"] = str(rng.choice(options, p=probs))
            record["inclusion_probs"] = str(probs)
            record["num_valid_votes"] = len(votes)
            records.append((idx, record))

        return pd.DataFrame(
            [record for _, record in records], index=[idx for idx, _ in records]
        )


async def extract_covariates(  # noqa: PLR0912
    input_df: pd.DataFrame,
    pipeline_context: PipelineContext,
    pipeline_stage_name: str,
    extract_type: ExtractType,
    llm: "APIModel",
    model_name: str,
    response_format: BaseModel | None = None,
    max_concurrent_requests: int | None = None,
) -> pd.DataFrame:
    """Extract information from reports using an LLM.

    This function processes the input DataFrame to extract structured
    information from a text report based on the specified extraction type.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing the reports to be processed.
    pipeline_context : PipelineContext
        Context for the pipeline execution, containing experiment and
        configuration details.
    pipeline_stage_name : str
        Name of the current pipeline stage, used for logging and tracking.
    extract_type : ExtractType
        The type of extraction to perform (e.g., relevance, treatment-outcome,
        known covariates, or imputations).
    llm : APIModel
        Language model instance used for processing the reports.
    model_name : str
        Name of the language model being used.
    response_format : BaseModel | None, optional, default=None
        Pydantic model defining the expected format of the LLM response.
        If None, no validation will be performed on the response.
    max_concurrent_requests : int | None, optional, default=None
        Maximum number of concurrent requests to the LLM. If None, defaults to 10 or
        the number of samples in the input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted covariates from the reports.

    Raises
    ------
    FileNotFoundError
        If the specified save path does not exist or cannot be created.
    ValueError
        If the input DataFrame does not contain the required 'report' field for prompt
        formatting.
    Exception
        If there is an error during the extraction process, such as issues with the LLM
        or data processing.

    Notes
    -----
    - If the save path already exists, the function will read the existing CSV file
      and return it as a DataFrame.

    """
    file_path = get_save_path(
        pipeline_context.save_path,
        pipeline_context.experiment.nct_id,
        pipeline_context.exp_name,
        model_name,
        extract_type.value,
        pipeline_context.outcome,
    )

    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path, index_col=0)
        input_df = input_df.loc[~input_df.index.isin(existing_data.index)]
        logger.info(
            f"Found {len(existing_data)} existing records, {len(input_df)} left to process."
        )
        if len(input_df) == 0:
            return existing_data

    # Set up the prompt and result queues for asynchronous processing
    num_samples = len(input_df)
    num_workers = min(max_concurrent_requests or min(10, num_samples), num_samples)
    queue_size = max(num_workers * 10, 1000)

    prompt_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    result_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

    # Create progress bars for the different tasks
    producer_pbar = _create_progress_bar(total=num_samples, desc="Creating prompts")
    worker_pbar = _create_progress_bar(total=num_samples, desc="Processing prompts")
    writer_pbar = _create_progress_bar(total=num_samples, desc="Writing results")

    # Create tasks to produce prompts
    producer_task = asyncio.create_task(
        _llm_task_producer(
            prompt_queue,
            input_df,
            pipeline_context.experiment,
            extract_type.value,
            pipeline_context.outcome,
            pipeline_context.source_name,
            producer_pbar,
        ),
        name="LLM-Task-Producer",
    )

    # Create worker tasks to process prompts
    worker_tasks = [
        asyncio.create_task(
            _prompt_processor(
                worker_id,
                prompt_queue,
                result_queue,
                llm,
                worker_pbar,
                extract_type,
                pipeline_stage_name,
                token_tracker=pipeline_context._token_tracker,
                response_format=response_format,
            ),
            name=f"Prompt-Processor-{worker_id}",
        )
        for worker_id in range(num_workers)
    ]

    # Create a CSV writer task to write results to a file
    csv_writer_task = asyncio.create_task(
        _csv_writer(result_queue, file_path, writer_pbar), name="CSV-Writer"
    )

    try:
        # Wait for the producer to finish producing tasks
        await producer_task
        await prompt_queue.join()  # Ensure all prompts are processed

        # Signal workers to stop by putting None in the queue
        for _ in range(len(worker_tasks)):
            await prompt_queue.put((None, None, None))  # Signal to stop processing

        # Wait for all workers to finish
        worker_errors = await asyncio.gather(*worker_tasks, return_exceptions=True)

        processing_error_count = 0
        for idx, error in enumerate(worker_errors):
            if isinstance(error, int):
                processing_error_count += error
            elif isinstance(error, BaseException):
                logging.error(
                    f"Worker {idx} encountered an exception: {type(error).__name__} - {error}",
                    exc_info=True,
                )
                processing_error_count += 1

        await result_queue.put(None)  # Signal the CSV writer to finish
        success_count = await csv_writer_task
        logger.info(
            f"Processing completed. {success_count} records written, "
            f"{processing_error_count} errors"
        )
    except BaseException as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        # Cancel any running tasks
        for task in [csv_writer_task, producer_task] + worker_tasks:
            if not task.done():
                task.cancel()
        raise
    finally:
        # Clean up progress bars
        for pbar in [producer_pbar, worker_pbar, writer_pbar]:
            if not pbar.disable:
                pbar.close()

    try:
        return pd.read_csv(file_path, index_col=0)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.warning(
            f"No results were written to {file_path}; returning an empty DataFrame."
        )
        return pd.DataFrame()


def _prompt_formatter(
    row: pd.Series,
    experiment: "Experiment",
    prompt_type: str,
    outcome: str,
    source_name: str,
) -> list[dict[str, str]]:
    """Format the prompt for a given row of data."""
    if "report" not in row:
        raise ValueError("Row must contain 'report' field for prompt formatting.")

    covariate_answers = None
    if prompt_type == "imputations":
        all_covariates = row[experiment.covariate_names].dropna()
        covariate_answers = all_covariates[all_covariates.str.lower() != "unknown"]
        covariate_answers = covariate_answers.to_dict()

    elif prompt_type == "sample_ty":
        to_sample = [col + "_discretized" for col in experiment.covariate_names]
        to_transform = {
            k.replace("_discretized", ""): v
            for k, v in row[to_sample].to_dict().items()
        }
        covariate_answers = experiment.apply_transform(
            to_transform, repr_type="language"
        )

    return experiment.build_prompt_for_report(
        prompt_type=prompt_type,
        outcome=outcome,
        source_name=source_name,
        report=row["report"],
        covariate_answers=covariate_answers,
        return_format="messages",
    )


async def _llm_task_producer(
    queue: asyncio.Queue,
    input_df: pd.DataFrame,
    experiment: "Experiment",
    prompt_type: str,
    outcome: str,
    source_name: str,
    pbar: tqdm,
) -> None:
    """Produce prompts for the LLM from the input DataFrame."""
    for idx, row in input_df.iterrows():
        try:
            messages = _prompt_formatter(
                row, experiment, prompt_type, outcome, source_name
            )
            await queue.put((idx, row, messages))
        except Exception as e:
            logging.error(
                f"Failed to format prompt for report at index {idx}: {type(e).__name__} - {e}",
                exc_info=True,
            )
            await queue.put((idx, row, None))  # Indicate failure with None
        finally:
            pbar.update(1)


async def _prompt_processor(
    worker_id: int,
    prompt_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    llm: "APIModel",
    pbar: tqdm,
    extract_type: ExtractType,
    pipeline_stage_name: str,
    token_tracker: "TokenTracker",
    response_format: BaseModel | None = None,
) -> int:
    """Worker function to process prompts.

    This function calls the LLM with formatted prompts and processes the results.
    """
    error_count = 0
    while True:
        try:
            index, row, messages = await prompt_queue.get()
            if index is None and row is None:
                break
        except asyncio.CancelledError:
            break

        try:
            if messages is None:  # prompt formatting failed
                logging.error(
                    f"Worker {worker_id} received None messages for item at index {index}"
                )
                error_count += 1
                await result_queue.put(False)  # Signal failure to writer
                continue

            response = await llm.ainvoke(
                messages, response_format=response_format, parse_output=True
            )
            token_tracker.add(pipeline_stage_name, response)
            processed_result = _result_processor(row, response, extract_type)

            if processed_result is not None:
                await result_queue.put(processed_result)
            else:
                logging.warning(
                    f"Worker {worker_id} received None result for item at index {index} with input {messages}."
                )
                error_count += 1
                await result_queue.put(False)  # Signal failure to writer

        except BaseException as e:
            logging.error(
                f"Worker {worker_id} failed on item at index {index}: {type(e).__name__} - {e}",
                exc_info=True,
            )
            error_count += 1
            await result_queue.put(False)  # Signal failure
        finally:
            pbar.update(1)
            prompt_queue.task_done()

    return error_count


def _result_processor(
    row: pd.Series, response: "ModelResponse", extract_type: ExtractType
) -> dict[str, Any] | None:
    """Process the LLM response and combine it with the original row data."""
    output: BaseModel | None = response.output_parsed
    if output is None:
        return None  # No output to process

    if extract_type == ExtractType.TY_FILTER:
        # append "_filter" to each key in the parsed data
        output = {
            f"{key}_filter": value
            for key, value in output.model_dump(mode="json").items()
        }
    elif extract_type == ExtractType.IMPUTATIONS:
        # append "_imputed" to each key in the parsed data
        output = {
            f"{key}_imputed": value
            for key, value in output.model_dump(mode="json").items()
        }

    # Combine original row data with parsed LLM data
    parsed_row_data = {"index": row.name}
    parsed_row_data.update(row.to_dict())
    parsed_row_data.update(output)

    return parsed_row_data
