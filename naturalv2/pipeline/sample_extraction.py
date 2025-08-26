"""Sample extraction stages of the NATURAL pipeline."""

import asyncio
import logging
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import yaml
from omegaconf import DictConfig
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from naturalv2.models.lm import build_lm_instance_from_cfg, get_message_content
from naturalv2.pipeline.constants import (
    INCLUSION_COL_NAME,
    OUTCOME_COL_NAME,
    TREATMENT_COL_NAME,
)
from naturalv2.pipeline.natural import PipelineContext, PipelineStage
from naturalv2.pipeline.utils import _create_progress_bar, _csv_writer
from naturalv2.utils import create_response_format, get_save_path


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.models.lm import LM, ResponseType

logger = logging.getLogger(__name__)


class ExtractType(str, Enum):
    """Enumeration for covariate extraction types."""

    RELEVANCE = "relevance"
    TY_FILTER = "ty_filter"
    KNOWNS = "knowns"
    IMPUTATIONS = "imputations"


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
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after extraction.
    llm : LM
        Lazy-loaded language model instance used for extraction.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.

    """

    def __init__(
        self, model_cfg: DictConfig, max_concurrent_workers: int | None = None
    ) -> None:
        """Initialize the class."""
        super().__init__(model_cfg)
        self.max_concurrent_workers = max_concurrent_workers
        self.data: pd.DataFrame | None = None
        self.extract_type: str | None = None

    def get_language_model(self) -> "LM":
        """Return the language model used in this stage.

        Returns
        -------
        LM
            An instance of the language model configured for this stage.
        """
        return build_lm_instance_from_cfg(self.model_cfg)

    def prompt_template(self):
        prompt_data: dict[str, Any] = {}
        if self.extract_type:
            prompts_dir = str(
                Path(__file__).resolve().parents[1] / "prompts" / "templates"
            )
            filepath = os.path.join(prompts_dir, f"{self.extract_type}.yaml")
            with open(filepath, "r") as stream:
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
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after relevance filtering.
    llm : LM
        Lazy-loaded language model instance used for relevance filtering.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.
    """

    def __init__(
        self, model_cfg: DictConfig, max_concurrent_workers: int | None = None
    ) -> None:
        super().__init__(model_cfg, max_concurrent_workers)
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
            experiment=context.experiment,
            source_name=context.source_name,
            outcome=context.outcome,
            extract_type=ExtractType.RELEVANCE,
            llm=self.llm,
            model_name=self._model_name,
            save_path=context.save_path,
            exp_name=context.exp_name,
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
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after treatment-outcome filtering.
    llm : LM
        Lazy-loaded language model instance used for treatment-outcome filtering.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.
    """

    def __init__(
        self, model_cfg: DictConfig, max_concurrent_workers: int | None = None
    ) -> None:
        super().__init__(model_cfg, max_concurrent_workers)
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
            experiment=context.experiment,
            source_name=context.source_name,
            outcome=context.outcome,
            extract_type=ExtractType.TY_FILTER,
            llm=self.llm,
            model_name=self._model_name,
            save_path=context.save_path,
            exp_name=context.exp_name,
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
    max_concurrent_workers : int | None, optional
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after extracting known covariates.
    llm : LM
        Lazy-loaded language model instance used for covariate extraction.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.

    """

    def __init__(
        self, model_cfg: DictConfig, max_concurrent_workers: int | None = None
    ) -> None:
        super().__init__(model_cfg, max_concurrent_workers)
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
            experiment=context.experiment,
            source_name=context.source_name,
            outcome=context.outcome,
            extract_type=ExtractType.KNOWNS,
            llm=self.llm,
            model_name=self._model_name,
            save_path=context.save_path,
            exp_name=context.exp_name,
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
    max_concurrent_workers : int | None, optional, default=None
        Maximum number of concurrent workers for processing. If None, defaults to 10.

    Attributes
    ----------
    max_concurrent_workers : int | None
        Maximum number of concurrent workers for processing.
    data : pd.DataFrame | None
        DataFrame containing the processed data after imputing missing covariates.
    llm : LM
        Lazy-loaded language model instance used for covariate extraction.
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    stage_name : str
        Name of the stage, derived from the class name.
    """

    def __init__(
        self, model_cfg: DictConfig, max_concurrent_workers: int | None = None
    ) -> None:
        super().__init__(model_cfg, max_concurrent_workers)
        self.extract_type = ExtractType.IMPUTATIONS.value

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        response_format = create_response_format(
            "ImputationsResponse",
            keys=context.experiment.covariate_names,
            # types={"Country": str, "Duration": int},
        )
        self.data = await extract_covariates(
            input_df=data,
            experiment=context.experiment,
            source_name=context.source_name,
            outcome=context.outcome,
            extract_type=ExtractType.IMPUTATIONS,
            llm=self.llm,
            model_name=self._model_name,
            save_path=context.save_path,
            exp_name=context.exp_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        self.data = context.experiment.discretize(self.data)
        logger.info(f"Final: {len(self.data)} reports after imputation.")
        return self.data


async def extract_covariates(  # noqa: PLR0912
    input_df: pd.DataFrame,
    experiment: "Experiment",
    source_name: str,
    outcome: str,
    extract_type: ExtractType,
    llm: "LM",
    model_name: str,
    save_path: str,
    exp_name: str,
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
    experiment : Experiment
        Experiment instance containing metadata and prompt templates.
    source_name : str
        Name of the source from which the reports are curated.
    outcome : str
        The outcome variable for which information is being extracted.
    extract_type : ExtractType
        The type of extraction to perform (e.g., relevance, treatment-outcome,
        known covariates, or imputations).
    llm : LM
        Language model instance used for processing the reports.
    model_name : str
        Name of the language model being used.
    save_path : str
        Base path where the processed data for this experiment will be saved.
    exp_name: str
        Identifier string for a particular run, included in results directory name.
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
        save_path, experiment.nct_id, exp_name, model_name, extract_type.value, outcome
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
            experiment,
            extract_type.value,
            outcome,
            source_name,
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

    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)

    return pd.DataFrame()  # Return empty DataFrame if file not found


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

    return experiment.build_prompt_for_report(
        prompt_type=prompt_type,
        outcome=outcome,
        source_name=source_name,
        report=row["report"],
        return_format="messages",
    )


def _result_processor(
    row: pd.Series,
    response: "ResponseType",
    extract_type: ExtractType,
    response_format: BaseModel | None = None,
) -> dict[str, Any] | None:
    """Process the LLM response and combine it with the original row data."""
    response_text = get_message_content(response)[0]
    if response_text is None:
        logging.warning(
            f"No content in LLM response for row index {row.name if hasattr(row, 'name') else 'unknown'}"
        )
        return None

    try:
        if response_format:
            parsed_data = response_format.model_validate_json(response_text).model_dump(
                mode="json"
            )
        else:  # No Pydantic validation, just return raw text or a simple dict
            parsed_data = {"llm_response": response_text}
    except Exception as e:
        logging.error(
            "Failed to validate/parse LLM response for row index "
            f"{row.name if hasattr(row, 'name') else 'unknown'}: {e}. "
            f"Response text: '{response_text[:200]}...'"
        )
        return None  # Signal error

    if extract_type == ExtractType.IMPUTATIONS:
        # append "_imputed" to each key in the parsed data
        parsed_data = {f"{key}_imputed": value for key, value in parsed_data.items()}

    # Combine original row data with parsed LLM data
    parsed_row_data = {"index": row.name}
    parsed_row_data.update(row.to_dict())
    parsed_row_data.update(parsed_data)

    return parsed_row_data


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
    llm: "LM",
    pbar: tqdm,
    extract_type: ExtractType,
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

            result = await llm(
                messages=messages,
                response_format=response_format or {"type": "json_object"},
            )
            processed_result = _result_processor(
                row, result, extract_type, response_format=response_format
            )

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
