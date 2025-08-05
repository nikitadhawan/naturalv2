"""Conditional extraction stages of the NATURAL pipeline."""

import asyncio
import logging
import os
import re
import string
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.special import softmax
from tqdm import tqdm

from naturalv2.evals.experiment import Experiment
from naturalv2.models.lm import build_lm_instance_from_cfg, get_prompt_logprobs
from naturalv2.pipeline import INCLUSION_COL_NAME, OUTCOME_COL_NAME, TREATMENT_COL_NAME
from naturalv2.pipeline.natural import PipelineContext, PipelineStage
from naturalv2.pipeline.utils import _create_progress_bar, _csv_writer
from naturalv2.utils import convert_enum_to_dicts, enumerate_strings, get_save_path


if TYPE_CHECKING:
    from naturalv2.models.lm import LM, ResponseType

logger = logging.getLogger(__name__)


class ConditionalsExtractType(str, Enum):
    """Enumeration for types of conditional probabilities to extract."""

    TY_GIVEN_X = "ty_given_x"
    Y_GIVEN_TX = "y_given_tx"
    INCLUSION = "inclusion"
    NONE = None  # No extraction needed


class ConditionalExtractionStage(PipelineStage):
    """Stage for computing conditional probabilities.

    At this stage, an LLM capabale of returning prompt log probabilities is
    used to compute conditional probabilities. This stage can be used to compute
    either the probability of the treatment and outcome given the covariates (P(T,Y|X))
    or the probability of the outcome given the treatment and covariates (P(Y|T,X)).

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    length_norm : bool, optional, default=False
        Whether to normalize the log probabilities by the length of the prompt,
        by default False.
    max_concurrent_workers : int | None, optional, default=None
        Maximum number of concurrent workers to use for making asynchronous
        calls to the LLM. If None, defaults to the minimum of 10 or the number of
        samples in the input data. This can be used to limit the number of concurrent
        requests to the LLM.

    """

    def __init__(
        self,
        model_cfg: DictConfig,
        length_norm: bool = False,
        max_concurrent_workers: int | None = None,
    ) -> None:
        """Initialize the conditional extraction stage."""
        super().__init__(model_cfg)
        self.length_norm = length_norm
        self.max_concurrent_workers = max_concurrent_workers

    def get_language_model(self) -> "LM":
        """Instantiate the language model for conditional extraction.

        Returns
        -------
        LM
            An instance of the language model configured for conditional extraction.

        """
        model = build_lm_instance_from_cfg(self.model_cfg)

        # Set mandatory parameters for conditional extraction
        if model.completion_type != "text":
            logger.warning(
                f"Model {self._model_name} is not configured for text completion, "
                "which is required for conditional/inclusion probability extraction."
                "Setting completion type to 'text'."
            )
            model.completion_type = "text"  # Ensure text completion type

        if "prompt_logprobs" not in model._request_params or (
            "prompt_logprobs" in model._request_params
            and model._request_params["prompt_logprobs"] not in [0, False]
        ):
            logger.warning(
                "The conditional/inclusion probability extraction stages requires "
                "the log probabilities of the tokens in the input prompt but the "
                "model is not configured to return them. "
                "Setting prompt_logprobs to 0."
            )
            model._request_params["prompt_logprobs"] = 0

        if "max_tokens" not in model._request_params:
            logger.warning(
                "The conditional/inclusion probability extraction stages does not "
                "require the model to generate any tokens. Setting max_tokens to 1 "
                "to improve throughput."
            )
            model._request_params["max_tokens"] = 1  # No generation needed
        if (
            "max_tokens" in model._request_params
            and model._request_params["max_tokens"] > 1
        ):
            logger.warning(
                "The conditional/inclusion probability extraction stages does not "
                "require the model to generate any tokens. Consider setting max_tokens "
                "to 1 to improve throughput."
            )

        return model

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Process the input data to extract conditional probabilities.

        Parameters
        ----------
        data : pd.DataFrame
            Input data containing reports to extract conditional probabilities from.
        context : PipelineContext
            Context for the pipeline execution, containing experiment and
            configuration details.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted conditional probabilities.

        Raises
        ------
        ValueError
            If the estimator type in context is not supported for conditional extraction


        Notes
        -----
        - The input DataFrame is discretized before the extraction of conditional
          probabilities.
        - The input DataFrame must contain a 'report' column with the text of the
          reports.
        """
        extract_type_map = {
            "NaturalIPW": ConditionalsExtractType.TY_GIVEN_X,
            "NaturalOI": ConditionalsExtractType.Y_GIVEN_TX,
            "NaturalMC": ConditionalsExtractType.TY_GIVEN_X,  # Sample T,Y from conditional for MC
        }
        extract_type = extract_type_map.get(context.estimator_type)
        if extract_type is None:
            raise ValueError(
                f"Estimator type '{context.estimator_type}' is not supported for "
                "conditional extraction. Supported estimators are: "
                f"{list(extract_type_map.keys())}."
            )

        self.data = await extract_conditionals(
            data,
            context.experiment,
            context.source_name,
            context.outcome,
            self.llm,
            self._model_name,
            context.save_path,
            extract_type,
            length_norm=self.length_norm,
            max_concurrent_requests=self.max_concurrent_workers,
        )
        logger.info(
            f"Extracted {extract_type.value} conditional probabilities from "
            f"{len(self.data)} reports."
        )
        return self.data


class InclusionProbStage(ConditionalExtractionStage):
    """Stage for extracting inclusion probabilities.

    This stage uses an LLM capabale of returning prompt log probabilities
    to compute the probabilities of the covariates matching the inclusion criteria
    given a text report (P(X ∈ Inclusion | report)).

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    length_norm : bool, optional, default=False
        Whether to normalize the log probabilities by the length of the prompt,
        by default False.
    max_concurrent_workers : int | None, optional, default=None
        Maximum number of concurrent workers to use for making asynchronous
        calls to the LLM. If None, defaults to the minimum of 10 or the number of
        samples in the input data. This can be used to limit the number of concurrent
        requests to the LLM.

    """

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Process the input data to extract inclusion probabilities.

        Parameters
        ----------
        data : pd.DataFrame
            Input data containing reports to extract inclusion probabilities from.
        context : PipelineContext
            Context for the pipeline execution, containing experiment and
            configuration details.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted inclusion probabilities.

        """
        extract_type = ConditionalsExtractType.INCLUSION
        self.data = await extract_conditionals(
            data,
            context.experiment,
            context.source_name,
            context.outcome,
            self.llm,
            self._model_name,
            context.save_path,
            extract_type,
            length_norm=self.length_norm,
            max_concurrent_requests=self.max_concurrent_workers,
        )
        logger.info(
            f"Extracted {extract_type.value} conditional probabilities from "
            f"{len(self.data)} reports."
        )
        return self.data


async def extract_conditionals(  # noqa: PLR0912
    input_df: pd.DataFrame,
    experiment: Experiment,
    source_name: str,
    outcome: str,
    llm: "LM",
    model_name: str,
    save_path: str,
    extract_type: ConditionalsExtractType,
    length_norm: bool = False,
    max_concurrent_requests: int | None = None,
) -> pd.DataFrame:
    """Extract conditional probabilities from input data.

    This function computes conditional probabilities based on the specified
    `extract_type` using an LLM. It prepares the input data, generates prompts,
    and processes the responses to extract the desired probabilities.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input data containing reports and other features with which to extract
        conditional probabilities.
    experiment : Experiment
        Experiment instance containing the options and methods for the experiment.
    source_name : str
        Name of the source from which the data was collected.
    outcome : str
        The outcome variable for which the conditional probabilities are computed.
    llm : LM
        Language model instance used to compute the conditional probabilities.
    model_name : str
        Name of the language model used for conditional extraction.
    save_path : str
        Path where the results will be saved.
    extract_type : ConditionalsExtractType
        Type of conditional probabilities to extract. Options are:
        - ConditionalsExtractType.TY_GIVEN_X: P(T,Y|X)
        - ConditionalsExtractType.Y_GIVEN_TX: P(Y|T,X)
        - ConditionalsExtractType.INCLUSION: P(X ∈ Inclusion | report)
        - ConditionalsExtractType.NONE: No extraction needed
    length_norm : bool, default=False
        Whether to normalize the log probabilities by the length of the prompt.
    max_concurrent_requests : int | None, optional, default=None
        Maximum number of concurrent requests to the LLM. If None, defaults to
        the minimum of 10 or the number of samples in the input data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original input data with additional columns
        for the extracted conditional probabilities.

    """

    # Return input if no extraction needed
    if extract_type == ConditionalsExtractType.NONE:
        logger.info("No conditional extraction needed. Returning input dataframe.")
        return input_df

    # Generate save path
    disallowed_chars = r"[^\w\-.]"
    file_path = get_save_path(
        save_path,
        experiment.nct_id,
        model_name,
        re.sub(disallowed_chars, "_", f"inclusion_probs_{outcome.lower()}")
        if extract_type == ConditionalsExtractType.INCLUSION
        else re.sub(disallowed_chars, "_", f"{extract_type.value}_{outcome}".lower()),
    )

    if os.path.exists(file_path):
        logging.info(f"File {file_path} already exists. Loading existing results.")
        return pd.read_csv(file_path, index_col=0)

    discretized_cols = [
        col + "_discretized"
        for col in experiment.covariate_names
        + experiment.extended_covariate_names
        + [TREATMENT_COL_NAME, OUTCOME_COL_NAME]
    ]
    if not input_df.empty and not all(
        col in input_df.columns for col in discretized_cols
    ):
        # Discretize input dataframe
        input_df = experiment.discretize(input_df)

    # Features to enumerate based on extraction type
    conditional_feature_mapping = {
        "ty_given_x": [TREATMENT_COL_NAME, outcome],
        "y_given_tx": [outcome],
        "inclusion": [INCLUSION_COL_NAME],
    }
    features_to_enumerate = conditional_feature_mapping[extract_type.value]
    _, interleaved_options, idx_to_feat = _prepare_for_conditional_extraction(
        experiment, features_to_enumerate
    )

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

    # Create task for producing prompts
    producer_task = asyncio.create_task(
        _llm_task_producer(
            prompt_queue,
            input_df,
            experiment,
            extract_type,
            interleaved_options,
            outcome,
            source_name,
            producer_pbar,
        ),
        name="LLM-Task-Producer",
    )

    # Create worker tasks to process prompts
    # Account for the number of interleaved options as each worker will process
    # a batch of prompts corresponding to these at once.
    worker_tasks = [
        asyncio.create_task(
            _prompt_processor(
                worker_id,
                prompt_queue,
                result_queue,
                llm,
                extract_type,
                length_norm,
                idx_to_feat,
                worker_pbar,
            ),
            name=f"Prompt-Processor-{worker_id}",
        )
        for worker_id in range(max(1, num_workers // len(interleaved_options)))
    ]

    # Create a CSV writer task to write results to a file
    csv_writer_task = asyncio.create_task(
        _csv_writer(result_queue, file_path, writer_pbar), name="CSV-Writer"
    )

    try:
        # Wait for the producer to finish producing tasks
        await producer_task

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

    return pd.read_csv(file_path, index_col=0)


def _prepare_for_conditional_extraction(
    experiment: Experiment, features_to_enumerate: list[str]
) -> tuple[list[str], list[str], list[dict[str, str] | dict[str, int]]]:
    """Prepare data structures for conditional extraction."""
    # Enumerate answer combinations for the specified keys
    answer_combinations = enumerate_strings(
        {key: experiment.options[key] for key in features_to_enumerate}
    )

    # Convert the combination to a list of dictionaries, where each dictionary
    # has the possible answers for each key in `keys_to_enumerate`
    enum_dicts = convert_enum_to_dicts(answer_combinations, features_to_enumerate)

    # Use the numerical representation of the answers for each key in each dictionary
    index_to_features = [
        experiment.apply_transform(enum_dict, repr_type="numeric")
        for enum_dict in enum_dicts
    ]

    # Build interleaved multiple choice questions
    # For each answer combination, add the question, choices and the answer
    # to a string
    question_prompts = experiment.get_question_prompts()
    interleaved_mcqa = _build_interleaved_multiple_choice_questions(
        {key: question_prompts[key] for key in features_to_enumerate},
        {key: experiment.options[key] for key in features_to_enumerate},
        answer_combinations,
        features_to_enumerate,
    )

    return answer_combinations, interleaved_mcqa, index_to_features


def _build_interleaved_multiple_choice_questions(
    question_lookup: dict[str, str],
    answer_choices: dict[str, list[str]],
    answer_combinations: list[str],
    enumeration_keys: list[str],
) -> list[str]:
    """Build interleaved multiple choice questions from answer combinations.

    For every answer combination, interleave the options for each question, including
    the question text and the corresponding answer choice.
    """
    all_interleaved_options = []
    for option in answer_combinations:
        interleaved_enum = "\n\nMultiple Choice Questions"  # Header for MCQA
        for index in range(len(enumeration_keys)):
            key = enumeration_keys[index]

            # Add question
            interleaved_enum += "\n\nQ: " + question_lookup[key]

            # Add options/choices
            num_choices = len(answer_choices[enumeration_keys[index]])
            option_labels = _get_alphabet_labels(num_choices)
            interleaved_enum += "\nOptions: "
            for i in range(num_choices):
                interleaved_enum += option_labels[i] + answer_choices[key][i] + " "

            # Add the answer choice
            split_option = [i.split(":") for i in option.split(",")]
            interleaved_enum += "\nA: " + split_option[index][1][1:]

        all_interleaved_options.append(interleaved_enum)
    return all_interleaved_options


async def _llm_task_producer(
    queue: asyncio.Queue,
    input_df: pd.DataFrame,
    experiment: Experiment,
    extract_type: ConditionalsExtractType,
    interleaved_mcqa: list[str],
    outcome: str,
    source_name: str,
    pbar: tqdm,
) -> None:
    """Produce prompts for the LLM based on the input DataFrame.

    This function formats the input data into prompts for the LLM, including
    the report text and any relevant covariates or treatment information based on
    the specified `extract_type`. It handles exceptions during prompt formatting
    and puts the prompts into the queue for processing by workers.
    """
    for idx, row in input_df.iterrows():
        try:
            report = row["report"]
            if extract_type != ConditionalsExtractType.INCLUSION:
                discretized_covariate_names = [
                    col + "_discretized" for col in experiment.covariate_names
                ]
                to_sample = (
                    discretized_covariate_names
                    if extract_type == ConditionalsExtractType.TY_GIVEN_X
                    else discretized_covariate_names
                    + [TREATMENT_COL_NAME + "_discretized"]
                )
                to_transform = {
                    k.replace("_discretized", ""): v
                    for k, v in row[to_sample].to_dict().items()
                }

                # Add question and answers for covariates to the report
                covariate_answers = experiment.apply_transform(
                    to_transform, repr_type="language"
                )
                question_prompts = experiment.get_question_prompts()
                qa_text = "\n\nQuestions and their correct answers:"
                for key in covariate_answers:
                    qa_text += (
                        "\nQ: "
                        + question_prompts[key]
                        + "A: "
                        + str(covariate_answers[key])
                        + "."
                    )
                report += qa_text

            # Repeat the report for all interleaved options
            reports = [report] * len(interleaved_mcqa)

            # Combine the report with the interleaved options
            reports_with_options = [
                report + option for report, option in zip(reports, interleaved_mcqa)
            ]

            # Format the prompts for the LLM
            prompts: list[str] = []
            for report_with_option in reports_with_options:
                prompt: str = experiment.build_prompt_for_report(
                    "conditionals",
                    outcome=outcome,
                    source_name=source_name,
                    report=report_with_option,
                    return_format="prompt",
                )
                prompts.append(prompt)

            # Put the prompts into the queue as a unit (must be processed together)
            await queue.put((idx, row, prompts))
        except Exception as e:
            logging.error(
                f"Failed to format prompts for item at index {idx}: {type(e).__name__} - {e}",
                exc_info=True,
            )
            # Put a None message to signal failure
            await queue.put((idx, row, None))
        finally:
            pbar.update(1)


async def _prompt_processor(
    worker_id: int,
    prompt_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    llm: "LM",
    extract_type: ConditionalsExtractType,
    length_norm: bool,
    index_to_features: list[dict[str, Any]],
    pbar: tqdm,
) -> int:
    """Worker function to process prompts.

    This function calls the LLM with formatted prompts and processes the results.
    """
    error_count = 0
    while True:
        try:
            index, row, prompts = await prompt_queue.get()
            if index is None and row is None:
                prompt_queue.task_done()
                break
        except asyncio.CancelledError:
            break

        try:
            if prompts is None:  # prompt formatting failed
                logging.error(
                    f"Worker {worker_id} received None prompts for item at index {index}"
                )
                error_count += 1
                await result_queue.put(False)  # Signal failure to writer
                continue

            # Make LLM call
            # The use of asyncio.TaskGroup ensures atomicity - if one of the calls
            # fails, all tasks are cancelled.
            async with asyncio.timeout(300.0), asyncio.TaskGroup() as tg:
                tasks: list[asyncio.Task] = []
                for prompt in prompts:
                    tasks.append(tg.create_task(llm(prompt=prompt), name="LLM-Call"))

            responses: list["ResponseType"] = [task.result() for task in tasks]
            processed_results = _result_processor(
                row,
                responses,
                length_norm=length_norm,
                index_to_features=index_to_features,
                extract_type=extract_type,
            )

            await result_queue.put(processed_results)
        except BaseExceptionGroup as eg:  # TaskGroup wraps exceptions in ExceptionGroup
            # log all exceptions
            for e in eg.exceptions:
                logging.error(
                    f"Worker {worker_id} encountered an error while processing "
                    f"item at index {index}: {type(e).__name__} - {e}",
                    exc_info=True,
                )
            error_count += 1
            await result_queue.put(False)
        finally:
            pbar.update(1)
            prompt_queue.task_done()

    return error_count


def _result_processor(
    row: pd.Series,
    responses: list["ResponseType"],
    length_norm: bool,
    index_to_features: list[dict[str, Any]],
    extract_type: ConditionalsExtractType,
) -> dict[str, Any]:
    """Process the LLM response and combine it with the original row data.

    Computes the conditional probabilities based on the responses and
    returns a dictionary with the original row data and the computed probabilities.

    """
    logprobs = []
    for response in responses:
        prompt_logprobs_obj = get_prompt_logprobs(response)
        if prompt_logprobs_obj is None:
            continue

        logprob = sum(prompt_logprobs_obj.logprobs)
        if length_norm:
            logprob = logprob / len(prompt_logprobs_obj.decoded_tokens)
        logprobs.append(logprob)

    probs: np.ndarray = softmax(np.array(logprobs), axis=0)
    sample_index = np.random.choice(len(probs), p=probs)

    sampled_features = index_to_features[sample_index]
    sampled_features = {
        f"{key}_sampled": value for key, value in sampled_features.items()
    }

    # Add the sampled features to the row data
    parsed_row_data: dict[str, Any] = row.to_dict()
    parsed_row_data.update(sampled_features)
    parsed_row_data[f"{extract_type.value}_probs"] = probs.tolist()

    return parsed_row_data


def _get_alphabet_labels(n: int) -> list[str]:
    """Generate alphabet labels for multiple choice options.

    For a given number n, generate labels like a), b), ..., z), aa), ab), etc.
    """
    labels = []
    alphabet = string.ascii_lowercase
    for i in range(n):
        label = ""
        idx = i
        while True:  # allow for more than 26 labels
            label = alphabet[idx % 26] + label
            idx = idx // 26 - 1
            if idx < 0:
                break
        labels.append(f"{label}) ")
    return labels
