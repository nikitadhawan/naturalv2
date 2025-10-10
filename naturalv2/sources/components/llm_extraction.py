"""LLM extraction helpers for curation stages.

This module provides the shared, asynchronous machinery for transforming
tabular inputs into LLM prompts, invoking the model with bounded
concurrency, and writing results to disk while tracking token usage.
"""

import asyncio
import logging
import os
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd
from tqdm.asyncio import tqdm

from naturalv2.models.utils import TokenTracker
from naturalv2.pipeline.utils import _create_progress_bar, _csv_writer
from naturalv2.prompts.utils import load_prompt
from naturalv2.utils import ListResponse


if TYPE_CHECKING:
    from naturalv2.models.lm import APIModel


logger = logging.getLogger(__name__)


class ExtractType(str, Enum):
    """Fixed extraction task types used across sources.

    Notes
    -----
    Dynamic task types (for example synonym tasks parameterized by an
    attribute, such as ``"synonym_treatment"``) may still use plain strings.
    Public APIs in this module accept either this enum or string values.
    """

    CONDITION = "condition"
    SYNONYM_TREATMENT = "synonym_treatment"


async def extract_curation_info(  # noqa: PLR0912
    extraction_inputs: pd.DataFrame,
    stage_name: str,
    source_name: str,
    extract_type: ExtractType | str,
    llm: "APIModel",
    file_path: str,
    token_tracker: TokenTracker,
    max_concurrent_requests: int | None = None,
) -> pd.DataFrame:
    """Extract curation information using an LLM.

    Parameters
    ----------
    extraction_inputs : pandas.DataFrame
        Input rows to transform into prompts and process with the LLM.
    stage_name : str
        Name of the current pipeline stage, used for token accounting.
    source_name : str
        Name of the source being curated from (e.g., ``"pubmed"``, ``"reddit"``).
    extract_type : ExtractType or str
        Extraction task identifier (e.g., ``ExtractType.CONDITION`` or
        ``"synonym_treatment"``).
    llm : APIModel
        Language model client used to perform extraction.
    file_path : str
        Destination CSV path where results are appended/written.
    token_tracker : TokenTracker
        Tracker for accumulating token usage statistics.
    max_concurrent_requests : int | None, optional, default=None
        Upper bound on concurrent LLM requests. If ``None``, a sensible
        default based on the input size is used.

    Returns
    -------
    pandas.DataFrame
        The saved result table if it exists, otherwise an empty DataFrame.
    """
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path, index_col=0)
        extraction_inputs = extraction_inputs.loc[
            ~extraction_inputs.index.isin(existing_data.index)
        ]
        logger.info(
            "Found %d existing records, %d left to process.",
            len(existing_data),
            len(extraction_inputs),
        )
        if len(extraction_inputs) == 0:
            return existing_data

    # Set up the prompt and result queues for asynchronous processing
    num_samples = len(extraction_inputs)
    num_workers = min(max_concurrent_requests or 10, num_samples)
    queue_size = max(num_workers * 10, 1000)

    prompt_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    result_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

    # Create progress bars for the different tasks
    producer_pbar = _create_progress_bar(total=num_samples, desc="Creating prompts")
    worker_pbar = _create_progress_bar(total=num_samples, desc="Processing prompts")
    writer_pbar = _create_progress_bar(total=num_samples, desc="Writing results")

    # Create tasks to produce prompts
    # Normalize extract type to a string for downstream helpers
    _extract_type_str = (
        extract_type.value
        if isinstance(extract_type, ExtractType)
        else str(extract_type)
    )

    producer_task = asyncio.create_task(
        _llm_task_producer(
            prompt_queue,
            extraction_inputs,
            source_name,
            _extract_type_str,
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
                stage_name,
                token_tracker,
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
        logger.error("Error during curation: %s", e, exc_info=True)
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


async def _llm_task_producer(
    queue: asyncio.Queue,
    extraction_inputs: pd.DataFrame,
    source_name: str,
    extract_type: str,
    pbar: tqdm,
) -> None:
    """Produce prompts for the LLM from the input DataFrame.

    Parameters
    ----------
    queue : asyncio.Queue
        Queue to receive tuples of ``(index, row, messages)`` for workers.
    extraction_inputs : pandas.DataFrame
        Input rows to convert into LLM prompt messages.
    source_name : str
        Name of the source being curated (e.g., ``"pubmed"``).
    extract_type : str
        Extraction task identifier (e.g., ``"condition"``).
    pbar : tqdm
        Progress bar to update per produced prompt.
    """
    for idx, row in extraction_inputs.iterrows():
        try:
            llm_inputs = row.to_dict()
            messages = load_prompt(
                base_dir="naturalv2/prompts/templates",
                prompt_type=f"{extract_type}_{source_name}",
                return_format="responses" if "synonym" in extract_type else "messages",
                **llm_inputs,
            )

            await queue.put((idx, row, messages))
        except Exception as e:
            logging.error(
                f"Failed to format prompt for index {idx}: {type(e).__name__} - {e}",
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
    stage_name: str,
    token_tracker: TokenTracker,
) -> int:
    """Process prompts with the LLM and enqueue results.

    Parameters
    ----------
    worker_id : int
        Identifier for logging and progress tracking.
    prompt_queue : asyncio.Queue
        Queue to consume items of the form ``(index, row, messages)``.
    result_queue : asyncio.Queue
        Queue to publish processed result dicts or ``False`` on failure.
    llm : APIModel
        Language model client used to perform extraction.
    pbar : tqdm
        Progress bar to update after each processed item.
    stage_name : str
        Name of the current pipeline stage, used for token accounting.
    token_tracker : TokenTracker
        Tracker for accumulating token usage statistics.

    Returns
    -------
    int
        The number of items that failed processing.
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
                messages, response_format=ListResponse, parse_output=True
            )
            token_tracker.add(stage_name, response)

            processed_result = {"index": index}
            processed_result.update(row.to_dict())
            processed_result["llm_output"] = response.output_parsed.output

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
