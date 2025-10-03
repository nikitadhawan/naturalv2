"""Shared curation stages and utilities."""

import ast
import asyncio
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
from tqdm.asyncio import tqdm

from naturalv2.models.utils import TokenTracker
from naturalv2.pipeline.utils import _create_progress_bar, _csv_writer
from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.curation import CurationContext, CurationStage, StageState
from naturalv2.utils import ListResponse, sanitize_filename


if TYPE_CHECKING:
    from naturalv2.models.lm import APIModel


logger = logging.getLogger(__name__)


class SynonymStage(CurationStage):
    """Stage to find synonyms for a given attribute.

    This stage uses an LLM to find synonyms for a specified attribute (e.g., treatments)
    from a given source (e.g., "pubmed", "reddit").

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    source_name : str
        Name of the source being curated from (e.g., "pubmed", "reddit").
    attribute : str
        The attribute for which synonyms are to be found (e.g., "treatments").
    max_concurrent_workers : int | None, optional, default=None
        Maximum number of concurrent workers for LLM requests.
    """

    def __init__(
        self,
        *,
        attribute: str,
        llm: "APIModel",
        max_concurrent_workers: int | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the class."""
        super().__init__(name=name)
        self.attribute = attribute
        self.llm = llm
        self.max_concurrent_workers = max_concurrent_workers

        self.extract_type = f"synonym_{attribute}"

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Get synonyms for ``attribute`` found on ``source``.

        This method takes a list of Experiments and generates synonyms for their ``attribute``
        found on ``source``, using a language model.

        Parameters
        ----------
        exp_list : list[Experiment]
            List of Experiments in a study.
        context : CurationContext
            Context for the stage execution.

        Returns
        -------
        list[Experiment]
            List of experiments with updated common names and synonyms.

        Raises
        ------
        Exception
            If there is an error during the extraction process.
        """
        if not context.experiments:
            logger.warning("No experiments provided to process.")
            return state

        llm_inputs = []
        for experiment in context.experiments:
            for keyword in getattr(experiment, f"{self.attribute}_names"):
                if context.source_name not in getattr(
                    experiment, f"{self.attribute}_common_names"
                ):
                    getattr(experiment, f"{self.attribute}_common_names").update(
                        {context.source_name: {}}
                    )
                if (
                    keyword
                    in getattr(experiment, f"{self.attribute}_common_names")[
                        context.source_name
                    ]
                ):
                    logger.debug(
                        f"Skipping {keyword} for {experiment.nct_id} - already have common names"
                    )
                    continue
                desc = getattr(experiment, f"{self.attribute}_desc")[keyword]
                llm_inputs.append(
                    {
                        "nct_id": experiment.nct_id,
                        "keyword": keyword,
                        "trial_title": experiment.title,
                        f"{self.attribute}_desc": desc,
                        "drugbank_names": experiment.drugbank_names[keyword],
                        "source": context.source_name,
                    }
                )

        input_df = pd.DataFrame(llm_inputs)
        if input_df.empty:
            logger.warning("No synonym tasks to process.")
            return state

        # Set up file path for saving results
        condition_safe = sanitize_filename(context.condition.lower())
        save_dir = os.path.join(context.save_dir, "curation_results", condition_safe)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(
            save_dir, f"{self.attribute}_synonyms_{context.experiment_name}.csv"
        )

        exp_dir = os.path.join(context.save_dir, "experiments")

        output_df = await extract_curation_info(
            input_df=input_df,
            stage_name=self.stage_name,
            source_name=context.source_name,
            extract_type=self.extract_type,
            llm=self.llm,
            file_path=file_path,
            token_tracker=context._token_tracker,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        num_keywords, num_synonyms = 0, 0
        for experiment in context.experiments:
            synonyms_dict = {}
            for keyword in getattr(experiment, f"{self.attribute}_names"):
                keyword_rows = output_df[output_df["keyword"] == keyword]
                if len(keyword_rows) > 0:
                    synonyms = ast.literal_eval(keyword_rows.iloc[0]["llm_output"])
                    synonyms_dict[keyword] = synonyms
                    num_keywords += 1
                    num_synonyms += len(synonyms)
            getattr(experiment, f"{self.attribute}_common_names")[
                context.source_name
            ].update(synonyms_dict)

            exp_file = os.path.join(exp_dir, f"{experiment.nct_id}.yaml")
            experiment.to_yaml(exp_file)

        state.metadata["num_keywords"] = num_keywords
        state.metadata["num_synonyms"] = num_synonyms
        return state


async def extract_curation_info(  # noqa: PLR0912
    input_df: pd.DataFrame,
    stage_name: str,
    source_name: str,
    extract_type: str,
    llm: "APIModel",
    file_path: str,
    token_tracker: TokenTracker,
    max_concurrent_requests: int | None = None,
) -> pd.DataFrame:
    """Extract curation information using an LLM.

    This function processes an input DataFrame using an LLM to extract curation information.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing input data for the LLM.
    stage_name : str
        Name of the current stage in the pipeline.
    source_name : str
        Name of the source being curated from (e.g., "pubmed", "reddit").
    extract_type : str
        Type of extraction being performed (e.g., "condition", "synonym_treatments").
    llm : APIModel
        An instance of the language model to use for extraction.
    file_path : str
        Path to save the output CSV file.
    token_tracker : TokenTracker
        Tracker for tokens used in LLM calls.
    max_concurrent_requests : int | None, optional, default=None
        Maximum number of concurrent requests to the LLM.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted curation information.

    Raises
    ------
    Exception
        If there is an error during the extraction process.
    """
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
    num_workers = min(max_concurrent_requests or 10, num_samples)
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
            source_name,
            extract_type,
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
        logger.error(f"Error during curation: {e}", exc_info=True)
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
    input_df: pd.DataFrame,
    source_name: str,
    extract_type: str,
    pbar: tqdm,
) -> None:
    """Produce prompts for the LLM from the input DataFrame."""
    for idx, row in input_df.iterrows():
        try:
            llm_inputs = row.to_dict()
            # Format the data for the LLM prompt
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
