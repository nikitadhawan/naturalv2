"""NATURAL Pipeline."""

import ast
import asyncio
import importlib.resources
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import pandas as pd
import yaml
from omegaconf import DictConfig
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table
from tqdm.asyncio import tqdm

from naturalv2.models.lm import LM, build_lm_instance_from_cfg, extract_list_response
from naturalv2.pipeline.utils import _create_progress_bar, _csv_writer
from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.pubmed import PubMedSet
from naturalv2.sources.reddit import RedditSource
from naturalv2.utils import ListResponse, sanitize_filename


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment


logger = logging.getLogger(__name__)


@dataclass
class CurationContext:
    """Context for the pipeline execution."""

    #: Disease category data is curated for
    condition: str

    #: All trials included in the curation.
    all_ncts: list[str]

    #: Train/val/test split that trials belong to.
    splits: list[str]

    #: An instance of the ``RedditSource`` or ``PubMedSet`` class.
    source_dataset: Union[RedditSource, PubMedSet]

    #: Whether or not the curated data should be filtered according to date.
    filter_by_date: bool

    #: The path where the processed data will be saved.
    save_path: str

    # Identifier string for a particular run, included in curated data directory name.
    exp_name: str


class CurationStage(ABC):
    """Base class for stages in a pipeline.

    Each stage executes a curation step and returns either information or data.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.

    Attributes
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    llm : LM | None
        Lazy-loaded language model instance.
    stage_name : str
        Name of the stage, derived from the class name.

    """

    def __init__(self, model_cfg: DictConfig, source_name: str) -> None:
        """Initialize the pipeline stage with model configuration."""
        self.model_cfg = model_cfg
        self.source_name = source_name
        self._llm: LM | None = None
        self._model_name: str = model_cfg.get("model_name", "")
        self._stats: dict[str, Any] = {}
        self.extract_type = None

    @property
    def stage_name(self) -> str:
        """Return the name of the stage."""
        return self.__class__.__name__

    @property
    def llm(self) -> LM:
        """Lazy-loaded language model property."""
        if self._llm is None:
            self._llm = self.get_language_model()
        return self._llm

    def get_language_model(self) -> LM:
        """Return the language model used in this stage.

        Returns
        -------
        LM
            An instance of the language model configured for this stage.
        """
        return build_lm_instance_from_cfg(self.model_cfg)

    @abstractmethod
    async def process(
        self, exp_list: list["Experiment"], context: CurationContext
    ) -> Any:
        """Use the input data and context to return information or curated data.

        Parameters
        ----------
        exp_list : list[Experiment]
            List of Experiments in a study.
        context : PipelineContext
            Context for the pipeline execution.

        Returns
        -------
        Any
            Information required for curation or curated data.
        """
        pass

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

    def get_stats(self) -> dict[str, Any]:
        """Return a dictionary of statistics collected during processing."""
        if "cost" not in self._stats:
            self._stats["cost"] = self.llm.cost
        if "total_prompt_tokens" not in self._stats:
            self._stats["total_prompt_tokens"] = self.llm.total_prompt_tokens
        if "total_completion_tokens" not in self._stats:
            self._stats["total_completion_tokens"] = self.llm.total_completion_tokens

        return self._stats

    def render_stats_table(self) -> None:
        """Print the statistics collected during processing in a table format."""
        stats_table = Table(title=f"Statistics for {self.stage_name}")
        stats_table.add_column("Key", style="cyan")
        stats_table.add_column("Value", style="magenta")

        for key, value in self.get_stats().items():
            stats_table.add_row(str(key), Pretty(value, expand_all=True))

        for key, value in self.prompt_template().items():
            stats_table.add_row(str(key), str(value))

        console = Console()
        console.print(stats_table)

    def add_stat(self, key: str, value: Any) -> None:
        """Add a statistic to the stage's stats dictionary.

        Parameters
        ----------
        key : str
            The key for the statistic.
        value : Any
            The value of the statistic.
        """
        self._stats[key] = value


class ConditionStage(CurationStage):
    def __init__(
        self,
        model_cfg: DictConfig,
        source_name: str,
        max_concurrent_workers: int | None = None,
    ) -> None:
        """Initialize the class."""
        super().__init__(model_cfg, source_name)
        self.source_name = source_name
        self.max_concurrent_workers = max_concurrent_workers
        self.data: Any = None
        self.extract_type = "condition"

    async def process(
        self, exp_list: list["Experiment"], context: CurationContext
    ) -> list[str]:
        """Find data dumps related to condition keywords.

        This method optionally uses an LLM to determine queries or other information
        to download data from ``source_name``.

        Parameters
        ----------
        exp_list : list[Experiment]
            List of Experiments in a study.
        context : CurationContext
            Context for the stage execution.

        Returns
        -------
        list[str]
            List of strings containing information or queries to download data.

        Raises
        ------
        Exception
            If there is an error during the extraction process.
        """
        # Get unique keywords
        all_condition_keywords = []
        for exp in exp_list:
            if exp.conditions:
                all_condition_keywords.extend(exp.conditions)
        all_condition_keywords = list(set(all_condition_keywords))

        # Get condition query data from source dataset
        condition_queries = await context.source_dataset.condition_filter(
            all_condition_keywords
        )

        if condition_queries.empty:
            logger.warning("No condition query data found to process.")
            return pd.DataFrame()

        # Set up file path for saving results
        condition_safe = sanitize_filename(context.condition.lower())
        save_dir = os.path.join(context.save_path, "curation_results", condition_safe)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"condition_queries_{context.exp_name}.csv")

        output_df = await extract_curation_info(
            input_df=condition_queries,
            source_name=self.source_name,
            extract_type=self.extract_type,
            llm=self.llm,
            file_path=file_path,
            max_concurrent_requests=self.max_concurrent_workers,
        )
        condition_metadata = []
        for output in output_df["llm_output"]:
            condition_metadata.extend(ast.literal_eval(output))

        condition_metadata = list(set(condition_metadata))
        self.add_stat("len_metadata", len(condition_metadata))
        return condition_metadata


class SynonymStage(CurationStage):
    def __init__(
        self,
        model_cfg: DictConfig,
        source_name: str,
        attribute: str,
        max_concurrent_workers: int | None = None,
    ) -> None:
        """Initialize the class."""
        super().__init__(model_cfg, source_name)
        self.source_name = source_name
        self.attribute = attribute
        self.max_concurrent_workers = max_concurrent_workers
        self.data: Any = None
        self.extract_type = f"synonym_{attribute}"

    async def process(
        self, exp_list: list["Experiment"], context: CurationContext
    ) -> list["Experiment"]:
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
        if not exp_list:
            logger.warning("No experiments provided to process.")
            return {}

        llm_inputs = []
        for exp in exp_list:
            for keyword in getattr(exp, f"{self.attribute}_names"):
                if self.source_name not in getattr(
                    exp, f"{self.attribute}_common_names"
                ):
                    getattr(exp, f"{self.attribute}_common_names").update(
                        {self.source_name: {}}
                    )
                if (
                    keyword
                    in getattr(exp, f"{self.attribute}_common_names")[self.source_name]
                ):
                    logger.debug(
                        f"Skipping {keyword} for {exp.nct_id} - already have common names"
                    )
                    continue
                desc = getattr(exp, f"{self.attribute}_desc")[keyword]
                llm_inputs.append(
                    {
                        "nct_id": exp.nct_id,
                        "keyword": keyword,
                        "trial_title": exp.title,
                        f"{self.attribute}_desc": desc,
                        "drugbank_names": exp.drugbank_names[keyword],
                        "source": self.source_name,
                    }
                )

        input_df = pd.DataFrame(llm_inputs)
        if input_df.empty:
            logger.warning("No synonym tasks to process.")
            return exp_list

        # Set up file path for saving results
        condition_safe = sanitize_filename(context.condition.lower())
        save_dir = os.path.join(context.save_path, "curation_results", condition_safe)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(
            save_dir, f"{self.attribute}_synonyms_{context.exp_name}.csv"
        )

        exp_dir = os.path.join(context.save_path, "experiments")

        output_df = await extract_curation_info(
            input_df=input_df,
            source_name=self.source_name,
            extract_type=self.extract_type,
            llm=self.llm,
            file_path=file_path,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        num_keywords, num_synonyms = 0, 0
        for exp in exp_list:
            synonyms_dict = {}
            for keyword in getattr(exp, f"{self.attribute}_names"):
                keyword_rows = output_df[output_df["keyword"] == keyword]
                if len(keyword_rows) > 0:
                    synonyms = ast.literal_eval(keyword_rows.iloc[0]["llm_output"])
                    synonyms_dict[keyword] = synonyms
                    num_keywords += 1
                    num_synonyms += len(synonyms)
            getattr(exp, f"{self.attribute}_common_names")[self.source_name].update(
                synonyms_dict
            )

            exp_file = os.path.join(exp_dir, f"{exp.nct_id}.yaml")
            exp.to_yaml(exp_file)

        self.add_stat("num_keywords", num_keywords)
        self.add_stat("num_synonyms", num_synonyms)
        return exp_list


async def extract_curation_info(  # noqa: PLR0912
    input_df: pd.DataFrame,
    source_name: str,
    extract_type: str,
    llm: LM,
    file_path: str,
    max_concurrent_requests: int | None = None,
) -> pd.DataFrame:
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
                return_format="messages",
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
    llm: LM,
    pbar: tqdm,
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

            result = await llm(messages=messages, response_format=ListResponse)
            llm_output = extract_list_response(result)[0]
            processed_result = {"index": index}
            processed_result.update(row.to_dict())
            processed_result["llm_output"] = llm_output

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
