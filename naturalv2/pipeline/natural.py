"""NATURAL Pipeline."""

import json
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.models import LM

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Exception raised when there is an error in processing a stage in the pipeline."""

    pass


@dataclass
class PipelineContext:
    """Context for the pipeline execution."""

    #: An instance of the ``Experiment`` class containing experiment details.
    experiment: "Experiment"

    #: The name of the source from which data is being processed.
    source_name: str

    #: The type of estimator being used in the pipeline.
    estimator_type: Literal["NaturalIPW", "NaturalMC", "NaturalOI"]

    #: The outcome variable being processed in the pipeline.
    outcome: str

    #: The path where the processed data will be saved.
    save_path: str

    # Identifier string for a particular run, included in results directory name.
    exp_name: str


class PipelineStage(ABC):
    """Base class for stages in a pipeline.

    Each stage processes input data and returns transformed data.

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

    def __init__(self, model_cfg: DictConfig) -> None:
        """Initialize the pipeline stage with model configuration."""
        self.model_cfg = model_cfg
        self._llm: "LM" | None = None
        self._model_name: str = model_cfg.get("model_name", "")
        self._stats: dict[str, Any] = {}

    @property
    def stage_name(self) -> str:
        """Return the name of the stage."""
        return self.__class__.__name__

    @property
    def llm(self) -> "LM":
        """Lazy-loaded language model property."""
        if self._llm is None:
            self._llm = self.get_language_model()
        return self._llm

    @abstractmethod
    def get_language_model(self) -> "LM":
        """Return the language model used in this stage."""
        pass

    @abstractmethod
    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Process the input data and return transformed data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to be processed.
        context : PipelineContext
            Context for the pipeline execution, containing experiment and
            configuration details.

        Returns
        -------
        pd.DataFrame
            Processed data after applying the stage's logic.
        """
        pass

    def prompt_template(self) -> dict[str, Any]:
        return {}

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

    def validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data. Override for stage-specific validation.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to validate.

        Raises
        -------
        ProcessingError
            If the input data is empty or does not meet stage-specific requirements.
        """
        if data.empty:
            raise ProcessingError(f"{self.stage_name}: Input data is empty")


class NATURALPipeline:
    """NATURAL pipeline for processing data through multiple stages.

    This pipeline executes a series of stages defined by the user, each of which
    processes the input data and passes it to the next stage. It also logs the
    time taken for each stage and collects statistics about the processing.

    Parameters
    ----------
    stages : list[PipelineStage]
        List of stages to be executed in the pipeline.

    Attributes
    ----------
    stages : list[PipelineStage]
        List of stages that make up the pipeline.

    """

    def __init__(self, stages: list[PipelineStage]) -> None:
        """Initialize the NATURAL pipeline with a list of stages."""
        self.stages = stages

        self._data_flow: dict[str, dict[str, Any]] = {}

    @asynccontextmanager
    async def _log_time(self, stage: PipelineStage):
        """Context manager to log time taken by a stage."""
        start_time = time.monotonic()
        try:
            yield
        finally:
            end_time = time.monotonic()
            duration = end_time - start_time
            logger.info(f"Stage {stage.stage_name} took {duration:.2f} seconds.")
            stage.add_stat("processing_time", f"{duration:.2f} seconds")

    async def run(
        self, input_df: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Run the pipeline on the input data.

        Parameters
        ----------
        input_df : pd.DataFrame
            Input dataframe to be processed by the pipeline.
        context : PipelineContext
            Context for the pipeline execution, containing experiment and
            configuration details.

        Returns
        -------
        pd.DataFrame
            Processed data after passing through all stages of the pipeline.

        Raises
        ------
        ProcessingError
            If the input dataframe is empty or if an error occurs during processing
            of any stage.
        """
        if input_df.empty:
            raise ProcessingError("Input dataframe is empty. Cannot run pipeline.")

        current_data = input_df.copy()

        for stage in self.stages:
            logger.info(f"Running stage: {stage.stage_name}")

            try:
                stage.validate_input(current_data)

                async with self._log_time(stage) as _:
                    current_data = await stage.process(current_data, context)

                    stage.add_stat("data_count", len(current_data))
                    stage.add_stat("model_name", stage._model_name)
                    stage.add_stat("model_request_params", stage.llm._request_params)
                    # TODO: add prompt template to stats
                    stage_stats = stage.get_stats()
                    logger.info(f"Stage {stage.stage_name} completed successfully.")
                    logger.info(f"Stats:\n{json.dumps(stage.get_stats(), indent=2)}")
                    for key, value in stage.prompt_template().items():
                        logger.info(f"{key}\n{str(value)}")
                    stage.render_stats_table()

                    self._data_flow[stage.stage_name] = stage_stats

                    if current_data.empty:
                        logger.warning(
                            f"Stage {stage.stage_name} returned an empty dataframe. "
                            "Skipping subsequent stages."
                        )
                        break
            except Exception as e:
                logger.error(
                    f"Error processing stage {stage.stage_name}: {e}", exc_info=True
                )
                raise ProcessingError from e

        logger.info(
            "Pipeline execution completed with the following data flow:\n"
            f"{json.dumps(self._data_flow, indent=2)}"
        )

        return current_data
