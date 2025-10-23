"""NATURAL Pipeline."""

import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import pandas as pd
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig

from naturalv2.logging_utils import build_kv_table, emit_table
from naturalv2.models.utils import TokenTracker


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.models.lm import Model

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

    #: Identifier string for a particular run, included in results directory name.
    exp_name: str

    #: Token tracker to monitor token usage across stages.
    _token_tracker: TokenTracker = TokenTracker()


class PipelineStage(ABC):
    """Base class for stages in a pipeline.

    Each stage processes input data and returns transformed data.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    name : str, optional, default=None
        Optional name for the stage. If not provided, the class name will be used.

    Attributes
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    llm : LM | None
        Lazy-loaded language model instance.
    stage_name : str
        Name of the stage, derived from the class name.

    """

    def __init__(self, model_cfg: DictConfig, name: str | None = None) -> None:
        """Initialize the pipeline stage with model configuration."""
        self.model_cfg = model_cfg
        self.name = name

        self._llm: Optional["Model"] = None
        self._model_name: str = model_cfg.get("model_id", None)
        if self._model_name is None:
            deployment_params = model_cfg.get("deployment_params", {})
            first_inner = next(iter(deployment_params.values()), None)
            self._model_name = first_inner["model"] if first_inner else ""

        self._model_name = self._model_name.split("/")[
            -1
        ]  # Get last part of model name
        self._stats: dict[str, Any] = {}

    @property
    def stage_name(self) -> str:
        """Return the name of the stage."""
        return self.name or self.__class__.__name__

    @property
    def llm(self) -> "Model":
        """Lazy-loaded language model property."""
        if self._llm is None:
            self._llm = self.get_language_model()
        return self._llm

    def get_language_model(self) -> "Model":
        """Return the language model used in this stage."""
        return hydra_instantiate(self.model_cfg, _convert_="partial")

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
        """Return the prompt template used in this stage, if applicable.

        Override in subclasses to provide stage-specific prompt details.

        Returns
        -------
        dict[str, Any]
            Dictionary containing prompt template details. Empty by default.
        """
        return {}

    def get_stats(self) -> dict[str, Any]:
        """Return a dictionary of statistics collected during processing.

        Returns
        -------
        dict[str, Any]
            Dictionary containing statistics such as cost, token usage, and
            other relevant metrics.
        """
        if "cost" not in self._stats:
            self._stats["cost"] = self.llm.cost

        return self._stats

    def render_stats_table(self) -> None:
        """Log the statistics collected during processing as a Rich table."""
        stats = list(self.get_stats().items())
        prompt_template = self.prompt_template()
        if prompt_template:
            stats.append(("--", "Prompt Template"))
            stats.extend(
                (f"prompt.{key}", value) for key, value in prompt_template.items()
            )
        stats_table = build_kv_table(f"{self.stage_name} Summary", stats)
        emit_table(stats_table, logger=logger)

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
            logger.info("Running stage: %s", stage.stage_name)

            try:
                stage.validate_input(current_data)

                async with self._log_time(stage):
                    current_data = await stage.process(current_data, context)

                    stage.add_stat("data_count", len(current_data))
                    stage.add_stat("model_name", stage._model_name)
                    stage.add_stat("lm_kwargs", stage.llm.kwargs)

                stage_stats = stage.get_stats()
                self._data_flow[stage.stage_name] = stage_stats

                logger.info("Stage %s completed successfully.", stage.stage_name)
                stage.render_stats_table()

                if current_data.empty:
                    logger.warning(
                        "Stage %s returned an empty dataframe. "
                        "Skipping subsequent stages.",
                        stage.stage_name,
                    )
                    break
            except Exception as e:
                logger.error(
                    "Error processing stage %s: %s", stage.stage_name, e, exc_info=True
                )
                raise ProcessingError from e

        if self._data_flow:
            data_flow_table = build_kv_table(
                "Pipeline Data Flow",
                ((stage, stats) for stage, stats in self._data_flow.items()),
            )
            emit_table(data_flow_table, logger=logger, render_console=False)

        context._token_tracker.log_table()

        return current_data
