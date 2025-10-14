"""Core curation primitives.

This module defines the core building blocks used by source-specific curation pipelines:

- ``CurationContext`` captures immutable run-time configuration and shared
  resources (e.g., experiment list, save directories, token tracking).
- ``StageState`` is a lightweight, mutable state object passed between stages
  in a pipeline to shuttle intermediate payloads and metadata.
- ``CurationStage`` is the abstract base for all curation stages.
- ``SourceStage`` extends ``CurationStage`` with convenience helpers for
  source-specific filesystem layout and dataset persistence.
- ``FilterCurateRunner`` executes a sequence of stages in order.

"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

from naturalv2.models.utils import TokenTracker
from naturalv2.utils import sanitize_filename


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.study import StudyDataset


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CurationContext:
    """Context for a curation run."""

    #: Name of the source being curated from (e.g., "pubmed", "reddit").
    source_name: str

    #: Condition/Disease category that the data is curated for.
    condition: str

    #: All trials included in the curation run.
    experiments: list["Experiment"]

    #: Train/val/test split identifiers that trials belong to.
    splits: list[str]

    #: Base directory where curated data, intermediate artefacts, and results
    #: should be written.
    save_dir: str

    #: Whether or not the curated data should be filtered according to date.
    filter_by_date: bool

    #: Instance of ``StudyDataset`` that is being curated/updated.
    study_dataset: "StudyDataset"

    #: Name of the current pipeline run; used to disambiguate artefacts.
    experiment_name: str

    #: Additional context-specific information.
    extras: dict[str, Any] = field(default_factory=dict)

    #: Internal tracker for LLM token usage across stages.
    _token_tracker: TokenTracker = field(default_factory=TokenTracker)

    def with_override(self, **overrides: Any) -> "CurationContext":
        """Return a copy of this context with selected fields overridden.

        This is useful when a stage needs to adjust a small subset of
        configuration values without mutating the original context.

        Parameters
        ----------
        **overrides
            Field names and replacement values to override on the returned
            context instance.

        Returns
        -------
        CurationContext
            A new context instance with the provided overrides applied.
        """

        values = {
            "source_name": self.source_name,
            "condition": self.condition,
            "experiments": self.experiments,
            "splits": self.splits,
            "save_dir": self.save_dir,
            "filter_by_date": self.filter_by_date,
            "study_dataset": self.study_dataset,
            "experiment_name": self.experiment_name,
            "extras": dict(self.extras),
            "_token_tracker": self._token_tracker,
        }
        values.update(overrides)
        return CurationContext(**values)


@dataclass(slots=True)
class StageState:
    """Mutable state passed between stages.

    Parameters
    ----------
    payload : Any | None, optional, default=None
        Primary stage output carried forward to the next stage. Its type is
        stage-specific.
    metadata : dict[str, Any], optional
        Auxiliary key-value map for sharing smaller pieces of information
        between stages.
    """

    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, **values: Any) -> None:
        """Update ``metadata`` with the provided key-value pairs.

        Parameters
        ----------
        **values
            Key-value pairs to merge into ``metadata``.
        """
        self.metadata.update(values)

    def require_metadata(self, key: str, *, stage: str | None = None) -> Any:
        """Return a metadata entry or raise if missing.

        Parameters
        ----------
        key : str
            Metadata key to retrieve.
        stage : str | None, optional, default=None
            Optional stage name to include in the error message for context.

        Returns
        -------
        Any
            The stored metadata value.

        Raises
        ------
        KeyError
            If ``key`` is not present in ``metadata``.
        """

        if key not in self.metadata:
            stage_name = stage or "Stage"
            raise KeyError(f"{stage_name}: required metadata '{key}' is missing")
        return self.metadata[key]


class CurationStage(ABC):
    """Abstract base class for curation pipeline stages.

    Subclasses must implement :meth:`run`.
    """

    stage_name: str

    def __init__(self, *, name: str | None = None) -> None:
        self.stage_name = name or self.__class__.__name__

    @abstractmethod
    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Execute the stage and return an updated ``StageState``.

        Parameters
        ----------
        context : Curationcontext
            Immutable configuration and shared resources for the run.
        state : StageState
            Mutable state produced by the previous stage.

        Returns
        -------
        StageState
            The updated state to pass to the next stage.
        """
        raise NotImplementedError

    def render_metadata(self, state: StageState) -> None:
        """Render avaialable metadata, up to the current stage, as a rich table.

        Parameters
        ----------
        state : StageState
            The state object containing the metadata to render.
        """
        rich_table = Table(title=f"Curation pipeline state at stage: {self.stage_name}")

        rich_table.add_column("Key", style="cyan")
        rich_table.add_column("Value", style="magenta")

        for key, value in state.metadata.items():
            rich_table.add_row(str(key), Pretty(value, max_length=10))

        console = Console(force_terminal=True)

        # Capture the table in a string instead of writing to the console
        with console.capture() as capture:
            console.print(rich_table)

        # Log the captured string.
        # This should now show up in both the terminal and the log file.
        logger.info(capture.get())


class SourceStage(CurationStage):
    """Stage with helpers for managing source-specific artefacts and paths."""

    def source_dir(self, context: CurationContext, *, ensure: bool = True) -> str:
        """Return the base directory for this source within ``save_dir``.

        Parameters
        ----------
        context : CurationContext
            The curation context.
        ensure : bool, default = True
            If ``True``, create the directory if it does not exist.

        Returns
        -------
        str
            Absolute path to the source directory (e.g., ``.../reddit_data``).
        """
        path = os.path.join(context.save_dir, f"{context.source_name}_data")
        if ensure:
            os.makedirs(path, exist_ok=True)
        return path

    def condition_dir(self, context: CurationContext, *, ensure: bool = True) -> str:
        """Return the condition-specific directory under the source directory.

        Parameters
        ----------
        context : CurationContext
            The curation context.
        ensure : bool, default=True
            If ``True``, create the directory if it does not exist.

        Returns
        -------
        str
            Absolute path to the condition directory.
        """
        condition_segment = sanitize_filename(context.condition.lower())
        path = os.path.join(self.source_dir(context), condition_segment)
        if ensure:
            os.makedirs(path, exist_ok=True)
        return path

    def results_dir(
        self, context: CurationContext, *subdirs: str, ensure: bool = True
    ) -> str:
        """Return a directory under ``curation_results`` for the condition.

        Parameters
        ----------
        context : CurationConext
            The curation context.
        *subdirs : str
            Subdirectory segments to append under the condition directory
            (e.g., ``"metrics"``, ``"plots"``).
        ensure : bool, default=True
            If ``True``, create the directory if it does not exist.

        Returns
        -------
        str
            Absolute path to the results directory.
        """

        condition_segment = sanitize_filename(context.condition.lower())
        path = os.path.join(context.save_dir, "curation_results", condition_segment)
        if subdirs:
            path = os.path.join(path, *subdirs)
        if ensure:
            os.makedirs(path, exist_ok=True)
        return path

    def _study_dataset_path(self, context: CurationContext) -> str | None:
        """Return the configured study dataset path, if available.

        Checks ``context.extras`` for ``"study_dataset_path"`` or the legacy
        key ``"study_dataset_file"``.

        Parameters
        ----------
        context : CurationContext
            The curation context.

        Returns
        -------
        str | None
            Path to the study dataset YAML file, if configured.
        """
        return context.extras.get("study_dataset_path") or context.extras.get(
            "study_dataset_file"
        )

    def persist_dataset(
        self,
        context: CurationContext,
        *,
        per_experiment_paths: dict[str, str] | None = None,
        namespace_paths: dict[str, list[str]] | None = None,
        per_experiment_sizes: dict[str, int] | None = None,
    ) -> None:
        """Update dataset metadata and persist the ``StudyDataset`` to disk.

        This is a convenience method for stages that produce curated artefacts
        and want to record their locations and sizes in the study dataset. If a
        dataset path is available in the context (see
        :meth:`_study_dataset_path`), the updated dataset is serialized to YAML.

        Parameters
        ----------
        context : CurationContext
            The curation context.
        per_experiment_paths : dict[str, str] | None, optional, default=None
            Mapping from ``experiment.nct_id`` to produced file path.
        namespace_paths : dict[str, list[str]] | None, optional, default=None
            Mapping from arbitrary namespaces to lists of paths. Useful for
            storing non per-experiment artefacts.
        per_experiment_sizes : dict[str, int] | None, optional, default=None
            Mapping from ``experiment.nct_id`` to the number of rows/items
            curated for that experiment.
        """

        if per_experiment_paths:
            context.study_dataset.data_paths.update(per_experiment_paths)
        if namespace_paths:
            for key, value in namespace_paths.items():
                context.study_dataset.data_paths[key] = value
        if per_experiment_sizes:
            context.study_dataset.data_sizes.update(per_experiment_sizes)

        dataset_path = self._study_dataset_path(context)
        if dataset_path:
            context.study_dataset.to_yaml(dataset_path)


class FilterCurateRunner:
    """Run a sequence of curation stages.

    Parameters
    ----------
    stages
        Iterable of stages to run in order.
    """

    def __init__(self, stages: Iterable[CurationStage]) -> None:
        self.stages = list(stages)

    async def _run_async(self, context: CurationContext) -> StageState:
        """Run stages asynchronously.

        Parameters
        ----------
        context
            The curation context to supply to each stage.

        Returns
        -------
        StageState
            The final state produced by the last stage.
        """
        state = StageState()
        for stage in self.stages:
            logger.info("Running stage %s", stage.stage_name)
            state = await stage.run(context, state)
            stage.render_metadata(state)
        return state

    def run(self, context: CurationContext) -> StageState:
        """Run stages synchronously using ``asyncio.run``.

        Parameters
        ----------
        context
            The curation context to supply to each stage.

        Returns
        -------
        StageState
            The final state produced by the last stage.
        """
        return asyncio.run(self._run_async(context))
