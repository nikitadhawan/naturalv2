"""Core components for curating data from various sources."""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from naturalv2.models.utils import TokenTracker
from naturalv2.utils import sanitize_filename


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.study import StudyDataset


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CurationContext:
    """Context for source curation."""

    #: Name of the source being curated from (e.g., "pubmed", "reddit").
    source_name: str

    #: Disease category data is curated for.
    condition: str

    #: All trials included in the curation.
    experiments: list["Experiment"]

    #: Train/val/test split that trials belong to.
    splits: list[str]

    #: The directory to save curated data to.
    save_dir: str

    #: Whether or not the curated data should be filtered according to date.
    filter_by_date: bool

    study_dataset: "StudyDataset"
    experiment_name: str

    #: Additional context-specific information.
    extras: dict[str, Any] = field(default_factory=dict)

    #: Tracker for tokens used in LLM calls.
    _token_tracker: TokenTracker = field(default_factory=TokenTracker)

    def with_override(self, **overrides: Any) -> "CurationContext":
        """Create a new context with overridden values."""

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
    """Mutable state shared between stages."""

    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, **values: Any) -> None:
        self.metadata.update(values)

    def require_metadata(self, key: str, *, stage: str | None = None) -> Any:
        """Return a metadata entry or raise a descriptive error if missing."""

        if key not in self.metadata:
            stage_name = stage or "Stage"
            raise KeyError(f"{stage_name}: required metadata '{key}' is missing")
        return self.metadata[key]


class CurationStage(ABC):
    """Abstract base class for pipeline stages."""

    stage_name: str

    def __init__(self, *, name: str | None = None) -> None:
        self.stage_name = name or self.__class__.__name__

    @abstractmethod
    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Execute the stage using ``state`` and return updated state."""
        raise NotImplementedError


class SourceStage(CurationStage):
    """Stage with helpers for managing source-specific artefacts."""

    def source_dir(self, context: CurationContext, *, ensure: bool = True) -> str:
        path = os.path.join(context.save_dir, f"{context.source_name}_data")
        if ensure:
            os.makedirs(path, exist_ok=True)
        return path

    def condition_dir(self, context: CurationContext, *, ensure: bool = True) -> str:
        condition_segment = sanitize_filename(context.condition.lower())
        path = os.path.join(self.source_dir(context), condition_segment)
        if ensure:
            os.makedirs(path, exist_ok=True)
        return path

    def results_dir(
        self, context: CurationContext, *subdirs: str, ensure: bool = True
    ) -> str:
        """Return a directory under ``curation_results`` for the condition."""

        condition_segment = sanitize_filename(context.condition.lower())
        path = os.path.join(context.save_dir, "curation_results", condition_segment)
        if subdirs:
            path = os.path.join(path, *subdirs)
        if ensure:
            os.makedirs(path, exist_ok=True)
        return path

    def _study_dataset_path(self, context: CurationContext) -> str | None:
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
        """Update the study dataset metadata and persist it to disk if possible."""

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
    def __init__(self, stages: Iterable[CurationStage]) -> None:
        self.stages = list(stages)

    async def run_async(self, context: CurationContext) -> StageState:
        state = StageState()
        for stage in self.stages:
            logger.info("Running stage %s", stage.stage_name)
            state = await stage.run(context, state)
        return state

    def run(self, context: CurationContext) -> StageState:
        return asyncio.run(self.run_async(context))
