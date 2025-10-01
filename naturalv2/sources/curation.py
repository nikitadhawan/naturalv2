"""Core components for curating data from various sources."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from naturalv2.models.utils import TokenTracker


if TYPE_CHECKING:
    from naturalv2.experiment import Experiment
    from naturalv2.study import StudyDataset


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CurationContext:
    """Context for source curation"""

    #: Name of the source being curated from (e.g., "pubmed", "reddit").
    source_name: str

    #: Disease category data is curated for
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

    #: Additional context-specific information.
    extras: dict[str, Any] = field(default_factory=dict)

    #: Tracker for tokens used in LLM calls.
    _token_tracker: TokenTracker = field(default_factory=TokenTracker)

    def with_override(self, **overrides: Any) -> "CurationContext":
        """Create a new context with overridden values.

        Parameters
        ----------
        **overrides : Any
            Values to override in the new context.

        Returns
        -------
        CurationContext
            A new context with the specified overrides.
        """
        values = {
            "source_name": self.source_name,
            "condition": self.condition,
            "experiments": self.experiments,
            "splits": self.splits,
            "save_path": self.save_dir,
            "filter_by_date": self.filter_by_date,
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


class CurationStage(ABC):
    """Abstract base class for pipeline stages."""

    stage_name: str

    def __init__(self, *, name: str | None = None) -> None:
        self.stage_name = name or self.__class__.__name__

    @abstractmethod
    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Execute the stage using ``state`` and return updated state."""
        raise NotImplementedError


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
