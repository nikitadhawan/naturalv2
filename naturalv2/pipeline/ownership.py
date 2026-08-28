"""Gate Reddit comments on ownership of the selected treatment."""

from enum import StrEnum
from typing import Literal

import pandas as pd

from naturalv2.pipeline.constants import TREATMENT_COL_NAME
from naturalv2.pipeline.natural import PipelineContext
from naturalv2.pipeline.sample_extraction import (
    SampleExtractionStage,
    extract_covariates,
)
from naturalv2.utils import create_response_format


OWNERSHIP_COL = "author_used_treatment"


class OwnershipExtractType(StrEnum):
    TREATMENT_OWNERSHIP = "treatment_ownership"


def gate_treatment_ownership(data: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where treatment use belongs to the author."""
    return data.loc[data[OWNERSHIP_COL].eq("Yes")]


class TreatmentOwnershipGateStage(SampleExtractionStage):
    """Check ownership, update ownership variable, and gate process based on that."""

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        result = data.copy()
        if context.source_name.casefold() != "reddit":
            self.data = result
            return result

        is_comment = result["report_type"].astype(str).str.casefold().eq("comment")
        result[OWNERSHIP_COL] = "Yes"
        result.loc[is_comment, OWNERSHIP_COL] = "Unclear"
        if not is_comment.any():
            self.data = result
            return result

        gate_input = result.loc[is_comment].copy()
        gate_input["report"] = (
            "SELECTED TREATMENT: "
            + gate_input[TREATMENT_COL_NAME].astype(str)
            + "\nTARGET COMMENT:\n"
            + gate_input["report_text"].astype(str)
        )
        response_format = create_response_format(
            "TreatmentOwnershipResponse",
            [OWNERSHIP_COL],
            {OWNERSHIP_COL: Literal["Yes", "No", "Unclear"]},
        )
        checked = await extract_covariates(
            input_df=gate_input,
            pipeline_context=context,
            pipeline_stage_name=self.stage_name,
            extract_type=OwnershipExtractType.TREATMENT_OWNERSHIP,
            llm=self.llm,
            model_name=self._model_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )
        checked = checked.loc[checked.index.isin(gate_input.index)]
        result.loc[checked.index, OWNERSHIP_COL] = checked[OWNERSHIP_COL]
        self.data = gate_treatment_ownership(result)
        return self.data
