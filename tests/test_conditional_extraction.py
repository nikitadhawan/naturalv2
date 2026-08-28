from types import SimpleNamespace

import pandas as pd
import pytest

from naturalv2.models.lm import VLLMModel
from naturalv2.pipeline.conditional_extraction import (
    ConditionalsExtractType,
    _detokenize_kwargs,
    extract_conditionals,
)
from naturalv2.pipeline.constants import TREATMENT_COL_NAME


def test_detokenize_kwargs():
    assert _detokenize_kwargs(object.__new__(VLLMModel)) == {"detokenize": False}
    assert _detokenize_kwargs(object()) == {}


@pytest.mark.asyncio
async def test_conditional_extraction_rejects_continuous_outcomes(tmp_path):
    experiment = SimpleNamespace(
        nct_id="NCT00000000",
        options={TREATMENT_COL_NAME: ["Control", "Treatment"], "outcome": []},
    )
    with pytest.raises(ValueError, match="SampleTYStage with NaturalMC"):
        await extract_conditionals(
            input_df=pd.DataFrame({"report": ["Outcome was 3.2."]}),
            experiment=experiment,
            source_name="test",
            outcome="outcome",
            llm=object(),
            model_name="test-model",
            save_path=str(tmp_path),
            exp_name="test-experiment",
            extract_type=ConditionalsExtractType.TY_GIVEN_X,
        )
