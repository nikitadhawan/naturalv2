from unittest.mock import Mock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from naturalv2.outcome_metadata import OutcomeBounds
from naturalv2.pipeline import (
    OUTCOME_COL_NAME,
    TREATMENT_COL_NAME,
    SampleValidationConfig,
)
from naturalv2.pipeline.sample_extraction import (
    _create_sample_ty_response_format,
    _filter_invalid_sampled_outcomes,
)


NCT_ID = "NCT012"
OUTCOME = "Functional Capacity"
TEN_PERCENT_POLICY = SampleValidationConfig(high_rejection_rate=0.10)
ALLOW_HIGH_REJECTION_POLICY = SampleValidationConfig(
    high_rejection_rate=0.10,
    allow_high_rejection_rate=True,
)
CHANGE_BOUNDS = OutcomeBounds(minimum=-55, maximum=55)


def _validate(
    extractions: pd.DataFrame,
    *,
    policy: SampleValidationConfig = TEN_PERCENT_POLICY,
    bounds: OutcomeBounds | None = None,
) -> pd.DataFrame:
    return _filter_invalid_sampled_outcomes(
        extractions,
        nct_id=NCT_ID,
        outcome=OUTCOME,
        bounds=bounds,
        sample_validation=policy,
    )


def test_response_schema_rejects_non_finite_outcome():
    experiment = Mock(
        options={TREATMENT_COL_NAME: ["Treatment A"]},
        is_binary_outcome=Mock(return_value=False),
    )
    response_format = _create_sample_ty_response_format(experiment, OUTCOME)

    with pytest.raises(ValidationError):
        response_format.model_validate(
            {TREATMENT_COL_NAME: "Treatment A", OUTCOME_COL_NAME: float("inf")}
        )


def test_cached_validation_filters_and_reports_each_invalid_outcome_cause():
    extractions = pd.DataFrame(
        {
            OUTCOME_COL_NAME: [
                -55,
                55,
                "not-a-number",
                pd.NA,
                float("inf"),
                -56,
                56,
            ],
        },
    )

    with patch("naturalv2.pipeline.sample_extraction.logger.error") as log:
        validated = _validate(
            extractions,
            policy=ALLOW_HIGH_REJECTION_POLICY,
            bounds=CHANGE_BOUNDS,
        )

    assert validated[OUTCOME_COL_NAME].tolist() == [-55, 55]
    assert log.call_args.kwargs["extra"]["rejection_reasons"] == {
        "unparsed": 2,
        "infinite": 1,
        "below_minimum": 1,
        "above_maximum": 1,
    }


def test_combined_rejection_rate_blocks_estimation():
    extractions = pd.DataFrame(
        {OUTCOME_COL_NAME: ([10.0] * 90 + [float("nan")] * 5 + [56.0] * 5)}
    )

    with pytest.raises(ValueError, match="high-rejection threshold"):
        _validate(extractions, bounds=CHANGE_BOUNDS)


def test_all_invalid_records_fail_even_with_override():
    extractions = pd.DataFrame({OUTCOME_COL_NAME: [float("inf")]})

    with pytest.raises(ValueError, match="No valid sampled outcomes remain"):
        _validate(extractions, policy=ALLOW_HIGH_REJECTION_POLICY)
