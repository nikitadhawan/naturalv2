import asyncio
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from naturalv2.pipeline import sample_extraction
from naturalv2.pipeline.constants import (
    OUTCOME_BASIS_COL_NAME,
    OUTCOME_COL_NAME,
    TREATMENT_COL_NAME,
)
from naturalv2.pipeline.natural import PipelineContext
from naturalv2.pipeline.sample_extraction import (
    SampleTYStage,
    filter_sampled_outcomes,
    sample_ty_response_format,
)


def make_sampled(values, bases=None):
    df = pd.DataFrame(
        {TREATMENT_COL_NAME: ["A"] * len(values), OUTCOME_COL_NAME: values}
    )
    if bases is not None:
        df[OUTCOME_BASIS_COL_NAME] = bases
    return df


# -- filter_sampled_outcomes ---------------------------------------------------


def test_missing_and_nonfinite_values_are_dropped():
    df = make_sampled([2.0, None, np.inf, -np.inf, 4.0])
    assert filter_sampled_outcomes(df)[OUTCOME_COL_NAME].tolist() == [2.0, 4.0]


def test_mixed_bases_are_warned_about_but_kept_without_configuration():
    # Three rows answered as a change from baseline, one slipped into an
    # absolute follow-up score (12.9 on a 1-49 scale) -- the #38 failure mode.
    # Nothing is dropped unless the expected basis is configured; the mix is
    # reported so it can be.
    df = make_sampled(
        [-11.0, -9.0, 12.9, -10.0],
        ["change_from_baseline", "change_from_baseline", "absolute"]
        + ["change_from_baseline"],
    )
    with patch.object(sample_extraction.logger, "warning") as warning:
        kept = filter_sampled_outcomes(df)
    assert kept.index.tolist() == [0, 1, 2, 3]
    message = warning.call_args.args[0] % warning.call_args.args[1:]
    assert "mix bases" in message
    assert "3 change_from_baseline" in message and "1 absolute" in message
    assert "outcome_basis" in message

    with patch.object(sample_extraction.logger, "warning") as warning:
        filter_sampled_outcomes(make_sampled([1.0, 2.0], ["absolute"] * 2))
    warning.assert_not_called()


def test_out_of_bounds_values_are_dropped_when_bounds_are_configured():
    df = make_sampled([12.0, 4_444_000.0, -3.0, 55.0], ["absolute"] * 4)
    kept = filter_sampled_outcomes(df, outcome_bounds=(0.0, 55.0))
    assert kept[OUTCOME_COL_NAME].tolist() == [12.0, 55.0]


def test_configured_bounds_apply_as_given_whatever_the_basis():
    # Bounds are stated on the trial's own basis by whoever configures them, so
    # a change score gets its signed span directly; nothing is derived.
    df = make_sampled([-11.3, 30.0, -48.0, -49.0], ["change_from_baseline"] * 4)
    kept = filter_sampled_outcomes(df, outcome_bounds=(-48.0, 48.0))
    assert kept[OUTCOME_COL_NAME].tolist() == [-11.3, 30.0, -48.0]


def test_without_configured_bounds_no_range_check_is_applied():
    df = make_sampled([4_444_000.0, 1.0], ["absolute"] * 2)
    assert len(filter_sampled_outcomes(df, outcome_bounds=None)) == 2


def test_filter_never_aborts_and_logs_what_it_dropped():
    df = make_sampled([np.nan, 999.0, 3.0], ["absolute"] * 3)
    with patch.object(sample_extraction.logger, "warning") as warning:
        kept = filter_sampled_outcomes(df, outcome_bounds=(0.0, 10.0))
    assert kept[OUTCOME_COL_NAME].tolist() == [3.0]
    message = warning.call_args.args[0] % warning.call_args.args[1:]
    assert "Dropped 2/3 sampled outcomes" in message
    assert "1 missing or non-finite" in message
    assert "1 outside [0, 10]" in message


def test_filter_leaves_empty_input_alone():
    df = make_sampled([])
    assert filter_sampled_outcomes(df).empty


# -- sample_ty_response_format ------------------------------------------------


def test_continuous_response_requires_a_finite_number_and_a_basis():
    fmt = sample_ty_response_format(["A", "B"], outcome_is_binary=False)

    parsed = fmt.model_validate(
        {
            TREATMENT_COL_NAME: "a",
            OUTCOME_COL_NAME: -11.3,
            OUTCOME_BASIS_COL_NAME: "Change_From_Baseline",
        }
    )
    assert getattr(parsed, TREATMENT_COL_NAME) == "A"  # case-insensitive literal
    assert getattr(parsed, OUTCOME_COL_NAME) == -11.3
    assert getattr(parsed, OUTCOME_BASIS_COL_NAME) == "change_from_baseline"

    with pytest.raises(ValidationError):  # null is not an answer; ty_filter gates
        fmt.model_validate(
            {
                TREATMENT_COL_NAME: "A",
                OUTCOME_COL_NAME: None,
                OUTCOME_BASIS_COL_NAME: "absolute",
            }
        )

    with pytest.raises(ValidationError):  # basis is required
        fmt.model_validate({TREATMENT_COL_NAME: "A", OUTCOME_COL_NAME: 3.0})
    with pytest.raises(ValidationError):  # and constrained
        fmt.model_validate(
            {
                TREATMENT_COL_NAME: "A",
                OUTCOME_COL_NAME: 3.0,
                OUTCOME_BASIS_COL_NAME: "delta",
            }
        )


def test_binary_response_format_is_unchanged():
    fmt = sample_ty_response_format(["A"], outcome_is_binary=True)
    assert set(fmt.model_fields) == {TREATMENT_COL_NAME, OUTCOME_COL_NAME}
    with pytest.raises(ValidationError):  # no null, no basis
        fmt.model_validate({TREATMENT_COL_NAME: "A", OUTCOME_COL_NAME: None})
    with pytest.raises(ValidationError):
        fmt.model_validate(
            {
                TREATMENT_COL_NAME: "A",
                OUTCOME_COL_NAME: "Yes",
                OUTCOME_BASIS_COL_NAME: "absolute",
            }
        )


# -- Concerns carried over from #50 / #51 -------------------------------------


def test_continuous_response_rejects_non_finite_values_at_parse_time():
    # #50: inf/nan must never get past the schema.
    fmt = sample_ty_response_format(["A"], outcome_is_binary=False)
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError):
            fmt.model_validate(
                {
                    TREATMENT_COL_NAME: "A",
                    OUTCOME_COL_NAME: bad,
                    OUTCOME_BASIS_COL_NAME: "absolute",
                }
            )


def test_cached_rows_of_every_invalid_kind_are_filtered_with_inclusive_bounds():
    # #51's cached-artifact case: values read back from CSV arrive as strings,
    # NA, inf, or just outside the instrument. FUNCAP55 is 0-55 reported as a
    # change, configured as the inclusive span [-55, 55].
    df = make_sampled(
        [-55, 55, "not-a-number", pd.NA, float("inf"), -56, 56],
        ["change_from_baseline"] * 7,
    )
    kept = filter_sampled_outcomes(df, outcome_bounds=(-55.0, 55.0))
    assert kept[OUTCOME_COL_NAME].tolist() == [-55, 55]


def test_all_invalid_rows_yield_an_empty_frame_without_raising():
    # #50/#51 aborted here; we leave that to the pipeline's existing
    # "no extractions" handling so one bad outcome cannot stop a whole run.
    df = make_sampled([float("inf"), None], ["absolute"] * 2)
    kept = filter_sampled_outcomes(df, outcome_bounds=(0.0, 10.0))
    assert kept.empty
    assert list(kept.columns) == list(df.columns)


def test_sample_ty_stage_filters_whatever_extraction_returns():
    # `extract_covariates` hands back cached rows alongside fresh ones, so the
    # filter has to sit in the stage, after the call, not only in the schema.
    experiment = Mock()
    experiment.nct_id = "NCT06366724"
    experiment.is_binary_outcome.return_value = False
    experiment.options = {TREATMENT_COL_NAME: ["A"]}
    experiment.discretize_ty.side_effect = lambda df, outcome: df
    context = PipelineContext(
        experiment=experiment,
        source_name="reddit",
        estimator_type="NaturalMC",
        outcome="Functional Capacity",
        save_path="unused",
        exp_name="unit",
    )
    stage = SampleTYStage(
        OmegaConf.create({"model_id": "unit/test"}),
        # Hydra hands these over as DictConfig / ListConfig, not plain dicts.
        outcome_bounds=OmegaConf.create(
            {"NCT06366724": {"Functional Capacity": [-55, 55]}}
        ),
        outcome_basis=OmegaConf.create(
            {"NCT06366724": {"Functional Capacity": "change_from_baseline"}}
        ),
    )
    stage._llm = object()  # never instantiate a real model
    # Three of four finite rows answered "absolute"; the configured basis says
    # change from baseline, so only those rows survive.
    extracted = make_sampled(
        [12.0, "inf", 4_444_000.0, None, 30.0, -12.0],
        ["absolute"] * 4 + ["change_from_baseline", "change_from_baseline"],
    )

    with patch.object(
        sample_extraction, "extract_covariates", AsyncMock(return_value=extracted)
    ) as extract:
        result = asyncio.run(stage.process(pd.DataFrame({"report": ["r"]}), context))

    assert result[OUTCOME_COL_NAME].tolist() == [30.0, -12.0]
    # The schema handed to the extractor is the continuous one.
    schema = extract.call_args.kwargs["response_format"]
    assert OUTCOME_BASIS_COL_NAME in schema.model_fields


# -- outcome_bounds configuration --------------------------------------------


def test_outcome_bounds_are_optional_and_scoped_to_trial_and_outcome():
    stage = SampleTYStage(
        OmegaConf.create({"model_id": "unit/test"}),
        outcome_bounds={"NCT1": {"Score": [0, 10]}},
    )
    assert stage.outcome_bounds == {"NCT1": {"Score": (0.0, 10.0)}}
    assert stage.outcome_bounds.get("NCT2", {}).get("Score") is None
    assert stage.outcome_bounds["NCT1"].get("Other") is None
    assert (
        SampleTYStage(OmegaConf.create({"model_id": "unit/test"})).outcome_bounds == {}
    )


@pytest.mark.parametrize("bad", [[10, 0], [5, 5]])
def test_outcome_bounds_must_be_an_increasing_pair(bad):
    with pytest.raises(ValueError, match="min < max"):
        SampleTYStage(
            OmegaConf.create({"model_id": "unit/test"}),
            outcome_bounds={"NCT1": {"Score": bad}},
        )


def test_configured_basis_drops_rows_on_any_other_basis():
    df = make_sampled(
        [-11.0, -9.0, 12.9, 13.4], ["change_from_baseline"] * 2 + ["absolute"] * 2
    )
    kept = filter_sampled_outcomes(df, outcome_basis="change_from_baseline")
    assert kept[OUTCOME_COL_NAME].tolist() == [-11.0, -9.0]
    kept = filter_sampled_outcomes(df, outcome_basis="absolute")
    assert kept[OUTCOME_COL_NAME].tolist() == [12.9, 13.4]


def test_outcome_basis_must_be_a_known_value():
    with pytest.raises(ValueError, match="change_from_baseline"):
        SampleTYStage(
            OmegaConf.create({"model_id": "unit/test"}),
            outcome_basis={"NCT1": {"Score": "delta"}},
        )
    stage = SampleTYStage(
        OmegaConf.create({"model_id": "unit/test"}),
        outcome_basis={"NCT1": {"Score": "absolute"}},
    )
    assert stage.outcome_basis == {"NCT1": {"Score": "absolute"}}
    assert stage.outcome_basis.get("NCT2", {}).get("Score") is None
