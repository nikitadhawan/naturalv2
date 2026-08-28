import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from naturalv2.cli.estimate_ate import (
    _calculate_treatment_responses,
    _load_sample_validation,
    _process_all_trials,
    _stratified_bootstrap_sample,
    _weight_by_inclusion,
)
from naturalv2.estimators import NaturalIPW, NaturalMC, NaturalOI
from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME


SAMPLE_TY_TARGET = "naturalv2.pipeline.sample_extraction.SampleTYStage"


def test_sample_validation_is_not_required_without_sample_ty():
    cfg = OmegaConf.create(
        {"pipeline": {"stages": {"other": {"_target_": "module.OtherStage"}}}}
    )

    assert _load_sample_validation(cfg) is None


@pytest.mark.asyncio
async def test_sample_ty_requires_sample_validation_before_processing():
    cfg = OmegaConf.create(
        {"pipeline": {"stages": {"sample_ty": {"_target_": SAMPLE_TY_TARGET}}}}
    )

    with pytest.raises(ValueError, match="`conf/common.yaml`"):
        await _process_all_trials(cfg)


class FakeExperiment:
    covariate_names = ["cov"]
    treatment_names = ["A", "B", "C"]
    status = "not_completed"
    apo_outcome_treatment = []

    def is_binary_outcome(self, outcome):
        return outcome == "binary_outcome"


def make_extractions(n, treatments, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cov_discretized": rng.integers(0, 2, size=n),
            f"{TREATMENT_COL_NAME}_discretized": treatments,
            f"{OUTCOME_COL_NAME}_discretized": rng.normal(size=n),
        }
    )


def test_stratified_bootstrap_preserves_group_sizes():
    df = make_extractions(40, [0, 0, 1] * 13 + [0])
    counts = df[f"{TREATMENT_COL_NAME}_discretized"].value_counts()
    for seed in range(10):
        boot = _stratified_bootstrap_sample(
            df, f"{TREATMENT_COL_NAME}_discretized", seed
        )
        assert (
            boot[f"{TREATMENT_COL_NAME}_discretized"].value_counts() == counts
        ).all()


@pytest.mark.parametrize("estimator_type", ["ipw", "oi"])
def test_unsupported_treatment_reported_as_nan(estimator_type):
    df = make_extractions(20, [0, 1] * 10)  # treatment 'C' never appears
    estimator = NaturalMC(FakeExperiment(), estimator_type=estimator_type)
    results, _ = _calculate_treatment_responses(
        FakeExperiment(),
        "dummy_outcome",
        estimator,
        df,
        bootstrap_size=10,
        seed=0,
        use_inclusion_weights=False,
    )
    by_treatment = {r["treatment"]: r for r in results}
    assert np.isnan(by_treatment["C"]["pred_response"])
    assert np.isnan(by_treatment["C"]["CI_lower"])
    assert not np.isnan(by_treatment["A"]["pred_response"])
    assert not np.isnan(by_treatment["B"]["pred_response"])


@pytest.mark.parametrize("estimator_cls", [NaturalIPW, NaturalOI])
def test_ipw_oi_reject_non_binary_outcome(estimator_cls):
    estimator = estimator_cls(FakeExperiment())
    df = make_extractions(10, [0, 1] * 5)
    with pytest.raises(ValueError, match="non-binary"):
        _calculate_treatment_responses(
            FakeExperiment(),
            "continuous_outcome",
            estimator,
            df,
            bootstrap_size=5,
            seed=0,
        )


@pytest.mark.parametrize("estimator_cls", [NaturalIPW, NaturalOI])
def test_ipw_oi_accept_binary_outcome_check(estimator_cls):
    # Should get past the guard (may still fail later for unrelated reasons,
    # e.g. missing conditional-probability columns -- not what's under test).
    estimator = estimator_cls(FakeExperiment())
    df = make_extractions(10, [0, 1] * 5)
    with pytest.raises(Exception) as exc_info:
        _calculate_treatment_responses(
            FakeExperiment(), "binary_outcome", estimator, df, bootstrap_size=5, seed=0
        )
    assert "non-binary" not in str(exc_info.value)


def test_weight_by_inclusion_falls_back_when_column_missing():
    responses = np.array([[1.0, 2.0, 3.0]])
    df = pd.DataFrame({"other_col": [0, 0, 0]})
    weighted = _weight_by_inclusion(responses, df, use_weights=True)
    assert weighted[0] == pytest.approx(2.0)  # uniform mean fallback


def test_weight_by_inclusion_uses_column_when_present():
    responses = np.array([[1.0, 2.0]])
    df = pd.DataFrame({"inclusion_probs": ["[0.9, 0.1]", "[0.1, 0.9]"]})
    weighted = _weight_by_inclusion(responses, df, use_weights=True)
    assert weighted[0] == pytest.approx((1.0 * 0.1 + 2.0 * 0.9) / (0.1 + 0.9))


class RowCountingEstimator:
    """Records how many rows it was called with, ignoring the estimator logic."""

    def __init__(self, num_treat):
        self.num_treat = num_treat
        self.seen_lengths = []

    def get_individual_treatment_effects(self, data):
        self.seen_lengths.append(len(data))
        return np.zeros((self.num_treat, len(data)))


@pytest.mark.parametrize("use_imputed_nones, expected_len", [(True, 4), (False, 3)])
def test_use_imputed_nones_drops_rows_with_missing_covariates(
    use_imputed_nones, expected_len
):
    df = pd.DataFrame({"cov": [1.0, 2.0, np.nan, 4.0]})
    estimator = RowCountingEstimator(num_treat=3)
    _calculate_treatment_responses(
        FakeExperiment(),
        "binary_outcome",
        estimator,
        df,
        bootstrap_size=2,
        seed=0,
        use_inclusion_weights=False,
        use_imputed_nones=use_imputed_nones,
    )
    assert estimator.seen_lengths[0] == expected_len
