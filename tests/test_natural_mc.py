import numpy as np
import pandas as pd
import pytest

from naturalv2.estimators.natural_mc import NaturalMC
from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME


class FakeExperiment:
    covariate_names = ["cov"]
    treatment_names = ["A", "B", "C"]


def make_data(n, treatments, seed=0):
    """``treatments``: list of treatment indices assigned per row (len n)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cov_discretized": rng.integers(0, 2, size=n),
            f"{TREATMENT_COL_NAME}_discretized": treatments,
            f"{OUTCOME_COL_NAME}_discretized": rng.normal(size=n),
        }
    )


@pytest.mark.parametrize("estimator_type", ["ipw", "oi"])
def test_two_arms_supported(estimator_type):
    data = make_data(20, [0, 1] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type=estimator_type)
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert ites.shape == (3, 20)
    assert not np.isnan(ites[:2, :]).any()  # treatment 'C' has no data, row 2 is 0/NaN


@pytest.mark.parametrize("estimator_type", ["ipw", "oi"])
def test_three_arms_supported(estimator_type):
    data = make_data(30, [0, 1, 2] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type=estimator_type)
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert not np.isnan(ites).any()


def test_ipw_missing_treatment_is_zero_not_nan():
    # Caller (`_calculate_treatment_responses`) is responsible for turning this
    # into NaN once it knows the treatment is genuinely unsupported.
    data = make_data(20, [0, 1] * 10)  # treatment 'C' (index 2) never appears
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert (ites[2, :] == 0).all()


def test_ipw_normalizes_each_treatment_arm():
    data = make_data(20, [0] * 10 + [1] * 10)
    data[f"{OUTCOME_COL_NAME}_discretized"] = [2.0] * 10 + [4.0] * 10
    outcomes = NaturalMC(FakeExperiment(), "ipw").get_individual_treatment_effects(
        data, "dummy"
    )
    assert outcomes.mean(axis=1)[:2] == pytest.approx([2.0, 4.0])


def test_oi_missing_treatment_extrapolates_without_crash():
    data = make_data(20, [0, 1] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type="oi")
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert ites.shape == (3, 20)
    assert not np.isnan(ites).any()


def test_ipw_fit_error_propagates_with_context():
    # LogisticRegression needs >=2 classes to fit at all.
    data = make_data(10, [0] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")

    with pytest.raises(ValueError) as exc_info:
        mc.get_individual_treatment_effects(data, outcome="dummy")

    assert "outcome='dummy'" in " ".join(exc_info.value.__notes__)


def test_oi_single_treatment_value_still_returns_finite_values():
    # LinearRegression has no such requirement -- it just extrapolates.
    data = make_data(10, [0] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type="oi")
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert not np.isnan(ites).any()


def test_missing_required_column_raises():
    data = make_data(5, [0, 1, 0, 1, 0]).drop(
        columns=[f"{TREATMENT_COL_NAME}_discretized"]
    )
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")
    with pytest.raises(ValueError, match="discretized"):
        mc.get_individual_treatment_effects(data, outcome="dummy")
