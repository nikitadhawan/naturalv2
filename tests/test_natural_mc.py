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


def make_two_arm_data():
    """Balanced arms with constant outcomes 2 (arm 0) and 4 (arm 1).

    The covariate alternates within each arm, so it carries no information about
    treatment and every propensity is 0.5.
    """
    return pd.DataFrame(
        {
            "cov_discretized": [0, 1] * 10,
            f"{TREATMENT_COL_NAME}_discretized": [0] * 10 + [1] * 10,
            f"{OUTCOME_COL_NAME}_discretized": [2.0] * 10 + [4.0] * 10,
        }
    )


@pytest.mark.parametrize("estimator_type", ["ipw", "oi"])
def test_two_arms_supported(estimator_type):
    data = make_data(20, [0, 1] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type=estimator_type)
    responses, weights = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert responses.shape == (3, 20)
    assert weights.shape == (3, 20)
    assert not np.isnan(responses[:2, :]).any()  # treatment 'C' has no data


@pytest.mark.parametrize("estimator_type", ["ipw", "oi"])
def test_three_arms_supported(estimator_type):
    data = make_data(30, [0, 1, 2] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type=estimator_type)
    responses, weights = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert not np.isnan(responses).any()
    assert not np.isnan(weights).any()


def test_ipw_missing_treatment_is_zero_not_nan():
    # Caller (`_calculate_treatment_responses`) is responsible for turning this
    # into NaN once it knows the treatment is genuinely unsupported.
    data = make_data(20, [0, 1] * 10)  # treatment 'C' (index 2) never appears
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")
    responses, weights = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert (responses[2, :] == 0).all()
    assert (weights[2, :] == 0).all()


def test_ipw_returns_observed_outcomes_and_inverse_propensity_weights():
    data = make_two_arm_data()
    responses, weights = NaturalMC(
        FakeExperiment(), "ipw"
    ).get_individual_treatment_effects(data, "dummy")

    # Row t carries the observed outcomes of arm t and zeros for everyone else.
    assert (responses[0, :10] == 2.0).all() and (responses[0, 10:] == 0).all()
    assert (responses[1, 10:] == 4.0).all() and (responses[1, :10] == 0).all()
    # Weights are 1 / P(T = t | x) on arm t (every propensity is 0.5 here) and
    # zero elsewhere; nothing is pre-normalised.
    np.testing.assert_allclose(weights[0, :10], 2.0)
    np.testing.assert_allclose(weights[1, 10:], 2.0)
    assert (weights[0, 10:] == 0).all() and (weights[1, :10] == 0).all()
    # The weighted mean per arm is the Hajek estimate.
    hajek = [np.average(responses[t], weights=weights[t]) for t in range(2)]
    assert hajek == pytest.approx([2.0, 4.0])


def test_ipw_other_arm_size_does_not_change_an_arm_estimate():
    # The old pooled normalisation divided every arm by the total weight, so
    # adding rows to arm 1 silently shrank arm 0's estimate.
    data = make_two_arm_data()
    extra_arm1 = pd.DataFrame(
        {
            "cov_discretized": [0, 1] * 8,
            f"{TREATMENT_COL_NAME}_discretized": [1] * 16,
            f"{OUTCOME_COL_NAME}_discretized": [4.0] * 16,
        }
    )
    mc = NaturalMC(FakeExperiment(), "ipw")

    r0, w0 = mc.get_individual_treatment_effects(data, "dummy")
    r1, w1 = mc.get_individual_treatment_effects(
        pd.concat([data, extra_arm1], ignore_index=True), "dummy"
    )
    assert np.average(r0[0], weights=w0[0]) == pytest.approx(2.0)
    assert np.average(r1[0], weights=w1[0]) == pytest.approx(2.0)


def test_oi_weights_are_uniform():
    data = make_two_arm_data()
    responses, weights = NaturalMC(
        FakeExperiment(), "oi"
    ).get_individual_treatment_effects(data, "dummy")
    assert (weights == 1.0).all()
    # Standardisation fills every row, so a plain mean is already the estimate.
    assert responses.mean(axis=1)[:2] == pytest.approx([2.0, 4.0])


def test_oi_missing_treatment_extrapolates_without_crash():
    data = make_data(20, [0, 1] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type="oi")
    responses, _ = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert responses.shape == (3, 20)
    assert not np.isnan(responses).any()


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
    responses, _ = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert not np.isnan(responses).any()


def test_missing_required_column_raises():
    data = make_data(5, [0, 1, 0, 1, 0]).drop(
        columns=[f"{TREATMENT_COL_NAME}_discretized"]
    )
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")
    with pytest.raises(ValueError, match="discretized"):
        mc.get_individual_treatment_effects(data, outcome="dummy")
