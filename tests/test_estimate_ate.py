import numpy as np
import pandas as pd
import pytest

from naturalv2.cli.estimate_ate import (
    _calculate_treatment_responses,
    _stratified_bootstrap_sample,
    _weight_by_inclusion,
)
from naturalv2.estimators import NaturalIPW, NaturalMC, NaturalOI
from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME


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


# -- Inclusion-weighted Hajek normalisation for NaturalMC-IPW ------------------

def make_two_arm_extractions(inclusion_probs=None):
    """Balanced arms with constant outcomes 2 (arm 0) and 4 (arm 1).

    The covariate alternates within each arm, so every propensity is 0.5. When
    ``inclusion_probs`` is given, entry ``i`` is P(include) for row ``i`` and is
    stored the way the pipeline stores it: a stringified ``[P(no), P(yes)]``.
    """
    df = pd.DataFrame(
        {
            "cov_discretized": [0, 1] * 10,
            f"{TREATMENT_COL_NAME}_discretized": [0] * 10 + [1] * 10,
            f"{OUTCOME_COL_NAME}_discretized": [2.0] * 10 + [4.0] * 10,
        }
    )
    if inclusion_probs is not None:
        df["inclusion_probs"] = [str([1 - p, p]) for p in inclusion_probs]
    return df


def test_weight_by_inclusion_forms_hajek_ratio_with_response_weights():
    # Row t holds the outcomes of units in arm t (zero elsewhere); weights hold
    # their inverse propensity weights. Inclusion probs are [0.5, 0.1, 0.9].
    responses = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 5.0]])
    weights = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
    df = pd.DataFrame({"inclusion_probs": ["[0.5, 0.5]", "[0.9, 0.1]", "[0.1, 0.9]"]})

    out = _weight_by_inclusion(
        responses, df, use_weights=True, response_weights=weights
    )

    arm1 = (0.1 * 1.0 * 3.0 + 0.9 * 2.0 * 5.0) / (0.1 * 1.0 + 0.9 * 2.0)
    assert out == pytest.approx([2.0, arm1])


def test_weight_by_inclusion_without_response_weights_is_weighted_mean():
    responses = np.array([[1.0, 2.0]])
    df = pd.DataFrame({"inclusion_probs": ["[0.9, 0.1]", "[0.1, 0.9]"]})
    plain = _weight_by_inclusion(responses, df, use_weights=True)
    explicit = _weight_by_inclusion(
        responses, df, use_weights=True, response_weights=np.ones_like(responses)
    )
    assert plain == pytest.approx(explicit)


def test_weight_by_inclusion_unsupported_treatment_is_nan():
    responses = np.zeros((1, 2))
    weights = np.zeros((1, 2))
    df = pd.DataFrame({"other_col": [0, 0]})
    out = _weight_by_inclusion(
        responses, df, use_weights=False, response_weights=weights
    )
    assert np.isnan(out[0])


def test_ipw_responses_recover_arm_means_with_uniform_inclusion():
    df = make_two_arm_extractions()
    results, _ = _calculate_treatment_responses(
        FakeExperiment(),
        "dummy",
        NaturalMC(FakeExperiment(), "ipw"),
        df,
        bootstrap_size=5,
        seed=0,
        use_inclusion_weights=False,
    )
    by_treatment = {r["treatment"]: r["pred_response"] for r in results}
    assert (by_treatment["A"], by_treatment["B"]) == pytest.approx((2.0, 4.0))


def test_ipw_responses_recover_arm_means_with_nonuniform_inclusion():
    # Arm 0's reports read as less eligible (p=0.2) than arm 1's (p=0.8).
    # Normalising inside the estimator with n / sum(w) and then averaging with
    # the inclusion probabilities gives [0.8, 6.4] here; the Hajek denominator
    # has to be formed with the same probabilities as the numerator.
    df = make_two_arm_extractions(inclusion_probs=[0.2] * 10 + [0.8] * 10)
    results, _ = _calculate_treatment_responses(
        FakeExperiment(),
        "dummy",
        NaturalMC(FakeExperiment(), "ipw"),
        df,
        bootstrap_size=5,
        seed=0,
        use_inclusion_weights=True,
    )
    by_treatment = {r["treatment"]: r["pred_response"] for r in results}
    assert (by_treatment["A"], by_treatment["B"]) == pytest.approx((2.0, 4.0))
