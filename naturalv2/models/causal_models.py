"""Causal models used by the NaturalMC estimator."""

from typing import TYPE_CHECKING, Self

import pandas as pd
from causallib.estimation import IPW, MarginalOutcomeEstimator, Standardization
from sklearn.linear_model import LinearRegression, LogisticRegression


if TYPE_CHECKING:
    from naturalv2.models.types import CausalData


class DifferenceInMeans(object):
    """Difference in Means Estimator for Causal Effects.

    This class provides a base implementation for estimating causal effects
    using the difference in means approach. It can be extended for specific
    causal models.

    Parameters
    ----------
    fit_y : bool, default=True
        Whether to fit the observed outcome variable Y during model fitting.
    outcome_y : bool, default=True
        Whether to use the observed outcome variable Y for estimating individual outcomes.

    """

    def __init__(self, fit_y: bool = True, outcome_y: bool = True) -> None:
        """Initialize the DifferenceInMeans estimator."""
        self.fit_y = fit_y
        self.outcome_y = outcome_y
        self._model = MarginalOutcomeEstimator(learner=None)

    def fit(self, data: "CausalData") -> Self:
        """Fit the model to the provided causal data.

        Parameters
        ----------
        data : CausalData
            The causal data containing covariates, treatment assignments, and outcomes.

        Returns
        -------
        Self
            Returns the fitted estimator instance.
        """
        data.validate()

        if self.fit_y:
            self._model.fit(data.X, data.T, data.Y)
        else:
            self._model.fit(data.X, data.T)

        return self

    def get_average_treatment_effects(self, data: "CausalData") -> pd.Series:
        """Calculate average treatment effects based on the fitted model.

        Parameters
        ----------
        data : CausalData
            The causal data containing covariates and treatment assignments.

        Returns
        -------
        pd.Series
            A series containing the estimated average treatment effects.
        """
        data.validate()

        if self.outcome_y:
            outcomes = self._model.estimate_population_outcome(data.X, data.T, data.Y)
        else:
            outcomes = self._model.estimate_population_outcome(data.X, data.T)
        return self._model.estimate_effect(outcomes[1], outcomes[0])["diff"]


class IPSW(DifferenceInMeans):
    """Inverse Propensity Score Weighting (IPSW) Estimator.

    This class implements the Inverse Propensity Score Weighting method for estimating
    individual treatment effects in causal inference. It uses logistic regression to
    model the propensity scores and applies the Hajek estimator for weighting.

    Attributes
    ----------
    fit_y : bool, default=False
        Whether to fit the observed outcome variable Y during model fitting.
    outcome_y : bool, default=True
        Whether to use the observed outcome variable Y for estimating individual
        outcomes. This attribute is not used in this class, but is included for
        consistency with the base class.
    """

    def __init__(self) -> None:
        """Initialize the IPSW estimator with a logistic regression model."""
        super().__init__(fit_y=False, outcome_y=True)
        learner = LogisticRegression(
            solver="lbfgs"
        )  # supports multi-class classification
        self._model = IPW(learner=learner)

    def get_individual_treatment_effects(self, data: "CausalData") -> pd.Series:
        """Calculate individual treatment effects using the IPSW method.

        Parameters
        ----------
        data : CausalData
            The causal data containing covariates, treatment assignments, and
            outcomes.

        Returns
        -------
        pd.Series
            A series containing the estimated individual treatment effects, computed
            as the product of observed outcomes and inverse propensity scores,
            normalized within each observed treatment arm.
        """
        data.validate()

        # ITE doesn't quite make sense for a MC version of IPW - we return
        # y/P(T=t|x) for each unit.
        ipw_scores = self._model.compute_weights(data.X, data.T)
        arm_weight_totals = ipw_scores.groupby(data.T).transform("sum")
        ipw_scores *= len(data.X) / arm_weight_totals  # Hajek estimator per arm
        return data.Y * ipw_scores


class OutcomeImputation(DifferenceInMeans):
    """Outcome Imputation Estimator.

    This class implements the Outcome Imputation method for estimating individual
    treatment effects in causal inference. It uses a linear regression model to
    predict (counterfactual) outcomes based on covariates and treatment assignments.

    Attributes
    ----------
    fit_y : bool, default=True
        Whether to fit the observed outcome variable Y during model fitting.
    outcome_y : bool, default=False
        Whether to use the observed outcome variable Y for estimating individual
        outcomes. This attribute is not used in this class, but is included for
        consistency with the base class.
    """

    def __init__(self) -> None:
        """Initialize the Outcome Imputation estimator with a linear regression model."""
        super().__init__(fit_y=True, outcome_y=False)
        learner = LinearRegression()
        self._model = Standardization(learner=learner)

    def get_individual_treatment_effects(
        self, data: "CausalData", treatment_values: list[int] | None = None
    ) -> pd.DataFrame:
        """Calculate individual treatment effects using the Outcome Imputation method.

        Parameters
        ----------
        data : CausalData
            The causal data containing covariates, treatment assignments, and
            outcomes.
        treatment_values : list[int] | None, optional, default=None
            Treatment values to predict counterfactual outcomes for. If ``None``,
            defaults to the treatment values observed in ``data.T``, which may
            omit values entirely absent from ``data`` (e.g. a rare treatment
            missing from a bootstrap resample).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the estimated individual treatment effects for each
            treatment group, where each column corresponds to a treatment value
            and rows are individuals: each column is a vector size (num_samples,)
            that contains the estimated outcome for each individual under the
            treatment value in the corresponding key
        """
        data.validate()

        return self._model.estimate_individual_outcome(
            data.X, data.T, treatment_values=treatment_values
        )
