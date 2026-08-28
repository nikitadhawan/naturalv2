"""Natural Monte Carlo Estimator."""

from typing import Literal

import numpy as np
import pandas as pd

from naturalv2.experiment import Experiment
from naturalv2.models.causal_models import IPSW, OutcomeImputation
from naturalv2.models.types import CausalData
from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME


class NaturalMC:
    """NATURAL Monte Carlo Estimator for individual treatment responses.
    TODO: Do not use for APOs; off-the-shelf estimators do not trivially extend to APOs.

    Parameters
    ----------
    experiment : Experiment
        The experiment containing treatment and covariate information.
    estimator_type : Literal["ipw", "oi"], default="ipw"
        The type of estimator to use for calculating ITEs. Options are:
        - "ipw": Inverse Probability Weighting
        - "oi": Outcome Imputation
    """

    def __init__(
        self,
        experiment: Experiment,
        estimator_type: Literal["ipw", "oi"] = "ipw",
    ) -> None:
        self.experiment = experiment
        self.estimator_type = estimator_type

        self._covariate_names = experiment.covariate_names
        self._num_treat = len(experiment.treatment_names)
        self._causal_models: dict[str, type[IPSW] | type[OutcomeImputation]] = {
            "ipw": IPSW,
            "oi": OutcomeImputation,
        }

    def get_individual_treatment_effects(
        self, observational_data: pd.DataFrame, outcome: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Individual Treatment Effects (ITE) for a given outcome.

        Parameters
        ----------
        observational_data : pd.DataFrame
            Data containing treatment, covariates, and outcome.
        outcome : str
            The name of the outcome column in `observational_data`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(responses, weights)``, both of shape ``[num_treatments, num_units]``.
            For ``"ipw"``, row ``t`` of ``responses`` holds the observed outcome
            ``y_i`` of units treated with ``t`` (zero elsewhere) and ``weights``
            holds their inverse propensity weights ``1 / P(T = t | x_i)`` (zero
            elsewhere). For ``"oi"``, ``responses`` holds the imputed outcome of
            every unit under each treatment and ``weights`` is all ones. The
            average potential outcome under ``t`` is the mean of ``responses[t]``
            weighted by ``p * weights[t]`` for any per-unit weights ``p`` (uniform
            or inclusion probabilities).

        Raises
        ------
        ValueError
            If required data is absent.
        Exception
            Any estimator fitting or prediction error is re-raised with trial,
            outcome, data-shape, and observed-treatment context.

        """
        sampled_treatment_col = f"{TREATMENT_COL_NAME}_discretized"
        if sampled_treatment_col not in observational_data.columns:
            raise ValueError(
                f"{sampled_treatment_col} must be in ``observational_data`` columns."
            )

        sampled_outcome_col = f"{OUTCOME_COL_NAME}_discretized"
        if sampled_outcome_col not in observational_data.columns:
            raise ValueError(
                f"{sampled_outcome_col} must be in ``observational_data`` columns."
            )

        discretized_covariate_names = [
            f"{cov}_discretized" for cov in self._covariate_names
        ]
        if not all(
            covariate in observational_data.columns
            for covariate in discretized_covariate_names
        ):
            raise ValueError(
                f"All covariates {discretized_covariate_names} must be in "
                "``observational_data`` columns."
            )

        model = self._causal_models[self.estimator_type]()

        data = CausalData(
            X=observational_data[discretized_covariate_names].copy(),  # covariates
            T=observational_data[sampled_treatment_col].copy(),  # treatment
            Y=observational_data[sampled_outcome_col].copy(),  # outcome
        )

        observed_treatments = sorted(data.T.unique().tolist())
        try:
            model.fit(data)
            all_ites = np.zeros((self._num_treat, len(observational_data)))
            all_weights = np.zeros_like(all_ites)

            if self.estimator_type == "ipw":
                outcomes = data.Y.to_numpy()
                ipw_weights = model.get_weights(data).to_numpy()
                for t in range(self._num_treat):
                    t_mask = data.T.eq(t).to_numpy()
                    all_ites[t, :] = outcomes * t_mask
                    all_weights[t, :] = ipw_weights * t_mask
            elif self.estimator_type == "oi":
                individual_outcomes = model.get_individual_treatment_effects(
                    data, treatment_values=list(range(self._num_treat))
                )
                for t in range(self._num_treat):
                    all_ites[t, :] = individual_outcomes[t].to_numpy()
                all_weights[:] = 1.0
            else:
                raise NotImplementedError(
                    f"Estimator type '{self.estimator_type}' not implemented."
                )
        except Exception as exc:
            trial_id = getattr(self.experiment, "nct_id", "unknown")
            exc.add_note(
                f"NaturalMC estimator_type={self.estimator_type!r} failed for "
                f"trial={trial_id!r}, outcome={outcome!r}, "
                f"data_shape={observational_data.shape}, "
                f"observed_treatments={observed_treatments}."
            )
            raise
        return all_ites, all_weights
