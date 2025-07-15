"""Natural Monte Carlo Estimator."""

from typing import Literal

import numpy as np
import pandas as pd

from naturalv2.evals.experiment import Experiment
from naturalv2.models.causal_models import (
    IPSW,
    CausalData,
    OutcomeImputation,
)
from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME


class NaturalMC:
    """NATURAL Monte Carlo Estimator for Individual Treatment Effects (ITE).

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

        self._num_treat = len(experiment.treatment_names)
        self._causal_models: dict[str, type[IPSW] | type[OutcomeImputation]] = {
            "ipw": IPSW,
            "oi": OutcomeImputation,
        }

    def get_individual_treatment_effects(
        self, observational_data: pd.DataFrame, outcome: str
    ) -> np.ndarray:
        """Calculate Individual Treatment Effects (ITE) for a given outcome.

        Parameters
        ----------
        observational_data : pd.DataFrame
            Data containing treatment, covariates, and outcome.
        outcome : str
            The name of the outcome column in `observational_data`.

        Returns
        -------
        np.ndarray
            An array of ITEs (treat2 - treat1) per unit corresponding to the
            specified outcome.

        Raises
        ------
        ValueError
            If the treatment column or covariates are not present in the data.

        """
        print(observational_data.columns)
        if TREATMENT_COL_NAME not in observational_data.columns:
            raise ValueError(
                f"{TREATMENT_COL_NAME} must be in ``observational_data`` columns."
            )

        if not all(
            covariate in observational_data.columns
            for covariate in self.experiment.covariate_names
        ):
            raise ValueError(
                f"All covariates {self.experiment.covariate_names} must be in "
                "``observational_data`` columns."
            )

        # array of ITEs (treat2 - treat1) per unit corresponding to {outcome}
        model = self._causal_models[self.estimator_type]()

        data = CausalData(
            X=observational_data[self.experiment.covariate_names].copy(),  # covariates
            T=observational_data[TREATMENT_COL_NAME].copy(),  # treatment
            Y=observational_data[OUTCOME_COL_NAME].copy(),  # outcome
        )

        model.fit(data)
        individual_outcomes = model.get_individual_treatment_effects(data)

        all_ites = np.zeros((self._num_treat, len(observational_data)))
        for t in range(self._num_treat):
            if self.estimator_type == "ipw":
                t_mask = [1 if treat == t else 0 for treat in data.T]
                all_ites[t, :] = (individual_outcomes * t_mask).to_numpy()
            elif self.estimator_type == "oi":
                all_ites[t, :] = individual_outcomes[t].to_numpy()
            else:
                raise NotImplementedError(
                    f"Estimator type '{self.estimator_type}' not implemented."
                )
        return all_ites
