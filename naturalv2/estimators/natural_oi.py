"""Natural Outcome Imputation Estimator."""

from ast import literal_eval

import numpy as np
import pandas as pd

from naturalv2.evals.experiment import Experiment
from naturalv2.pipeline import TREATMENT_COL_NAME
from naturalv2.utils import convert_enum_to_dicts, enumerate_strings


class NaturalOI:
    """NATURAL Outcome Imputation Estimator for Individual Treatment Effects (ITE).

    Parameters
    ----------
    experiment : Experiment
        The experiment object containing treatment and covariate information.
    """

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

        self._covariate_names = experiment.covariate_names
        self._num_treat = len(experiment.treatment_names)
        self._conditional_shape = [2]  # binary outcomes

    def get_individual_treatment_effects(
        self, conditionals: pd.DataFrame
    ) -> np.ndarray:
        """Calculate Individual Treatment Effects (ITE) given conditionals.

        Given a dataframe containing P(Y | X, T) for each treatment T and covariate X,
        this method computes the ITEs for each treatment and covariate combination.

        Parameters
        ----------
        conditionals : pd.DataFrame
            DataFrame containing the conditional probabilities of outcomes given
            treatments and covariates. It should include columns for covariates,
            treatment, and outcomes.

        Returns
        -------
        np.ndarray
            An array of shape (num_treatments, num_samples) containing the ITEs for
            each treatment and covariate combination.

        Raises
        ------
        ValueError
            If the conditionals DataFrame does not contain ``y_given_tx_probs`` column
            or if it does not contain the covariate names defined in the experiment.
        """
        if "y_given_tx_probs" not in conditionals.columns:
            raise ValueError(
                "Conditionals DataFrame must contain 'y_given_tx_probs' column."
            )

        discretized_covariate_cols = [
            f"{cov}_discretized" for cov in self._covariate_names
        ]
        if not all(
            covariate in conditionals.columns
            for covariate in discretized_covariate_cols
        ):
            raise ValueError(
                "Conditionals DataFrame must contain all covariates: "
                f"{discretized_covariate_cols}."
            )

        # array of ITEs (treat2 - treat1) per unit corresponding to {outcome}
        conditionals = conditionals.copy()
        # outcome_idx = self.experiment.outcome_names.index(outcome)

        options = enumerate_strings(
            {
                covariate: self.experiment.options[covariate]
                for covariate in self._covariate_names
            }
        )
        idx_to_feat = convert_enum_to_dicts(options, self._covariate_names)
        feat_dicts = [
            self.experiment.apply_transform(dct, repr_type="numeric")
            for dct in idx_to_feat
        ]

        conditionals.loc[:, "y_given_tx_probs"] = conditionals.apply(
            lambda row: np.array(literal_eval(row["y_given_tx_probs"])).reshape(
                self._conditional_shape
            ),
            axis=1,
        )
        # choose probs corresponding to {outcome}
        # conditionals.loc[:, "y_given_tx_probs"] = conditionals.apply(
        #     lambda row: row["y_given_tx_probs"][2 * outcome_idx : 2 * (outcome_idx + 1)], axis=1
        # )

        self.outcome_conditionals = self._compute_outcome_conditionals(conditionals)
        all_ites = np.zeros((self._num_treat, len(conditionals)))
        for i, (_, row) in enumerate(conditionals.iterrows()):
            x = {
                k.replace("_discretized", ""): v
                for k, v in row[discretized_covariate_cols].to_dict().items()
            }
            x_idx = feat_dicts.index(x)
            for t in range(self._num_treat):
                all_ites[t, i] = self.outcome_conditionals[x_idx, t]

        return all_ites

    def _compute_outcome_conditionals(self, conditionals: pd.DataFrame) -> np.ndarray:
        """Compute the outcome conditionals for each treatment and covariate combination."""
        options = enumerate_strings(
            {
                covariate: self.experiment.options[covariate]
                for covariate in self._covariate_names
            }
        )
        idx_to_feat = convert_enum_to_dicts(options, self._covariate_names)
        feat_dicts = [
            self.experiment.apply_transform(dct, repr_type="numeric")
            for dct in idx_to_feat
        ]

        outcome_conditionals = np.zeros((len(feat_dicts), self._num_treat))

        for i in range(len(feat_dicts)):
            features = feat_dicts[i]
            subset = conditionals.copy()

            # restrict posts using sampled features
            for key in self._covariate_names:
                subset = subset.loc[subset[key + "_discretized"] == features[key]]

            for t in range(self._num_treat):
                subset_t = subset.loc[subset[TREATMENT_COL_NAME + "_discretized"] == t]

                if len(subset_t) > 0:
                    py1_given_xt = np.array(
                        [
                            sum([j * prob[j] for j in range(len(prob))])
                            for prob in subset_t["y_given_tx_probs"]
                        ]
                    )
                    outcome_conditionals[i, t] = np.mean(py1_given_xt)

        return outcome_conditionals
