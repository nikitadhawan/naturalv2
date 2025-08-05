"""Natural Inverse Probability Weighting (IPW) Estimator."""

import ast

import numpy as np
import pandas as pd

from naturalv2.evals.experiment import Experiment
from naturalv2.utils import convert_enum_to_dicts, enumerate_strings


class NaturalIPW:
    """NATURAL Inverse Probability Weighting (IPW) Estimator.

    This class computes Individual Treatment Effects (ITE) using the IPW method.
    It calculates the propensity scores for each treatment given the covariates
    and uses these scores to estimate the ITEs from the conditional probabilities
    of treatment and outcome given covariates.

    Parameters
    ----------
    experiment : Experiment
        The experiment object containing treatment and covariate information.

    """

    def __init__(self, experiment: Experiment) -> None:
        """Initialize the NaturalIPW estimator."""
        self.experiment = experiment

        self._covariate_names = experiment.covariate_names
        self._num_treat = len(experiment.treatment_names)
        self._conditional_shape = [self._num_treat, 2]  # binary outcomes

    def get_individual_treatment_effects(
        self, conditionals: pd.DataFrame
    ) -> np.ndarray:
        """Calculate Individual Treatment Effects (ITE) given conditionals.

        Given a dataframe containing P(T, Y | X) for each treatment T and covariate X,
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
        """
        discretized_covariate_cols = [
            f"{cov}_discretized" for cov in self._covariate_names
        ]
        if not all(col in conditionals.columns for col in discretized_covariate_cols):
            raise ValueError(
                "Conditionals DataFrame must contain discretized covariate columns: "
                f"{discretized_covariate_cols}."
            )

        if "ty_given_x_probs" not in conditionals.columns:
            raise ValueError(
                "Conditionals DataFrame must contain 'ty_given_x_probs' column."
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
        ]  # dataset should have already been discretized and the transforms ready

        conditionals.loc[:, "ty_given_x_probs"] = conditionals.apply(
            lambda row: np.array(ast.literal_eval(row["ty_given_x_probs"])).reshape(
                self._conditional_shape
            ),
            axis=1,
        )
        # choose probs corresponding to {outcome}
        # conditionals.loc[:, "ty_given_x_probs"] = conditionals.apply(
        #     lambda row: row["ty_given_x_probs"][:, 2 * outcome_idx : 2 * (outcome_idx + 1)], axis=1
        # )

        self.prop_score_lst = self._compute_prop_score(conditionals)
        all_ites = np.zeros((self._num_treat, len(conditionals)))
        for i, (_, row) in enumerate(conditionals.iterrows()):  # Fixed PLW2901
            probs = row["ty_given_x_probs"]
            x = {
                k.replace("_discretized", ""): v
                for k, v in row[discretized_covariate_cols].to_dict().items()
            }

            # enumerate treatments
            for t in range(self._num_treat):
                # propensity score given x features
                x_idx = feat_dicts.index(x)
                t_given_x = self.prop_score_lst[x_idx, t]
                # enumerate binary outcomes
                for y in range(2):
                    # probability of this enumerated possibility
                    posterior = probs[t, y]
                    # ignore propensity scores of 0
                    if t_given_x > 0:
                        all_ites[t, i] += y * posterior / t_given_x

        return all_ites

    def _compute_prop_score(self, conditionals: pd.DataFrame) -> np.ndarray:
        """Compute propensity scores for each treatment given covariates."""
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
        prop_score_lst = []

        for i in range(len(feat_dicts)):
            features = feat_dicts[i]
            subset = conditionals.copy()

            # restrict posts using sampled features
            for key in self._covariate_names:
                subset = subset.loc[subset[key + "_discretized"] == features[key]]
            if len(subset) == 0:
                prop_scores = [0 for _ in range(self._num_treat)]
            else:
                # marginalize out Y
                propensity = subset[["ty_given_x_probs"]].apply(
                    lambda row: np.sum(row["ty_given_x_probs"], axis=-1), axis=1
                )
                # average over posts
                prop_scores = []
                for t in range(self._num_treat):
                    prop_t = propensity.apply(lambda arr, t=t: arr[t]).sum() / len(
                        subset
                    )  # Fixed B023
                    prop_scores.append(prop_t)
            prop_score_lst.append(prop_scores)
        return np.array(prop_score_lst)
