import numpy as np

from naturalv2.utils import enum_to_dcts, enumerate_strings


class NaturalOI:
    def __init__(self, experiment):
        self.experiment = experiment
        self.covariate_names = experiment.covariate_names
        self.num_treat = len(experiment.treatment_names)
        # self.num_out = len(experiment.outcome_names)
        self.conditional_shape = [2]  # binary outcomes

    def compute_outcome_cond(self, conditionals):
        options = enumerate_strings(self.experiment.get_options(self.covariate_names))
        idx_to_feat = enum_to_dcts(options, self.covariate_names)
        feat_dicts = [self.experiment.transform_samples(dct) for dct in idx_to_feat]

        outcome_conditionals = np.zeros((len(feat_dicts), self.num_treat))

        for i in range(len(feat_dicts)):
            features = feat_dicts[i]
            subset = conditionals.copy()
            # restrict posts using sampled features
            for key in self.covariate_names:
                subset = subset.loc[subset[key] == features[key]]
            for t in range(self.num_treat):
                subset_t = subset.loc[subset["treatment"] == t]

                if len(subset_t) > 0:
                    py1_given_xt = np.array(
                        [
                            sum([j * prob[j] for j in range(len(prob))])
                            for prob in subset_t["probs"]
                        ]
                    )
                    outcome_conditionals[i, t] = np.mean(py1_given_xt)

        return outcome_conditionals

    def get_ites(self, conditionals):
        # array of ITEs (treat2 - treat1) per unit corresponding to {outcome}
        conditionals = conditionals.copy()
        # outcome_idx = self.experiment.outcome_names.index(outcome)
        options = enumerate_strings(self.experiment.get_options(self.covariate_names))
        idx_to_feat = enum_to_dcts(options, self.covariate_names)
        feat_dicts = [self.experiment.transform_samples(dct) for dct in idx_to_feat]

        conditionals.loc[:, "probs"] = conditionals.apply(
            lambda row: np.array(
                [float(prob) for prob in row["probs"][1:-1].split()]
            ).reshape(self.conditional_shape),
            axis=1,
        )
        # choose probs corresponding to {outcome}
        # conditionals.loc[:, "probs"] = conditionals.apply(
        #     lambda row: row["probs"][2 * outcome_idx : 2 * (outcome_idx + 1)], axis=1
        # )

        self.outcome_conditionals = self.compute_outcome_cond(conditionals)
        all_ites = np.zeros((self.num_treat, len(conditionals)))
        for i, (_, row) in enumerate(conditionals.iterrows()):
            x = row[self.covariate_names].to_dict()
            x_idx = feat_dicts.index(x)
            for t in range(self.num_treat):
                all_ites[t, i] = self.outcome_conditionals[x_idx, t]

        return all_ites
