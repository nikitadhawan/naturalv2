"""Create a study for clinical trials based on specified conditions."""

import logging
import os

import hydra
import yaml
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map

from naturalv2.evals.clinical_trial import ClinicalTrial, Mesh, download_clinical_trials
from naturalv2.evals.experiment import Experiment
from naturalv2.utils import check_trial, get_nested_value


logger = logging.getLogger(__name__)


class Study:
    def __init__(
        self,
        retro_trials: list[tuple[str, str | None]],
        test_trials: list[tuple[str, str | None]],
        cfg: DictConfig,
    ) -> None:
        """Initialize the Study object with retrospective and test trials.

        Parameters
        ----------
        retro_trials : list[tuple[str, str | None]]
            List of tuples containing NCT IDs and their completion dates for
            retrospective trials.
        test_trials : list[tuple[str, str | None]]
            List of tuples containing NCT IDs and their expected completion dates
            for test trials.
        cfg : DictConfig
            Configuration object containing study parameters.

        """
        # order retro_trials by completion date and split into train/val according to train_ratio
        retro_trials.sort(key=lambda x: (x[1] is None, x[1]))
        train_size = int(len(retro_trials) * cfg.train_ratio)
        train_trials, val_trials = retro_trials[:train_size], retro_trials[train_size:]

        self.conditions: list[str] = list(cfg.conditions)
        self.train_ratio: float = cfg.train_ratio

        train_exp = [
            Experiment(cfg.data_path, nct_id, status="completed")
            for (nct_id, _) in train_trials
        ]
        self.train_trials = [
            {exp.nct_id: [exp.title, exp.date] + list(exp.references)}
            for exp in train_exp
            if exp.effect_sizes and exp.outcome_treatment
        ]
        self.num_train_labels = sum(
            [len(exp.effect_sizes) for exp in train_exp if exp.effect_sizes]
        )

        val_exp = [
            Experiment(cfg.data_path, nct_id, status="completed")
            for (nct_id, _) in val_trials
        ]
        self.val_trials = [
            {exp.nct_id: [exp.title, exp.date] + list(exp.references)}
            for exp in val_exp
            if exp.effect_sizes and exp.outcome_treatment
        ]
        self.num_val_labels = sum(
            [len(exp.effect_sizes) for exp in val_exp if exp.effect_sizes]
        )

        test_exp = [
            Experiment(cfg.data_path, nct_id, status="active")
            for (nct_id, _) in test_trials
        ]
        self.test_trials = [
            {exp.nct_id: [exp.title, exp.date] + list(exp.references)}
            for exp in test_exp
            if exp.outcome_treatment
        ]
        self.num_test_to_predict = sum(
            [len(exp.outcome_treatment) for exp in test_exp if exp.outcome_treatment]
        )

        # Collect all baseline measures and their frequency
        covariates_dict: dict[str, int] = {}
        for exp in train_exp + val_exp + test_exp:
            for covariate_name in exp.covariate_names:
                covariates_dict[covariate_name] = (
                    covariates_dict.get(covariate_name, 0) + 1
                )
        self.covariates = sorted(
            covariates_dict.items(), key=lambda item: item[1], reverse=True
        )

        self.num_train_trials = len(self.train_trials)
        self.num_val_trials = len(self.val_trials)
        self.num_test_trials = len(self.test_trials)

        self.num_keywords = len(
            {
                kw
                for exp in train_exp + val_exp + test_exp
                for kw in (exp.conditions if exp.conditions else [])
            }
        )
        self.num_treatments = len(
            {
                treatment
                for exp in train_exp + val_exp + test_exp
                for treatment in (exp.treatment_names if exp.treatment_names else [])
            }
        )
        self.num_outcomes = len(
            {
                outcome
                for exp in train_exp + val_exp + test_exp
                for outcome in (exp.outcome_names if exp.outcome_names else [])
            }
        )

        self._log_study_summary()

    def _log_study_summary(self):
        logger.info(
            """
            Study created for %s with:
            Train: %s trials, %s labels
            Val: %s trials, %s labels
            Test: %s trials, %s labels to predict
            Condition keywords: %s
            Treatments: %s
            Outcomes: %s
            """,
            self.conditions,
            self.num_train_trials,
            self.num_train_labels,
            self.num_val_trials,
            self.num_val_labels,
            self.num_test_trials,
            self.num_test_to_predict,
            self.num_keywords,
            self.num_treatments,
            self.num_outcomes,
        )

    def to_yaml(self, filename: str) -> None:
        """Save the Study object to a YAML file.

        Parameters
        ----------
        filename : str
            The path to the YAML file where the Study data will be saved.
        """
        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    @classmethod
    def from_yaml(cls: type["Study"], filename: str) -> "Study":
        """Load a Study object from a YAML file.

        Parameters
        ----------
        filename : str
            The path to the YAML file containing the Study data.

        Returns
        -------
        Study
            An instance of the Study class populated with data from the YAML file.
        """
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        study = cls.__new__(cls)
        study.__dict__.update(data)

        study._log_study_summary()
        return study

    def __repr__(self) -> str:
        """String representation of the Study object."""
        return (
            f"Study(conditions={self.conditions}, "
            f"num_train_trials={self.num_train_trials}, "
            f"num_train_labels={self.num_train_labels}, "
            f"num_val_trials={self.num_val_trials}, "
            f"num_val_labels={self.num_val_labels}, "
            f"num_test_trials={self.num_test_trials}, "
            f"num_test_to_predict={self.num_test_to_predict})"
        )


def find_valid_ncts(data_path: str, test: bool = False) -> list[str]:
    """Find valid NCT IDs from clinical trial reports.

    This function processes clinical trial JSON files to identify valid trials
    based on specific criteria such as randomization, control groups, and
    healthy participants. It returns a list of valid NCT IDs.

    Parameters
    ----------
    data_path : str
        The path to the directory containing clinical trial JSON files.
    test : bool, default=False
        If True, processes test data; otherwise, processes training data.
        Defaults to False.

    Returns
    -------
    list[str]
        A list of valid NCT IDs that meet the specified criteria.
    """
    stats = {
        "total": 0,
        "randomized": 0,
        "multiple_noncontrol": 0,
        "nonhealthy": 0,
        "binary_endpoint": 0,
    }
    trial_path = os.path.join(data_path, "nct_reports" + ("_test" if test else ""))
    valid_nct_path = os.path.join(trial_path, "valid_binary_nct_ids.txt")

    if not os.path.exists(trial_path):
        download_clinical_trials(trial_path, test)

    if not os.path.exists(valid_nct_path):
        file_list = [(filename, trial_path) for filename in os.listdir(trial_path)]

        results: list[tuple[str, dict[str, int], bool]] = process_map(
            _process_trial_file,
            file_list,
            desc="Finding valid trials" + (" (test)" if test else ""),
            chunksize=1,
        )
        with open(valid_nct_path, "a") as valid_file:
            for nct_id, trial_stats, check in results:
                if nct_id and trial_stats:
                    for key, value in trial_stats.items():
                        stats[key] += value
                    if check:
                        valid_file.write(f"{nct_id}\n")
        logger.info("Benchmark Stats: %s", stats)

    with open(valid_nct_path, "r") as valid_file:
        return [line.strip() for line in valid_file.readlines()]


def find_condition_ncts(
    nct_ids: list[str], data_path: str, conditions: list[str], test: bool = False
) -> list[tuple[str, str | None]]:
    """Find NCT IDs of trials related to specific conditions.

    This function processes clinical trial JSON files to identify trials that
    are related to specified medical conditions. It returns a list of tuples
    containing NCT IDs and their corresponding expected completion dates.

    Parameters
    ----------
    nct_ids : list[str]
        A list of NCT IDs to process.
    data_path : str
        The path to the directory containing clinical trial JSON files.
    conditions : list[str]
        A list of medical conditions to filter trials by.
    test : bool, default=False
        If True, processes test data; otherwise, processes training data.

    Returns
    -------
    list[tuple[str, str | None]]
        A list of tuples where each tuple contains an NCT ID and its expected
        completion date (or None if not available). The list is filtered to
        include only trials that match the specified conditions.
    """
    trial_path = os.path.join(data_path, "nct_reports" + ("_test" if test else ""))
    condition_nct_path = os.path.join(
        trial_path, f"valid_binary_{conditions[0]}_nct_ids.txt"
    )
    condition_trials: list[tuple[str, str | None]] = []
    conditions_set = {cond.replace("_", " ").lower() for cond in conditions}

    results: list[tuple[str, str | None]] = process_map(
        _process_condition_trial,
        [(nct_id, trial_path, conditions_set, test) for nct_id in nct_ids],
        desc="Finding condition trials" + (" (test)" if test else ""),
        chunksize=1,
    )

    for nct_id, result_date in results:
        if nct_id:
            condition_trials.append((nct_id, result_date))
            with open(condition_nct_path, "a") as condition_file:
                condition_file.write(f"{nct_id}\n")

    # For debugging:
    # for nct_id in tqdm(nct_ids, desc="Finding condition trials" + (" (test)" if test else "")):
    #     result_nct_id, result_date = _process_condition_trial((nct_id, trial_path, conditions_set, test))
    #     if result_nct_id:
    #         condition_trials.append((result_nct_id, result_date))
    #         with open(condition_nct_path, "a") as condition_file:
    #             condition_file.write(f"{result_nct_id}\n")

    return condition_trials


def _process_trial_file(args: tuple[str, str]) -> tuple[str, dict[str, int], bool]:
    """Process a single clinical trial JSON file to extract its NCT ID and statistics."""
    filename, trial_path = args
    if filename.endswith(".json"):
        trial = ClinicalTrial.from_json_file(os.path.join(trial_path, filename))
        trial_stats, check = check_trial(trial)
        return trial.protocolSection.identificationModule.nctId, trial_stats, check
    return "", {}, False


def _process_condition_trial(
    args: tuple[str, str, set[str], bool],
) -> tuple[str, str | None]:
    """Process a clinical trial to find if it matches specified conditions."""
    nct_id, trial_path, conditions_set, test = args

    trial = ClinicalTrial.from_json_file(os.path.join(trial_path, f"{nct_id}.json"))

    mesh_ancestors: list[Mesh] | None = get_nested_value(
        trial, "derivedSection.conditionBrowseModule.ancestors"
    )
    trial_disease_mesh = (
        [ancestor.term for ancestor in mesh_ancestors] if mesh_ancestors else []
    )
    trial_mesh_set = {mesh.term.lower() for mesh in trial_disease_mesh}

    # Remove "disease" or "diseases" from the set
    trial_mesh_set = {
        term for term in trial_mesh_set if term not in {"disease", "diseases"}
    }

    # Check if any of the conditions match the trial's disease mesh
    matching_conditions = [
        trial_mesh
        for trial_mesh in trial_mesh_set
        if any(condition in trial_mesh for condition in conditions_set)
        or any(trial_mesh in condition for condition in conditions_set)
    ]

    if matching_conditions:
        result_date: str | None = (
            get_nested_value(
                trial, "protocolSection.statusModule.completionDateStruct.date"
            )
            if test
            else get_nested_value(
                trial, "protocolSection.statusModule.resultsFirstPostDateStruct.date"
            )
        )
        return nct_id, result_date
    return "", None


def run_study_and_get_stats(cfg: DictConfig) -> dict:
    nct_list = find_valid_ncts(cfg.data_path)
    test_nct_list = find_valid_ncts(cfg.data_path, test=True)
    logger.info(
        "Total valid trials: %s Completed and %s Test",
        len(nct_list),
        len(test_nct_list),
    )

    retro_trials = find_condition_ncts(nct_list, cfg.data_path, cfg.conditions)
    test_trials = find_condition_ncts(
        test_nct_list, cfg.data_path, cfg.conditions, test=True
    )

    study = Study(retro_trials, test_trials, cfg)
    os.makedirs(os.path.join(cfg.save_path, "studies"), exist_ok=True)
    study.to_yaml(
        os.path.join(
            cfg.save_path,
            "studies",
            cfg.conditions[0].lower().replace(" ", "_") + "_study.yaml",
        )
    )

    return {
        "conditions": cfg.conditions,
        "train_trials": study.num_train_trials,
        "train_labels": study.num_train_labels,
        "val_trials": study.num_val_trials,
        "val_labels": study.num_val_labels,
        "test_trials": study.num_test_trials,
        "test_labels": study.num_test_to_predict,
        "num_keywords": study.num_keywords,
        "num_treatments": study.num_treatments,
        "num_outcomes": study.num_outcomes,
    }


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    stats = run_study_and_get_stats(cfg)
    print(yaml.safe_dump(stats, sort_keys=False))


if __name__ == "__main__":
    main()
