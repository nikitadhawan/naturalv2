import logging
import os
from multiprocessing import Pool, cpu_count
from typing import Optional

import hydra
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from naturalv2.evals.clinical_trial import ClinicalTrial, download_clinical_trials
from naturalv2.evals.experiment import Experiment
from naturalv2.utils import check_trial, get_nested_value


logger = logging.getLogger(__name__)


def _process_trial_file(
    args: tuple[str, str],
) -> tuple[Optional[str], Optional[dict[str, int]], Optional[bool]]:
    filename, trial_path = args
    if filename.endswith(".json"):
        trial = ClinicalTrial.from_json_file(os.path.join(trial_path, filename))
        trial_stats, check = check_trial(trial)
        return trial.protocolSection.identificationModule.nctId, trial_stats, check
    return None, None, None


def find_valid_ncts(data_path: str, test: bool = False) -> list[str]:
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
        with open(valid_nct_path, "a") as valid_file, Pool(cpu_count()) as pool:
            file_list = [(filename, trial_path) for filename in os.listdir(trial_path)]

            results = list(
                tqdm(
                    pool.imap(_process_trial_file, file_list),
                    desc="Finding valid trials" + (" (test)" if test else ""),
                    total=len(file_list),
                )
            )
            for nct_id, trial_stats, check in results:
                if nct_id:
                    for key, value in trial_stats.items():
                        stats[key] += value
                    if check:
                        valid_file.write(f"{nct_id}\n")
        logger.info("Benchmark Stats: %s", stats)

    with open(valid_nct_path, "r") as valid_file:
        return [line.strip() for line in valid_file.readlines()]


def _process_condition_trial(
    args: tuple[str, str, set[str], bool],
) -> tuple[Optional[str], Optional[str]]:
    nct_id, trial_path, conditions_set, test = args
    trial = ClinicalTrial.from_json_file(os.path.join(trial_path, f"{nct_id}.json"))
    trial_conditions: Optional[list[str]] = get_nested_value(
        trial, "protocolSection.conditionsModule.conditions"
    )
    trial_keywords: Optional[list[str]] = get_nested_value(
        trial, "protocolSection.conditionsModule.keywords"
    )
    trial_conditions_set = {
        word.lower() for word in (trial_conditions or []) + (trial_keywords or [])
    }

    matching_conditions = [
        trial_condition
        for trial_condition in trial_conditions_set
        if any(condition in trial_condition for condition in conditions_set)
    ]
    if matching_conditions:
        result_date: Optional[str] = (
            get_nested_value(
                trial, "protocolSection.statusModule.completionDateStruct.date"
            )
            if test
            else get_nested_value(
                trial, "protocolSection.statusModule.resultsFirstPostDateStruct.date"
            )
        )
        return nct_id, result_date
    return None, None


def find_condition_ncts(
    nct_ids: list[str], data_path: str, conditions: list[str], test=False
) -> list[tuple[str, Optional[str]]]:
    trial_path = os.path.join(data_path, "nct_reports" + ("_test" if test else ""))
    condition_nct_path = os.path.join(
        trial_path, f"valid_binary_{conditions[0]}_nct_ids.txt"
    )
    condition_trials: list[tuple[str, Optional[str]]] = []
    conditions_set = {cond.replace("_", " ").lower() for cond in conditions}

    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(
                    _process_condition_trial,
                    [(nct_id, trial_path, conditions_set, test) for nct_id in nct_ids],
                ),
                desc="Finding condition trials" + (" (test)" if test else ""),
                total=len(nct_ids),
            )
        )
        for nct_id, result_date in results:
            if nct_id:
                condition_trials.append((nct_id, result_date))
                with open(condition_nct_path, "a") as condition_file:
                    condition_file.write(f"{nct_id}\n")

    return condition_trials


class Study:
    def __init__(
        self,
        retro_trials: list[tuple[str, Optional[str]]],
        test_trials: list[tuple[str, Optional[str]]],
        cfg: DictConfig,
    ):
        # order retro_trials by completion date and split into train/val according to train_ratio
        retro_trials.sort(key=lambda x: (x[1] is None, x[1]))
        train_size = int(len(retro_trials) * cfg.train_ratio)
        train_trials, val_trials = retro_trials[:train_size], retro_trials[train_size:]

        self.conditions: list[str] = list(cfg.conditions)
        self.train_ratio: float = cfg.train_ratio

        train_exp = [
            Experiment(cfg.data_path, nct_id, split="train")
            for (nct_id, _) in train_trials
        ]
        self.train_trials = [
            {exp.nct_id: [exp.title, exp.date] + exp.references}
            for exp in train_exp
            if exp.effect_sizes and exp.outcome_treatment
        ]
        self.num_train_labels = sum(
            [len(exp.effect_sizes) for exp in train_exp if exp.effect_sizes]
        )

        val_exp = [
            Experiment(cfg.data_path, nct_id, split="val") for (nct_id, _) in val_trials
        ]
        self.val_trials = [
            {exp.nct_id: [exp.title, exp.date] + exp.references}
            for exp in val_exp
            if exp.effect_sizes and exp.outcome_treatment
        ]
        self.num_val_labels = sum(
            [len(exp.effect_sizes) for exp in val_exp if exp.effect_sizes]
        )

        test_exp = [
            Experiment(cfg.data_path, nct_id, split="test")
            for (nct_id, _) in test_trials
        ]
        self.test_trials = [
            {exp.nct_id: [exp.title, exp.date] + exp.references}
            for exp in test_exp
            if exp.outcome_treatment
        ]
        self.num_test_to_predict = sum(
            [len(exp.outcome_treatment) for exp in test_exp if exp.outcome_treatment]
        )

        self.num_train_trials = len(self.train_trials)
        self.num_val_trials = len(self.val_trials)
        self.num_test_trials = len(self.test_trials)

        logger.info(
            """
            Study created for %s with:
            Train: %s trials, %s labels
            Val: %s trials, %s labels
            Test: %s trials, %s labels to predict
            """,
            self.conditions,
            self.num_train_trials,
            self.num_train_labels,
            self.num_val_trials,
            self.num_val_labels,
            self.num_test_trials,
            self.num_test_to_predict,
        )

    def to_yaml(self, filename):
        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        study = cls.__new__(cls)
        study.__dict__.update(data)
        return study


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # find nct_ids of valid retrospective and test trials
    nct_list = find_valid_ncts(cfg.data_path)
    test_nct_list = find_valid_ncts(cfg.data_path, test=True)
    logger.info(
        "Total valid trials: %s Completed and %s Test",
        len(nct_list),
        len(test_nct_list),
    )

    # find nct_ids of retrospective and test trials related to {condition}
    retro_trials = find_condition_ncts(nct_list, cfg.data_path, cfg.conditions)
    test_trials = find_condition_ncts(
        test_nct_list, cfg.data_path, cfg.conditions, test=True
    )

    study = Study(retro_trials, test_trials, cfg)
    os.makedirs(os.path.join(cfg.save_path, "studies"), exist_ok=True)
    study.to_yaml(
        os.path.join(cfg.save_path, "studies", cfg.conditions[0] + "_study.yaml")
    )


if __name__ == "__main__":
    main()
