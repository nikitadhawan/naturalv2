"""Create a study for clinical trials based on specified conditions."""

import logging
import os

import hydra
import yaml
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map

from naturalv2.clinical_trial import ClinicalTrial, Mesh, download_clinical_trials
from naturalv2.study import Study, get_study_filepaths
from naturalv2.utils import check_trial, get_nested_value


logger = logging.getLogger(__name__)


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
            leave=False,
            dynamic_ncols=True,
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
        leave=False,
        dynamic_ncols=True,
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
    trial_mesh_set = {mesh.lower() for mesh in trial_disease_mesh}

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
    study_filepath = get_study_filepaths(cfg.data_path, cfg.conditions[0])["study"]
    study.to_yaml(study_filepath)

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


# TODO: improve on relative path for config
@hydra.main(config_path="../../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    stats = run_study_and_get_stats(cfg)
    print(yaml.safe_dump(stats, sort_keys=False))


if __name__ == "__main__":
    main()
