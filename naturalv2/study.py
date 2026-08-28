"""Module for managing studies in the NaturalV2 project."""

import logging
import os

import yaml
from omegaconf import DictConfig

from naturalv2.experiment import Experiment
from naturalv2.utils import get_experiment_filepath, sanitize_filename


logger = logging.getLogger(__name__)


class Study:
    """Represents a study with retrospective and test trials.

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

    def __init__(
        self,
        retro_trials: list[tuple[str, str | None]],
        test_trials: list[tuple[str, str | None]],
        cfg: DictConfig,
    ) -> None:
        """Initialize the Study object with retrospective and test trials."""
        # order retro_trials by completion date and split into train/val according to train_ratio
        retro_trials.sort(key=lambda x: (x[1] is None, x[1]))
        train_size = int(len(retro_trials) * cfg.train_ratio)
        train_trials, val_trials = retro_trials[:train_size], retro_trials[train_size:]

        self.conditions: list[str] = list(cfg.conditions)
        self.train_ratio: float = cfg.train_ratio
        self.ate: bool = cfg.ate
        self.experiment_name: str = cfg.experiment_name

        require_binary_endpoint: bool = cfg.trial_filters.binary_endpoint
        ot = lambda exp: (
            exp.apo_outcome_treatment if not self.ate else exp.outcome_treatment
        )
        es = lambda exp: (
            exp.avg_potential_outcomes if not self.ate else exp.effect_sizes
        )

        build_exp = lambda nct_id, status: Experiment(
            cfg.save_path,
            nct_id,
            self.experiment_name,
            status=status,
            require_binary_endpoint=require_binary_endpoint,
            outcome_bounds=cfg.get("outcome_bounds", {}).get(nct_id, {}),
        )

        train_exp = [build_exp(nct_id, "completed") for (nct_id, _) in train_trials]
        self.train_trials = [
            {exp.nct_id: [exp.title, exp.date] + list(exp.references)}
            for exp in train_exp
            if es(exp) and ot(exp)
        ]
        self.num_train_labels = sum([len(es(exp)) for exp in train_exp if es(exp)])

        val_exp = [build_exp(nct_id, "completed") for (nct_id, _) in val_trials]
        self.val_trials = [
            {exp.nct_id: [exp.title, exp.date] + list(exp.references)}
            for exp in val_exp
            if es(exp) and ot(exp)
        ]
        self.num_val_labels = sum([len(es(exp)) for exp in val_exp if es(exp)])

        test_exp = [build_exp(nct_id, "active") for (nct_id, _) in test_trials]
        self.test_trials = [
            {exp.nct_id: [exp.title, exp.date] + list(exp.references)}
            for exp in test_exp
            if ot(exp)
        ]
        self.num_test_to_predict = sum([len(ot(exp)) for exp in test_exp if ot(exp)])

        # Persist each experiment so later pipeline steps load it.
        for exp in train_exp + val_exp + test_exp:
            exp.to_yaml(
                get_experiment_filepath(cfg.save_path, exp.nct_id, self.experiment_name)
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

    def _log_study_summary(self) -> None:
        """Log a summary of the study's configuration and data."""
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


class StudyDataset:
    """Represents a dataset for a study, containing conditions and sources.

    Parameters
    ----------
    conditions : list[str]
        List of conditions for the study.
    sources : list[str]
        List of sources from which data is collected.

    """

    def __init__(
        self,
        conditions: list[str],
        sources: list[str],
    ) -> None:
        """Initialize the StudyDataset with conditions and sources."""
        self.conditions = list(conditions)
        self.sources = {source: {} for source in sources}
        self.data_sizes = {}
        self.data_paths = {}

    def to_yaml(self, filename: str) -> None:
        """Save the study dataset to a YAML file.

        Parameters
        ----------
        filename : str
            The path to the YAML file where the dataset will be saved.

        """
        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    @classmethod
    def from_yaml(cls, filename: str) -> "StudyDataset":
        """Load a study dataset from a YAML file.

        Parameters
        ----------
        filename : str
            The path to the YAML file from which the dataset will be loaded.

        Returns
        -------
        StudyDataset
            An instance of `StudyDataset` populated with data from the YAML file.
        """
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        study_dataset = cls.__new__(cls)
        study_dataset.__dict__.update(data)
        return study_dataset


def get_study_filepaths(
    base_dir: str, condition: str, experiment_name: str, ate: bool = True
) -> dict[str, str]:
    """Get file paths for the study and study dataset YAML files.

    Parameters
    ----------
    base_dir : str
        The base directory where the study files will be stored.
    condition : str
        The condition for which the study files are being created.
    experiment_name : str
        The experiment name, included in the filename so that multiple studies
        for the same condition (e.g. different trial filters) do not collide.
    ate : bool, default=True
        If False, the ``_apo`` estimand suffix is appended to the filename.

    Returns
    -------
    dict[str, str]
        A dictionary containing the file paths for the study and study dataset
        YAML files under the keys 'study' and 'study_dataset', respectively.
    """
    # Ensure that base_dir exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create the 'studies' directory if it doesn't exist
    studies_dir = os.path.join(base_dir, "studies")
    os.makedirs(studies_dir, exist_ok=True)

    name = (
        f"{sanitize_filename(condition.lower())}_{sanitize_filename(experiment_name)}"
    )
    if not ate:
        name = name + "_apo"
    return {
        "study": os.path.join(studies_dir, f"{name}_study.yaml"),
        "study_dataset": os.path.join(studies_dir, f"{name}_study_dataset.yaml"),
    }
