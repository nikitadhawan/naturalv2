import os
import yaml
import pandas as pd
from dotenv import load_dotenv

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from create_study import Study
from naturalv2.evals.experiment import Experiment

# ruff: noqa

load_dotenv(".env")


class StudyDataset:
    def __init__(
        self,
        conditions: list[str],
        sources: list[str],
    ):
        self.conditions = list(conditions)
        self.sources = list(sources)
        self.data_sizes = {}
        self.data_paths = {}

    def to_yaml(self, filename):
        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        study_dataset = cls.__new__(cls)
        study_dataset.__dict__.update(data)
        return study_dataset


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    study_file = os.path.join(
        cfg.save_path, "studies", cfg.conditions[0] + "_study.yaml"
    )
    study = Study.from_yaml(study_file)
    train_ncts = [list(trial.keys())[0] for trial in study.train_trials]
    val_ncts = [list(trial.keys())[0] for trial in study.val_trials]
    test_ncts = [list(trial.keys())[0] for trial in study.test_trials]

    study_dataset_file = os.path.join(
        cfg.data_path, "studies", cfg.conditions[0] + "_study_dataset.yaml"
    )
    if os.path.exists(study_dataset_file):
        study_dataset = StudyDataset.from_yaml(study_dataset_file)
    else:
        study_dataset = StudyDataset(study.conditions, cfg.sources)

    for source_name in cfg.sources:
        source_dataset = instantiate(cfg[source_name])

        # search for keyword list + download
        if f"{source_name}_condition_filtered" not in study_dataset.data_paths:
            condition_filter_paths = source_dataset.condition_filter(study.conditions)
            study_dataset.data_paths.update(
                {f"{source_name}_condition_filtered": condition_filter_paths}
            )

        # rule based filter + format datapoints
        if f"{source_name}_cleaned" not in study_dataset.data_paths:
            clean_path, data_size = source_dataset.clean_data(study.conditions[0])
            study_dataset.data_paths.update({f"{source_name}_cleaned": clean_path})
            study_dataset.data_sizes.update({f"{source_name}_cleaned": data_size})
        clean_path = study_dataset.data_paths[f"{source_name}_cleaned"]

        # search for treatment and outcome to curate data for each experiment
        os.makedirs(os.path.join(cfg.save_path, "experiments"), exist_ok=True)
        for nct_id in train_ncts + val_ncts + test_ncts:
            if f"{source_name}_{nct_id}" not in study_dataset.data_paths:
                exp_file = os.path.join(cfg.save_path, f"experiments/{nct_id}.yaml")
                # Load Experiment from exisiting file or create a new one
                try:
                    exp = Experiment.from_yaml(exp_file)
                except:
                    exp = Experiment(cfg.data_path, nct_id, split="train")
                # Track the studies of which this Experiment is a part
                if cfg.conditions[0] not in exp.studies:
                    exp.studies.append(cfg.conditions[0])
                # Curate a dataset for this Experiment from {source_name}
                if source_name in exp.source_paths:
                    exp_data_path = exp.source_paths[source_name]
                    exp_data_size = len(pd.read_csv(exp_data_path, index_col=0))
                else:
                    for attribute in ["treatment", "outcome"]:
                        exp.set_common_names(
                            attribute,
                            source_name,
                            cfg.sample_model,
                            source_dataset.get_common_name_prompts(),
                        )
                    exp_data_path, exp_data_size = source_dataset.experiment_data(
                        exp, study.conditions[0], cfg.filter_by_date, clean_path
                    )
                    # Track Experiment data and save to yaml
                    exp.source_paths[source_name] = exp_data_path
                    exp.to_yaml(exp_file)
                # Track Experiment data in study dataset
                study_dataset.data_paths.update(
                    {f"{source_name}_{nct_id}": exp_data_path}
                )
                study_dataset.data_sizes.update(
                    {f"{source_name}_{nct_id}": exp_data_size}
                )

    # Save paths to data for in study_dataset yaml
    study_dataset.to_yaml(
        os.path.join(
            cfg.data_path, "studies", cfg.conditions[0] + "_study_dataset.yaml"
        )
    )


if __name__ == "__main__":
    main()
