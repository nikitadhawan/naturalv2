"""Pipeline for filtering and curating experiments using LLMs."""

import asyncio
import json
import logging
import os
from typing import Literal, Union

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from naturalv2.experiment import Experiment
from naturalv2.pipeline import CurationContext, CurationStage
from naturalv2.sources import PubMedSet, RedditSource
from naturalv2.study import Study, StudyDataset, get_study_filepaths


load_dotenv()
is_weave_available = os.getenv("USE_WEAVE", "false").lower() == "true"

LOGGING_CONFIG = {
    "version": 1,
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        }
    },
    "formatters": {
        "http": {
            "format": "%(levelname)s [%(asctime)s] %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "loggers": {
        "httpx": {
            "handlers": ["default"],
            "level": "WARNING",
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def _get_nct_ids(study: Study) -> list[str]:
    """Get NCT IDs based on the split."""
    train_ncts = [list(trial.keys())[0] for trial in study.train_trials]
    val_ncts = [list(trial.keys())[0] for trial in study.val_trials]
    test_ncts = [list(trial.keys())[0] for trial in study.test_trials]
    all_ncts = train_ncts + val_ncts + test_ncts
    splits = (
        ["train"] * len(train_ncts)
        + ["val"] * len(val_ncts)
        + ["test"] * len(test_ncts)
    )
    # TODO: remove after testing
    all_ncts = ["NCT03828539"]
    splits = ["val"]
    return all_ncts, splits


def _get_curated_dataset(exp_list, context, source_name, clean_data_paths):
    all_exp_data_paths, all_exp_data_sizes = {}, {}
    for exp in exp_list:
        exp_data_path, exp_data_size = context.source_dataset.curate_experiment_data(
            exp, context.condition, context.filter_by_date, clean_data_paths
        )
        exp.source_paths[source_name].extend(exp_data_path)

        all_exp_data_paths[exp.nct_id] = exp_data_path
        all_exp_data_sizes[exp.nct_id] = exp_data_size
    return all_exp_data_paths, all_exp_data_sizes


# TODO: improve on relative path for config
@hydra.main(config_path="../../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    if is_weave_available:
        import weave  # type: ignore # noqa: PLC0415

        weave.init("naturalv2")

    # Load study from yaml
    study_file = get_study_filepaths(cfg.save_path, cfg.conditions[0])["study"]
    study = Study.from_yaml(study_file)

    all_ncts, splits = _get_nct_ids(study)

    # Create study dataset
    study_dataset_file = get_study_filepaths(cfg.save_path, cfg.conditions[0])[
        "study_dataset"
    ]
    if os.path.exists(study_dataset_file):
        study_dataset = StudyDataset.from_yaml(study_dataset_file)
    else:
        study_dataset = StudyDataset(study.conditions, cfg.sources)

    async def process_all_sources() -> None:
        for source_name in cfg.sources:
            source_dataset: Union[RedditSource, PubMedSet] = instantiate(
                cfg[source_name]
            )

            curation_context = CurationContext(
                condition=cfg.conditions[0],
                all_ncts=all_ncts,
                splits=splits,
                source_dataset=source_dataset,
                filter_by_date=cfg.filter_by_date,
                save_path=cfg.save_path,
                exp_name=cfg.experiment_name,
            )

            # Collect experiments for curation
            exp_list = []
            for nct_id, split in zip(all_ncts, splits):
                # Load or create experiment
                exp_file = os.path.join(cfg.save_path, "experiments", f"{nct_id}.yaml")
                try:
                    exp = Experiment.from_yaml(exp_file)
                except (FileNotFoundError, ValueError):
                    status: Literal["completed", "active"] = (
                        "active" if split == "test" else "completed"
                    )
                    exp = Experiment(cfg.data_path, nct_id, status=status)
                exp_list.append(exp)

            condition_stage: CurationStage = instantiate(
                cfg.condition_config, source_name=source_name
            )
            treat_synonym_stage: CurationStage = instantiate(
                cfg.synonym_config, source_name=source_name, attribute="treatment"
            )

            # Get condition related queries to download data from ``source_name``.
            condition_metadata = await condition_stage.process(
                exp_list, curation_context
            )
            study_dataset.sources[source_name] = condition_metadata
            study_dataset.to_yaml(study_dataset_file)

            logger.info(f"Stage {condition_stage.stage_name} completed successfully.")
            logger.info(f"Stats:\n{json.dumps(condition_stage.get_stats(), indent=2)}")
            for key, value in condition_stage.prompt_template().items():
                logger.info(f"{key}\n{str(value)}")
            condition_stage.render_stats_table()

            # Clean and download data.
            all_clean_paths = await source_dataset.clean_data(condition_metadata)
            study_dataset.data_paths[f"{source_name}_cleaned"] = all_clean_paths
            study_dataset.to_yaml(study_dataset_file)
            logger.info(f"Data cleaning for {source_name} completed successfully.")

            # Get treatment synonyms and curate data based on string-matching.
            exp_list = await treat_synonym_stage.process(exp_list, curation_context)

            logger.info(
                f"Stage {treat_synonym_stage.stage_name} completed successfully."
            )
            logger.info(
                f"Stats:\n{json.dumps(treat_synonym_stage.get_stats(), indent=2)}"
            )
            for key, value in treat_synonym_stage.prompt_template().items():
                logger.info(f"{key}\n{str(value)}")
            treat_synonym_stage.render_stats_table()

            # Get curated dataset and its size for each experiment.
            all_exp_data_paths, all_exp_data_sizes = _get_curated_dataset(
                exp_list, curation_context, source_name, all_clean_paths
            )
            logger.info(f"Curation for {source_name} completed successfully.")
            logger.info(f"Data sizes:\n{json.dumps(all_exp_data_sizes, indent=2)}")

            study_dataset.data_paths.update(all_exp_data_paths)
            study_dataset.data_sizes.update(all_exp_data_sizes)
            study_dataset.to_yaml(study_dataset_file)

    asyncio.run(process_all_sources())


if __name__ == "__main__":
    main()
