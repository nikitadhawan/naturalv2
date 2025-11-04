"""Pipeline for filtering data sources and curating trial-specific data."""

import asyncio
import logging
import os
from typing import Literal

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

import naturalv2.hydra_setup  # noqa: F401 # Ensure custom resolvers are registered
from naturalv2.experiment import Experiment
from naturalv2.sources.core import CurationContext, CurationStage, FilterCurateRunner
from naturalv2.study import Study, StudyDataset, get_study_filepaths
from naturalv2.utils import get_experiment_filepath


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


def _get_nct_ids(study: Study) -> tuple[list[str], list[str]]:
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
    return all_ncts, splits


def _build_pipeline(source_cfg: DictConfig) -> FilterCurateRunner:
    """Return `FilterCurateRunner` with curation stages defined in `source_cfg`."""
    stages: list[CurationStage] = []
    for name, stage_cfg in source_cfg.stages.items():
        if stage_cfg is not None:
            stages.append(
                hydra.utils.instantiate(stage_cfg, name=name, _convert_="partial")
            )
    return FilterCurateRunner(stages)


# TODO: improve on relative path for config
@hydra.main(
    config_path="../../conf/", config_name="filter_curate.yaml", version_base="1.2"
)
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    """Main function to run the filter and curate pipeline."""
    asyncio.run(main=_async_main(cfg))


async def _async_main(cfg: DictConfig) -> None:
    """Asynchronous main function to run the filter and curate pipeline."""
    if is_weave_available:
        import weave  # type: ignore # noqa: PLC0415

        weave.init(project_name="naturalv2")

    study_filepaths: dict[str, str] = get_study_filepaths(
        base_dir=cfg.save_path, condition=cfg.conditions[0]
    )

    # Load study from yaml
    study: Study = Study.from_yaml(filename=study_filepaths["study"])

    # Create study dataset
    study_dataset_file: str = study_filepaths["study_dataset"]
    if os.path.exists(path=study_dataset_file):
        study_dataset: StudyDataset = StudyDataset.from_yaml(
            filename=study_dataset_file
        )
    else:
        study_dataset = StudyDataset(study.conditions, cfg.sources)

    if cfg.nct_id:
        all_ncts: list[str] = [cfg.nct_id]
        splits: list[str] = [cfg.split]
        logger.info(msg=f"Curating data for trial {cfg.nct_id}.")
    else:
        all_ncts, splits = _get_nct_ids(study)

    # Collect experiments for curation
    experiment_list: list[Experiment] = []
    for nct_id, split in zip(all_ncts, splits):
        experiment_filepath = get_experiment_filepath(cfg.save_path, nct_id)
        try:
            # Load existing experiment if available
            experiment: Experiment = Experiment.from_yaml(filename=experiment_filepath)
        except (FileNotFoundError, ValueError):
            # Otherwise create a new experiment instance
            status: Literal["completed", "active"] = (
                "active" if split == "test" else "completed"
            )
            experiment = Experiment(cfg.data_path, nct_id, status=status)
        experiment_list.append(experiment)

    for source_name, source_cfg in cfg.sources.items():
        logger.info("Running pipeline for source %s", source_name)
        context = CurationContext(
            source_name=source_name,
            condition=cfg.conditions[0],
            experiments=experiment_list,
            splits=splits,
            save_dir=cfg.save_path,
            filter_by_date=cfg.filter_by_date,
            study_dataset=study_dataset,
            experiment_name=cfg.experiment_name,
            extras={"study_dataset_path": study_filepaths["study_dataset"]},
        )

        pipeline: FilterCurateRunner = _build_pipeline(source_cfg)
        _ = await pipeline.run(context)

        context.study_dataset.to_yaml(filename=study_filepaths["study_dataset"])


if __name__ == "__main__":
    main()
