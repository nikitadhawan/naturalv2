"""Pipeline for filtering and curating experiments using LLMs."""

import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Union

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from naturalv2.clinical_trial import ClinicalTrial
from naturalv2.experiment import Experiment
from naturalv2.models.lm import LM, build_lm_instance_from_cfg, extract_list_response
from naturalv2.prompts import get_common_name_prompts
from naturalv2.sources import PubMedSet, RedditSource
from naturalv2.study import Study, StudyDataset, get_study_filepaths
from naturalv2.utils import ListResponse


load_dotenv()

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


@dataclass
class LLMTask:
    """Represents a single LLM call task"""

    nct_id: str
    attribute: str  # "treatment" or "outcome"
    name: str  # specific treatment/outcome name
    messages: list[dict[str, str]]
    source_name: str
    task_id: str  # unique identifier for this task


@dataclass
class LLMResult:
    """Represents the result of an LLM call"""

    task_id: str
    nct_id: str
    attribute: str
    name: str
    common_names: list[str]
    success: bool
    error: Optional[str] = None


@dataclass
class ExperimentTask:
    """Represents data for processing a single experiment"""

    nct_id: str
    split: str
    source_name: str
    experiment_instance: Experiment


class _DataCurator:
    """Curates experiments.

    Prepares tasks, executes LLM calls, and processes results.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing paths and settings.
    study : Study
        The study object containing conditions and trials.
    source_dataset : Union[RedditSource, PubMedSet]
        The source dataset from which experiments are curated.
    study_dataset : StudyDataset
        The dataset for the study containing conditions and sources.
    clean_paths : list[str]
        List of paths to the cleaned data.
    language_model : LM
        Language model instance used for LLM calls.

    """

    def __init__(
        self,
        cfg: DictConfig,
        study: Study,
        source_dataset: Union[RedditSource, PubMedSet],
        study_dataset: StudyDataset,
        clean_paths: list[str],
        language_model: LM,
    ):
        self.cfg = cfg
        self.study = study
        self.source_dataset = source_dataset
        self.study_dataset = study_dataset
        self.clean_paths = clean_paths
        self.language_model = language_model

        # Simple in-memory tracking
        self._completed_experiments: set[str] = set()

        self._experiment_dir = os.path.join(cfg.save_path, "experiments")
        os.makedirs(self._experiment_dir, exist_ok=True)

    def prepare_experiment_tasks(
        self, nct_ids: list[str], splits: list[str], source_name: str
    ) -> list[ExperimentTask]:
        """Prepare experiment tasks.

        Parameters
        ----------
        nct_ids : list[str]
            List of NCT IDs for the experiments.
        splits : list[str]
            Corresponding splits for each NCT ID (e.g., "train", "val", "test").
        source_name : str
            Name of the source dataset (e.g., "pubmed", "reddit").

        Returns
        -------
        list[ExperimentTask]
            List of ``ExperimentTask`` objects for each NCT ID and split.
        """
        experiment_tasks: list[ExperimentTask] = []

        for nct_id, split in tqdm(
            zip(nct_ids, splits), desc="Preparing experiments", unit="exp"
        ):
            experiment_id = f"{source_name}_{nct_id}"

            # Skip if already processed or in study dataset
            if (
                experiment_id in self._completed_experiments
                or experiment_id in self.study_dataset.data_paths
            ):
                logger.debug(f"Skipping already processed experiment: {experiment_id}")
                continue

            # Load or create experiment
            exp_file = os.path.join(self._experiment_dir, f"{nct_id}.yaml")
            try:
                exp = Experiment.from_yaml(exp_file)
                logger.debug(f"Loaded existing experiment: {nct_id}")
            except (FileNotFoundError, ValueError):
                status: Literal["completed", "active"] = (
                    "active" if split == "test" else "completed"
                )
                exp = Experiment(self.cfg.data_path, nct_id, status=status)
                logger.debug(f"Created new experiment: {nct_id}")

            # Track the studies of which this Experiment is a part
            study_condition = self.cfg.conditions[0]
            if study_condition not in [study[0] for study in exp.studies]:
                exp.studies.append([study_condition, split])

            # Create experiment task
            experiment_tasks.append(
                ExperimentTask(
                    nct_id=nct_id,
                    split=split,
                    source_name=source_name,
                    experiment_instance=exp,
                )
            )

        return experiment_tasks

    async def get_common_names_with_llm(  # noqa: PLR0915
        self,
        experiment_tasks: list[ExperimentTask],
        semaphore_limit: int = 10,
    ) -> dict[str, LLMResult]:
        """Execute LLM calls to get common names for treatments and outcomes.

        Parameters
        ----------
        experiment_tasks : list[ExperimentTask]
            List of ``ExperimentTask`` objects to process.
        semaphore_limit : int, default=10
            Maximum number of concurrent LLM calls.

        Returns
        -------
        dict[str, LLMResult]
            Dictionary mapping task IDs to their results. This includes
            the common names extracted from the LLM responses, along with
            metadata about the task such as NCT ID, attribute (treatment/outcome),
            and success status.
        """
        task_queue: asyncio.Queue[LLMTask] = asyncio.Queue(maxsize=100)
        result_queue: asyncio.Queue[LLMResult] = asyncio.Queue()

        # Shared counter for progress tracking
        progress = {"total": 0, "completed": 0, "failed": 0}
        pbar = None

        async def producer():
            """Generate tasks and put them in the queue."""
            nonlocal pbar
            task_generator = self._generate_llm_tasks_for_experiments(experiment_tasks)

            for task in task_generator:
                await task_queue.put(task)
                progress["total"] += 1

                # Initialize progress bar on first task
                if pbar is None:
                    pbar = tqdm(
                        desc="Executing LLM tasks", unit="task", dynamic_ncols=True
                    )

        async def consumer() -> None:
            """Process tasks from the queue."""
            while True:
                task = await task_queue.get()
                if task is None:  # Producer finished
                    task_queue.task_done()
                    break

                try:
                    logger.debug(f"Executing LLM task: {task.task_id}")
                    lm_response = await self.language_model(
                        messages=task.messages, response_format=ListResponse
                    )
                    parsed_response = extract_list_response(lm_response)
                    common_names = parsed_response[0] if parsed_response else []

                    result = LLMResult(
                        task_id=task.task_id,
                        nct_id=task.nct_id,
                        attribute=task.attribute,
                        name=task.name,
                        common_names=common_names,
                        success=bool(common_names),  # Success if we got any names
                    )
                except Exception as e:
                    error_msg = f"LLM call failed for {task.nct_id}/{task.attribute}/{task.name}: {str(e)}"
                    logger.warning(error_msg)
                    result = LLMResult(
                        task_id=task.task_id,
                        nct_id=task.nct_id,
                        attribute=task.attribute,
                        name=task.name,
                        common_names=[],
                        success=False,
                        error=error_msg,
                    )
                finally:
                    await result_queue.put(result)
                    task_queue.task_done()

                    # Update progress
                    progress["completed"] += 1
                    if not result.success:
                        progress["failed"] += 1

                    if pbar is not None:
                        pbar.total = progress["total"]
                        success_count = progress["completed"] - progress["failed"]
                        pbar.set_postfix(
                            {
                                "success": success_count,
                                "failed": progress["failed"],
                                "rate": f"{success_count / max(1, progress['completed']) * 100:.1f}%",
                            }
                        )
                        pbar.update(1)

        # Start consumers
        consumer_tasks = [
            asyncio.create_task(consumer()) for _ in range(semaphore_limit)
        ]

        # Start producer
        producer_task = asyncio.create_task(producer())

        # Collect results
        all_results = {}
        failed_count = 0

        try:
            # Wait for producer to finish
            await producer_task

            # Wait for all tasks to be processed
            await task_queue.join()

            # Signal completion to all consumers
            for _ in range(semaphore_limit):
                await task_queue.put(None)  # type: ignore

            # Wait for consumers to finish
            await asyncio.gather(*consumer_tasks)

            # Collect all results
            while not result_queue.empty():
                result = await result_queue.get()
                all_results[result.task_id] = result
                if not result.success:
                    failed_count += 1

        finally:
            # Close progress bar
            if pbar is not None:
                pbar.close()

        success_count = len(all_results) - failed_count
        logger.info(
            f"Completed LLM tasks: {success_count} successful, {failed_count} failed "
            f"({success_count / max(1, len(all_results)) * 100:.1f}% success rate)"
        )

        return all_results

    def process_experiments(
        self, experiment_tasks: list[ExperimentTask], llm_results: dict[str, LLMResult]
    ) -> list[tuple[str, str, int]]:
        """Process experiments with LLM results.

        Parameters
        ----------
        experiment_tasks : list[ExperimentTask]
            List of ``ExperimentTask`` objects to process.
        llm_results : dict[str, LLMResult]
            Dictionary mapping task IDs to their results.

        Returns
        -------
        list[tuple[str, str, int]]
            List of tuples containing experiment ID, data path, and data size for
            each processed experiment.
        """
        if not experiment_tasks:
            logger.info("No experiments to process")
            return []

        # remove empty files in clean_paths
        clean_data_paths = [
            path
            for path in self.clean_paths
            if os.path.exists(path) and os.path.getsize(path) > 0
        ]
        grouped_llm_results = self._group_llm_results_by_experiment(llm_results)
        final_results_list = []
        failed_count = 0
        for experiment_task in tqdm(
            experiment_tasks,
            desc="Processing experiments",
            unit="exp",
            dynamic_ncols=True,
            position=0,
        ):
            common_name_dict = grouped_llm_results.get(experiment_task.nct_id, {})
            experiment_id = f"{experiment_task.source_name}_{experiment_task.nct_id}"

            try:
                # Apply LLM results to the experiment object
                if common_name_dict:
                    attribute = "treatment"
                    if attribute in common_name_dict:
                        common_names = common_name_dict[attribute]
                        getattr(
                            experiment_task.experiment_instance,
                            f"{attribute}_common_names",
                        ).update({experiment_task.source_name: common_names})

                # Save the modified experiment object to YAML
                exp_file = os.path.join(
                    self._experiment_dir, f"{experiment_task.nct_id}.yaml"
                )
                experiment_task.experiment_instance.to_yaml(exp_file)

                # Run experiment data curation using the source_dataset instance
                exp_data_path, exp_data_size = (
                    self.source_dataset.curate_experiment_data(
                        experiment_task.experiment_instance,
                        self.study.conditions[0],
                        self.cfg.filter_by_date,
                        clean_data_paths,
                    )
                )

                # Update experiment source paths on the copied experiment instance
                current_paths = experiment_task.experiment_instance.source_paths.get(
                    experiment_task.source_name, []
                )
                experiment_task.experiment_instance.source_paths[
                    experiment_task.source_name
                ] = current_paths + [exp_data_path]

                # Save the modified experiment object to YAML
                experiment_task.experiment_instance.to_yaml(exp_file)
                final_results_list.append((experiment_id, exp_data_path, exp_data_size))
            except Exception as e:
                failed_count += 1
                logger.error(
                    f"Error processing experiment {experiment_id}: {str(e)}",
                    exc_info=True,
                )

        success_rate = len(final_results_list) / max(1, len(experiment_tasks)) * 100
        logger.info(
            f"Experiment processing complete: {len(final_results_list)}/{len(experiment_tasks)} "
            f"successful ({success_rate:.1f}% success rate)"
        )
        if failed_count > 0:
            logger.warning(f"{failed_count} experiments failed to process.")

        return final_results_list

    def _generate_llm_tasks_for_experiments(
        self, experiment_tasks: list[ExperimentTask]
    ) -> Iterator[LLMTask]:
        """Generator that yields LLM tasks on demand to save memory"""
        for exp_task in experiment_tasks:
            exp = exp_task.experiment_instance
            nct_id = exp_task.nct_id
            source_name = exp_task.source_name

            attribute = "treatment"
            # Skip if already have common names for this source
            if source_name in getattr(exp, f"{attribute}_common_names"):
                logger.debug(
                    f"Skipping {attribute} for {nct_id} - already have common names"
                )
                continue

            attribute_names = getattr(exp, f"{attribute}_names")
            if not attribute_names:
                logger.debug(f"No {attribute} names found for {nct_id}")
                continue

            for i, name in enumerate(attribute_names):
                # Create unique task ID
                task_id = f"{nct_id}_{attribute}_{i}_{abs(hash(name)) % 10000}"

                str_substitutes = {"keyword": name, "trial_title": exp.title}
                if attribute == "treatment":
                    str_substitutes["treatment_desc"] = exp.treatment_desc[name]
                    str_substitutes["drugbank_names"] = exp.drugbank_names[name]

                # Prepare messages
                messages = get_common_name_prompts(
                    attribute, source_name, **str_substitutes
                )

                yield LLMTask(
                    nct_id=nct_id,
                    attribute=attribute,
                    name=name,
                    messages=messages,
                    source_name=source_name,
                    task_id=task_id,
                )

    def _group_llm_results_by_experiment(
        self, llm_results: dict[str, LLMResult]
    ) -> dict[str, dict[str, dict[str, list[str]]]]:
        grouped: defaultdict[str, defaultdict[str, dict[str, list[str]]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        for result in llm_results.values():
            if result.success and result.common_names:
                grouped[result.nct_id][result.attribute][result.name] = list(
                    set(result.common_names)
                )

        # Convert to regular dict
        final_grouped: dict[str, dict[str, dict[str, list[str]]]] = {}
        for nct_id, attributes in grouped.items():
            final_grouped[nct_id] = {}
            for attribute, name_map in attributes.items():
                final_grouped[nct_id][attribute] = dict(name_map)

        return final_grouped


async def _curate_experiments(
    cfg: DictConfig,
    study: Study,
    source_dataset: Union[RedditSource, PubMedSet],
    study_dataset: StudyDataset,
    clean_paths: list[str],
    language_model: LM,
    source_name: str,
    all_ncts: list[str],
    splits: list[str],
) -> None:
    """Main async function to curate experiments in parallel"""
    curator = _DataCurator(
        cfg, study, source_dataset, study_dataset, clean_paths, language_model
    )

    # Prepare experiment tasks (no LLM tasks yet)
    logger.info(
        f"Preparing experiment tasks for {source_name} with {len(all_ncts)} NCT IDs"
    )
    experiment_tasks = curator.prepare_experiment_tasks(
        nct_ids=all_ncts, splits=splits, source_name=source_name
    )

    if not experiment_tasks:
        logger.info(f"No experiments to process for {source_name}")
        return

    # Phase 1: Execute LLM calls
    llm_results = await curator.get_common_names_with_llm(
        experiment_tasks=experiment_tasks,
        semaphore_limit=cfg.get("curate_max_concurrency", 10),
    )

    # Phase 2: Process all experiments
    processing_results = curator.process_experiments(
        experiment_tasks=experiment_tasks, llm_results=llm_results
    )

    # Update study dataset with results
    for experiment_id, exp_data_path, exp_data_size in processing_results:
        study_dataset.data_paths[experiment_id] = exp_data_path
        study_dataset.data_sizes[experiment_id] = exp_data_size

    logger.info(
        f"Completed curation of {source_name}: {len(processing_results)} experiments "
        "successfully processed"
    )


# TODO: improve on relative path for config
@hydra.main(config_path="../../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Load study from yaml
    study_file = get_study_filepaths(cfg.save_path, cfg.conditions[0])["study"]
    study = Study.from_yaml(study_file)

    # Extract train, val, and test NCT IDs
    # train_ncts = [list(trial.keys())[0] for trial in study.train_trials]
    # val_ncts = [list(trial.keys())[0] for trial in study.val_trials]
    # test_ncts = [list(trial.keys())[0] for trial in study.test_trials]

    # TODO: use all trials after testing
    # splits = (
    #     ["train"] * len(train_ncts)
    #     + ["val"] * len(val_ncts)
    #     + ["test"] * len(test_ncts)
    # )
    # all_ncts = train_ncts + val_ncts + test_ncts
    all_ncts = ["NCT03828539"]
    splits = ["val"]

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

            # Use a trial's `conditions` to search for related data in the source
            all_condition_keywords: list[str] = []
            for nct_id, split in tqdm(
                zip(all_ncts, splits),
                desc=f"Collecting all condition keywords for {source_name}",
                leave=False,
            ):
                if split == "test":
                    trial_path = os.path.join(
                        cfg.data_path, f"nct_reports_test/{nct_id}.json"
                    )
                else:
                    trial_path = os.path.join(
                        cfg.data_path, f"nct_reports/{nct_id}.json"
                    )

                trial = ClinicalTrial.from_json_file(trial_path)

                conditions_mod = trial.protocolSection.conditionsModule
                if conditions_mod is not None:
                    condition_keywords = conditions_mod.conditions
                    all_condition_keywords.extend(condition_keywords)

            # Ensure unique keywords
            all_condition_keywords = list(set(all_condition_keywords))

            # Filter and download data related to condition keywords
            await source_dataset.condition_filter(
                all_condition_keywords, study_dataset, study_dataset_file
            )

            all_clean_paths = await source_dataset.clean_data()
            study_dataset.data_paths[f"{source_name}_cleaned"] = all_clean_paths
            study_dataset.to_yaml(study_dataset_file)

            # Use sample model for common names
            sample_model = build_lm_instance_from_cfg(cfg.sample_model)

            # Process experiments in batches with async LLM calls
            await _curate_experiments(
                cfg,
                study,
                source_dataset,
                study_dataset,
                all_clean_paths,
                sample_model,
                source_name,
                all_ncts,
                splits,
            )
            study_dataset.to_yaml(study_dataset_file)

    asyncio.run(process_all_sources())


if __name__ == "__main__":
    main()
