"""Pipeline for filtering and curating experiments using LLMs."""

import asyncio
import copy
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from string import Template
from typing import Iterator, Optional, Union

import hydra
import psutil
import yaml
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from create_study import Study
from naturalv2.evals.experiment import Experiment
from naturalv2.models.lm import LM, build_lm_instance_from_cfg, extract_list_response
from naturalv2.sources.pubmed import PubMedSet
from naturalv2.sources.reddit import RedditSource
from naturalv2.utils import ListResponse, load_prompt


load_dotenv(".env")

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
        self.conditions = list(conditions)
        self.sources = list(sources)
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
    clean_path : str
        Path to the cleaned data directory.
    language_model : LM
        Language model instance used for LLM calls.

    """

    def __init__(
        self,
        cfg: DictConfig,
        study: Study,
        source_dataset: Union[RedditSource, PubMedSet],
        study_dataset: StudyDataset,
        clean_path: str,
        language_model: LM,
    ):
        self.cfg = cfg
        self.study = study
        self.source_dataset = source_dataset
        self.study_dataset = study_dataset
        self.clean_path = clean_path
        self.language_model = language_model

        # Simple in-memory tracking
        self._completed_experiments = set()

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
                logger.info(f"Skipping already processed experiment: {experiment_id}")
                continue

            # Load or create experiment
            exp_file = os.path.join(self._experiment_dir, f"{nct_id}.yaml")
            try:
                exp = Experiment.from_yaml(exp_file)
                logger.debug(f"Loaded existing experiment: {nct_id}")
            except FileNotFoundError:
                status = "active" if split == "test" else "completed"
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

    def _generate_llm_tasks_for_experiments(
        self, experiment_tasks: list[ExperimentTask], source_name: str
    ) -> Iterator[LLMTask]:
        """Generator that yields LLM tasks on demand to save memory"""
        prompt_dct = self.source_dataset.get_common_name_prompts()

        for exp_task in experiment_tasks:
            exp = exp_task.experiment_instance
            nct_id = exp_task.nct_id

            for attribute in ["treatment", "outcome"]:
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

                    str_substitutes = {
                        "keyword": name,
                        "trial_title": exp.title,
                    }
                    if attribute == "treatment":
                        str_substitutes["treatment_desc"] = exp.treatment_desc[name]
                    elif attribute == "outcome":
                        str_substitutes["outcome_desc"] = exp.outcome_desc[name]

                    # Prepare messages
                    messages = copy.deepcopy(prompt_dct[attribute])
                    messages[0]["content"] = Template(
                        messages[0]["content"]
                    ).safe_substitute(str_substitutes)

                    yield LLMTask(
                        nct_id=nct_id,
                        attribute=attribute,
                        name=name,
                        messages=messages,
                        source_name=source_name,
                        task_id=task_id,
                    )

    async def execute_llm_tasks_in_batches(
        self,
        experiment_tasks: list[ExperimentTask],
        source_name: str,
        batch_size: int = 100,
        semaphore_limit: Optional[int] = 50,
    ) -> dict[str, LLMResult]:
        """Execute LLM tasks in batches to manage memory usage.

        Parameters
        ----------
        experiment_tasks : list[ExperimentTask]
            List of ``ExperimentTask`` objects to process.
        source_name : str
            Name of the source dataset (e.g., "pubmed", "reddit").
        batch_size : int
            Number of tasks to process in each batch.
        semaphore_limit : Optional[int]
            Maximum number of concurrent LLM calls. If None, no limit is applied.

        Returns
        -------
        dict[str, LLMResult]
            Dictionary mapping task IDs to their results.
        """

        # First, collect all tasks to get total count
        all_tasks = list(
            self._generate_llm_tasks_for_experiments(experiment_tasks, source_name)
        )

        if not all_tasks:
            logger.info("No LLM tasks to execute for the given experiments.")
            return {}

        total_tasks = len(all_tasks)
        logger.info(f"Processing {total_tasks} LLM tasks in batches of {batch_size}")

        semaphore = (
            asyncio.Semaphore(semaphore_limit) if semaphore_limit else nullcontext()
        )
        all_results = {}
        failed_count = 0

        async def execute_single_task(task: LLMTask) -> LLMResult:
            async with semaphore:
                try:
                    logger.debug(f"Executing LLM task: {task.task_id}")
                    lm_response = await self.language_model(
                        messages=task.messages, response_format=ListResponse
                    )
                    parsed_response = extract_list_response(lm_response)
                    common_names = parsed_response[0] if parsed_response else []

                    return LLMResult(
                        task_id=task.task_id,
                        nct_id=task.nct_id,
                        attribute=task.attribute,
                        name=task.name,
                        common_names=common_names,
                        success=True,
                    )
                except Exception as e:
                    error_msg = f"LLM call failed for {task.nct_id}/{task.attribute}/{task.name}: {str(e)}"
                    logger.warning(error_msg)
                    return LLMResult(
                        task_id=task.task_id,
                        nct_id=task.nct_id,
                        attribute=task.attribute,
                        name=task.name,
                        common_names=[],
                        success=False,
                        error=str(e),
                    )

        # Process in batches
        with tqdm(total=total_tasks, desc="Executing LLM tasks", unit="task") as pbar:
            for i in range(0, total_tasks, batch_size):
                batch_tasks = all_tasks[i : i + batch_size]
                batch_number = (i // batch_size) + 1
                total_batches = (total_tasks + batch_size - 1) // batch_size

                # Execute current batch
                batch_coroutines = [execute_single_task(task) for task in batch_tasks]

                for coro in asyncio.as_completed(batch_coroutines):
                    try:
                        result = await coro
                        all_results[result.task_id] = result

                        if not result.success:
                            failed_count += 1

                        # Update progress bar
                        success_count = len(all_results) - failed_count
                        pbar.set_postfix(
                            {
                                "batch": f"{batch_number}/{total_batches}",
                                "success": success_count,
                                "failed": failed_count,
                                "rate": f"{success_count / max(1, len(all_results)) * 100:.1f}%",
                            }
                        )
                    except Exception as e:
                        logger.error(f"Unexpected error in LLM task: {e}")
                        failed_count += 1
                    finally:
                        pbar.update(1)

                # Clear batch tasks to free memory
                del batch_tasks
                del batch_coroutines

        success_count = len(all_results) - failed_count
        logger.info(
            f"Completed LLM tasks: {success_count} successful, {failed_count} failed "
            f"({success_count / max(1, len(all_results)) * 100:.1f}% success rate)"
        )

        return all_results

    def _group_llm_results_by_experiment(
        self, llm_results: dict[str, LLMResult]
    ) -> dict[str, dict[str, list[str]]]:
        """Group LLM results by experiment and attribute"""
        grouped: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for result in llm_results.values():
            if result.success and result.common_names:
                grouped[result.nct_id][result.attribute].extend(result.common_names)

        # Remove duplicates and convert to regular dict
        final_grouped: dict[str, dict[str, list[str]]] = {}
        for nct_id, attributes in grouped.items():
            final_grouped[nct_id] = {}
            for attribute, names in attributes.items():
                final_grouped[nct_id][attribute] = list(set(names))

        return final_grouped

    @staticmethod
    def _mp_worker_process_single_experiment(args_bundle):
        """
        Worker function for processing a single experiment.
        Designed to be run in a separate process.
        """
        # Unpack arguments
        (
            exp_task,
            exp_llm_data,
            source_dataset,
            study_condition_0,
            cfg_filter_by_date,
            clean_path,
            experiment_dir,
        ) = args_bundle

        # Type annotations
        exp_task: ExperimentTask
        exp_llm_data: dict[str, list[str]]
        source_dataset: Union[RedditSource, PubMedSet]
        study_condition_0: str
        cfg_filter_by_date: bool
        clean_path: str
        experiment_dir: str

        experiment_id = f"{exp_task.source_name}_{exp_task.nct_id}"

        try:
            # Apply LLM results to the experiment object
            if exp_llm_data:
                for attribute in ["treatment", "outcome"]:
                    if attribute in exp_llm_data:
                        common_names = exp_llm_data[attribute]
                        getattr(
                            exp_task.experiment_instance, f"{attribute}_common_names"
                        ).update({exp_task.source_name: common_names})

            # Run experiment data curation using the source_dataset instance
            exp_data_path, exp_data_size = source_dataset.curate_experiment_data(
                exp_task.experiment_instance,
                study_condition_0,
                cfg_filter_by_date,
                clean_path,
            )

            # Update experiment source paths on the copied experiment instance
            current_paths = exp_task.experiment_instance.source_paths.get(
                exp_task.source_name, []
            )
            exp_task.experiment_instance.source_paths[exp_task.source_name] = (
                current_paths + [exp_data_path]
            )

            # Save the modified experiment object to YAML
            exp_file = os.path.join(experiment_dir, f"{exp_task.nct_id}.yaml")
            exp_task.experiment_instance.to_yaml(exp_file)

            return {
                "status": "success",
                "experiment_id": experiment_id,
                "exp_data_path": exp_data_path,
                "exp_data_size": exp_data_size,
                "nct_id": exp_task.nct_id,
            }

        except Exception as e:
            # Return error information for logging in the main process
            return {
                "status": "error",
                "experiment_id": experiment_id,
                "nct_id": exp_task.nct_id,
                "error_message": f"Error in worker for {experiment_id} (NCT: {exp_task.nct_id}): {str(e)}",
            }

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

        grouped_llm_results = self._group_llm_results_by_experiment(llm_results)

        tasks_args_bundles = []
        for exp_task in experiment_tasks:
            exp_llm_data_for_task = grouped_llm_results.get(exp_task.nct_id, {})

            # Bundle arguments for the worker function
            args_bundle = (
                exp_task,
                exp_llm_data_for_task,
                self.source_dataset,
                self.study.conditions[0],
                self.cfg.filter_by_date,
                self.clean_path,
                self._experiment_dir,
            )
            tasks_args_bundles.append(args_bundle)

        final_results_list = []
        processed_count = 0
        failed_count = 0

        max_workers = self.cfg.get("max_workers_curate") or max(
            1, (psutil.cpu_count(logical=False) or 1) // 4
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_worker_outputs = executor.map(
                _DataCurator._mp_worker_process_single_experiment, tasks_args_bundles
            )

            for worker_output in tqdm(
                future_worker_outputs,
                total=len(tasks_args_bundles),
                desc=f"Processing experiments [{max_workers} workers]",
                unit="exp",
            ):
                if worker_output["status"] == "success":
                    final_results_list.append(
                        (
                            worker_output["experiment_id"],
                            worker_output["exp_data_path"],
                            worker_output["exp_data_size"],
                        )
                    )
                    # Update completed_experiments in the main process
                    self._completed_experiments.add(worker_output["experiment_id"])
                    processed_count += 1
                    logger.debug(
                        "Successfully processed experiment: "
                        f"{worker_output['experiment_id']} (size: "
                        f"{worker_output['exp_data_size']})"
                    )
                else:  # status == "error"
                    failed_count += 1
                    logger.error(worker_output["error_message"])

        success_rate = processed_count / max(1, len(experiment_tasks)) * 100
        logger.info(
            f"Experiment processing complete: {processed_count}/{len(experiment_tasks)} "
            f"successful ({success_rate:.1f}% success rate)"
        )
        if failed_count > 0:
            logger.warning(f"{failed_count} experiments failed to process.")

        return final_results_list


def _get_keywords_from_llm(
    study: Study, source_name: str, lm_cfg: DictConfig
) -> list[str]:
    """Extract keywords for a condition using an LLM.

    Parameters
    ----------
    study : Study
        The study object containing conditions.
    source_name : str
        Name of the source dataset (e.g., "pubmed", "reddit").
    lm_cfg : DictConfig
        Configuration for the language model to use.

    Returns
    -------
    list[str]
        List of keywords extracted by the LLM for the specified condition.

    Raises
    ------
    ValueError
        If the LLM does not return any keywords.
    """
    lm = build_lm_instance_from_cfg(lm_cfg)
    condition = study.conditions[0]

    messages: list[dict[str, str]] = load_prompt(
        base_dir="naturalv2/prompts",
        prompt_type="conditions_keywords_reddit"
        if source_name.lower() == "reddit"
        else "conditions_keywords_pubmed",
        return_format="messages",
        condition=condition,
    )

    response = lm.call_sync(messages=messages, response_format=ListResponse)
    keywords = extract_list_response(response)
    if keywords:
        final_keywords = keywords[0]
        logger.info(
            f"LLM extracted the following keywords for {condition}: {final_keywords}"
        )
        return final_keywords

    raise ValueError(
        "The LLM did not return any keywords. Please check the model and the prompt."
    )


async def _curate_experiments(
    cfg: DictConfig,
    study: Study,
    source_dataset: Union[RedditSource, PubMedSet],
    study_dataset: StudyDataset,
    clean_path: str,
    language_model: LM,
    source_name: str,
    train_ncts: list[str],
    val_ncts: list[str],
    test_ncts: list[str],
) -> None:
    """Main async function to curate experiments in parallel"""
    curator = _DataCurator(
        cfg, study, source_dataset, study_dataset, clean_path, language_model
    )

    # prepare NCT IDs and splits
    splits = (
        ["train"] * len(train_ncts)
        + ["val"] * len(val_ncts)
        + ["test"] * len(test_ncts)
    )
    all_ncts = train_ncts + val_ncts + test_ncts

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

    # Phase 1: Execute LLM calls in batches
    batch_size = cfg.get("curate_batch_size", 200)
    semaphore_limit = cfg.get("curate_max_concurrency", 100)

    llm_results = await curator.execute_llm_tasks_in_batches(
        experiment_tasks=experiment_tasks,
        source_name=source_name,
        batch_size=batch_size,
        semaphore_limit=semaphore_limit,
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


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # load study from yaml
    study_file = os.path.join(
        cfg.save_path,
        "studies",
        cfg.conditions[0].lower().replace(" ", "_") + "_study.yaml",
    )
    study = Study.from_yaml(study_file)

    # extract train, val, and test NCT IDs
    train_ncts = [list(trial.keys())[0] for trial in study.train_trials]
    val_ncts = [list(trial.keys())[0] for trial in study.val_trials]
    test_ncts = [list(trial.keys())[0] for trial in study.test_trials]

    # create study dataset
    study_dataset_file = os.path.join(
        cfg.data_path,
        "studies",
        cfg.conditions[0].lower().replace(" ", "_") + "_study_dataset.yaml",
    )
    if os.path.exists(study_dataset_file):
        study_dataset = StudyDataset.from_yaml(study_dataset_file)
    else:
        study_dataset = StudyDataset(study.conditions, cfg.sources)

    # initialize sample model
    sample_model = build_lm_instance_from_cfg(cfg.sample_model)

    async def process_all_sources():
        for source_name in cfg.sources:
            source_dataset: Union[RedditSource, PubMedSet] = instantiate(
                cfg[source_name]
            )

            if f"{source_name}_condition_filtered" not in study_dataset.data_paths:
                keywords = _get_keywords_from_llm(
                    study, source_name, cfg[source_name].lm_cfg
                )
                condition_filter_paths = source_dataset.condition_filter(keywords)
                study_dataset.data_paths.update(
                    {f"{source_name}_condition_filtered": condition_filter_paths}
                )

            if f"{source_name}_cleaned" not in study_dataset.data_paths:
                clean_path, data_size = source_dataset.clean_data(study.conditions[0])
                study_dataset.data_paths.update({f"{source_name}_cleaned": clean_path})
                study_dataset.data_sizes.update({f"{source_name}_cleaned": data_size})

            study_dataset.to_yaml(study_dataset_file)
            clean_path = study_dataset.data_paths[f"{source_name}_cleaned"]

            # Process experiments in batches with async LLM calls
            await _curate_experiments(
                cfg,
                study,
                source_dataset,
                study_dataset,
                clean_path,
                sample_model,
                source_name,
                train_ncts,
                val_ncts,
                test_ncts,
            )
            study_dataset.to_yaml(study_dataset_file)

    asyncio.run(process_all_sources())


if __name__ == "__main__":
    main()
