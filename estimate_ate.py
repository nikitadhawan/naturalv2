"""Estimate Average Treatment Effects (ATE) using the NATURAL pipeline."""

import asyncio
import logging
import os

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from create_study import Study
from naturalv2.estimators.natural_ipw import NaturalIPW
from naturalv2.estimators.natural_mc import NaturalMC
from naturalv2.estimators.natural_oi import NaturalOI
from naturalv2.evals.experiment import Experiment
from naturalv2.pipeline.natural import NATURALPipeline, PipelineContext, PipelineStage


load_dotenv(".env")

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


def weight_by_inclusion(ites: np.ndarray, inclusion_probs: pd.DataFrame) -> np.ndarray:
    """Weight ITEs by inclusion probabilities.

    Parameters
    ----------
    ites : np.ndarray
        Array of individual treatment effects (ITE) with shape
        ``[num_treatments, num_datapoints]``.
    inclusion_probs : pd.DataFrame
        DataFrame containing inclusion probabilities for each treatment.
        It should have a column named 'inclusion_probs' with stringified lists
        of probabilities for each treatment.

    Returns
    -------
    np.ndarray
        Weighted average treatment effects for each treatment, with shape
        ``[num_treatments]``.
    """
    # ites has shape [num_treatments, num_datapoints]
    probs = inclusion_probs.apply(
        lambda row: [float(prob) for prob in row["inclusion_probs"][1:-1].split()][1],
        axis=1,
    ).to_numpy()
    return np.average(ites, axis=1, weights=probs)


def calculate_treatment_effects(
    experiment: Experiment,
    outcome: str,
    estimator: NaturalIPW | NaturalMC | NaturalOI,
    extractions: pd.DataFrame,
) -> list[dict]:
    """Calculate treatment effects for all outcome-treatment pairs.

    Parameters
    ----------
    experiment : Experiment
        The experiment object containing treatment and outcome information.
    outcome : str
        The outcome for which to calculate treatment effects.
    estimator : NaturalIPW | NaturalMC | NaturalOI
        The estimator to use for calculating treatment effects.
    extractions : pd.DataFrame
        DataFrame containing the extractions from the pipeline.

    Returns
    -------
    list[dict]
        A list of dictionaries containing the predicted and true ATEs, along with
        absolute errors if available.
    """
    result_dicts = []

    if hasattr(estimator, "estimator_type"):
        all_ites = estimator.get_ites(extractions, outcome)
    else:
        all_ites = estimator.get_ites(extractions)
    weighted_effects = weight_by_inclusion(all_ites, extractions)  # len: num_treatments

    for i, treat1 in enumerate(experiment.treatment_names):
        for j, treat2 in enumerate(experiment.treatment_names):
            if i < j:
                pred_ate = weighted_effects[j] - weighted_effects[i]
                results = {
                    "estimator": estimator.__class__.__name__,
                    "outcome": outcome,
                    "treatments": f"{treat2}-{treat1}",
                    "pred_ate": pred_ate,
                }
                logger.info(f"Predicted ATE: {pred_ate}")
                if experiment.status == "completed":
                    effect_idx = experiment.outcome_treatment.index(
                        (outcome, (treat1, treat2))
                    )
                    true_ate = experiment.effect_sizes[effect_idx]
                    error = abs(pred_ate - true_ate)
                    results.update({"true_ate": true_ate, "abs_error": error})
                    logger.info(f"True ATE: {true_ate}")
                    logger.info(f"Absolute Error: {error}")
                result_dicts.append(results)

    return result_dicts


def _save_results(results: list[dict], save_path: str, nct_id: str) -> None:
    """Save results to CSV file."""
    result_df = pd.DataFrame(results)
    results_path = os.path.join(save_path, "results", f"{nct_id}/ate_results.csv")

    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path, index_col=0)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    result_df.to_csv(results_path)


def _get_nct_ids(cfg: DictConfig, study: Study) -> list[str]:
    """Get NCT IDs based on the split."""
    if cfg.split == "train":
        return [list(trial.keys())[0] for trial in study.train_trials]
    if cfg.split == "val":
        return [list(trial.keys())[0] for trial in study.val_trials]

    return [list(trial.keys())[0] for trial in study.test_trials]


def _process_trial(cfg: DictConfig, nct_id: str) -> None:
    """Process a single trial to estimate treatment effects."""
    if "NCT03828539" not in nct_id:
        return  # use only NCT03828539 for testing purposes; TODO: remove later

    # Load the experiment configuration
    try:
        exp_file = os.path.join(cfg.save_path, "experiments", f"{nct_id}.yaml")
        experiment = Experiment.from_yaml(exp_file)
    except (FileNotFoundError, ValueError) as e:
        logger.error(
            f"Experiment file for {nct_id} not found or invalid. Skipping trial. "
            f"Error: {e}",
            exc_info=True,
        )
        return
    os.makedirs(
        os.path.join(cfg.save_path, "results", f"{experiment.nct_id}"),
        exist_ok=True,
    )

    for outcome in experiment.outcome_names:
        for source_name in cfg.sources:
            pipeline_context = PipelineContext(
                experiment=experiment,
                source_name=source_name,
                estimator_type=cfg.estimator._target_.split(".")[-1],
                outcome=outcome,
                save_path=cfg.save_path,
            )

            pipeline_stages = []
            for stage_config in cfg.pipeline_stages:
                stage: PipelineStage = instantiate(stage_config)
                pipeline_stages.append(stage)

            pipeline = NATURALPipeline(pipeline_stages)

            try:
                logger.info(
                    f"Running pipeline for {nct_id} with source '{source_name}' "
                    f"and outcome '{outcome}'"
                )

                # Load curated data
                curated_df = pd.concat(
                    [
                        pd.read_csv(path, index_col=0)
                        for path in experiment.source_paths[source_name]
                    ],
                    ignore_index=True,
                )
                # TODO: remove subsampling after testing
                curated_df = curated_df.sample(
                    frac=0.05, random_state=cfg.seed, ignore_index=True
                )
                logger.info(
                    f"Initial number of curated reports: {len(curated_df)} reports."
                )

                # Run the pipeline
                extractions = asyncio.run(pipeline.run(curated_df, pipeline_context))
                if extractions.empty:
                    logger.warning(
                        f"No extractions found for {source_name} and outcome '{outcome}'. "
                        "Cannot calculate treatment effects."
                    )
                    continue

                # Calculate and save treatment effects
                estimator = instantiate(cfg.estimator, experiment=experiment)
                results = calculate_treatment_effects(
                    experiment, outcome, estimator, extractions
                )

                for result in results:
                    result.update(pipeline._data_flow)
                    result["source_name"] = source_name
                    result["initial_curated"] = len(curated_df)

                _save_results(results, cfg.save_path, experiment.nct_id)
            except Exception as e:
                logger.error(
                    f"Error processing {nct_id} with source '{source_name}' and outcome '{outcome}': {e}"
                )
                continue


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main function to estimate average treatment effects."""
    # Load study object from YAML file
    study_file = os.path.join(
        cfg.save_path,
        "studies",
        cfg.conditions[0].lower().replace(" ", "_") + "_study.yaml",
    )
    study = Study.from_yaml(study_file)

    if cfg.split not in ["train", "val", "test"]:
        raise ValueError(
            f"Invalid split '{cfg.split}'. Must be one of 'train', 'val', or 'test'."
        )

    # Get NCT IDs based on the split
    nct_ids = _get_nct_ids(cfg, study)
    logger.info(f"Processing {len(nct_ids)} trials for split '{cfg.split}'.")

    for nct_id in nct_ids:
        _process_trial(cfg, nct_id)


if __name__ == "__main__":
    main()
