"""Estimate Average Treatment Effects (ATE) using the NATURAL pipeline."""

import asyncio
import logging
import os
from ast import literal_eval

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.stats import norm

import naturalv2.hydra_setup  # noqa: F401 # Ensure custom resolvers are registered
from naturalv2.estimators import NaturalIPW, NaturalMC, NaturalOI
from naturalv2.experiment import Experiment
from naturalv2.pipeline import NATURALPipeline, PipelineContext, PipelineStage
from naturalv2.study import Study, get_study_filepaths
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


def _weight_by_inclusion(
    ites: np.ndarray, inclusion_probs: pd.DataFrame, use_weights=True
) -> np.ndarray:
    """Weight individual treatment effects (ITE) by inclusion probabilities.

    Parameters
    ----------
    ites : np.ndarray
        Array of individual treatment effects (ITE) with shape
        ``[num_treatments, num_datapoints]``.
    inclusion_probs : pd.DataFrame
        DataFrame containing inclusion probabilities for each datapoint.
        It should have a column named 'inclusion_probs' with stringified lists
        of probabilities for each datapoint.

    Returns
    -------
    np.ndarray
        Weighted average treatment effects for each treatment, with shape
        ``[num_treatments]``.
    """
    # ites has shape [num_treatments, num_datapoints]
    if not use_weights:
        probs = np.ones(ites.shape[1])
    else:
        probs = inclusion_probs.apply(
            lambda row: literal_eval(row["inclusion_probs"])[1], axis=1
        ).to_numpy()
    return np.average(ites, axis=1, weights=probs)


def _calculate_treatment_effects(
    experiment: Experiment,
    outcome: str,
    estimator: NaturalIPW | NaturalMC | NaturalOI,
    extractions: pd.DataFrame,
    bootstrap_size: int,
    seed: int,
    alpha: float = 0.05,
    use_inclusion_weights: bool = True,
    use_imputed_nones: bool = True,
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
    if not use_imputed_nones:
        # Filter out rows with ``None`` in any covariate columns
        extractions = extractions.dropna(subset=experiment.covariate_names)

    result_dicts = []

    if isinstance(estimator, NaturalMC):
        all_ites = estimator.get_individual_treatment_effects(extractions, outcome)
    else:
        all_ites = estimator.get_individual_treatment_effects(extractions)

    weighted_effects = _weight_by_inclusion(
        all_ites, extractions, use_inclusion_weights
    )  # len: num_treatments

    bootstrap_weighted_effects = np.zeros(
        (bootstrap_size, len(experiment.treatment_names))
    )
    for b in range(bootstrap_size):
        bootstrap_data = extractions.copy().sample(
            n=len(extractions),
            replace=True,
            random_state=seed + b,
            axis="index",
        )

        if isinstance(estimator, NaturalMC):
            all_ites = estimator.get_individual_treatment_effects(
                bootstrap_data, outcome
            )
        else:
            all_ites = estimator.get_individual_treatment_effects(bootstrap_data)

        bootstrap_weighted_effects[b, :] = _weight_by_inclusion(
            all_ites, bootstrap_data, use_inclusion_weights
        )  # len: num_treatments

    for i, treat1 in enumerate(experiment.treatment_names):
        for j, treat2 in enumerate(experiment.treatment_names):
            if i < j:
                try:
                    pred_ate = weighted_effects[j] - weighted_effects[i]
                    bootstrap_pred_ate = (
                        bootstrap_weighted_effects[:, j]
                        - bootstrap_weighted_effects[:, i]
                    )  # len: bootstrap_size
                    avg_bootstrap_pred_ate = np.mean(bootstrap_pred_ate)
                    sample_variance = np.sum(
                        (bootstrap_pred_ate - avg_bootstrap_pred_ate) ** 2
                    ) / (bootstrap_size - 1)
                    conf_delta = norm.ppf(1 - alpha / 2) * np.sqrt(sample_variance)
                    results = {
                        "estimator": estimator.__class__.__name__,
                        "outcome": outcome,
                        "treatments": f"{treat2}-{treat1}",
                        "pred_ate": pred_ate,
                        "CI_lower": pred_ate - conf_delta,
                        "CI_upper": pred_ate + conf_delta,
                    }
                    logger.info(
                        f"Predicted ATE: {pred_ate}, ({pred_ate - conf_delta}, {pred_ate + conf_delta})"
                    )
                    if experiment.status == "completed":
                        effect_idx = experiment.outcome_treatment.index(
                            [outcome, [treat1, treat2]]
                        )
                        true_ate = experiment.effect_sizes[effect_idx]
                        error = abs(pred_ate - true_ate)
                        results.update({"true_ate": true_ate, "abs_error": error})
                        logger.info(f"True ATE: {true_ate}")
                        logger.info(f"Absolute Error: {error}")
                    result_dicts.append(results)

                except:
                    logger.info(
                        f"ATE prediction errored for {treat1}, {treat2}, {outcome}."
                    )
                    continue

    return result_dicts


def _save_results(
    results: list[dict], save_path: str, nct_id: str, exp_name: str
) -> None:
    """Save results to CSV file."""
    result_df = pd.DataFrame(results)
    results_path = os.path.join(
        save_path, "results", f"{nct_id}_{exp_name}/ate_results.csv"
    )

    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path, index_col=0)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    result_df.to_csv(results_path)


def _get_nct_ids(split: str, study: Study) -> list[str]:
    """Get NCT IDs based on the split."""
    if split == "train":
        return [list(trial.keys())[0] for trial in study.train_trials]
    if split == "val":
        return [list(trial.keys())[0] for trial in study.val_trials]

    return [list(trial.keys())[0] for trial in study.test_trials]


async def _process_trial(cfg: DictConfig, nct_id: str) -> None:
    """Process a single trial to estimate treatment effects."""

    # Load the experiment configuration
    try:
        exp_file = get_experiment_filepath(cfg.save_path, nct_id)
        experiment = Experiment.from_yaml(exp_file)
    except (FileNotFoundError, ValueError) as e:
        logger.error(
            f"Experiment file for {nct_id} not found or invalid. Skipping trial. "
            f"Error: {e}",
            exc_info=True,
        )
        return
    os.makedirs(
        os.path.join(
            cfg.save_path, "results", f"{experiment.nct_id}_{cfg.experiment_name}"
        ),
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
                exp_name=cfg.experiment_name,
            )

            pipeline_stages: list[PipelineStage] = []
            for name, stage_config in cfg.pipeline.stages.items():
                pipeline_stages.append(
                    instantiate(stage_config, name=name, _recursive_=False)
                )

            pipeline = NATURALPipeline(pipeline_stages)

            try:
                logger.info(
                    f"Running pipeline for {nct_id} with source '{source_name}' "
                    f"and outcome '{outcome}'"
                )

                # Load curated data
                curated_df = pd.read_csv(experiment.source_paths[source_name])
                # TODO: remove subsampling after testing
                # curated_df = curated_df.sample(
                #     frac=0.05, random_state=cfg.seed, ignore_index=True
                # )
                logger.info(
                    f"Initial number of curated reports: {len(curated_df)} reports."
                )

                # Run the pipeline
                extractions = await pipeline.run(curated_df, pipeline_context)
                if extractions.empty:
                    logger.warning(
                        f"No extractions found for {source_name} and outcome '{outcome}'. "
                        "Cannot calculate treatment effects."
                    )
                    result = {}
                    result.update(pipeline._data_flow)
                    result["source_name"] = source_name
                    result["initial_curated"] = len(curated_df)
                    _save_results(
                        [result], cfg.save_path, experiment.nct_id, cfg.experiment_name
                    )
                    continue

                # Calculate and save treatment effects
                estimator = instantiate(cfg.estimator, experiment=experiment)
                results = _calculate_treatment_effects(
                    experiment,
                    outcome,
                    estimator,
                    extractions,
                    cfg.get("bootstrap_size", 10),
                    cfg.get("seed", 0),
                    cfg.get("alpha", 0.05),
                    cfg.get("use_inclusion_weights", True),
                    cfg.get("use_imputed_nones", True),
                )

                for result in results:
                    result.update(pipeline._data_flow)
                    result["source_name"] = source_name
                    result["initial_curated"] = len(curated_df)

                _save_results(
                    results, cfg.save_path, experiment.nct_id, cfg.experiment_name
                )
            except Exception as e:
                logger.error(
                    f"Error processing {nct_id} with source '{source_name}' and outcome '{outcome}': {e}",
                    exc_info=True,
                )
                continue


async def _process_all_trials(cfg: DictConfig) -> None:
    """Process all trials in the specified split to estimate treatment effects."""

    if cfg.nct_id:
        nct_ids = [cfg.nct_id]
        logger.info(f"Processing trial {cfg.nct_id}.")

    else:
        # Load study object from YAML file
        study_file = get_study_filepaths(cfg.save_path, cfg.conditions[0])["study"]
        study = Study.from_yaml(study_file)

        if cfg.split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split '{cfg.split}'. Must be one of 'train', 'val', or 'test'."
            )

        # Get NCT IDs based on the split
        nct_ids = _get_nct_ids(cfg.split, study)
        logger.info(f"Processing {len(nct_ids)} trials for split '{cfg.split}'.")

    for nct_id in nct_ids:
        await _process_trial(cfg, nct_id)


# TODO: improve on relative path for config
@hydra.main(
    config_path="../../conf", config_name="estimate_ate.yaml", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    """Main function to estimate average treatment effects."""
    if is_weave_available:
        import weave  # type: ignore # noqa: PLC0415

        weave.init("naturalv2")

    asyncio.run(_process_all_trials(cfg))


if __name__ == "__main__":
    main()
