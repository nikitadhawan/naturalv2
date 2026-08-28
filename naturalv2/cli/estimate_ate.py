"""Estimate Average Treatment Effects (ATE) using the NATURAL pipeline."""

import asyncio
import logging
import os
from ast import literal_eval
from collections.abc import Mapping

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm
from tqdm.contrib.logging import logging_redirect_tqdm

import naturalv2.hydra_setup  # noqa: F401 # Ensure custom resolvers are registered
from naturalv2.estimators import NaturalIPW, NaturalMC, NaturalOI
from naturalv2.experiment import Experiment
from naturalv2.logging_utils import build_kv_table, emit_table
from naturalv2.pipeline import (
    TREATMENT_COL_NAME,
    NATURALPipeline,
    PipelineContext,
    PipelineStage,
    SampleValidationConfig,
)
from naturalv2.pipeline.sample_extraction import SampleTYStage
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
    responses: np.ndarray, inclusion_probs: pd.DataFrame, use_weights=True
) -> np.ndarray:
    """Weight individual treatment responses by inclusion probabilities.

    Parameters
    ----------
    responses : np.ndarray
        Array of individual treatment responses with shape
        ``[num_treatments, num_datapoints]``.
    inclusion_probs : pd.DataFrame
        DataFrame containing inclusion probabilities for each datapoint.
        It should have a column named 'inclusion_probs' with stringified lists
        of probabilities for each datapoint.

    Returns
    -------
    np.ndarray
        Weighted average treatment response for each treatment, with shape
        ``[num_treatments]``.
    """
    # responses has shape [num_treatments, num_datapoints]
    if use_weights and "inclusion_probs" not in inclusion_probs.columns:
        # e.g. the `inclusion_prob` stage was explicitly skipped in the pipeline.
        logger.warning(
            "use_inclusion_weights=True but no 'inclusion_probs' column is present; "
            "falling back to uniform weights."
        )
        use_weights = False

    if not use_weights:
        probs = np.ones(responses.shape[1])
    else:
        probs = inclusion_probs.apply(
            lambda row: literal_eval(row["inclusion_probs"])[1], axis=1
        ).to_numpy()
    return np.average(responses, axis=1, weights=probs)


def _stratified_bootstrap_sample(
    extractions: pd.DataFrame, stratify_col: str, random_state: int
) -> pd.DataFrame:
    """Resample with replacement within each ``stratify_col`` group.

    Unlike a flat resample of the whole DataFrame, this guarantees every group
    that has at least one row keeps exactly its original row count in the
    resample, so a treatment already present in ``extractions`` can never
    vanish from a bootstrap replicate by chance. A treatment with zero rows to
    begin with is unaffected either way -- there's nothing to resample from.
    """
    resampled_groups = [
        group.sample(n=len(group), replace=True, random_state=random_state)
        for _, group in extractions.groupby(stratify_col)
    ]
    return pd.concat(resampled_groups)


def _calculate_treatment_responses(
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
    """Calculate treatment responses for all treatments.

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

    if isinstance(
        estimator, (NaturalIPW, NaturalOI)
    ) and not experiment.is_binary_outcome(outcome):
        # These estimators only model a binary Yes/No conditional distribution.
        raise ValueError(
            f"{estimator.__class__.__name__} doesn't support non-binary outcome "
            f"'{outcome}'; use NaturalMC instead."
        )

    sampled_treatment_col = f"{TREATMENT_COL_NAME}_discretized"
    unsupported_treatments: set[str] = set()
    if isinstance(estimator, NaturalMC):
        observed_treatment_idxs = set(extractions[sampled_treatment_col].unique())
        unsupported_treatments = {
            treatment
            for i, treatment in enumerate(experiment.treatment_names)
            if i not in observed_treatment_idxs
        }
        if unsupported_treatments:
            logger.warning(
                "No reports mention treatment(s) %s for outcome '%s'; their "
                "responses will be reported as NaN.",
                sorted(unsupported_treatments),
                outcome,
            )

    if isinstance(estimator, NaturalMC):
        all_responses = estimator.get_individual_treatment_effects(extractions, outcome)
    else:
        all_responses = estimator.get_individual_treatment_effects(extractions)

    weighted_responses = _weight_by_inclusion(
        all_responses, extractions, use_inclusion_weights
    )  # len: num_treatments

    bootstrap_weighted_responses = np.zeros(
        (bootstrap_size, len(experiment.treatment_names))
    )
    for b in range(bootstrap_size):
        if isinstance(estimator, NaturalMC):
            bootstrap_data = _stratified_bootstrap_sample(
                extractions, sampled_treatment_col, seed + b
            )
        else:
            bootstrap_data = extractions.copy().sample(
                n=len(extractions),
                replace=True,
                random_state=seed + b,
                axis="index",
            )

        if isinstance(estimator, NaturalMC):
            all_responses = estimator.get_individual_treatment_effects(
                bootstrap_data, outcome
            )
        else:
            all_responses = estimator.get_individual_treatment_effects(bootstrap_data)

        bootstrap_weighted_responses[b, :] = _weight_by_inclusion(
            all_responses, bootstrap_data, use_inclusion_weights
        )  # len: num_treatments

    for i, treatment in enumerate(experiment.treatment_names):
        if treatment in unsupported_treatments:
            pred_response = np.nan
            conf_delta = np.nan
        else:
            pred_response = weighted_responses[i]
            bootstrap_response = bootstrap_weighted_responses[:, i]
            avg_bootstrap_response = np.mean(bootstrap_response)
            sample_variance = np.sum(
                (bootstrap_response - avg_bootstrap_response) ** 2
            ) / (bootstrap_size - 1)
            conf_delta = norm.ppf(1 - alpha / 2) * np.sqrt(sample_variance)
        results = {
            "estimator": estimator.__class__.__name__,
            "outcome": outcome,
            "treatment": treatment,
            "pred_response": pred_response,
            "CI_lower": pred_response - conf_delta,
            "CI_upper": pred_response + conf_delta,
        }
        logger.info(
            "Predicted Response: %f, (%f, %f)",
            pred_response,
            pred_response - conf_delta,
            pred_response + conf_delta,
        )
        if (
            experiment.status == "completed"
            and [outcome, treatment] in experiment.apo_outcome_treatment
        ):
            response_idx = experiment.apo_outcome_treatment.index([outcome, treatment])
            true_response = experiment.avg_potential_outcomes[response_idx]
            error = abs(pred_response - true_response)
            true_stats = experiment.apo_stats[response_idx]
            dispersion_type, dispersion, cohort_size = true_stats
            results.update(
                {
                    "true_response": true_response,
                    "abs_error": error,
                    "true_dispersion_type": dispersion_type,
                    "true_dispersion": dispersion,
                    "true_cohort_size": cohort_size,
                }
            )
            logger.info(
                "True Response: %f, %s: %s, Cohort Size: %s",
                true_response,
                str(dispersion_type),
                str(dispersion),
                str(cohort_size),
            )
            logger.info("Absolute Error: %f", error)
        result_dicts.append(results)

    return result_dicts, weighted_responses


def _calculate_treatment_effects(
    experiment,
    outcome,
    estimator,
    weighted_responses,
):
    result_dicts = []
    for i, treat1 in enumerate(experiment.treatment_names):
        for j, treat2 in enumerate(experiment.treatment_names):
            if i < j:
                try:
                    pred_ate = weighted_responses[j] - weighted_responses[i]
                    results = {
                        "estimator": estimator.__class__.__name__,
                        "outcome": outcome,
                        "treatments": f"{treat2}-{treat1}",
                        "pred_ate": pred_ate,
                    }
                    logger.info("Predicted ATE: %f", pred_ate)
                    if (
                        experiment.status == "completed"
                        and [outcome, [treat1, treat2]] in experiment.outcome_treatment
                    ):
                        effect_idx = experiment.outcome_treatment.index(
                            [outcome, [treat1, treat2]]
                        )
                        true_ate = experiment.effect_sizes[effect_idx]
                        error = abs(pred_ate - true_ate)
                        results.update({"true_ate": true_ate, "abs_error": error})
                        logger.info("True ATE: %f", true_ate)
                        logger.info("Absolute Error: %f", error)
                    result_dicts.append(results)

                except:
                    logger.info(
                        "ATE prediction errored for %s, %s, %s.",
                        treat1,
                        treat2,
                        outcome,
                    )
                    continue

    return result_dicts


def _warn_on_stale_bounds(
    experiment: Experiment,
    nct_id: str,
    configured_bounds: Mapping[str, Mapping[str, float]],
) -> None:
    for outcome, wanted in configured_bounds.items():
        persisted = experiment.outcome_bounds.get(outcome)
        in_use = persisted.model_dump(mode="json") if persisted else None
        if in_use != wanted:
            logger.warning(
                "Configured bounds for %s / %r differ from the experiment YAML "
                "(config %s, in use %s). Bounds are persisted by create_study, so "
                "rebuild the experiment to apply the change.",
                nct_id,
                outcome,
                wanted,
                in_use,
            )


def _save_results(
    results: list[dict], save_path: str, nct_id: str, exp_name: str, eval_type: str
) -> None:
    """Save results to CSV file."""
    result_df = pd.DataFrame(results)
    results_path = os.path.join(
        save_path, "results", f"{nct_id}_{exp_name}/{eval_type}_results.csv"
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


def _load_sample_validation(cfg: DictConfig) -> SampleValidationConfig | None:
    """Load the rejection policy when the configured pipeline samples outcomes."""
    sample_ty_target = f"{SampleTYStage.__module__}.{SampleTYStage.__name__}"
    uses_sample_ty = any(
        stage_config.get("_target_") == sample_ty_target
        for stage_config in cfg.pipeline.stages.values()
    )
    if not uses_sample_ty:
        return None
    if "sample_validation" not in cfg:
        raise ValueError(
            "SampleTYStage requires a `sample_validation` section. Inherit "
            "`conf/common.yaml` or define the policy explicitly."
        )
    return SampleValidationConfig.model_validate(
        OmegaConf.to_container(cfg.sample_validation, resolve=True)
    )


async def _process_trial(  # noqa: PLR0912, PLR0915
    cfg: DictConfig,
    nct_id: str,
    sample_validation: SampleValidationConfig | None,
) -> None:
    """Process a single trial to estimate treatment effects."""

    # Load the experiment configuration
    try:
        exp_file = get_experiment_filepath(cfg.save_path, nct_id, cfg.experiment_name)
        experiment = Experiment.from_yaml(exp_file)
    except (FileNotFoundError, ValueError) as e:
        logger.error(
            "Experiment file for %s not found or invalid. Skipping trial. Error: %s",
            nct_id,
            e,
            exc_info=True,
        )
        return
    _warn_on_stale_bounds(
        experiment,
        nct_id,
        (
            OmegaConf.to_container(cfg.get("outcome_bounds", {}) or {}, resolve=True)
            or {}
        ).get(nct_id, {}),
    )

    # If the experiment has no _avg_potential_outcomes or it is an empty list, calculate them from the trial.
    # Note: we can remove this once all our experiment yamls are updated to include APOs.
    if (
        not hasattr(experiment, "_avg_potential_outcomes")
        or not experiment._avg_potential_outcomes
    ):
        from naturalv2.clinical_trial import ClinicalTrial  # noqa: PLC0415

        trial = ClinicalTrial.from_json_file(experiment.trial_path)
        experiment._avg_potential_outcomes = []
        experiment._set_outcome_treatment_effects(trial)
        experiment.to_yaml(exp_file)
    os.makedirs(
        os.path.join(
            cfg.save_path, "results", f"{experiment.nct_id}_{cfg.experiment_name}"
        ),
        exist_ok=True,
    )
    for source_name in cfg.sources:
        for outcome in experiment.outcome_names:
            logger.info(
                "Running pipeline for %s with source '%s' and outcome '%s'",
                nct_id,
                source_name,
                outcome,
            )

            estimator_type = cfg.estimator._target_.split(".")[-1]
            pipeline_context = PipelineContext(
                experiment=experiment,
                source_name=source_name,
                estimator_type=estimator_type,
                outcome=outcome,
                save_path=cfg.save_path,
                exp_name=cfg.experiment_name,
                sample_validation=sample_validation,
            )

            pipeline_stages: list[PipelineStage] = []
            for name, stage_config in cfg.pipeline.stages.items():
                pipeline_stages.append(
                    instantiate(stage_config, name=name, _recursive_=False)
                )

            pipeline = NATURALPipeline(pipeline_stages)

            try:
                # Load curated data
                if cfg.filter_by_date:
                    curated_filepath = experiment.source_paths.get(source_name)
                else:
                    curated_filepath = experiment.source_paths.get(
                        f"{source_name}_no_date_filter"
                    )

                if not curated_filepath or not os.path.exists(curated_filepath):
                    logger.warning(
                        "The source '%s' does not have a curated dataset for the "
                        "experiment %s. Skipping...",
                        source_name,
                        nct_id,
                    )
                    continue

                if os.path.isdir(curated_filepath):
                    curated_df = pd.read_parquet(curated_filepath)
                else:
                    curated_df = pd.read_csv(curated_filepath)
                # TODO: remove subsampling after testing
                # curated_df = curated_df.sample(
                #     frac=0.05, random_state=cfg.seed, ignore_index=True
                # )
                context_table = build_kv_table(
                    "Pipeline Context",
                    [
                        ("NCT ID", nct_id),
                        ("Source", source_name),
                        ("Outcome", outcome),
                        ("Estimator type", estimator_type),
                        ("Number of curated reports", len(curated_df)),
                    ],
                )
                emit_table(context_table, logger=logger)
                if len(curated_df) > 100000:
                    logger.warning(
                        f"{nct_id} has more than 100k datapoints - double check if pipeline should be run."
                    )
                    continue

                # Run the pipeline
                extractions = await pipeline.run(curated_df, pipeline_context)
                if extractions.empty:
                    logger.warning(
                        "No extractions found for %s and outcome '%s'. "
                        "Cannot calculate treatment effects.",
                        source_name,
                        outcome,
                    )
                    result = {}
                    result.update(pipeline._data_flow)
                    result["source_name"] = source_name
                    result["initial_curated"] = len(curated_df)
                    result["conditions"] = cfg.conditions
                    result["filter_by_date"] = cfg.filter_by_date
                    _save_results(
                        [result],
                        cfg.save_path,
                        experiment.nct_id,
                        cfg.experiment_name,
                        "apo",
                    )
                    continue

                # Calculate and save treatment effects
                estimator = instantiate(cfg.estimator, experiment=experiment)
                results, weighted_responses = _calculate_treatment_responses(
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
                    result["conditions"] = cfg.conditions
                    result["filter_by_date"] = cfg.filter_by_date

                _save_results(
                    results,
                    cfg.save_path,
                    experiment.nct_id,
                    cfg.experiment_name,
                    "apo",
                )

                if cfg.ate:
                    ate_results = _calculate_treatment_effects(
                        experiment, outcome, estimator, weighted_responses
                    )
                    for ate_result in ate_results:
                        ate_result.update(pipeline._data_flow)
                        ate_result["source_name"] = source_name
                        ate_result["initial_curated"] = len(curated_df)
                        ate_result["conditions"] = cfg.conditions
                        ate_result["filter_by_date"] = cfg.filter_by_date
                    _save_results(
                        ate_results,
                        cfg.save_path,
                        experiment.nct_id,
                        cfg.experiment_name,
                        "ate",
                    )

            except Exception as e:
                logger.error(
                    "Error processing %s with source '%s' and outcome '%s': %s",
                    nct_id,
                    source_name,
                    outcome,
                    e,
                    exc_info=True,
                )
                continue


async def _process_all_trials(cfg: DictConfig) -> None:
    """Process all trials in the specified split to estimate treatment effects."""
    sample_validation = _load_sample_validation(cfg)

    if cfg.nct_id:
        nct_ids = [cfg.nct_id]
        logger.info("Processing trial %s", cfg.nct_id)

    else:
        # Load study object from YAML file
        study_file = get_study_filepaths(
            cfg.save_path, cfg.conditions[0], cfg.experiment_name, ate=cfg.ate
        )["study"]
        study = Study.from_yaml(study_file)

        if cfg.split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split '{cfg.split}'. Must be one of 'train', 'val', or 'test'."
            )

        # Get NCT IDs based on the split
        nct_ids = _get_nct_ids(cfg.split, study)
        logger.info("Processing %s trials for '%s' split", len(nct_ids), cfg.split)

    for nct_id in nct_ids:
        await _process_trial(cfg, nct_id, sample_validation)


# TODO: improve on relative path for config
@hydra.main(
    config_path="../../conf", config_name="estimate_ate.yaml", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    """Main function to estimate average treatment effects."""
    if is_weave_available:
        import weave  # type: ignore # noqa: PLC0415

        weave.init("naturalv2")

    with logging_redirect_tqdm():
        asyncio.run(_process_all_trials(cfg))


if __name__ == "__main__":
    main()
