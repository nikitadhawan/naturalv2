import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Literal, Optional

import hydra
import nest_asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from scipy.special import softmax
from tqdm import tqdm

from naturalv2.evals.experiment import Experiment
from naturalv2.models.lm import LM, get_message_content, get_prompt_logprobs
from naturalv2.utils import (
    create_response_format,
    enum_to_dcts,
    enumerate_strings,
    get_sample_text,
    qa_interleaved_enum,
)


if TYPE_CHECKING:  # so that script can run without installing vllm, unless required
    from naturalv2.models.vllm import VLLM

load_dotenv(".env")


def get_save_path(
    base_path: str,
    nct_id: str,
    model_name: str,
    extract_type: str,
    outcome: str,
) -> str:
    """Generate save path for extracted data."""
    return os.path.join(
        base_path,
        "results",
        f"{nct_id}",
        f"{model_name.replace('/', '-')}_{extract_type}_{outcome}.csv",
    )


async def _extract_covariates_from_report(
    model: LM, messages: list[dict[str, str]], response_format: BaseModel
) -> dict[str, Any]:
    response = model(messages=messages, response_format=response_format)
    response_text = get_message_content(response)[0]
    return response_format.model_validate_json(response_text).model_dump(mode="json")


async def extract_covariates(
    input_df: pd.DataFrame,
    experiment: Experiment,
    source_name: str,
    outcome: str,
    model_cfg: DictConfig,
    save_path: str,
    extract_type: Literal["relevance", "ty_filter", "knowns", "imputations"],
    batch_size: int = 1,
    response_format: Optional[type[BaseModel]] = None,
) -> pd.DataFrame:
    """Extract covariates from input data using LLM.

    Parameters
    ----------
    input_df: pd.DataFrame
        Input dataframe with reports
    experiment: Experiment
        Experiment object
    source_name: str
        Source of data, according to which prompts will be constructed
    outcome: str
        The outcome of interest
    model_cfg: DictConfig
        Model configuration
    save_path: str
        Base path to save results
    extract_type: Literal["relevance", "ty_filter", "knowns", "imputations"]
        Type of extraction to perform
    batch_size: int
        Number of samples to process in each batch
    response_format: Optional[type[BaseModel]]
        Pydantic model for response format validation

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted covariates

    """
    file_path = get_save_path(
        save_path, experiment.nct_id, model_cfg.model, extract_type, outcome
    )

    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)

    model = LM(**model_cfg)

    system_message = {
        "role": "system",
        "content": experiment.get_system_prompt(extract_type, outcome, source_name),
    }
    user_prompt_template = "\nText Report\n>{report}"

    out_dicts = []

    def _get_messages(reports: list[str]) -> list[list[dict[str, str]]]:
        return [
            [
                system_message,
                {
                    "role": "user",
                    "content": user_prompt_template.format(report=report),
                },
            ]
            for report in reports
        ]

    for start in tqdm(range(0, len(input_df), batch_size)):
        batch_df = input_df.iloc[start : start + batch_size]

        reports = batch_df["report"].tolist()
        messages = _get_messages(reports)

        tasks = [
            _extract_covariates_from_report(model, message, response_format)
            for message in messages
        ]
        results = await asyncio.gather(*tasks)

        out_dicts.extend(
            [{**results[j], **{"report": reports[j]}} for j in range(len(batch_df))]
        )

    llm_samples_df = pd.DataFrame.from_dict(out_dicts)
    # TODO later: Remove to use only new extractions - shouldn't change results much.
    if extract_type == "imputations":
        input_df.update(llm_samples_df, overwrite=False)
        llm_samples_df = input_df.copy()

    # if extract_type != "ty_filter":
    #     llm_samples_df = experiment.discretize(
    #         llm_samples_df, hard_filter=False, inf=False
    #     )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    llm_samples_df.to_csv(file_path)
    return llm_samples_df


def prepare_conditional_inputs(
    input_df: pd.DataFrame,
    experiment: Experiment,
    extract_type: str,
    reports: list[str],
) -> list[str]:
    """Prepare inputs for conditional extraction."""
    if extract_type == "inclusion":
        return reports

    for idx, report in enumerate(reports):
        row = input_df.loc[input_df["report"] == report]
        if len(row) == 0:
            continue

        to_sample = (
            experiment.covariate_names
            if extract_type == "ty_given_x"
            else experiment.covariate_names + ["treatment"]
        )
        row = row[to_sample].to_dict("records")[0]
        sample_text = get_sample_text(row, experiment.question_prompts)
        reports[idx] += sample_text

    return reports


def process_local_model(
    model: "VLLM", reports: list[str], interleaved_options: list[str]
) -> tuple[np.ndarray, list[int]]:
    """Process inputs with local VLLM model."""
    probs, sample_indices, _ = model.compute_input_probs(reports, interleaved_options)
    return probs, sample_indices


def process_remote_model(
    model: LM,
    system_prompt: str,
    llm_inputs: list[str],
    num_reports: int,
    num_options: int,
    length_norm: bool,
) -> tuple[np.ndarray, list[int]]:
    """Process inputs with remote LM model."""
    responses = [
        model(prompt=system_prompt + "\n\nText Report\n" + llm_input)
        for llm_input in llm_inputs
    ]

    logprobs = []
    for response in responses:
        prompt_logprobs_obj = get_prompt_logprobs(response)
        if prompt_logprobs_obj is None:
            continue

        logprob = sum(prompt_logprobs_obj.logprobs)
        if length_norm:
            logprob = logprob / len(prompt_logprobs_obj.decoded_tokens)
        logprobs.append(logprob)

    probs = softmax(
        np.array(logprobs).reshape((num_reports, num_options)),
        axis=1,
    )
    sample_indices = [np.random.choice(len(prob), p=prob) for prob in probs]

    return probs, sample_indices


def prepare_for_conditional_extraction(
    experiment: Experiment, to_enum: list[str]
) -> tuple[list[str], list[str], list[dict]]:
    """Prepare data structures for conditional extraction."""
    options = enumerate_strings({key: experiment.options[key] for key in to_enum})
    interleaved_options = qa_interleaved_enum(
        {key: experiment.question_prompts[key] for key in to_enum},
        {key: experiment.options[key] for key in to_enum},
        options,
        to_enum,
    )
    idx_to_feat = enum_to_dcts(options, to_enum)
    idx_to_feat = [
        experiment.apply_transform(dct, repr_type="numeric") for dct in idx_to_feat
    ]

    return options, interleaved_options, idx_to_feat


def extract_conditionals(
    input_df: pd.DataFrame,
    experiment: Experiment,
    source_name: str,
    outcome: str,
    model_cfg: DictConfig,
    save_path: str,
    extract_type: Literal["ty_given_x", "y_given_tx", "inclusion", None],
    length_norm: bool = False,
    batch_size: int = 1,
) -> pd.DataFrame:
    """Extract conditional probabilities from input data.

    Parameters
    ----------
    input_df: pd.DataFrame
        Input dataframe with reports
    experiment: Experiment
        Experiment object
    outcome: str
        Outcome of interest
    source_name: str
        Source of data, according to which prompts will be constructed
    model_cfg: DictConfig
        Model configuration
    save_path: str
        Base path to save results
    extract_type: Literal["ty_given_x", "y_given_tx", "inclusion", None]
        Type of conditional to extract
    length_norm: bool
        Whether to normalize logprobs by length
    batch_size: int
        Number of samples to process in each batch

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted conditional probabilities

    """

    # Return input if no extraction needed
    if extract_type is None:
        return input_df

    # Define features to enumerate based on extraction type
    to_enum_map = {
        "ty_given_x": ["treatment", outcome],
        "y_given_tx": outcome,
        "inclusion": ["inclusion"],
    }
    to_enum = to_enum_map[extract_type]

    # Generate save path
    file_path = get_save_path(
        save_path,
        experiment.nct_id,
        model_cfg.model,
        f"{extract_type}_{outcome}_probs"
        if extract_type == "inclusion"
        else f"{extract_type}",
    )

    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)

    # Validate model configuration
    assert model_cfg.get("completion_type") == "text", "Model type must be 'text'."
    assert model_cfg.get("prompt_logprobs") == 0, "Prompt logprobs must be 0."
    local = model_cfg.get("local", None)

    # Initialize model
    if local:
        from naturalv2.models.vllm import VLLM

        model = VLLM(**model_cfg)
    else:
        model = LM(**model_cfg)

    # Discretize input dataframe
    input_df = experiment.discretize(input_df)

    # Get system prompt and prepare options
    system_prompt = experiment.get_system_prompt("conditionals", outcome, source_name)
    _, interleaved_options, idx_to_feat = prepare_for_conditional_extraction(
        experiment, to_enum
    )

    if local:
        model.system_prompt = system_prompt

    llm_probs_df = pd.DataFrame()

    for start in tqdm(range(0, len(input_df), batch_size)):
        batch_df = input_df.iloc[start : start + batch_size].reset_index(drop=True)

        # Prepare input reports
        reports = batch_df["report"].tolist()
        reports = prepare_conditional_inputs(
            input_df, experiment, extract_type, reports
        )

        # Repeat reports and options for all combinations
        reports_repeated = [
            report for report in reports for _ in range(len(interleaved_options))
        ]
        options_repeated = interleaved_options * len(reports)
        llm_inputs = [
            report + option
            for report, option in zip(reports_repeated, options_repeated)
        ]

        # Select columns to include in output
        cols = experiment.covariate_names + [outcome, "treatment", "report"]
        rows = batch_df[cols]

        # Process inputs based on model type
        if local:
            probs, sample_indices = process_local_model(
                model, reports, interleaved_options
            )
        else:
            probs, sample_indices = process_remote_model(
                model,
                system_prompt,
                llm_inputs,
                len(reports),
                len(interleaved_options),
                length_norm,
            )

        # Prepare results for saving
        dict_to_save = [
            {
                **rows.iloc[j].to_dict(),
                **idx_to_feat[sample_indices[j]],
                **{"probs": probs[j]},
            }
            for j in range(len(reports))
        ]

        # TODO [fcogidi]: avoid saving to disk at every iteration?
        # Append to output dataframe
        df_to_save = pd.DataFrame.from_dict(dict_to_save)
        llm_probs_df = pd.concat([llm_probs_df, df_to_save], ignore_index=True)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        llm_probs_df.to_csv(file_path)

    return pd.read_csv(file_path, index_col=0)


def weight_by_inclusion(ites: np.ndarray, inclusion_probs: pd.DataFrame) -> np.ndarray:
    """Weight ITEs by inclusion probabilities."""
    # ites has shape [num_treatments, num_datapoints]
    probs = inclusion_probs.apply(
        lambda row: [float(prob) for prob in row["probs"][1:-1].split()][1], axis=1
    ).to_numpy()
    return np.average(ites, axis=1, weights=probs)


def calculate_treatment_effects(
    experiment: Experiment,
    outcome: str,
    estimator,
    conditionals: pd.DataFrame,
    inclusion_probs: pd.DataFrame,
    data_flow: dict[str, int],
) -> list[dict]:
    """Calculate treatment effects for all outcome-treatment pairs."""
    result_dicts = []

    if hasattr(estimator, "estimator_type"):
        all_ites = estimator.get_ites(conditionals, outcome)
    else:
        all_ites = estimator.get_ites(conditionals)
    weighted_effects = weight_by_inclusion(
        all_ites, inclusion_probs
    )  # len: num_treatments

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
                logging.info(f"Predicted ATE: {pred_ate}")
                if experiment.split != "test":
                    effect_idx = experiment.outcome_treatment.index(
                        (outcome, (treat1, treat2))
                    )
                    true_ate = experiment.effect_sizes[effect_idx]
                    error = abs(pred_ate - true_ate)
                    results.update({"true_ate": true_ate, "abs_error": error})
                    logging.info(f"True ATE: {true_ate}")
                    logging.info(f"Absolute Error: {error}")
                results.update(data_flow)
                result_dicts.append(results)

    return result_dicts


def save_results(results: list[dict], save_path: str, nct_id: str) -> None:
    """Save results to CSV file."""
    result_df = pd.DataFrame(results)
    results_path = os.path.join(save_path, "results", f"{nct_id}/ate_results.csv")

    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path, index_col=0)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    result_df.to_csv(results_path)


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main function to estimate average treatment effects."""
    exp_file = os.path.join(cfg.save_path, "experiments", f"{cfg.eval.nct_id}.yaml")
    experiment = Experiment.from_yaml(exp_file)
    os.makedirs(
        os.path.join(cfg.save_path, "results", f"{experiment.nct_id}"), exist_ok=True
    )
    outcome = cfg.outcome if cfg.outcome is not None else experiment.outcome_names[0]
    assert outcome in experiment.outcome_names, (
        f"This experiment didn't measure {outcome}."
    )

    nest_asyncio.apply()

    data_flow = {}

    # Load curated data for the first source in {cfg.sources}
    # TODO: remove subsampling after testing
    source_name = cfg.sources[0]
    curated_df = pd.read_csv(experiment.source_paths[source_name], index_col=0).sample(
        frac=0.05, random_state=cfg.seed, ignore_index=True
    )
    data_flow["curated"] = len(curated_df)
    logging.info(f"Initial number of curated reports: {len(curated_df)} reports.")

    # Find reports relevant to the problem setting (uncomment for automated pipeline)
    relevance_response_format = create_response_format(
        "RelevanceResponse", ["relevant"], {"relevant": Literal["Yes", "No"]}
    )
    curated_df = asyncio.run(
        extract_covariates(
            curated_df,
            experiment,
            source_name,
            outcome,
            cfg.cheap_model,
            cfg.save_path,
            "relevance",
            response_format=relevance_response_format,
        )
    )
    curated_df = curated_df[curated_df["relevant"].lower() == "yes"]
    data_flow["relevant"] = len(curated_df)
    logging.info(f"After relevance filter: {len(curated_df)} reports.")

    # Filter out reports that do not contain t,y info
    ty_filter_response_format = create_response_format(
        "TYFilterResponse", experiment.treatment_names + [outcome]
    )
    ty_samples = asyncio.run(
        extract_covariates(
            curated_df,
            experiment,
            source_name,
            outcome,
            cfg.cheap_model,
            cfg.save_path,
            "ty_filter",
            response_format=ty_filter_response_format,
        )
    )
    ty_filtered_df = experiment.hard_filter_ty(ty_samples)
    data_flow["ty_filtered"] = len(ty_filtered_df)
    logging.info(f"After treatment-outcome filter: {len(ty_filtered_df)} reports.")

    # Extract samples from reports, allowing LLM to output "unknown" for missing info
    knowns_response_format = create_response_format(
        "KnownsResponse", experiment.covariate_names
    )
    samples_with_unknown = asyncio.run(
        extract_covariates(
            ty_filtered_df,
            experiment,
            source_name,
            outcome,
            cfg.sample_model,
            cfg.save_path,
            "knowns",
            response_format=knowns_response_format,
        )
    )

    # Filter reports known to violate inclusion criteria
    inclusion_filtered = experiment.hard_filter_inclusion(samples_with_unknown)
    data_flow["inclusion_filtered"] = len(inclusion_filtered)
    logging.info(f"After inclusion filter: {len(inclusion_filtered)} reports.")

    # Impute samples from reports, imputing missing info
    imputations_response_format = create_response_format(
        "ImputationsResponse", experiment.covariate_names
    )
    imputed_samples = asyncio.run(
        extract_covariates(
            inclusion_filtered,
            experiment,
            source_name,
            outcome,
            cfg.sample_model,
            cfg.save_path,
            "imputations",
            response_format=imputations_response_format,
        )
    )

    # Drop rows with missing covariates even after imputation
    # imputed_samples = imputed_samples.dropna(
    #     subset=experiment.covariate_names
    # ).reset_index(drop=True)
    data_flow["final"] = len(imputed_samples)
    logging.info(f"Final: {len(imputed_samples)} reports.")

    # Extract conditionals depending on the estimator type
    estimator_type = cfg.estimator._target_.split(".")[-1]
    extract_type_map = {
        "NaturalIPW": "ty_given_x",
        "NaturalOI": "y_given_tx",
        "NaturalMC": None,
    }
    extract_type = extract_type_map.get(estimator_type)

    conditionals = extract_conditionals(
        imputed_samples,
        experiment,
        source_name,
        outcome,
        cfg.probs_model,
        cfg.save_path,
        extract_type=extract_type,
    )

    # Extract inclusion probabilities of the form P(X in I | R)
    inclusion_probs = extract_conditionals(
        imputed_samples,
        experiment,
        source_name,
        outcome,
        cfg.probs_model,
        cfg.save_path,
        extract_type="inclusion",
    )

    # Calculate and save treatment effects
    estimator = instantiate(cfg.estimator, experiment=experiment)
    results = calculate_treatment_effects(
        experiment, outcome, estimator, conditionals, inclusion_probs, data_flow
    )

    save_results(results, cfg.save_path, experiment.nct_id)


if __name__ == "__main__":
    main()
