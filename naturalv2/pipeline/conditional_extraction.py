import logging
import os
from typing import TYPE_CHECKING, Dict, Literal

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.special import softmax
from tqdm import tqdm

from naturalv2.evals.experiment import Experiment
from naturalv2.models.lm import LM, build_lm_instance_from_cfg, get_prompt_logprobs
from naturalv2.pipeline.natural import PipelineStage
from naturalv2.utils import (
    enum_to_dcts,
    enumerate_strings,
    get_sample_text,
    get_save_path,
    qa_interleaved_enum,
)


if TYPE_CHECKING:  # so that script can run without installing vllm, unless required
    from naturalv2.models.vllm import VLLM

logger = logging.getLogger(__name__)


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
        f"{extract_type}_{outcome}_probs",
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
        model = build_lm_instance_from_cfg(model_cfg)

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
                **batch_df.iloc[j].to_dict(),
                **idx_to_feat[sample_indices[j]],
                **{f"{extract_type}_probs": probs[j]},
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


class ConditionalExtractionStage(PipelineStage):
    """Stage for imputing missing information."""

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.extract_type_map = {
            "NaturalIPW": "ty_given_x",
            "NaturalOI": "y_given_tx",
            "NaturalMC": None,
        }

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self.extract_type = self.extract_type_map.get(self.estimator_type)
        self.data = extract_conditionals(
            data,
            self.experiment,
            self.source_name,
            self.outcome,
            self.model_cfg,
            self.save_path,
            extract_type=self.extract_type,
        )
        logger.info(
            f"Extracted {self.extract_type} conditionals from {len(self.data)} reports."
        )
        return self.data

    def get_stats(self) -> Dict[str, int]:
        return {"conditionals_extracted": len(self.data)}


class InclusionProbStage(PipelineStage):
    """Stage for imputing missing information."""

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.extract_type_map = {
            "NaturalIPW": "ty_given_x",
            "NaturalOI": "y_given_tx",
            "NaturalMC": None,
        }

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self.extract_type = self.extract_type_map.get(self.estimator_type)
        self.data = extract_conditionals(
            data,
            self.experiment,
            self.source_name,
            self.outcome,
            self.model_cfg,
            self.save_path,
            extract_type=self.extract_type,
        )
        logger.info(
            f"Extracted {self.extract_type} conditionals from {len(self.data)} reports."
        )
        return self.data

    def get_stats(self) -> Dict[str, int]:
        return {"inclusion_probs": len(self.data)}
