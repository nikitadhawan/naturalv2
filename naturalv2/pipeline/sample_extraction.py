import asyncio
import logging
import os
from typing import Any, Dict, Literal, Optional

import pandas as pd
from omegaconf import DictConfig
from pydantic import BaseModel
from tqdm import tqdm

from naturalv2.evals.experiment import Experiment
from naturalv2.models.lm import LM, build_lm_instance_from_cfg, get_message_content
from naturalv2.pipeline.natural import PipelineStage
from naturalv2.utils import create_response_format, get_save_path


logger = logging.getLogger(__name__)


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
        save_path, experiment.nct_id, model_cfg.model, extract_type
    )

    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)

    model = build_lm_instance_from_cfg(model_cfg)

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


class RelevanceFilterStage(PipelineStage):
    """Stage for filtering relevant reports."""

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self.response_format = create_response_format(
            "RelevanceResponse", ["relevant"], {"relevant": Literal["Yes", "No"]}
        )
        filtered_data = asyncio.run(
            extract_covariates(
                data,
                self.experiment,
                self.source_name,
                self.outcome,
                self.model_cfg,
                self.save_path,
                "relevance",
                response_format=self.response_format,
            )
        )
        self.data = filtered_data[filtered_data["relevant"].lower() == "yes"]
        logger.info(f"After relevance filter: {len(self.data)} reports.")
        return self.data

    def get_stats(self) -> Dict[str, int]:
        return {"relevant": len(self.data)}


class TreatmentOutcomeFilterStage(PipelineStage):
    """Stage for filtering reports with treatment and outcome information."""

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self.response_format = create_response_format(
            "TYFilterResponse", self.experiment.treatment_names + [self.outcome]
        )
        ty_samples = asyncio.run(
            extract_covariates(
                data,
                self.experiment,
                self.source_name,
                self.outcome,
                self.model_cfg,
                self.save_path,
                "ty_filter",
                response_format=self.response_format,
            )
        )
        self.data = self.experiment.hard_filter_ty(ty_samples)
        logger.info(f"After treatment-outcome filter: {len(self.data)} reports.")
        return self.data

    def get_stats(self) -> Dict[str, int]:
        return {"ty_filtered": len(self.data)}


class KnownsStage(PipelineStage):
    """Stage for extracting knowns information, allowing 'Unknown' for missing info."""

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self.response_format = create_response_format(
            "KnownsResponse", self.experiment.covariate_names
        )
        self.data = asyncio.run(
            extract_covariates(
                data,
                self.experiment,
                self.source_name,
                self.outcome,
                self.model_cfg,
                self.save_path,
                "knowns",
                response_format=self.response_format,
            )
        )
        self.data = self.experiment.hard_filter_inclusion(data)
        logger.info(f"After inclusion filter: {len(self.data)} reports.")
        return self.data

    def get_stats(self) -> Dict[str, int]:
        return {"inclusion_filtered": len(self.data)}


class ImputationsStage(PipelineStage):
    """Stage for imputing missing information."""

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self.response_format = create_response_format(
            "ImputationsResponse",
            self.experiment.covariate_names,  # TODO: include treatment according ot extract_type
        )
        self.data = asyncio.run(
            extract_covariates(
                data,
                self.experiment,
                self.source_name,
                self.outcome,
                self.model_cfg,
                self.save_path,
                "imputations",
                response_format=self.response_format,
            )
        )
        # Drop rows with missing covariates even after imputation
        # self.data = self.data.dropna(
        #     subset=self.experiment.covariate_names
        # ).reset_index(drop=True)
        logger.info(f"Final: {len(self.data)} reports after imputation.")
        return self.data

    def get_stats(self) -> Dict[str, int]:
        return {"final": len(self.data)}
