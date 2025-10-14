"""Synonym expansion stage using an LLM.

This module provides a reusable stage that queries a language model to expand
keyword lists (e.g., treatment) with common names and synonyms for a given
source (such as ``"pubmed"`` or ``"reddit"``).
"""

import ast
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.components import extract_curation_info
from naturalv2.sources.core import CurationContext, SourceStage, StageState


if TYPE_CHECKING:
    from naturalv2.models.lm import APIModel


logger = logging.getLogger(__name__)


class SynonymStage(SourceStage):
    """Find synonyms for a specific experiment attribute.

    The stage queries an LLM to expand a chosen attribute of each experiment
    (for example, ``"treatment"``) into a set of common names and synonyms
    that are specific to a source (e.g., ``"pubmed"``, ``"reddit"``). Results
    are written to per-experiment YAML files and summarized in the stage state.

    Output files are written to ``results_dir(context, "synonyms")`` with the
    naming pattern ``{attribute}_synonyms_{experiment_name}.csv``.

    Parameters
    ----------
    attribute : str
        Name of the experiment attribute to expand (e.g., ``"treatment"``).
        The corresponding fields are expected in ``Experiment`` as
        ``{attribute}_names``, ``{attribute}_desc``, and
        ``{attribute}_common_names``.
    llm : APIModel
        Language model client used to perform the extraction.
    max_concurrent_workers : int | None, optional, default=None
        Maximum number of concurrent workers used during extraction.
    name : str | None, optional, default=None
        Optional explicit stage name; defaults to the class name.
    """

    def __init__(
        self,
        *,
        attribute: str,
        llm: "APIModel",
        max_concurrent_workers: int | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the stage."""
        super().__init__(name=name)
        self.attribute = attribute
        self.llm = llm
        self.max_concurrent_workers = max_concurrent_workers

        self.extract_type = f"synonym_{attribute}"

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Generate synonyms for ``attribute`` values in the experiments.

        Parameters
        ----------
        context : CurationContext
            Curation context providing experiments, source name, save paths and
            token tracking.
        state : StageState
            Incoming state from previous stage; will be updated with summary
            metrics.

        Returns
        -------
        StageState
            State updated with ``num_keywords`` and ``num_synonyms``.
        """
        if not context.experiments:
            logger.warning("No experiments provided to process.")
            return state

        llm_inputs = []
        for experiment in context.experiments:
            for keyword in getattr(experiment, f"{self.attribute}_names"):
                if context.source_name not in getattr(
                    experiment, f"{self.attribute}_common_names"
                ):
                    getattr(experiment, f"{self.attribute}_common_names").update(
                        {context.source_name: {}}
                    )
                if (
                    keyword
                    in getattr(experiment, f"{self.attribute}_common_names")[
                        context.source_name
                    ]
                ):
                    logger.debug(
                        f"Skipping {keyword} for {experiment.nct_id} - already have common names"
                    )
                    continue
                desc = getattr(experiment, f"{self.attribute}_desc")[keyword]
                llm_inputs.append(
                    {
                        "nct_id": experiment.nct_id,
                        "keyword": keyword,
                        "trial_title": experiment.title,
                        f"{self.attribute}_desc": desc,
                        "drugbank_names": experiment.drugbank_names[keyword],
                        "source": context.source_name,
                    }
                )

        extraction_inputs = pd.DataFrame(llm_inputs)
        if extraction_inputs.empty:
            logger.warning("No synonym tasks to process.")
            return state

        # Set up file path for saving results
        results_dir = self.results_dir(context)
        file_path = os.path.join(
            results_dir,
            f"{self.attribute}_synonyms_{context.experiment_name}.csv",
        )

        output_df = await extract_curation_info(
            extraction_inputs,
            stage_name=self.stage_name,
            source_name=context.source_name,
            extract_type=self.extract_type,
            llm=self.llm,
            file_path=file_path,
            token_tracker=context._token_tracker,
            max_concurrent_requests=self.max_concurrent_workers,
        )

        experiments_dir = os.path.join(context.save_dir, "experiments")
        num_keywords, num_synonyms = 0, 0
        for experiment in context.experiments:
            synonyms_dict = {}
            for keyword in getattr(experiment, f"{self.attribute}_names"):
                keyword_rows = output_df[output_df["keyword"] == keyword]
                if len(keyword_rows) > 0:
                    synonyms = ast.literal_eval(keyword_rows.iloc[0]["llm_output"])
                    synonyms_dict[keyword] = synonyms
                    num_keywords += 1
                    num_synonyms += len(synonyms)
            getattr(experiment, f"{self.attribute}_common_names")[
                context.source_name
            ].update(synonyms_dict)

            exp_file = os.path.join(experiments_dir, f"{experiment.nct_id}.yaml")
            experiment.to_yaml(exp_file)

        state.update(num_keywords=num_keywords, num_synonyms=num_synonyms)
        logger.info(
            "%s: found %d synonyms for %d keywords",
            self.stage_name,
            num_synonyms,
            num_keywords,
        )
        context._token_tracker.log_table()

        # Add prompt template to metadata for logging
        prompt_id = f"{self.extract_type}_{context.source_name}"
        template = load_prompt(
            base_dir="naturalv2/prompts/templates",
            prompt_type=prompt_id,
            return_format="prompt",
        )
        state.metadata.setdefault("prompt_templates", {})[prompt_id] = template
        return state
