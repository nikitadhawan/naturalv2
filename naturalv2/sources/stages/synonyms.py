"""LLM-powered synonym expansion stage shared across sources."""

import ast
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

from naturalv2.sources.components.llm_extraction import extract_curation_info
from naturalv2.sources.core import CurationContext, SourceStage, StageState


if TYPE_CHECKING:
    from naturalv2.models.lm import APIModel


logger = logging.getLogger(__name__)


class SynonymStage(SourceStage):
    """Stage to find synonyms for a given attribute.

    This stage uses an LLM to find synonyms for a specified attribute (e.g., treatments)
    from a given source (e.g., "pubmed", "reddit").

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for the language model used in this stage.
    source_name : str
        Name of the source being curated from (e.g., "pubmed", "reddit").
    attribute : str
        The attribute for which synonyms are to be found (e.g., "treatments").
    max_concurrent_workers : int | None, optional, default=None
        Maximum number of concurrent workers for LLM requests.
    """

    def __init__(
        self,
        *,
        attribute: str,
        llm: "APIModel",
        max_concurrent_workers: int | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the class."""
        super().__init__(name=name)
        self.attribute = attribute
        self.llm = llm
        self.max_concurrent_workers = max_concurrent_workers

        self.extract_type = f"synonym_{attribute}"

    async def run(self, context: CurationContext, state: StageState) -> StageState:
        """Get synonyms for ``attribute`` found on ``source``.

        This method takes a list of Experiments and generates synonyms for their ``attribute``
        found on ``source``, using a language model.

        Parameters
        ----------
        exp_list : list[Experiment]
            List of Experiments in a study.
        context : CurationContext
            Context for the stage execution.

        Returns
        -------
        list[Experiment]
            List of experiments with updated common names and synonyms.

        Raises
        ------
        Exception
            If there is an error during the extraction process.
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

        input_df = pd.DataFrame(llm_inputs)
        if input_df.empty:
            logger.warning("No synonym tasks to process.")
            return state

        # Set up file path for saving results
        results_dir = self.results_dir(context, "synonyms")
        file_path = os.path.join(
            results_dir,
            f"{self.attribute}_synonyms_{context.experiment_name}.csv",
        )

        exp_dir = os.path.join(context.save_dir, "experiments")

        output_df = await extract_curation_info(
            input_df=input_df,
            stage_name=self.stage_name,
            source_name=context.source_name,
            extract_type=self.extract_type,
            llm=self.llm,
            file_path=file_path,
            token_tracker=context._token_tracker,
            max_concurrent_requests=self.max_concurrent_workers,
        )

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

            exp_file = os.path.join(exp_dir, f"{experiment.nct_id}.yaml")
            experiment.to_yaml(exp_file)

        state.update(num_keywords=num_keywords, num_synonyms=num_synonyms)
        return state
