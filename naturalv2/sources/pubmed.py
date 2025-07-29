import json
import os
from typing import Optional

import pandas as pd
from omegaconf import DictConfig

from naturalv2.evals.experiment import Experiment

from .pubmed_utils import fetch_articles, search_pubmed


class PubMedSet:
    def __init__(
        self,
        data_path: str,
        lm_cfg: DictConfig,
        download: bool = False,
        api_key: Optional[str] = None,
    ):
        self.data_path = data_path
        self.lm_cfg = lm_cfg
        self.api_key = api_key

        # TODO
        # self.treatment_names
        # self.outcome_words
        # self.trial_keywords

    def get_search_query(self, keyword: str) -> str:
        return (
            f'("{keyword}"[All Fields]) AND '
            '"english"[Language] AND '
            '"case reports"[Publication Type] AND '
            "hasabstract[Filter] AND "
            '"humans"[MeSH Terms]'
        )

    def condition_filter(self, keywords: list[str]) -> list[str]:
        self.data_files = []
        for keyword in keywords:
            keyword_data_path = self.data_path + f"{keyword}_case_reports.json"
            if not os.path.exists(keyword_data_path):
                query = self.get_search_query(keyword)
                webenv, query_key = search_pubmed(query, self.api_key)
                case_reports = fetch_articles(
                    webenv, query_key, self.api_key, self.data_path
                )
                with open(keyword_data_path, "w") as f:
                    json.dump(case_reports, f, indent=2)
                # TODO: else: check how many are cached already
                print(f"For query: {keyword}, {len(case_reports)} case reports found!")
            self.data_files.append(keyword_data_path)

        return self.data_files

    def clean_data(self) -> tuple[str, int]:
        pass

    def curate_experiment_data(
        self,
        experiment: Experiment,
        study_name: str,
        filter_by_date: bool,
        clean_data_path: str,
    ) -> tuple[str, int]:
        rule_filtered_df = pd.DataFrame()
        save_path = os.path.join(self.data_path, f"{experiment.nct_id}_pubmed.csv")

        if not os.path.exists(save_path):
            # TODO: curate experiment data
            pass
        else:
            rule_filtered_df = pd.read_csv(save_path, index_col=0)

        return save_path, len(rule_filtered_df)
