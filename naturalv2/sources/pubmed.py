import datetime
import json
import os

import pandas as pd

from naturalv2.sources.reddit_utils import (
    date_filter,
    get_context_post_df,
    rule_based_filter,
)

from .pubmed_utils import fetch_articles, pubmed_queries_llm, search_pubmed


class PubMedSet:
    def __init__(
        self, data_path, trial, match_method, llm, download=False, api_key=None
    ):
        self.data_path = data_path
        self.trial = trial
        self.llm = llm
        query_keywords = self.get_keywords(match_method, trial)

        if download:
            self.download_data(api_key, query_keywords)

        # TODO
        # self.treatment_names
        # self.outcome_words
        # self.trial_keywords

    def get_keywords(self, method, trial):
        if method == "string_match":
            return (
                trial.keywords
                + trial.conditions
                + [i.title for i in trial.interventions]
            )

        if method == "llm":
            return pubmed_queries_llm(trial, self.llm)

        raise ValueError(
            f"Unknown match method {method}. Please use 'string_match' or 'llm'."
        )

    def get_query(self, keyword: str) -> str:
        return (
            f'("{keyword}"[All Fields]) AND '
            '"english"[Language] AND '
            '"case reports"[Publication Type] AND '
            "hasabstract[Filter] AND "
            '"humans"[MeSH Terms]'
        )

    def download_data(self, api_key: str, keywords: list[str]) -> None:
        self.data_files = []
        for keyword in keywords:
            keyword_data_path = self.data_path + f"{keyword}_case_reports.json"
            if not os.path.exists(keyword_data_path):
                query = self.get_query(keyword)
                webenv, query_key = search_pubmed(query, api_key)
                case_reports = fetch_articles(
                    webenv, query_key, api_key, self.data_path
                )
                with open(keyword_data_path, "w") as f:
                    json.dump(case_reports, f, indent=2)
                # TODO: else: check how many are cached already
                print(f"For query: {keyword}, {len(case_reports)} case reports found!")
            self.data_files.append(keyword_data_path)

    def curate_data(self, filter_by_date: bool = False) -> pd.DataFrame:
        rule_filtered_df = pd.DataFrame()
        save_path = self.data_path + f"{self.trial.nctid}_pubmed_rule_based.csv"
        # treatment_names = get_reddit_synonyms([i.title for i in self.trial.interventions], self.llm)
        # outcome_words = get_reddit_synonyms([o.title for o in self.trial.primary_endpoints], self.llm)
        if not os.path.exists(save_path):
            for sub in self.subreddits:
                submissions = pd.read_csv(self.data_files[f"{sub}_submissions"])
                comments = pd.read_csv(self.data_files[f"{sub}_comments"])
                if date_filter:
                    trial_date = datetime.datetime.strptime(
                        self.trial.results_first_posted, "%Y-%m-%d"
                    )
                    trial_date_utc = int(
                        trial_date.replace(tzinfo=datetime.timezone.utc).timestamp()
                    )
                    submissions = filter_by_date(submissions, trial_date_utc)
                    comments = filter_by_date(comments, trial_date_utc)
                submissions = rule_based_filter(submissions, "selftext")
                comments = rule_based_filter(comments, "body")
                merged_df = get_context_post_df(
                    submissions, comments, self.treatment_names, self.outcome_words
                )
                rule_filtered_df = pd.concat(
                    [rule_filtered_df, merged_df], ignore_index=True
                )
                rule_filtered_df.to_csv(save_path)
            rule_filtered_df = rule_filtered_df.drop_duplicates("post")
            rule_filtered_df.to_csv(save_path)
        else:
            rule_filtered_df = pd.read_csv(save_path, index_col=0)
        self.curated_data = rule_filtered_df
        return self.curated_data
