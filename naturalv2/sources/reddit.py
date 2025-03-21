import os

import pandas as pd

from naturalv2.models.lm import LM
from naturalv2.sources.reddit_utils import (
    date_filter,
    download_sub_data,
    get_context_post_df,
    get_sub_about_info,
    rule_based_filter,
    subreddit_relevance_llm,
)


class RedditSource:
    def __init__(self, data_path, match_method, lm_cfg):
        self.data_path = data_path
        self.match_method = match_method
        self.lm_cfg = lm_cfg

        self.subs_about = get_sub_about_info(self.data_path)

    def condition_filter(self, keywords):
        self.relevant_subs = []
        for row in self.subs_about.iterrows():
            sub_name, desc, public_desc = row[1].to_list()
            desc = f"Subreddit: r/{sub_name}.\nDescription: {desc}\nPublic description: {public_desc}"
            if self.match_method == "string_match":
                if any(keyword.lower() in desc.lower() for keyword in keywords):
                    self.relevant_subs.append(sub_name)
                    print(f"{sub_name} is relevant.")
            elif self.match_method == "llm":
                lm = LM(**self.lm_cfg)
                answer = subreddit_relevance_llm(desc, keywords, lm)
                if answer.lower().startswith("yes"):
                    self.relevant_subs.append(sub_name)
                    print(f"{sub_name} is relevant.")
        print(len(self.relevant_subs), "relevant subreddits found!")

        condition_data_paths = []
        for sub in self.relevant_subs:
            submissions_path = os.path.join(self.data_path, f"{sub}_submissions.csv")
            comments_path = os.path.join(self.data_path, f"{sub}_comments.csv")
            if not os.path.exists(submissions_path):
                download_sub_data(sub, "submissions", self.data_path)
            if not os.path.exists(comments_path):
                download_sub_data(sub, "comments", self.data_path)
            condition_data_paths.extend([submissions_path, comments_path])

        return condition_data_paths

    def clean_data(self, study_name):
        os.makedirs(os.path.join(self.data_path, study_name), exist_ok=True)
        save_path = os.path.join(self.data_path, f"{study_name}/reddit_cleaned.csv")
        if os.path.exists(save_path):
            cleaned_data = pd.read_csv(save_path, index_col=0)
            return save_path, len(cleaned_data)

        rule_filtered_df = pd.DataFrame()
        for sub in self.relevant_subs:
            submissions = pd.read_csv(
                os.path.join(self.data_path, f"{sub}_submissions.csv"), index_col=0
            )
            comments = pd.read_csv(
                os.path.join(self.data_path, f"{sub}_comments.csv"), index_col=0
            )
            submissions = rule_based_filter(submissions, "selftext")
            comments = rule_based_filter(comments, "body")
            merged_df = get_context_post_df(submissions, comments)
            rule_filtered_df = pd.concat(
                [rule_filtered_df, merged_df], ignore_index=True
            )
            rule_filtered_df.to_csv(save_path)
        rule_filtered_df = rule_filtered_df.drop_duplicates("post")
        rule_filtered_df.to_csv(save_path)
        return save_path, len(rule_filtered_df)

    def experiment_data(self, exp, study_name, filter_by_date, clean_data_path):
        # check treatment/outcome mention, filter by date
        save_path = os.path.join(
            self.data_path, f"{study_name}/reddit_{exp.nct_id}.csv"
        )
        if os.path.exists(save_path):
            exp_df = pd.read_csv(save_path, index_col=0)
            return save_path, len(exp_df)

        clean_data = pd.read_csv(clean_data_path, index_col=0)
        exp_df = pd.DataFrame(columns=clean_data.columns)
        treatment_names = exp.treatment_common_names["reddit"]
        outcome_names = exp.outcome_common_names["reddit"]

        for _, row in clean_data.iterrows():
            t_matches = [
                x
                for x in treatment_names
                if x in row["subreddit"].lower()
                or x in row["title"].lower()
                or x in row["post"].lower()
            ]
            if isinstance(row["initial_post"], str) and not pd.isna(
                row["initial_post"]
            ):
                t_matches += [
                    x for x in treatment_names if x in row["initial_post"].lower()
                ]
            t_matches = set(t_matches)
            o_matches = [
                x
                for x in outcome_names
                if x in row["subreddit"].lower()
                or x in row["title"].lower()
                or x in row["post"].lower()
            ]
            if isinstance(row["initial_post"], str) and not pd.isna(
                row["initial_post"]
            ):
                o_matches += [
                    x for x in outcome_names if x in row["initial_post"].lower()
                ]
            o_matches = set(o_matches)
            if len(t_matches) > 0 and len(o_matches) > 0:
                row["treatments"] = list(t_matches)
                row["outcomes"] = list(o_matches)
                exp_df = pd.concat([exp_df, pd.DataFrame([row])], ignore_index=True)

        if filter_by_date:
            exp_df = date_filter(exp_df, exp.date)
        exp_df.to_csv(save_path)
        return save_path, len(exp_df)

    def get_common_name_prompts(self):
        system_prompt = "You are a helpful medical assistant who can translate medical terminology into common terms."
        t_prompt = "\n\nWhat are common brand names or terms that people use when discussing the treatment, {keyword}, specifically when posting on Reddit?"
        o_prompt = "\n\nWhat are key common terms that people must use when discussing the outcome, {keyword}, specifically when posting on Reddit?"
        final_prompt = "\n\nReturn only a Python list of at most 5 individual words, without any other text or formatting."
        return {
            "system": system_prompt,
            "treatment": t_prompt + final_prompt,
            "outcome": o_prompt + final_prompt,
        }
