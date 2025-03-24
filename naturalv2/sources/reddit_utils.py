import datetime
import json
import logging.handlers
import os
import urllib
import warnings
from time import time
from urllib import request

import pandas as pd
import wget
import zstandard


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def download_subs_list(data_path):
    if not os.path.exists(data_path + "subs_list.txt"):
        url = "https://the-eye.eu/redarcs/"
        response = request.urlopen(url)
        html = response.read().decode("utf-8")

        # Extract subreddit names from links
        subs = []
        for line in html.split("\n"):
            if "href=" in line and ".zst" in line:
                sub = line.split("href=")[1].split("_")[0].split("/")[-1]
                if sub not in subs:
                    subs.append(sub)

        with open(data_path + "subs_list.txt", "w") as f:
            f.write("\n".join(subs))
        print(len(subs), " subreddits listed.")


def download_sub_data(subreddit, data_type, data_path):
    _ = wget.download(
        "https://the-eye.eu/redarcs/files/{}_{}.zst".format(subreddit, data_type),
        out=data_path,
    )

    log = logging.getLogger("bot")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler())

    def read_and_decode(
        reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0
    ):
        chunk = reader.read(chunk_size)
        bytes_read += chunk_size
        if previous_chunk is not None:
            chunk = previous_chunk + chunk
        try:
            return chunk.decode()
        except UnicodeDecodeError as err:
            if bytes_read > max_window_size:
                raise UnicodeError(
                    f"Unable to decode frame after reading {bytes_read:,} bytes"
                ) from err
            log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
            return read_and_decode(
                reader, chunk_size, max_window_size, chunk, bytes_read
            )

    def read_lines_zst(file_name):
        with open(file_name, "rb") as file_handle:
            buffer = ""
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(
                file_handle
            )
            while True:
                chunk = read_and_decode(reader, 2**27, (2**29) * 2)
                if not chunk:
                    break
                lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line, file_handle.tell()
            buffer = lines[-1]
        reader.close()

    file_path = data_path + "{}_{}.zst".format(subreddit, data_type)
    # file_size = os.stat(file_path).st_size
    file_lines = 0
    # file_bytes_processed = 0
    # created = None
    bad_lines = 0
    data = []

    for line, _ in read_lines_zst(file_path):
        try:
            obj = json.loads(line)
            data += [obj]
        except (KeyError, json.JSONDecodeError):
            bad_lines += 1
        file_lines += 1
        # if file_lines % 100000 == 0:
        #     log.info(
        #         f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%"
        #     )

    save_path = data_path + "{}_{}.csv".format(subreddit, data_type)
    data_csv = pd.DataFrame(data)
    data_csv.to_csv(save_path)
    os.remove(file_path)
    log.info(f" Complete : {file_lines:,} : {bad_lines:,}")


def download_from_url(url_str, max_retries=5, retry_delay=120):
    for attempt in range(max_retries):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            req = request.Request(url_str, headers=headers)
            data = json.load(request.urlopen(req))
            time.sleep(2)
            return data
        except urllib.error.HTTPError as e:
            if e.code != 429:
                print(f"Error fetching data from {url_str}: {e}")
                return {"error": str(e)}
            if attempt < max_retries - 1:
                print(f"Rate limited, waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries exceeded for {url_str}")
                return {"error": "Max retries exceeded"}
        except Exception as e:
            print(f"Unexpected error fetching {url_str}: {e}")
            return {"error": str(e)}
    return {"error": "Failed to fetch data after maximum retries"}


def get_sub_about_info(data_path):
    if os.path.exists(os.path.join(data_path, "subs_about.csv")):
        return pd.read_csv(data_path + "subs_about.csv", index_col=0)

    if not os.path.exists(os.path.join(data_path, "subs_list.txt")):
        download_subs_list(data_path)

    with open(data_path + "subs_list.txt", "r") as f:
        subs_list = f.read().splitlines()

    about_csv_path = data_path + "subs_about.csv"
    about_df = pd.DataFrame(columns=["sub", "description", "public_description"])
    about_jsons = data_path + "subs_about"
    if not os.path.exists(about_jsons):
        os.makedirs(about_jsons)
    for sub in subs_list:
        about_file = data_path + f"subs_about/{sub}_about.json"
        if os.path.exists(about_file):
            with open(about_file, "r") as f:
                data = json.load(f)
        else:
            about_url = f"https://www.reddit.com/r/{sub}/about.json"
            data = download_from_url(about_url)
            if not data:
                continue
            with open(about_file, "w") as f:
                json.dump(data, f)

        if "data" in list(data.keys()):
            descr = data["data"].get("description", "")
            public_descr = data["data"].get("public_description", "")
            row = {"sub": sub, "description": descr, "public_description": public_descr}
            about_df = pd.concat([about_df, pd.DataFrame([row])], ignore_index=True)
        else:
            assert "429" not in data["error"]
    about_df.to_csv(about_csv_path)
    return about_df


def get_submission_permalink(permalink):
    return "/" + permalink.split("/")[-2] + "/"


def get_comment_permalink(permalink):
    return "/" + permalink.split("/")[-3] + "/"


def date_filter(df, date_str):
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    utc_date_cutoff = int(date_obj.replace(tzinfo=datetime.timezone.utc).timestamp())
    idx = df["date_created"].apply(lambda x: int(x) <= utc_date_cutoff)
    return df[idx]


def get_date(utc_timestamp):
    dt = datetime.datetime.fromtimestamp(utc_timestamp, tz=datetime.timezone.utc)
    return dt.strftime("%B %d, %Y")


def rule_based_filter(post_df, text_field):
    # remove rows where the text field is not of type str
    idx = post_df[text_field].apply(lambda x: isinstance(x, str))
    post_df = post_df.loc[idx]
    idx = post_df["permalink"].apply(lambda x: isinstance(x, str))
    post_df = post_df.loc[idx]
    # remove rows without a score
    post_df = post_df.loc[post_df["score"] != None]
    # remove rows where the submission is deleted or removed
    post_df = post_df.loc[post_df[text_field] != "[deleted]"]
    post_df = post_df.loc[post_df[text_field] != "[removed]"]
    # remove very short comments
    if text_field == "body":
        idx = post_df[text_field].apply(lambda x: len(x.split()) >= 10)
        post_df = post_df.loc[idx]
    # remove posts with "bot" in the author's name
    idx = post_df["author"].apply(lambda x: "bot" not in x.lower())
    post_df = post_df.loc[idx]
    for i, row in post_df.iterrows():
        body = row[text_field]
        # unescape some common html tags
        body = body.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        body = body.replace("\n", " ").replace("\t", " ")
        body = body.strip()
        # drop if there is no space in first 2048 characters
        try:
            _ = body[: body.rindex(" ", 0, 2048)]
        except ValueError:
            post_df = post_df.drop([i])
            continue
        # drop everything with less than 50% alphabetic characters; space counts
        length_characters = float(len(body))
        filtered = [c for c in body if c.isalpha()]
        if float(len(filtered)) / length_characters < 0.5:
            post_df = post_df.drop([i])
            continue
    return post_df


# def check_treatment_mention(lst, treatment_names):
#     filtered_lst = []
#     for elem in lst:
#         matches = [x for x in treatment_names if x in elem["subreddit"].lower()]
#         matches += [x for x in treatment_names if x in elem["title"].lower()]
#         matches += [x for x in treatment_names if x in elem["post"].lower()]
#         if "initial_post" in list(elem.keys()):
#             matches += [x for x in treatment_names if x in elem["initial_post"].lower()]
#         matches = set(matches)
#         if len(matches) > 0:
#             elem["treatments"] = list(matches)
#             filtered_lst.append(elem)
#     return filtered_lst


# def check_outcome_mention(lst, outcome_words):
#     filtered_lst = []
#     for elem in lst:
#         matches = [x for x in outcome_words if x in elem["subreddit"].lower()]
#         matches += [x for x in outcome_words if x in elem["title"].lower()]
#         matches += [x for x in outcome_words if x in elem["post"].lower()]
#         if "initial_post" in list(elem.keys()):
#             matches += [x for x in outcome_words if x in elem["initial_post"].lower()]
#         matches = set(matches)
#         if len(matches) > 0:
#             elem["outcome_words"] = list(matches)
#             filtered_lst.append(elem)
#     return filtered_lst


def get_context_post_df(submissions, comments):
    merged_df = pd.DataFrame(
        columns=[
            "subreddit",
            "title",
            "initial_post",
            "post",
            "score",
            "date_created",
            "permalink",
            "treatments",
            "outcome_words",
            "author_replies",
        ]
    )
    comments["permalink_processed"] = comments["permalink"].map(
        lambda x: get_comment_permalink(x)
    )
    for _, submission in submissions.iterrows():
        subreddit = submission["subreddit"]
        title = submission["title"]
        submission_text = submission["selftext"]
        score = int(submission["score"])
        created_utc = int(submission["created_utc"])
        date_created = get_date(created_utc)
        submission_permalink = get_submission_permalink(submission["permalink"])
        submission_comments = comments[
            comments["permalink_processed"] == submission_permalink
        ]
        submission_author_comments = submission_comments[
            submission_comments["author"] == submission["author"]
        ]
        submission_comments = submission_comments.drop(submission_author_comments.index)
        if len(submission_author_comments["body"].to_list()) > 0:
            submission_text += (
                "\n\nThe author also replied with the following in the thread:"
            )
            for reply in submission_author_comments["body"].to_list():
                submission_text += "\n> " + reply
        to_append = [
            {
                "subreddit": subreddit,
                "title": title,
                "initial_post": "",
                "post": submission_text,
                "score": score,
                "date_created": date_created,
                "permalink": submission_permalink,
                "author_replies": submission_author_comments["body"].to_list(),
            }
        ]
        to_append += [
            {
                "subreddit": subreddit,
                "title": title,
                "initial_post": submission_text,
                "post": str(submission_comments.iloc[j]["body"]),
                "score": str(submission_comments.iloc[j]["score"]),
                "date_created": get_date(
                    int(submission_comments.iloc[j]["created_utc"])
                ),
                "permalink": submission_comments.iloc[j]["permalink"],
                "author_replies": [],
            }
            for j in range(len(submission_comments))
        ]
        # to_append = check_treatment_mention(to_append, treatment_names)
        # to_append = check_outcome_mention(to_append, outcome_words)
        if len(to_append) > 0:
            df_to_append = pd.DataFrame.from_dict(to_append)
            merged_df = pd.concat([merged_df, df_to_append], ignore_index=True)
    return merged_df


def get_reddit_synonyms(keywords, llm):
    system_prompt = "You are a helpful medical assistant who can translate medical terminology into common terms."
    user_prompts = [
        f"What are common brand names or terms that people specifically use when discussing any of {str(keywords)}, especially on platforms like Reddit?"
    ]
    user_prompts = [
        prompt
        + " Return only a Python list of at most 10 individual words, without any other text or formatting."
        for prompt in user_prompts
    ]
    llm_keywords = llm.get_outputs(system_prompt, user_prompts)
    all_keywords = []
    # process outputs into a list of individual words
    for keyword in llm_keywords:
        processed_keyword = (
            keyword[keyword.find("[") + 1 : keyword.find("]")]
            .replace('"', "")
            .replace("'", "")
        )
        keyword_list = [k.strip() for k in processed_keyword.split(",")]
        all_keywords.extend(keyword_list)
        # for k in keyword_list:
        #     words = k.split()
        #     all_keywords.extend(words)
    print(all_keywords)
    return [k.lower() for k in all_keywords]


def subreddit_relevance_llm(desc, keywords, lm):
    system_prompt = "You are an expert in analyzing online forums for relevant discussions about clinical conditions."
    system_prompt += "Your task is to determine if a subreddit is likely to contain personal experiences related to a set of related conditions based on the subreddit description."

    user_prompt = "Evaluate the subreddit relevance to the conditions listed. Consider if the subreddit likely contains personal experiences with the condition. Answer Yes if relevant, No if not."
    user_prompt += (
        f"\n\n**Conditions:** {keywords}\n\n**Subreddit Description:** {desc}"
    )
    user_prompt += "Is the subreddit likely to contain personal experiences relevant to the listed conditions? Answer with Yes or No."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return lm(messages=messages)[0]
