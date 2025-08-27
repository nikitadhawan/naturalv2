"""Utilities for retrieving and processing PubMed case reports."""

import ast
import json
import os
import xml.etree.ElementTree as ET

import requests


def search_pubmed(query, api_key):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 10000,
        "usehistory": "y",
        "api_key": api_key,
    }
    response = requests.get(search_url, params=params)
    root = ET.fromstring(response.content)

    webenv = root.find("WebEnv").text
    query_key = root.find("QueryKey").text
    return webenv, query_key


def fetch_articles(webenv, query_key, api_key, data_path):
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retmax": 10000,
        "retmode": "xml",
        "api_key": api_key,
    }

    response = requests.get(fetch_url, params=params)
    root = ET.fromstring(response.content)
    case_reports = []
    for article in root.findall(".//PubmedArticle"):
        article_data = {}
        # Check if it's a case report
        is_case_report = False
        for pub_type in article.findall(".//PublicationType"):
            if pub_type.text.lower() == "case reports":
                is_case_report = True
        assert is_case_report
        # Extract basic metadata
        article_data["pmid"] = article.find(".//PMID").text
        article_data["title"] = article.find(".//ArticleTitle").text
        article_data["abstract"] = article.find(".//Abstract/AbstractText")
        article_data["abstract"] = (
            article_data["abstract"].text
            if article_data["abstract"] is not None
            else "No abstract available"
        )
        article_data["abstract"] = (
            article_data["abstract"]
            if article_data["abstract"] is not None
            else "No abstract available"
        )
        # Extract authors
        article_data["authors"] = []
        for author in article.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None and fore_name is not None:
                article_data["authors"].append(f"{fore_name.text} {last_name.text}")
        # Extract publication date
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.find("Year")
            month = pub_date.find("Month")
            day = pub_date.find("Day")
            article_data["publication_date"] = (
                f"{year.text if year is not None else ''}-{month.text if month is not None else ''}-{day.text if day is not None else ''}"
            )
        # Check for PMC full-text
        article_id_list = article.find(".//ArticleIdList")
        if article_id_list is not None:
            for article_id in article_id_list.findall("ArticleId"):
                if article_id.get("IdType") == "pmc":
                    article_data["pmc_id"] = article_id.text
                    try:
                        article_data["full_text"] = fetch_pmc_fulltext(
                            data_path, article_data["pmc_id"], api_key
                        )
                    except Exception as e:
                        print(
                            f"Error retrieving full text for PMC{article_data['pmc_id']}: {str(e)}"
                        )
                    break
        case_reports.append(article_data)
    return case_reports


def get_cached_fulltext(cache_dir, pmc_id):
    cache_file = os.path.join(cache_dir, f"{pmc_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def cache_fulltext(cache_dir, pmc_id, fulltext):
    cache_file = os.path.join(cache_dir, f"{pmc_id}.json")
    with open(cache_file, "w") as f:
        json.dump(fulltext, f)


def parse_section(section):
    content = []
    title = section.find("title")
    if title is not None:
        content.append(f"## {title.text}")
    for child in section:
        if child.tag == "p":
            content.append(
                ET.tostring(child, encoding="unicode", method="text").strip()
            )
        elif child.tag == "sec":
            content.extend(parse_section(child))
    return content


def fetch_pmc_fulltext(data_path, pmc_id, api_key):
    cache_dir = data_path + "pmc_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cached_fulltext = get_cached_fulltext(cache_dir, pmc_id)
    if cached_fulltext:
        return cached_fulltext

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmc_id,
        "rettype": "fulltext",
        "retmode": "xml",
        "api_key": api_key,
    }
    response = requests.get(fetch_url, params=params)
    root = ET.fromstring(response.content)
    article = root.find(".//article")
    if article is None:
        return None
    structured_text = {}
    # Extract front matter (title, abstract)
    front = article.find("front")
    if front is not None:
        title = front.find(".//article-title")
        if title is not None:
            structured_text["title"] = title.text
        abstract = front.find(".//abstract")
        if abstract is not None:
            structured_text["abstract"] = "\n".join(parse_section(abstract))
    # Extract body sections
    body = article.find("body")
    if body is not None:
        structured_text["body"] = []
        for section in body.findall("sec"):
            structured_text["body"].extend(parse_section(section))
    # Extract back matter (references, appendices)
    back = article.find("back")
    if back is not None:
        structured_text["back"] = []
        for section in back.findall("sec"):
            structured_text["back"].extend(parse_section(section))
    cache_fulltext(cache_dir, pmc_id, structured_text)
    return structured_text


def pubmed_queries_llm(trial, llm):
    system_prompt = "You are a language model that generates a plain Python list output of keywords for PubMed case report searches related to a clinical trial. "
    system_prompt += "Use the trial's title, keywords, conditions, interventions, and endpoints to create the list. "
    system_prompt += "Include synonyms and abbreviations where relevant."
    user_prompt = f"""
    Generate a Python list of 10 search keywords for PubMed case reports using the following clinical trial details:

    - **Title:** {trial.brief_title}
    - **Keywords:** {trial.keywords}
    - **Conditions:** {trial.conditions}
    - **Interventions:** {trial.interventions}
    - **Primary Endpoint(s):** {trial.primary_endpoints}

    The list should contain relevant synonyms and variations without any additional text or formatting.
    """
    answer = llm.get_outputs(system_prompt, [user_prompt])
    return ast.literal_eval(answer)
