import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import aiohttp
from lxml import etree as ET  # noqa: N812
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_delay,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm

from naturalv2.utils import concurrency_limited


logger = logging.getLogger(__name__)

_PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_RETMAX = "100000"  # Maximum number of records to return in a single request
_MAX_GET_TERM_LENGTH = 1800  # PubMed GET requests start failing beyond ~2KB URLs


_MAX_RATE_LIMIT_WAIT_SECONDS = 300  # Cap server-directed sleeps to 5 minutes.


def _should_retry(exception: BaseException) -> bool:
    """Determine if the exception is transient and should be retried."""
    if isinstance(exception, aiohttp.ClientResponseError):
        if exception.status == 429:
            return True
        return 500 <= exception.status < 600
    if isinstance(exception, aiohttp.ClientError):
        return True
    return isinstance(exception, asyncio.TimeoutError)


async def search_pubmed(
    query: str, api_key: str | None, executor: ThreadPoolExecutor
) -> tuple[str | None, str | None]:
    """Search PubMed for the given query and return WebEnv and QueryKey.

    Returns (WebEnv, QueryKey) or (None, None) on failure.

    Parameters
    ----------
    query : str
        The search query string.
    api_key : str | None
        Optional PubMed API key.
    executor : ThreadPoolExecutor
        Executor for blocking operations.

    Returns
    -------
    tuple[str | None, str | None]
        A tuple containing WebEnv and QueryKey, or (None, None) on failure.
    """
    search_url = _PUBMED_BASE_URL + "esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": _RETMAX,
        "usehistory": "y",
    }

    pubmed_api_key = api_key if api_key else os.getenv("PUBMED_API_KEY")
    if pubmed_api_key:
        params["api_key"] = pubmed_api_key

    http_method = "POST" if len(query) > _MAX_GET_TERM_LENGTH else "GET"

    try:
        async with aiohttp.ClientSession() as session:
            root = await _get_xml_root(
                search_url,
                params,
                "Failed to retrieve PubMed search results",
                executor,
                session,
                http_method=http_method,
            )
    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            logger.warning(
                "PubMed rate limit exhausted while searching query '%s'; skipping.",
                query,
            )
            return None, None
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError):
        log_params = {k: v for k, v in params.items() if k != "api_key"}
        logger.warning(
            "PubMed network failure after retries while searching query '%s'; params=%s",
            query,
            log_params,
        )
        return None, None
    if root is None:
        return None, None

    # Extract WebEnv and QueryKey
    webenv = root.find("WebEnv")
    query_key = root.find("QueryKey")
    if webenv is None or query_key is None:
        logger.error("WebEnv or QueryKey not found in PubMed response.")
        return None, None

    return webenv.text, query_key.text


async def fetch_articles(
    webenv: str | None,
    query_key: str | None,
    data_path: str,
    api_key: str | None,
    executor: ThreadPoolExecutor,
) -> list[dict[str, str]]:
    """Fetch articles from PubMed using WebEnv and QueryKey.

    Parameters
    ----------
    webenv : str | None
        The WebEnv string from PubMed search results.
    query_key : str | None
        The QueryKey string from PubMed search results.
    data_path : str
        Path to store cached full texts.
    api_key : str | None
        Optional PubMed API key.
    executor : ThreadPoolExecutor
        Executor for blocking operations.

    Returns
    -------
    list[dict[str, str]]
        A list of dictionaries containing article metadata and full text.
    """
    if webenv is None:
        logger.error("WebEnv is None, cannot fetch articles.")
        return []

    if query_key is None:
        logger.error("QueryKey is None, cannot fetch articles.")
        return []

    fetch_url = _PUBMED_BASE_URL + "efetch.fcgi"
    params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retmax": _RETMAX,
        "retmode": "xml",
    }

    pubmed_api_key = api_key if api_key else os.getenv("PUBMED_API_KEY")
    if pubmed_api_key:
        params["api_key"] = pubmed_api_key

    try:
        async with aiohttp.ClientSession() as session:
            root = await _get_xml_root(
                fetch_url,
                params,
                "Error fetching articles from PubMed",
                executor,
                session,
            )
    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            logger.warning(
                "PubMed rate limit exhausted while fetching article batch (query_key=%s); skipping batch.",
                query_key,
            )
            return []
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError):
        log_params = {k: v for k, v in params.items() if k != "api_key"}
        logger.warning(
            "PubMed network failure after retries while fetching article batch (query_key=%s); params=%s",
            query_key,
            log_params,
        )
        return []
    if root is None:
        return []

    # Extract case reports
    articles = root.findall(".//PubmedArticle")

    return await _extract_case_reports(articles, data_path, api_key, executor)


async def _extract_case_reports(
    articles: list[ET.Element],
    data_path: str,
    api_key: str | None,
    executor: ThreadPoolExecutor,
) -> list[dict[str, str]]:
    """Extract case reports from the list of articles."""
    case_reports: list[dict[str, str]] = []
    batch_size = 100
    loop = asyncio.get_running_loop()
    for i in range(0, len(articles), batch_size):
        batch = articles[i : min(i + batch_size, len(articles))]

        # Create all metadata extraction tasks
        tasks = [
            loop.run_in_executor(executor, _extract_common_metadata, article)
            for article in batch
            if _is_case_report(article)
        ]

        # Wait for all CPU-bound metadata tasks to complete
        all_metadata = await tqdm.gather(
            *tasks, desc="Extracting metadata", unit="article", leave=False, position=1
        )

        # Create I/O-bound full-text fetching tasks
        full_text_fetch_tasks = []
        for article_data, pmc_ids in all_metadata:
            if not article_data:
                continue

            if pmc_ids:
                task = _fetch_and_process_fulltexts(
                    pmc_ids,
                    article_data,
                    data_path,
                    api_key,
                    executor,
                )
                full_text_fetch_tasks.append(task)
            else:
                case_reports.append(article_data)

        # Run all I/O-bound tasks concurrently
        processed_articles = await tqdm.gather(
            *full_text_fetch_tasks,
            desc="Fetching full texts",
            unit="article",
            leave=False,
            position=2,
            dynamic_ncols=True,
        )
        for processed_article_list in processed_articles:
            case_reports.extend(processed_article_list)

    return case_reports


async def _fetch_and_process_fulltexts(
    pmc_ids: list[str],
    article_data: dict[str, str],
    data_path: str,
    api_key: str | None,
    executor: ThreadPoolExecutor,
) -> list[dict[str, str]]:
    """Fetch and process full texts for the given PMC IDs."""

    async def _fetch_with_id(pmc_id: str) -> tuple[str, dict[str, str] | None]:
        result = await _fetch_pmc_fulltext(pmc_id, data_path, api_key, executor)
        return pmc_id, result

    semaphore = asyncio.Semaphore(10)
    tasks = [
        asyncio.create_task(
            concurrency_limited(_fetch_with_id(pmc_id), semaphore),
            name=f"fetch_pmc_{pmc_id}",
        )
        for pmc_id in pmc_ids
    ]

    full_text_articles = []
    for coro in asyncio.as_completed(tasks):
        try:
            pmc_id, structured_text = await coro
            if structured_text:
                full_text = "\n\n".join(
                    [
                        structured_text.get("title", ""),
                        structured_text.get("abstract", ""),
                        structured_text.get("body", ""),
                        structured_text.get("back", ""),
                    ]
                )
                article_data["pmc_id"] = pmc_id
                article_data["full_text"] = full_text.strip()
                full_text_articles.append(article_data)
        except Exception as e:
            logger.error(
                f"Unexpected error while fetching PMC full text: {e}", exc_info=True
            )
    return full_text_articles


async def _fetch_pmc_fulltext(
    pmc_id: str,
    data_path: str,
    api_key: str | None,
    executor: ThreadPoolExecutor,
) -> dict[str, str] | None:
    """Fetch and process the full text of a PMC article by its PMC ID."""
    # Return cached full text if available
    cache_dir = os.path.join(data_path, ".pmc_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{pmc_id}.json")

    loop = asyncio.get_running_loop()
    cached_fulltext = await loop.run_in_executor(executor, _read_cache_file, cache_file)
    if cached_fulltext:
        return cached_fulltext

    # If not cached, fetch from PMC
    fetch_url = _PUBMED_BASE_URL + "efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmc_id,
        "rettype": "fulltext",
        "retmode": "xml",
    }
    pubmed_api_key = api_key if api_key else os.getenv("PUBMED_API_KEY")
    if pubmed_api_key:
        params["api_key"] = pubmed_api_key

    try:
        async with aiohttp.ClientSession() as session:
            root = await _get_xml_root(
                fetch_url,
                params,
                "Error fetching full text from PMC",
                executor,
                session,
            )
    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            logger.warning(
                "PubMed rate limit exhausted while fetching PMC full text for %s; skipping full text.",
                pmc_id,
            )
            return None
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError):
        log_params = {k: v for k, v in params.items() if k != "api_key"}
        logger.warning(
            "PubMed network failure after retries while fetching PMC full text for %s; params=%s",
            pmc_id,
            log_params,
        )
        return None
    if root is None:
        return None

    # Parse the XML to structured text
    structured_text = await loop.run_in_executor(executor, _parse_fulltext_xml, root)

    if structured_text:
        # Cache the full text
        await loop.run_in_executor(
            executor, _write_cache_file, cache_file, structured_text
        )

    return structured_text


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_delay(600),
    retry=retry_if_exception(_should_retry),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
    reraise=True,
)
async def _get_xml_root(
    url: str,
    params: dict[str, str],
    client_error_msg: str,
    executor: ThreadPoolExecutor,
    session: aiohttp.ClientSession,
    *,
    http_method: str = "GET",
) -> ET.Element | None:
    """Fetch XML content from the given URL with parameters and return the root element."""
    method = http_method.upper()
    if method not in {"GET", "POST"}:
        raise ValueError(f"Unsupported HTTP method: {http_method}")

    request_kwargs: dict[str, dict[str, str]]
    request_kwargs = {"params": params} if method == "GET" else {"data": params}

    try:
        requester = session.get if method == "GET" else session.post
        async with requester(url, **request_kwargs) as response:
            response.raise_for_status()
            content = await response.read()

        # Parse XML content in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _parse_xml_content, content)
    except aiohttp.ClientResponseError as e:
        log_params = {k: v for k, v in params.items() if k != "api_key"}
        if e.status == 400:
            logger.debug(
                "Bad request for URL %s with params %s: %s", url, log_params, e.message
            )
            return None
        if e.status != 429 and not (500 <= e.status < 600):
            logger.error(
                "%s: %s - %s, URL: %s, Params: %s",
                client_error_msg,
                e.status,
                e.message,
                url,
                log_params,
            )
        raise


def _is_case_report(article: ET.Element) -> bool:
    """Check if the article is a case report based on its PublicationType."""
    return any(
        pub_type.text and pub_type.text.lower() == "case reports"
        for pub_type in article.findall(".//PublicationType")
    )


def _extract_common_metadata(
    article: ET.Element,
) -> tuple[dict[str, str] | None, list[str]]:
    """Process article metadata synchronously in thread pool."""
    article_data = _extract_basic_metadata(article)
    if article_data is None:
        return None, []

    article_data["abstract"] = _extract_abstract(article)
    article_data["authors"] = _extract_authors(article)
    pub_date = _extract_publication_date(article)
    if pub_date:
        article_data["publication_date"] = pub_date

    pmc_ids = _extract_pmc_ids(article)

    return article_data, pmc_ids


def _extract_basic_metadata(article: ET.Element) -> dict[str, str] | None:
    """Extract basic metadata (PMID and title) from the article."""
    pmid_element = article.find(".//PMID")
    title_element = article.find(".//ArticleTitle")

    if pmid_element is None or not pmid_element.text:
        logger.debug("PMID is empty, skipping article.")
        return None

    if title_element is None or not title_element.text:
        logger.debug("Title is empty, skipping article.")
        return None

    return {"pmid": pmid_element.text, "title": title_element.text}


def _extract_abstract(article: ET.Element) -> str:
    """Extract the abstract text from the article."""
    abstract_element = article.find(".//Abstract/AbstractText")
    return (
        abstract_element.text.strip()
        if abstract_element is not None and abstract_element.text is not None
        else "No abstract available"
    )


def _extract_authors(article: ET.Element) -> str:
    """Extract the list of authors from the article."""
    authors = []
    for author in article.findall(".//Author"):
        last_name_elem = author.find("LastName")
        fore_name_elem = author.find("ForeName")
        if (
            last_name_elem is not None
            and fore_name_elem is not None
            and (last_name_elem.text and fore_name_elem.text)
        ):
            author_name = f"{fore_name_elem.text} {last_name_elem.text}"
            authors.append(author_name)
    return ", ".join(authors) if authors else "No authors listed"


def _extract_publication_date(article: ET.Element) -> str:
    """Extract the publication date from the article."""
    pub_date_elem = article.find(".//PubDate")
    if pub_date_elem is not None:
        year = pub_date_elem.find("Year")
        month = pub_date_elem.find("Month")
        day = pub_date_elem.find("Day")
        return f"{year.text if year is not None else ''}-{month.text if month is not None else ''}-{day.text if day is not None else ''}"
    return ""


def _extract_pmc_ids(article: ET.Element) -> list[str]:
    """Extract PMC IDs from the article."""
    article_id_list = article.find(".//ArticleIdList")
    pmc_ids = []
    if article_id_list is not None:
        for article_id in article_id_list.findall("ArticleId"):
            if article_id.get("IdType") == "pmc" and article_id.text:
                pmc_ids.append(article_id.text)
    return pmc_ids


def _parse_fulltext_xml(root: ET.Element) -> dict[str, str]:
    """Parse the full text XML and return structured text sections."""
    article = root.find(".//article")
    if article is None:
        return {}

    structured_text = {}
    # Extract front matter (title, abstract)
    front = article.find("front")
    if front is not None:
        title = front.find(".//article-title")
        if title is not None and title.text:
            structured_text["title"] = title.text
        abstract = front.find(".//abstract")
        if abstract is not None:
            structured_text["abstract"] = "\n".join(_parse_section(abstract))

    # Extract body sections
    body = article.find("body")
    if body is not None:
        body_sections = []
        for section in body.findall("sec"):
            body_sections.extend(_parse_section(section))
        structured_text["body"] = "\n".join(body_sections)

    # Extract back matter (references, appendices)
    back = article.find("back")
    if back is not None:
        back_sections = []
        for section in back.findall("sec"):
            back_sections.extend(_parse_section(section))
        structured_text["back"] = "\n".join(back_sections)

    return structured_text


def _parse_section(section: ET.Element) -> list[str]:
    """Recursively parse a section element and return its content as a list of strings."""
    content = []
    title = section.find("title")
    if title is not None and title.text:
        content.append(f"## {title.text}")

    for child in section:
        if child.tag == "p":
            content.append(
                ET.tostring(child, encoding="unicode", method="text").strip()
            )
        elif child.tag == "sec":
            content.extend(_parse_section(child))
    return content


def _read_cache_file(cache_file: str) -> dict[str, Any] | None:
    """Synchronous cache file reading to be run in thread pool."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading cache file {cache_file}: {e}")
            return None
    return None


def _write_cache_file(cache_file: str, data: dict[str, str]) -> None:
    """Synchronous cache file writing to be run in thread pool."""
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
    except IOError as e:
        logger.error(f"Error writing cache file {cache_file}: {e}")


def _parse_xml_content(content: bytes) -> ET.Element | None:
    """Parse XML content and return the root element."""
    try:
        return ET.fromstring(content)
    except ET.ParseError as e:
        logger.error(f"Error parsing XML: {e}")
        return None
