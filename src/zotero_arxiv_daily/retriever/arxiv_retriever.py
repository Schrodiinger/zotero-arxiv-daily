from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    processing_delay = 0

    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        entries = [
            entry for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            if entries:
                entries = entries[:5]
                logger.info(f"Debug mode: using {len(entries)} papers from RSS.")
            else:
                logger.warning(
                    "Debug mode enabled, but RSS returned no papers. "
                    "Falling back to the 5 most recent arXiv papers."
                )
                client = arxiv.Client(num_retries=1, delay_seconds=3)
                fallback_query = " OR ".join(
                    f"cat:{category}"
                    for category in self.config.source.arxiv.category
                )
        
                fallback_search = arxiv.Search(
                    query=fallback_query,
                    max_results=5,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )
        
                try:
                    fallback_results = list(client.results(fallback_search))
                    logger.info(
                        f"Debug fallback retrieved {len(fallback_results)} recent arXiv papers."
                    )
                    return fallback_results
                except Exception as exc:
                    logger.error(f"Debug fallback failed: {exc}")
                    return []

        raw_papers = [self._result_from_rss_entry(entry) for entry in entries]
        logger.info(f"Using arXiv RSS metadata for {len(raw_papers)} candidate papers.")
        return raw_papers

    @staticmethod
    def _result_from_rss_entry(entry) -> ArxivResult:
        paper_id = entry.id.removeprefix("oai:arXiv.org:")
        versionless_id = paper_id.rsplit("v", 1)[0] if paper_id.rsplit("v", 1)[-1].isdigit() else paper_id
        entry_id = entry.get("link") or f"https://arxiv.org/abs/{versionless_id}"
        summary = entry.get("summary", "")
        if "Abstract:" in summary:
            summary = summary.split("Abstract:", 1)[1].strip()

        authors = [
            ArxivResult.Author(name.strip())
            for author in entry.get("authors", [])
            for name in author.get("name", "").split(",")
            if name.strip()
        ]
        if not authors and entry.get("dc_creator"):
            creator = entry.get("dc_creator")
            author_names = creator if isinstance(creator, list) else [creator]
            authors = [
                ArxivResult.Author(name.strip())
                for value in author_names
                for name in value.split(",")
                if name.strip()
            ]
        links = [
            ArxivResult.Link(
                href=link.get("href", ""),
                title=link.get("title"),
                rel=link.get("rel", ""),
                content_type=link.get("type"),
            )
            for link in entry.get("links", [])
            if link.get("href")
        ]
        if not any(link.title == "pdf" for link in links):
            links.append(ArxivResult.Link(f"https://arxiv.org/pdf/{versionless_id}", title="pdf"))
        categories = [tag.get("term", "") for tag in entry.get("tags", []) if tag.get("term")]
        primary = entry.get("arxiv_primary_category", {}).get("term", "")
        return ArxivResult(
            entry_id=entry_id,
            title=entry.get("title", ""),
            authors=authors,
            summary=summary,
            primary_category=primary or (categories[0] if categories else ""),
            categories=categories,
            links=links,
        )

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[a.name for a in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
        )

    def enrich_paper(self, paper: Paper) -> None:
        """Extract full text only after this paper survives reranking."""
        raw_paper = ArxivResult(
            entry_id=paper.url,
            title=paper.title,
            links=[ArxivResult.Link(paper.pdf_url, title="pdf")] if paper.pdf_url else [],
        )
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        paper.full_text = full_text


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
