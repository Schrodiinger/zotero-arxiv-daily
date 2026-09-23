"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(
        arxiv_retriever.arxiv,
        "Client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("arXiv API should not be called")),
    )

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    allowed_types = {"new", "cross"} if config.source.arxiv.include_cross_list else {"new"}
    expected_entries = [
        entry for entry in mock_feedparser.entries
        if entry.get("arxiv_announce_type", "new") in allowed_types
    ]
    assert len(papers) == len(expected_entries)
    assert {paper.title for paper in papers} == {entry.title for entry in expected_entries}
    assert papers[0].authors == ["Chunhua Liu", "Kabir Manandhar Shrestha", "Sukai Huang"]
    assert papers[0].abstract.startswith("As large language models")
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2508.13426"


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
