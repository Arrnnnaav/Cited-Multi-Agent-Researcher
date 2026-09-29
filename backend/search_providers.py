"""Tavily web search: one result per URL with text extracted from that page.

Compared with Gemini grounding this gives source-specific evidence directly:
each result's `content` is the most relevant excerpt of that page, so a
citation [N] is checked against text that actually came from source N. It
also needs no LLM call for the search step.
"""

from __future__ import annotations

import time

import httpx

from backend.agents._gemini import _record
from backend.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS
from backend.evidence import MAX_PASSAGE_CHARS
from backend.schemas import RawResult

TAVILY_URL = "https://api.tavily.com/search"

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30)
    return _client


class SearchError(RuntimeError):
    pass


def parse_tavily(data: dict) -> list[RawResult]:
    out = []
    for r in data.get("results") or []:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        content = " ".join((r.get("content") or "").split())[:MAX_PASSAGE_CHARS]
        out.append(
            RawResult(
                url=url,
                title=r.get("title") or url,
                snippet=content,
                evidence_status="grounded" if content else "metadata_only",
            )
        )
    return out


async def tavily_search(query: str, max_results: int | None = None) -> list[RawResult]:
    if not TAVILY_API_KEY:
        raise SearchError("TAVILY_API_KEY not set")
    started = time.perf_counter()
    try:
        r = await client().post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={
                "query": query[:400],
                "max_results": max_results or TAVILY_MAX_RESULTS,
                "search_depth": "basic",
                "include_answer": False,
            },
        )
    except httpx.TransportError as e:
        _record("tavily", "search", started, "unavailable")
        raise SearchError(f"tavily transport error: {e}") from e
    if r.status_code >= 400:
        _record("tavily", "search", started, f"http_{r.status_code}")
        raise SearchError(f"tavily HTTP {r.status_code}: {r.text[:200]}")
    _record("tavily", "search", started, "ok")
    return parse_tavily(r.json())
