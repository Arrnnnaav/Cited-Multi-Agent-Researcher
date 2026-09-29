import asyncio
import json
import time
import uuid
from typing import Literal

from backend import records
from backend import llm
from backend.agents import _gemini, citation_agent, search_agent, synthesis_agent
from backend.config import PIPELINE_VERSION, SUBAGENT_CAP
from backend.eval.citation_checks import check_citations
from backend.schemas import ResearchResponse


def _get_subagent_count(query_type: str, num_subtopics: int) -> int:
    if query_type == "fact":
        return 1
    return min(num_subtopics, SUBAGENT_CAP)


async def _classify(query: str) -> Literal["fact", "comparison"]:
    text = await llm.generate_text(
        f'Classify this query as exactly "fact" or "comparison" (one word only):\n{query}',
        stage="classify",
    )
    text = text.strip().lower()
    return "comparison" if "comparison" in text else "fact"


async def _decompose(query: str, query_type: str) -> list[str]:
    if query_type == "fact":
        return [query]
    text = await llm.generate_text(
        f"""Break this comparison query into sub-questions, one per comparison axis.
Return a JSON array of strings only. No markdown.
Query: {query}""",
        stage="decompose",
    )
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return [query]
    if (
        not isinstance(parsed, list)
        or not all(isinstance(q, str) for q in parsed)
        or not parsed
    ):
        return [query]
    return parsed


async def run(query: str, trace: bool = True) -> ResearchResponse:
    start = time.monotonic()
    run_id = uuid.uuid4().hex[:16]
    calls: list[dict] = []
    token = _gemini.call_log.set(calls)
    stages: dict[str, int] = {}
    rec = {"run_id": run_id, "config_version": PIPELINE_VERSION, "query": query,
           "model_calls": calls, "stages": stages, "status": "error"}  # fmt: skip

    def mark(name: str, t0: float) -> None:
        stages[name] = int((time.monotonic() - t0) * 1000)

    try:
        t0 = time.monotonic()
        query_type = await _classify(query)
        mark("classify", t0)
        t0 = time.monotonic()
        sub_questions = await _decompose(query, query_type)
        mark("decompose", t0)
        n = _get_subagent_count(query_type, len(sub_questions))
        rec.update(query_type=query_type, subquestions=sub_questions[:n])

        t0 = time.monotonic()
        searched = await asyncio.gather(
            *[search_agent.run(q) for q in sub_questions[:n]]
        )
        mark("search", t0)
        raw_results = [r for results, _ in searched for r in results]
        cache_hits = sum(hit for _, hit in searched)

        sources = citation_agent.run(raw_results)
        t0 = time.monotonic()
        answer = await synthesis_agent.run(query, sources)
        mark("synthesis", t0)
        findings = check_citations(answer, sources)

        latency_ms = int((time.monotonic() - start) * 1000)
        rec.update(
            sources=[s.model_dump() for s in sources], answer=answer,
            findings=[f.model_dump() for f in findings], cache_hits=cache_hits,
            latency_ms=latency_ms, status="ok",
        )  # fmt: skip
        return ResearchResponse(
            answer=answer,
            sources=sources,
            query_type=query_type,
            subagents_used=n,
            latency_ms=latency_ms,
            run_id=run_id,
            findings=findings,
            cache_hits=cache_hits,
        )
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}"[:500],
                   latency_ms=int((time.monotonic() - start) * 1000))  # fmt: skip
        raise
    finally:
        _gemini.call_log.reset(token)
        if trace:
            try:
                await records.save_run(rec)
            except Exception:  # tracing must never break a request
                pass
