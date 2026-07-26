import asyncio
import json
import time
from typing import Literal

import google.generativeai as genai

from backend.config import GOOGLE_API_KEY, SUBAGENT_CAP
from backend.schemas import ResearchResponse
from backend.agents import citation_agent, search_agent, synthesis_agent, _gemini

genai.configure(api_key=GOOGLE_API_KEY)


def _get_subagent_count(query_type: str, num_subtopics: int) -> int:
    if query_type == "fact":
        return 1
    return min(num_subtopics, SUBAGENT_CAP)


async def _classify(query: str) -> Literal["fact", "comparison"]:
    response = await _gemini.generate(
        lambda name: genai.GenerativeModel(model_name=name),
        f'Classify this query as exactly "fact" or "comparison" (one word only):\n{query}',
    )
    text = response.text.strip().lower()
    return "comparison" if "comparison" in text else "fact"


async def _decompose(query: str, query_type: str) -> list[str]:
    if query_type == "fact":
        return [query]
    response = await _gemini.generate(
        lambda name: genai.GenerativeModel(model_name=name),
        f"""Break this comparison query into sub-questions, one per comparison axis.
Return a JSON array of strings only. No markdown.
Query: {query}""",
    )
    text = response.text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return [query]


async def run(query: str) -> ResearchResponse:
    start = time.monotonic()

    query_type = await _classify(query)
    sub_questions = await _decompose(query, query_type)
    n = _get_subagent_count(query_type, len(sub_questions))

    raw_results_nested = await asyncio.gather(
        *[search_agent.run(q) for q in sub_questions[:n]]
    )
    raw_results = [r for sublist in raw_results_nested for r in sublist]

    sources = citation_agent.run(raw_results)
    answer = await synthesis_agent.run(query, sources)

    latency_ms = int((time.monotonic() - start) * 1000)
    return ResearchResponse(
        answer=answer,
        sources=sources,
        query_type=query_type,
        subagents_used=n,
        latency_ms=latency_ms,
    )
