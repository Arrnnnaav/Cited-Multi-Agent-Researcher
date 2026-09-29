import logging

from backend import config, redis_layer, search_providers
from backend.agents import _gemini
from backend.evidence import join_passages, passages_by_chunk
from backend.schemas import RawResult


def parse_grounded(response) -> list[RawResult]:
    """One RawResult per grounding chunk, carrying only the passages that the
    grounding metadata attributes to that chunk."""
    results: list[RawResult] = []
    try:
        metadata = response.candidates[0].grounding_metadata
        chunks = list(metadata.grounding_chunks or []) if metadata else []
        passages = passages_by_chunk(metadata) if metadata else {}
    except (IndexError, AttributeError):
        chunks, passages = [], {}
    for idx, chunk in enumerate(chunks):
        web = getattr(chunk, "web", None)
        if not web:
            continue
        own = passages.get(idx, [])
        results.append(
            RawResult(
                url=web.uri or "",
                title=web.title or web.uri or "Unknown",
                snippet=join_passages(own),
                evidence_status="grounded" if own else "metadata_only",
            )
        )
    if not results:
        # No web grounding: keep the text, but label it for what it is.
        results.append(
            RawResult(
                url="",
                title="Model answer (no web source)",
                snippet=(response.text or "")[:500],
                evidence_status="model_only",
            )
        )
    return results


log = logging.getLogger(__name__)


async def _gemini_search(sub_question: str) -> list[RawResult]:
    response = await _gemini.generate(
        f"Research this question and provide detailed findings:\n{sub_question}",
        stage="search",
        search=True,
    )
    return parse_grounded(response)


async def _search(sub_question: str) -> list[RawResult]:
    if config.SEARCH_PROVIDER == "tavily":
        try:
            results = await search_providers.tavily_search(sub_question)
            if results:
                return results
        except search_providers.SearchError as e:
            log.warning("tavily failed (%s)", e)
        if not config.GOOGLE_API_KEY:
            return []
        log.warning("falling back to Gemini grounding for %r", sub_question[:60])
    return await _gemini_search(sub_question)


async def run(sub_question: str) -> tuple[list[RawResult], bool]:
    """Returns (results, cache_hit)."""
    version = f"{config.PIPELINE_VERSION}:{config.SEARCH_PROVIDER}"
    cached = await redis_layer.cache_get(sub_question, version)
    if cached is not None:
        return [RawResult(**r) for r in cached], True
    results = await _search(sub_question)
    # Model-only fallbacks are not cached: next time grounding may succeed.
    if any(r.evidence_status != "model_only" for r in results):
        await redis_layer.cache_set(
            sub_question, version, [r.model_dump() for r in results]
        )
    return results, False
