import google.generativeai as genai

from backend import redis_layer
from backend.agents import _gemini
from backend.config import GOOGLE_API_KEY, PIPELINE_VERSION
from backend.evidence import join_passages, passages_by_chunk
from backend.schemas import RawResult

genai.configure(api_key=GOOGLE_API_KEY)


def _search_tool() -> genai.protos.Tool:
    # Gemini 2.0/2.5 grounding uses the `google_search` tool. The older
    # `google_search_retrieval` (dynamic retrieval) is Gemini 1.5 only and the
    # API rejects it for 2.x models.
    return genai.protos.Tool(google_search=genai.protos.Tool.GoogleSearch())


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
                snippet=response.text[:500],
                evidence_status="model_only",
            )
        )
    return results


async def run(sub_question: str) -> tuple[list[RawResult], bool]:
    """Returns (results, cache_hit)."""
    cached = await redis_layer.cache_get(sub_question, PIPELINE_VERSION)
    if cached is not None:
        return [RawResult(**r) for r in cached], True

    def make_model(name: str):
        return genai.GenerativeModel(model_name=name, tools=[_search_tool()])

    response = await _gemini.generate(
        make_model,
        f"Research this question and provide detailed findings:\n{sub_question}",
        stage="search",
    )
    results = parse_grounded(response)
    # Model-only fallbacks are not cached: next time grounding may succeed.
    if any(r.evidence_status != "model_only" for r in results):
        await redis_layer.cache_set(
            sub_question, PIPELINE_VERSION, [r.model_dump() for r in results]
        )
    return results, False
