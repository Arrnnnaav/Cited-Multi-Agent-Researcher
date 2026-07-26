import google.generativeai as genai
from backend.config import GOOGLE_API_KEY
from backend.schemas import RawResult
from backend.agents import _gemini

genai.configure(api_key=GOOGLE_API_KEY)


def _search_tool() -> genai.protos.Tool:
    # Gemini 2.0/2.5 grounding uses the `google_search` tool. The older
    # `google_search_retrieval` (dynamic retrieval) is Gemini 1.5 only and the
    # API rejects it for 2.x models.
    return genai.protos.Tool(google_search=genai.protos.Tool.GoogleSearch())


async def run(sub_question: str) -> list[RawResult]:
    def make_model(name: str):
        return genai.GenerativeModel(model_name=name, tools=[_search_tool()])

    response = await _gemini.generate(
        make_model,
        f"Research this question and provide detailed findings:\n{sub_question}",
    )

    results: list[RawResult] = []
    try:
        metadata = response.candidates[0].grounding_metadata
        if metadata and metadata.grounding_chunks:
            for chunk in metadata.grounding_chunks:
                if chunk.web:
                    results.append(
                        RawResult(
                            url=chunk.web.uri or "",
                            title=chunk.web.title or chunk.web.uri or "Unknown",
                            snippet=response.text[:300],
                        )
                    )
    except (IndexError, AttributeError):
        pass

    if not results:
        results.append(
            RawResult(
                url="",
                title="Gemini Knowledge Base",
                snippet=response.text[:500],
            )
        )

    return results
