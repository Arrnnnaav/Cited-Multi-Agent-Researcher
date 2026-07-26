import google.generativeai as genai
from backend.config import GOOGLE_API_KEY
from backend.schemas import CitedSource
from backend.agents import _gemini

genai.configure(api_key=GOOGLE_API_KEY)


async def run(query: str, sources: list[CitedSource]) -> str:
    source_block = "\n".join(
        f"[{s.id}] {s.title} ({s.url})\n{s.snippet}" for s in sources
    )
    prompt = f"""You are a research assistant. Answer the query using ONLY the provided sources.
Cite sources inline using [N] notation. Every factual claim must have a citation.

Query: {query}

Sources:
{source_block}

Answer:"""

    response = await _gemini.generate(
        lambda name: genai.GenerativeModel(model_name=name), prompt
    )
    return response.text
