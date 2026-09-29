from backend.agents import _gemini
from backend.schemas import CitedSource

_STATUS_NOTE = {
    "grounded": "",
    "metadata_only": " (no passage available: do not cite for specific facts)",
    "model_only": " (NOT a web source: never cite)",
}


def build_prompt(query: str, sources: list[CitedSource]) -> str:
    source_block = "\n".join(
        f"[{s.id}] {s.title} ({s.url or 'no url'}){_STATUS_NOTE[s.evidence_status]}\n"
        f"Passage: {s.snippet or '(none)'}"
        for s in sources
    )
    return f"""You are a research assistant. Answer the query using ONLY the provided sources.
Cite inline with [N]. Cite a source for a claim only if its Passage supports that claim.
If no passage supports something, say it is uncertain instead of citing.

Query: {query}

Sources:
{source_block}

Answer:"""


async def run(query: str, sources: list[CitedSource]) -> str:
    response = await _gemini.generate(
        build_prompt(query, sources),
        stage="synthesis",
    )
    return response.text
