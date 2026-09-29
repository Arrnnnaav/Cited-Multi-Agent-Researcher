"""Provider-agnostic text generation for the non-search agents."""

from backend.agents import _gemini, _openai_compat
from backend.config import LLM_PROVIDER


async def generate_text(prompt: str, *, stage: str = "") -> str:
    if LLM_PROVIDER == "openai_compat":
        return await _openai_compat.generate_text(prompt, stage=stage)
    resp = await _gemini.generate(prompt, stage=stage)
    return resp.text or ""
