import json

import google.generativeai as genai

from backend.config import GOOGLE_API_KEY
from backend.schemas import CitedSource, JudgeScore
from backend.agents import _gemini

genai.configure(api_key=GOOGLE_API_KEY)


async def run(query: str, answer: str, sources: list[CitedSource]) -> JudgeScore:
    source_list = "\n".join(f"[{s.id}] {s.title} — {s.url}" for s in sources)
    prompt = f"""You are an evaluator. Score the answer on two dimensions.

Query: {query}

Answer: {answer}

Sources:
{source_list}

Return ONLY valid JSON in this exact format:
{{"factuality": <float 0.0-1.0>, "citation_coverage": <float 0.0-1.0>, "reasoning": "<one sentence>"}}

factuality: Are the claims factually accurate? (1.0 = fully accurate)
citation_coverage: Are factual claims backed by [N] citations? (1.0 = every claim cited)"""

    response = await _gemini.generate(
        lambda name: genai.GenerativeModel(model_name=name), prompt
    )
    text = response.text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)
    return JudgeScore(
        factuality=float(data["factuality"]),
        citation_coverage=float(data["citation_coverage"]),
        reasoning=str(data["reasoning"]),
    )
