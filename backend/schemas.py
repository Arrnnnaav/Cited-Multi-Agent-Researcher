from typing import Literal
from pydantic import BaseModel


class ResearchRequest(BaseModel):
    query: str


class RawResult(BaseModel):
    url: str
    title: str
    snippet: str


class CitedSource(BaseModel):
    id: int
    url: str
    title: str
    snippet: str
    credibility_score: float


class ResearchResponse(BaseModel):
    answer: str
    sources: list[CitedSource]
    query_type: Literal["fact", "comparison"]
    subagents_used: int
    latency_ms: int


class JudgeScore(BaseModel):
    factuality: float
    citation_coverage: float
    reasoning: str


class EvalResult(BaseModel):
    query: str
    answer: str
    sources: list[CitedSource]
    scores: JudgeScore
