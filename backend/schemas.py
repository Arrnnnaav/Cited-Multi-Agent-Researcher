from typing import Literal

from pydantic import BaseModel, Field

# grounded:      passage text attributed to this source by grounding metadata
# metadata_only: the source was retrieved but no passage was attributed to it
# model_only:    no web source at all; the text is the model's own answer
EvidenceStatus = Literal["grounded", "metadata_only", "model_only"]


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class RawResult(BaseModel):
    url: str
    title: str
    snippet: str
    evidence_status: EvidenceStatus = "metadata_only"


class CitedSource(BaseModel):
    id: int
    url: str
    title: str
    snippet: str
    credibility_score: float
    evidence_status: EvidenceStatus = "metadata_only"


class Finding(BaseModel):
    kind: Literal[
        "invalid_citation",
        "cites_model_only",
        "cites_unverified",
        "uncited_sentence",
        "duplicate_source",
        "unused_source",
    ]
    severity: Literal["error", "warning", "info"]
    detail: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[CitedSource]
    query_type: Literal["fact", "comparison"]
    subagents_used: int
    latency_ms: int
    run_id: str = ""
    findings: list[Finding] = Field(default_factory=list)
    cache_hits: int = 0
    config_version: str = ""


class JudgeScore(BaseModel):
    factuality: float
    citation_coverage: float
    reasoning: str


class EvalResult(BaseModel):
    query: str
    answer: str
    sources: list[CitedSource]
    scores: JudgeScore
    findings: list[Finding] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    rating: int | None = Field(default=None, ge=-1, le=1)
    issue: (
        Literal[
            "unsupported_claim", "wrong_citation", "missing_info", "outdated", "other"
        ]
        | None
    ) = None
    comment: str = Field(default="", max_length=2000)
    citation_id: int | None = None
