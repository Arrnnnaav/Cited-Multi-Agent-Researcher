"""Trace recording for the self-improving loop.

Records each research run with all relevant telemetry for later analysis.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from enum import Enum

from backend.config import GEMINI_MODEL, GEMINI_MODELS, SUBAGENT_CAP
from backend.schemas import ResearchResponse, CitedSource


class AgentStage(str, Enum):
    """Pipeline stages where failures can occur."""
    CLASSIFY = "classify"
    DECOMPOSE = "decompose"
    SEARCH = "search"
    CITATION = "citation"
    SYNTHESIS = "synthesis"
    JUDGE = "judge"


class FailureCategory(str, Enum):
    """Categorized failure types for targeted improvement."""
    LOW_CITATION_COVERAGE = "low_citation_coverage"
    LOW_FACTUALITY = "low_factuality"
    SEARCH_NO_RESULTS = "search_no_results"
    SEARCH_IRRELEVANT = "search_irrelevant"
    DEDUP_FAILED = "dedup_failed"
    CLASSIFICATION_WRONG = "classification_wrong"
    DECOMPOSITION_BAD = "decomposition_bad"
    SYNTHESIS_HALLUCINATION = "synthesis_hallucination"
    SYNTHESIS_MISSING_CITATIONS = "synthesis_missing_citations"
    JUDGE_PARSE_ERROR = "judge_parse_error"
    TIMEOUT = "timeout"
    MODEL_QUOTA = "model_quota"
    UNKNOWN = "unknown"


@dataclass
class AgentTrace:
    """Trace for a single agent execution."""
    stage: AgentStage
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    latency_ms: int
    model_name: str
    prompt_version: str
    error: Optional[str] = None
    token_estimate: Optional[int] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunTrace:
    """Complete trace for one research request."""
    run_id: str
    timestamp: str
    query: str
    config_version: str
    agent_version: str
    
    # Configuration snapshot
    subagent_cap: int
    gemini_model: str
    gemini_fallbacks: list[str]
    
    # Pipeline traces
    traces: list[AgentTrace] = field(default_factory=list)
    
    # Final outcome
    response: Optional[ResearchResponse] = None
    judge_scores: Optional[dict[str, float]] = None  # factuality, citation_coverage
    judge_reasoning: Optional[str] = None
    
    # Failure analysis
    failure_categories: list[FailureCategory] = field(default_factory=list)
    failure_details: dict[str, Any] = field(default_factory=dict)
    
    # Feedback (explicit or derived)
    explicit_feedback: Optional[dict[str, Any]] = None  # e.g., {"rating": 1-5, "comment": "..."}
    derived_signals: dict[str, Any] = field(default_factory=dict)  # schema_valid, citations_present, etc.
    
    total_latency_ms: int = 0
    total_token_estimate: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        data = asdict(self)
        # Convert enums to values
        data["traces"] = [
            {**t, "stage": t["stage"].value if isinstance(t["stage"], AgentStage) else t["stage"]}
            for t in data["traces"]
        ]
        data["failure_categories"] = [fc.value for fc in data["failure_categories"]]
        if self.response:
            data["response"] = self.response.model_dump()
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunTrace":
        """Create from dict (for loading stored traces)."""
        # Convert enums back
        traces = []
        for t in data.get("traces", []):
            t_copy = dict(t)
            t_copy["stage"] = AgentStage(t["stage"])
            traces.append(AgentTrace(**t_copy))
        
        failure_cats = [FailureCategory(fc) for fc in data.get("failure_categories", [])]
        
        response = None
        if data.get("response"):
            response = ResearchResponse(**data["response"])
        
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            query=data["query"],
            config_version=data["config_version"],
            agent_version=data["agent_version"],
            subagent_cap=data["subagent_cap"],
            gemini_model=data["gemini_model"],
            gemini_fallbacks=data["gemini_fallbacks"],
            traces=traces,
            response=response,
            judge_scores=data.get("judge_scores"),
            judge_reasoning=data.get("judge_reasoning"),
            failure_categories=failure_cats,
            failure_details=data.get("failure_details", {}),
            explicit_feedback=data.get("explicit_feedback"),
            derived_signals=data.get("derived_signals", {}),
            total_latency_ms=data.get("total_latency_ms", 0),
            total_token_estimate=data.get("total_token_estimate", 0),
        )


class TraceStore:
    """Persistent store for run traces."""
    
    def __init__(self, base_path: Path = Path("improvement/traces")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._current_run: Optional[RunTrace] = None
    
    def start_run(
        self,
        query: str,
        config_version: str,
        agent_version: str,
    ) -> RunTrace:
        """Start a new trace."""
        self._current_run = RunTrace(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(timespec="seconds"),
            query=query,
            config_version=config_version,
            agent_version=agent_version,
            subagent_cap=SUBAGENT_CAP,
            gemini_model=GEMINI_MODEL,
            gemini_fallbacks=GEMINI_MODELS[1:],
        )
        return self._current_run
    
    def add_trace(self, trace: AgentTrace) -> None:
        """Add an agent trace to the current run."""
        if self._current_run:
            self._current_run.traces.append(trace)
    
    def finalize(
        self,
        response: Optional[ResearchResponse] = None,
        judge_scores: Optional[dict[str, float]] = None,
        judge_reasoning: Optional[str] = None,
        failure_categories: Optional[list[FailureCategory]] = None,
        failure_details: Optional[dict[str, Any]] = None,
        explicit_feedback: Optional[dict[str, Any]] = None,
    ) -> RunTrace:
        """Finalize the current run with outcome data."""
        if not self._current_run:
            raise RuntimeError("No active run to finalize")
        
        run = self._current_run
        run.response = response
        run.judge_scores = judge_scores
        run.judge_reasoning = judge_reasoning
        run.failure_categories = failure_categories or []
        run.failure_details = failure_details or {}
        run.explicit_feedback = explicit_feedback
        
        # Compute derived signals
        run.derived_signals = self._compute_derived_signals(run)
        
        # Aggregate latency and tokens
        run.total_latency_ms = sum(t.latency_ms for t in run.traces)
        run.total_token_estimate = sum(t.token_estimate or 0 for t in run.traces)
        
        # Persist
        self._persist(run)
        self._current_run = None
        return run
    
    def _compute_derived_signals(self, run: RunTrace) -> dict[str, Any]:
        """Compute objective signals from the trace."""
        signals: dict[str, Any] = {}
        
        if run.response:
            # Schema validity - always true for Pydantic models
            signals["schema_valid"] = True
            
            # Citation presence
            answer = run.response.answer
            citations_in_answer = len([c for c in answer.split("[") if "]" in c and c[1:].split("]")[0].isdigit()])
            signals["citation_count_in_answer"] = citations_in_answer
            signals["source_count"] = len(run.response.sources)
            signals["citation_coverage_estimate"] = min(1.0, citations_in_answer / max(1, len(run.response.sources)))
            
            # Answer length
            signals["answer_length"] = len(answer)
            signals["answer_word_count"] = len(answer.split())
        
        # Check for judge parse errors
        for trace in run.traces:
            if trace.stage == AgentStage.JUDGE and trace.error:
                signals["judge_parse_error"] = True
                break
        else:
            signals["judge_parse_error"] = False
        
        # Check for search failures
        search_traces = [t for t in run.traces if t.stage == AgentStage.SEARCH]
        signals["search_agent_count"] = len(search_traces)
        signals["search_errors"] = sum(1 for t in search_traces if t.error)
        
        return signals
    
    def _persist(self, run: RunTrace) -> None:
        """Save trace to disk."""
        date_dir = self.base_path / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        file_path = date_dir / f"{run.run_id}.json"
        file_path.write_text(json.dumps(run.to_dict(), indent=2))
    
    def load_traces(
        self,
        days: int = 30,
        failure_category: Optional[FailureCategory] = None,
    ) -> list[RunTrace]:
        """Load traces from disk, optionally filtered."""
        traces: list[RunTrace] = []
        cutoff = datetime.now().timestamp() - (days * 86400)
        
        for file_path in self.base_path.rglob("*.json"):
            # Check file age
            if file_path.stat().st_mtime < cutoff:
                continue
            
            try:
                data = json.loads(file_path.read_text())
                trace = RunTrace.from_dict(data)
                
                if failure_category:
                    if failure_category not in trace.failure_categories:
                        continue
                
                traces.append(trace)
            except Exception:
                continue  # Skip corrupted files
        
        # Sort by timestamp, newest first
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        return traces
    
    def get_failure_stats(self, days: int = 30) -> dict[FailureCategory, int]:
        """Get failure category counts."""
        traces = self.load_traces(days=days)
        stats: dict[FailureCategory, int] = {}
        for trace in traces:
            for fc in trace.failure_categories:
                stats[fc] = stats.get(fc, 0) + 1
        return stats


# Global trace store instance
trace_store = TraceStore()