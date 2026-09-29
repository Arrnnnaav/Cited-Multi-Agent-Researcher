"""Failure analysis and categorization for the self-improving loop.

Analyzes traces to identify recurring failure patterns and group them
into actionable categories with supporting evidence.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.improvement.traces import (
    AgentStage,
    FailureCategory,
    RunTrace,
    trace_store,
)


@dataclass
class FailureCluster:
    """A cluster of related failures with supporting examples."""
    category: FailureCategory
    count: int
    examples: list[RunTrace]  # Top N most recent examples
    common_patterns: dict[str, Any] = field(default_factory=dict)
    suggested_fix_surface: str = ""  # e.g., "prompt", "retrieval", "workflow"
    first_seen: str = ""
    last_seen: str = ""


class FailureAnalyzer:
    """Analyzes traces to find recurring failure patterns."""
    
    # Thresholds for automatic failure detection
    CITATION_COVERAGE_THRESHOLD = 0.6
    FACTUALITY_THRESHOLD = 0.6
    MIN_EXAMPLES_FOR_CLUSTER = 2
    
    def __init__(self, trace_store_instance: Optional[Any] = None):
        self.store = trace_store_instance or trace_store
    
    def analyze_recent(self, days: int = 7) -> list[FailureCluster]:
        """Analyze recent traces for failure patterns."""
        traces = self.store.load_traces(days=days)
        if not traces:
            return []
        
        # Group by failure category
        by_category: dict[FailureCategory, list[RunTrace]] = defaultdict(list)
        for trace in traces:
            for cat in trace.failure_categories:
                by_category[cat].append(trace)
        
        # Build clusters
        clusters = []
        for category, cat_traces in by_category.items():
            if len(cat_traces) >= self.MIN_EXAMPLES_FOR_CLUSTER:
                cluster = self._build_cluster(category, cat_traces)
                clusters.append(cluster)
        
        # Sort by count descending
        clusters.sort(key=lambda c: c.count, reverse=True)
        return clusters
    
    def _build_cluster(self, category: FailureCategory, traces: list[RunTrace]) -> FailureCluster:
        """Build a failure cluster from traces of the same category."""
        # Sort by timestamp
        traces.sort(key=lambda t: t.timestamp)
        
        # Get common patterns
        patterns = self._extract_patterns(category, traces)
        
        return FailureCluster(
            category=category,
            count=len(traces),
            examples=traces[-3:],  # Keep 3 most recent as examples
            common_patterns=patterns,
            suggested_fix_surface=self._suggest_fix_surface(category, patterns),
            first_seen=traces[0].timestamp,
            last_seen=traces[-1].timestamp,
        )
    
    def _extract_patterns(self, category: FailureCategory, traces: list[RunTrace]) -> dict[str, Any]:
        """Extract common patterns from failure traces."""
        patterns: dict[str, Any] = {}
        
        if category == FailureCategory.LOW_CITATION_COVERAGE:
            patterns = self._patterns_low_citation(traces)
        elif category == FailureCategory.LOW_FACTUALITY:
            patterns = self._patterns_low_factuality(traces)
        elif category == FailureCategory.SEARCH_NO_RESULTS:
            patterns = self._patterns_search_no_results(traces)
        elif category == FailureCategory.CLASSIFICATION_WRONG:
            patterns = self._patterns_classification(traces)
        elif category == FailureCategory.DECOMPOSITION_BAD:
            patterns = self._patterns_decomposition(traces)
        elif category == FailureCategory.SYNTHESIS_MISSING_CITATIONS:
            patterns = self._patterns_synthesis_missing_citations(traces)
        
        # Add generic patterns
        patterns["avg_latency_ms"] = sum(t.total_latency_ms for t in traces) / len(traces)
        patterns["query_types"] = list(set(t.response.query_type if t.response else "unknown" for t in traces))
        patterns["subagent_counts"] = list(set(t.response.subagents_used if t.response else 0 for t in traces))
        
        return patterns
    
    def _patterns_low_citation(self, traces: list[RunTrace]) -> dict[str, Any]:
        """Patterns for low citation coverage."""
        return {
            "avg_citation_coverage": sum(
                t.derived_signals.get("citation_coverage_estimate", 0) for t in traces
            ) / len(traces),
            "avg_sources": sum(
                t.derived_signals.get("source_count", 0) for t in traces
            ) / len(traces),
            "avg_citations_in_answer": sum(
                t.derived_signals.get("citation_count_in_answer", 0) for t in traces
            ) / len(traces),
        }
    
    def _patterns_low_factuality(self, traces: list[RunTrace]) -> dict[str, Any]:
        """Patterns for low factuality."""
        return {
            "avg_factuality": sum(
                t.judge_scores.get("factuality", 0) if t.judge_scores else 0
                for t in traces
            ) / len(traces),
            "common_query_topics": self._extract_topics(traces),
        }
    
    def _patterns_search_no_results(self, traces: list[RunTrace]) -> dict[str, Any]:
        """Patterns for search returning no results."""
        search_errors = sum(t.derived_signals.get("search_errors", 0) for t in traces)
        return {
            "search_error_rate": search_errors / max(1, sum(t.derived_signals.get("search_agent_count", 0) for t in traces)),
            "fallback_model_used": any(
                any("flash" in str(t.get("model_name", "")).lower() for t in trace.traces if t.stage == AgentStage.SEARCH)
                for trace in traces
            ),
        }
    
    def _patterns_classification(self, traces: list[RunTrace]) -> dict[str, Any]:
        """Patterns for wrong classification."""
        return {
            "misclassified_as": list(set(
                t.response.query_type if t.response else "unknown" for t in traces
            )),
            "queries": [t.query for t in traces],
        }
    
    def _patterns_decomposition(self, traces: list[RunTrace]) -> dict[str, Any]:
        """Patterns for bad decomposition."""
        subq_counts = []
        for t in traces:
            for trace in t.traces:
                if trace.stage == AgentStage.DECOMPOSE:
                    output = trace.output_data
                    if isinstance(output, dict) and "sub_questions" in output:
                        subq_counts.append(len(output["sub_questions"]))
        return {
            "avg_subquestions": sum(subq_counts) / len(subq_counts) if subq_counts else 0,
            "queries": [t.query for t in traces],
        }
    
    def _patterns_synthesis_missing_citations(self, traces: list[RunTrace]) -> dict[str, Any]:
        """Patterns for synthesis missing citations."""
        return {
            "avg_citation_coverage": sum(
                t.derived_signals.get("citation_coverage_estimate", 0) for t in traces
            ) / len(traces),
            "avg_answer_length": sum(
                t.derived_signals.get("answer_length", 0) for t in traces
            ) / len(traces),
        }
    
    def _suggest_fix_surface(self, category: FailureCategory, patterns: dict[str, Any]) -> str:
        """Suggest which part of the system to fix for this failure category."""
        mapping = {
            FailureCategory.LOW_CITATION_COVERAGE: "prompt",  # synthesis_agent prompt
            FailureCategory.LOW_FACTUALITY: "prompt",  # synthesis_agent or search_agent prompt
            FailureCategory.SEARCH_NO_RESULTS: "retrieval",  # search tool config
            FailureCategory.SEARCH_IRRELEVANT: "retrieval",  # search prompt or reranking
            FailureCategory.DEDUP_FAILED: "workflow",  # citation_agent logic
            FailureCategory.CLASSIFICATION_WRONG: "prompt",  # orchestrator classify prompt
            FailureCategory.DECOMPOSITION_BAD: "prompt",  # orchestrator decompose prompt
            FailureCategory.SYNTHESIS_HALLUCINATION: "prompt",  # synthesis_agent prompt
            FailureCategory.SYNTHESIS_MISSING_CITATIONS: "prompt",  # synthesis_agent prompt
            FailureCategory.JUDGE_PARSE_ERROR: "prompt",  # judge_agent prompt
        }
        return mapping.get(category, "unknown")
    
    def _extract_topics(self, traces: list[RunTrace]) -> list[str]:
        """Extract common query topics (simplified)."""
        topics = []
        for t in traces:
            # Simple keyword extraction from query
            words = t.query.lower().split()
            topics.extend([w for w in words if len(w) > 4])
        # Return top 5 most common
        from collections import Counter
        return [w for w, _ in Counter(topics).most_common(5)]


class FailureClusterStore:
    """Persistent store for failure clusters and their analysis."""
    
    def __init__(self, base_path: Path = Path("improvement/clusters")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_cluster(self, cluster: FailureCluster) -> None:
        """Save a failure cluster."""
        file_path = self.base_path / f"{cluster.category.value}.json"
        data = {
            "category": cluster.category.value,
            "count": cluster.count,
            "example_run_ids": [ex.run_id for ex in cluster.examples],
            "common_patterns": cluster.common_patterns,
            "suggested_fix_surface": cluster.suggested_fix_surface,
            "first_seen": cluster.first_seen,
            "last_seen": cluster.last_seen,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        file_path.write_text(json.dumps(data, indent=2))
    
    def load_cluster(self, category: FailureCategory) -> Optional[FailureCluster]:
        """Load a failure cluster."""
        file_path = self.base_path / f"{category.value}.json"
        if not file_path.exists():
            return None
        data = json.loads(file_path.read_text())
        # Reconstruct with example run IDs (traces loaded separately)
        return FailureCluster(
            category=category,
            count=data["count"],
            examples=[],  # Loaded separately if needed
            common_patterns=data["common_patterns"],
            suggested_fix_surface=data["suggested_fix_surface"],
            first_seen=data["first_seen"],
            last_seen=data["last_seen"],
        )
    
    def load_all(self) -> list[FailureCluster]:
        """Load all clusters."""
        clusters = []
        for file_path in self.base_path.glob("*.json"):
            try                data = json.loads(file_path.read_text())
                cat = FailureCategory(data["category"])
                clusters.append(FailureCluster(
                    category=cat,
                    count=data["count"],
                    examples=[],
                    common_patterns=data["common_patterns"],
                    suggested_fix_surface=data["suggested_fix_surface"],
                    first_seen=data["first_seen"],
                    last_seen=data["last_seen"],
                ))
            except Exception:
                continue
        return clusters


# Global analyzer instance
failure_analyzer = FailureAnalyzer()
cluster_store = FailureClusterStore()