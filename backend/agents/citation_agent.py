from backend.evidence import join_passages, normalize_url
from backend.schemas import CitedSource, RawResult

_RANK = {"grounded": 2, "metadata_only": 1, "model_only": 0}


def _score_credibility(url: str) -> float:
    """Domain-type heuristic only. It says nothing about relevance or whether
    the source supports a claim; claim support is checked separately."""
    if not url:
        return 0.3
    u = url.lower()
    if any(d in u for d in [".gov", ".edu"]):
        return 0.95
    if any(
        d in u
        for d in [
            "wikipedia.org",
            "reuters.com",
            "bbc.com",
            "nature.com",
            "scholar.google",
        ]
    ):
        return 0.85
    if ".org" in u:
        return 0.75
    return 0.6


def run(raw_results: list[RawResult]) -> list[CitedSource]:
    """Deduplicate on normalized URL (title for URL-less results), merging the
    passages of duplicates so no evidence is dropped, then number sources."""
    merged: dict[str, dict] = {}
    for r in raw_results:
        key = normalize_url(r.url) or f"title:{r.title}"
        if key not in merged:
            merged[key] = {"r": r, "passages": [r.snippet] if r.snippet else []}
            continue
        entry = merged[key]
        if r.snippet and r.snippet not in entry["passages"]:
            entry["passages"].append(r.snippet)
        if _RANK[r.evidence_status] > _RANK[entry["r"].evidence_status]:
            entry["r"] = r.model_copy(update={"snippet": entry["r"].snippet})
    cited: list[CitedSource] = []
    for idx, entry in enumerate(merged.values(), start=1):
        r = entry["r"]
        cited.append(
            CitedSource(
                id=idx,
                url=r.url,
                title=r.title,
                snippet=join_passages(entry["passages"]),
                credibility_score=_score_credibility(r.url),
                evidence_status=r.evidence_status,
            )
        )
    return cited
