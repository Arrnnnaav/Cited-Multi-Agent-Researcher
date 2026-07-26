from backend.schemas import RawResult, CitedSource


def _score_credibility(url: str) -> float:
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
    seen: set[str] = set()
    cited: list[CitedSource] = []
    idx = 1
    for result in raw_results:
        key = result.url if result.url else result.title
        if key in seen:
            continue
        seen.add(key)
        cited.append(
            CitedSource(
                id=idx,
                url=result.url,
                title=result.title,
                snippet=result.snippet,
                credibility_score=_score_credibility(result.url),
            )
        )
        idx += 1
    return cited
