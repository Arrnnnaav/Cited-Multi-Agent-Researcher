from backend.agents.citation_agent import run, _score_credibility
from backend.schemas import RawResult


def _make_result(url: str, title: str = "T", snippet: str = "s") -> RawResult:
    return RawResult(url=url, title=title, snippet=snippet)


def test_assigns_sequential_ids():
    results = [_make_result("https://a.com"), _make_result("https://b.gov")]
    cited = run(results)
    assert cited[0].id == 1
    assert cited[1].id == 2


def test_deduplicates_same_url():
    results = [
        _make_result("https://a.com"),
        _make_result("https://a.com", title="dup"),
    ]
    cited = run(results)
    assert len(cited) == 1


def test_deduplicates_empty_url_by_title():
    results = [
        RawResult(url="", title="Same Title", snippet="x"),
        RawResult(url="", title="Same Title", snippet="y"),
    ]
    cited = run(results)
    assert len(cited) == 1


def test_credibility_gov():
    assert _score_credibility("https://nih.gov/page") == 0.95


def test_credibility_edu():
    assert _score_credibility("https://mit.edu/paper") == 0.95


def test_credibility_wikipedia():
    assert _score_credibility("https://en.wikipedia.org/wiki/X") == 0.85


def test_credibility_org():
    score = _score_credibility("https://somenonprofit.org")
    assert 0.7 <= score <= 0.8


def test_credibility_no_url():
    assert _score_credibility("") == 0.3


def test_credibility_unknown_domain():
    assert _score_credibility("https://randomsite.com") == 0.6


def test_cited_source_has_correct_credibility_score():
    results = [_make_result("https://example.gov")]
    cited = run(results)
    assert cited[0].credibility_score == 0.95
