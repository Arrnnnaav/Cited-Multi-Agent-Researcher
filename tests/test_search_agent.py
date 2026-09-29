from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest
from fakeredis import FakeServer, aioredis

from backend import redis_layer
from backend.agents.search_agent import parse_grounded, run


def _response(
    text: str, urls: list[str], supports: list[tuple[str, list[int]]] | None = None
):
    chunks = [NS(web=NS(uri=u, title=f"Page: {u}")) for u in urls]
    sup = [
        NS(segment=NS(text=t), grounding_chunk_indices=idx) for t, idx in supports or []
    ]
    meta = NS(grounding_chunks=chunks, grounding_supports=sup)
    resp = MagicMock()
    resp.text = text
    resp.candidates = [NS(grounding_metadata=meta)]
    return resp


def test_each_source_gets_only_its_own_passages():
    """Regression: previously every source got the same first 300 chars of the
    answer as its snippet, so [2] 'supported' whatever [1] said."""
    resp = _response(
        "Paris is the capital. Lyon is the third largest city.",
        ["https://a.com", "https://b.com"],
        [("Paris is the capital.", [0]), ("Lyon is the third largest city.", [1])],
    )
    a, b = parse_grounded(resp)
    assert a.snippet == "Paris is the capital." and a.evidence_status == "grounded"
    assert b.snippet == "Lyon is the third largest city."
    assert a.snippet != b.snippet


def test_source_without_attributed_passage_is_metadata_only():
    resp = _response("x", ["https://a.com", "https://b.com"], [("Only A.", [0])])
    a, b = parse_grounded(resp)
    assert b.evidence_status == "metadata_only" and b.snippet == ""


def test_no_grounding_is_labelled_model_only_not_a_web_source():
    resp = _response("Some knowledge about X.", [])
    [r] = parse_grounded(resp)
    assert r.url == "" and r.evidence_status == "model_only"
    assert "Some knowledge" in r.snippet


@pytest.fixture
def fake_redis():
    r = aioredis.FakeRedis(server=FakeServer(), decode_responses=True)
    redis_layer.set_client(r)
    yield r


async def test_second_identical_search_is_served_from_cache(fake_redis):
    resp = _response("Paris.", ["https://a.com"], [("Paris.", [0])])
    with patch("backend.agents.search_agent.genai.GenerativeModel") as M:
        M.return_value.generate_content.return_value = resp
        first, hit1 = await run("capital of  France?")
        second, hit2 = await run("Capital of France?")  # normalized key
    assert (hit1, hit2) == (False, True)
    assert M.return_value.generate_content.call_count == 1
    assert second == first
    assert (await redis_layer.metrics()) == {"cache_miss": 1, "cache_hit": 1}


async def test_model_only_results_are_not_cached(fake_redis):
    with patch("backend.agents.search_agent.genai.GenerativeModel") as M:
        M.return_value.generate_content.return_value = _response("guess", [])
        await run("q")
        _, hit = await run("q")
    assert hit is False
