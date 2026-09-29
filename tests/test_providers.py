import json

import httpx
import pytest
from fakeredis import FakeServer, aioredis

from backend import config, llm, redis_layer, search_providers
from backend.agents import _openai_compat, search_agent
from backend.schemas import RawResult


@pytest.fixture
def fake_redis():
    r = aioredis.FakeRedis(server=FakeServer(), decode_responses=True)
    redis_layer.set_client(r)
    yield r


def test_parse_tavily_keeps_per_url_text():
    data = {"results": [
        {"url": "https://a.com", "title": "A", "content": "Webb launched on 25 December 2021."},
        {"url": "https://b.com", "title": "B", "content": ""},
        {"url": "", "title": "no url", "content": "x"},
    ]}  # fmt: skip
    a, b = search_providers.parse_tavily(data)
    assert a.snippet.startswith("Webb launched") and a.evidence_status == "grounded"
    assert b.evidence_status == "metadata_only"


async def test_tavily_request_shape_and_errors(monkeypatch):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        if seen["body"]["query"] == "fail":
            return httpx.Response(432, text="plan limit")
        return httpx.Response(
            200,
            json={"results": [{"url": "https://a.com", "title": "A", "content": "c"}]},
        )

    monkeypatch.setattr(search_providers, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(
        search_providers,
        "_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    [r] = await search_providers.tavily_search("james webb launch")
    assert (
        seen["auth"] == "Bearer tvly-test"
        and seen["body"]["max_results"] == config.TAVILY_MAX_RESULTS
    )
    assert r.url == "https://a.com"
    with pytest.raises(search_providers.SearchError):
        await search_providers.tavily_search("fail")


async def test_search_agent_uses_tavily_then_falls_back_to_gemini(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_PROVIDER", "tavily")
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "g")
    calls = []

    async def tavily_fail(q):
        calls.append("tavily")
        raise search_providers.SearchError("down")

    async def gemini(q):
        calls.append("gemini")
        return [
            RawResult(
                url="https://g.com", title="G", snippet="s", evidence_status="grounded"
            )
        ]

    monkeypatch.setattr(search_providers, "tavily_search", tavily_fail)
    monkeypatch.setattr(search_agent, "_gemini_search", gemini)
    results, hit = await search_agent.run("q")
    assert (
        calls == ["tavily", "gemini"] and results[0].url == "https://g.com" and not hit
    )


async def test_search_agent_tavily_without_google_key_returns_empty(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_PROVIDER", "tavily")
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")

    async def tavily_fail(q):
        raise search_providers.SearchError("down")

    monkeypatch.setattr(search_providers, "tavily_search", tavily_fail)
    assert await search_agent.run("q") == ([], False)


def _compat_transport(behaviour: dict):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.content)["model"]
        seen.append(model)
        status = behaviour[model]
        if status == 200:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": f"hi from {model}"}}]}
            )
        return httpx.Response(status, text="err")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


async def test_openai_compat_falls_through_rate_limited_and_missing_models(
    fake_redis, monkeypatch
):
    client, seen = _compat_transport({"a": 429, "b": 404, "c": 200})
    monkeypatch.setattr(_openai_compat, "_client", client)
    assert (
        await _openai_compat.generate_text("p", models=["a", "b", "c"]) == "hi from c"
    )
    assert seen == ["a", "b", "c"]
    ttl_a = await fake_redis.ttl("research:quota_exhausted:a")
    ttl_b = await fake_redis.ttl("research:quota_exhausted:b")
    assert 0 < ttl_a <= 60 < ttl_b  # rate limit parks briefly, missing model for a day


async def test_openai_compat_auth_error_is_not_swallowed(fake_redis, monkeypatch):
    client, _ = _compat_transport({"a": 401, "b": 200})
    monkeypatch.setattr(_openai_compat, "_client", client)
    with pytest.raises(_openai_compat.CompatError):
        await _openai_compat.generate_text("p", models=["a", "b"])


async def test_llm_dispatches_on_provider(monkeypatch):
    async def compat(prompt, stage=""):
        return "compat"

    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai_compat")
    monkeypatch.setattr(llm._openai_compat, "generate_text", compat)
    assert await llm.generate_text("p") == "compat"


def test_nim_is_tried_before_openrouter(monkeypatch):
    monkeypatch.setattr(_openai_compat, "NVIDIA_API_KEY", "nvapi-x")
    monkeypatch.setattr(_openai_compat, "NIM_MODELS", ["meta/llama-3.3-70b-instruct"])
    monkeypatch.setattr(_openai_compat, "OPENAI_COMPAT_MODELS", ["some/model:free"])
    order = [(base.split("//")[1].split("/")[0], m) for base, _, m in _openai_compat.routes()]
    assert order[0] == ("integrate.api.nvidia.com", "meta/llama-3.3-70b-instruct")
    assert order[1][1] == "some/model:free"
