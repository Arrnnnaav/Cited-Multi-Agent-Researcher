"""Evidence path (M0), traces/feedback (M1), Redis quota/rate/jobs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fakeredis import FakeServer, aioredis
from fastapi.testclient import TestClient
from google.genai import errors

from backend import jobs, records, redis_layer
from backend.agents import _gemini, citation_agent
from backend.eval.citation_checks import check_citations, cited_ids, summarize
from backend.evidence import normalize_url
from backend.schemas import CitedSource, RawResult


def src(i, url="https://a.com", status="grounded", snippet="p"):
    return CitedSource(id=i, url=url, title="t", snippet=snippet, credibility_score=0.6,
                       evidence_status=status)  # fmt: skip


# ---- M0: evidence + checks ------------------------------------------------


def test_normalize_url_strips_tracking_fragment_case_and_slash():
    assert (
        normalize_url("HTTPS://Example.com/a/?utm_source=x&id=2#top")
        == "https://example.com/a?id=2"
    )
    assert normalize_url("https://example.com/a") == normalize_url(
        "https://EXAMPLE.com/a/"
    )


def test_citation_agent_merges_duplicate_urls_and_keeps_both_passages():
    cited = citation_agent.run([
        RawResult(url="https://a.com/x?utm_medium=m", title="A", snippet="one", evidence_status="grounded"),
        RawResult(url="https://A.com/x/", title="A2", snippet="two", evidence_status="grounded"),
        RawResult(url="https://b.com", title="B", snippet="", evidence_status="metadata_only"),
    ])  # fmt: skip
    assert [c.id for c in cited] == [1, 2]
    assert "one" in cited[0].snippet and "two" in cited[0].snippet


def test_checks_catch_each_failure_type():
    sources = [src(1), src(2, "", "model_only"), src(3, "https://c.com", "metadata_only"),
               src(4, "https://A.com/"), src(5, "https://e.com")]  # fmt: skip
    answer = ("Paris has 2.1 million residents [1]. It was founded in 250 BC [2]. "
              "The Seine crosses it [3]. Tourism brought 44 million visitors in 2023 [9]. "
              "The Louvre opened to the public in 1793 as a museum.")  # fmt: skip
    kinds = summarize(check_citations(answer, sources))
    assert kinds["invalid_citation"] == 1  # [9]
    assert kinds["cites_model_only"] == 1  # [2]
    assert kinds["cites_unverified"] == 1  # [3]
    assert kinds["duplicate_source"] == 1  # [4] == [1]
    assert kinds["unused_source"] == 2  # [4], [5]
    assert kinds["uncited_sentence"] == 1  # Louvre


def test_clean_answer_has_no_errors():
    answer = "Paris is the capital of France [1]. It is on the Seine [1, 2]."
    findings = check_citations(answer, [src(1), src(2, "https://b.com")])
    assert [f for f in findings if f.severity == "error"] == []
    assert cited_ids(answer) == [1, 1, 2]


# ---- Redis: shared quota state + rate limit ------------------------------


@pytest.fixture
def fake_redis():
    r = aioredis.FakeRedis(server=FakeServer(), decode_responses=True)
    redis_layer.set_client(r)
    yield r


def _fake_call(behaviour: dict):
    """_gemini._call stand-in: model name -> exception to raise or value to return."""
    seen = []

    async def call(model, prompt, config):
        seen.append(model)
        out = behaviour[model]
        if isinstance(out, Exception):
            raise out
        return out

    return call, seen


async def test_exhausted_model_is_skipped_by_every_later_call(fake_redis, monkeypatch):
    monkeypatch.setattr(_gemini, "GEMINI_MODELS", ["m1", "m2"])
    call, seen = _fake_call({"m1": errors.ClientError(429, {"error": {"message": "quota"}}), "m2": "ok"})
    monkeypatch.setattr(_gemini, "_call", call)
    assert await _gemini.generate("p") == "ok"
    assert await _gemini.generate("p") == "ok"
    assert seen == ["m1", "m2", "m2"]  # second call never touches m1
    assert await redis_layer.is_exhausted("m1")


async def test_call_log_records_model_actually_used(fake_redis, monkeypatch):
    monkeypatch.setattr(_gemini, "GEMINI_MODELS", ["m1"])
    call, _ = _fake_call({"m1": "ok"})
    monkeypatch.setattr(_gemini, "_call", call)
    log: list = []
    token = _gemini.call_log.set(log)
    await _gemini.generate("p", stage="search")
    _gemini.call_log.reset(token)
    assert log[0]["model"] == "m1" and log[0]["stage"] == "search" and log[0]["outcome"] == "ok"


async def test_search_calls_request_google_search_tool(monkeypatch):
    monkeypatch.setattr(_gemini, "GEMINI_MODELS", ["m1"])
    seen = {}

    async def call(model, prompt, config):
        seen["config"] = config
        return "ok"

    monkeypatch.setattr(_gemini, "_call", call)
    await _gemini.generate("p", search=True)
    assert seen["config"].tools[0].google_search is not None
    await _gemini.generate("p")
    assert seen["config"] is None


async def test_rate_limit_counts_per_model_window(fake_redis):
    for _ in range(3):
        await redis_layer.acquire_rate_slot("m1", rpm=5)
    keys = [k async for k in fake_redis.scan_iter("research:rate:m1:*")]
    assert len(keys) == 1 and int(await fake_redis.get(keys[0])) == 3


async def test_runs_without_redis():
    redis_layer.set_client(None)
    assert await redis_layer.cache_get("q", "v") is None
    await redis_layer.acquire_rate_slot("m1")  # no-op, no error


# ---- Eval job queue --------------------------------------------------------


async def test_eval_job_succeeds(fake_redis):
    job = await jobs.enqueue(fake_redis)
    handled = await jobs.process_once(
        fake_redis, "c", AsyncMock(return_value={"n": 10}), block_ms=10
    )
    assert handled == 1
    st = await jobs.status(fake_redis, job)
    assert st["status"] == "done" and st["attempts"] == "1"


async def test_eval_job_retries_then_dead_letters(fake_redis):
    job = await jobs.enqueue(fake_redis)
    boom = AsyncMock(side_effect=RuntimeError("quota"))
    for _ in range(jobs.MAX_ATTEMPTS):
        await jobs.process_once(fake_redis, "c", boom, block_ms=10)
    st = await jobs.status(fake_redis, job)
    assert st["status"] == "dead" and st["attempts"] == str(jobs.MAX_ATTEMPTS)
    assert await fake_redis.xlen(jobs.DLQ) == 1


# ---- M1: traces + feedback -------------------------------------------------


async def test_run_trace_and_feedback_join(tmp_path):
    db = str(tmp_path / "r.db")
    await records.save_run({"run_id": "r1", "config_version": "v", "query": "q",
                            "status": "ok", "model_calls": [{"model": "m"}]}, db)  # fmt: skip
    assert await records.add_feedback(
        {"run_id": "r1", "rating": -1, "issue": "wrong_citation"}, db
    )
    assert not await records.add_feedback({"run_id": "nope"}, db)
    run = await records.get_run("r1", db)
    assert run["model_calls"] == [{"model": "m"}]
    assert run["feedback"][0]["issue"] == "wrong_citation"


async def test_orchestrator_records_trace_with_findings():
    from backend.agents import orchestrator

    raw = [RawResult(url="https://a.com", title="A", snippet="Paris is the capital.",
                     evidence_status="grounded")]  # fmt: skip
    with patch.object(orchestrator, "_classify", AsyncMock(return_value="fact")), \
         patch.object(orchestrator.search_agent, "run", AsyncMock(return_value=(raw, True))), \
         patch.object(orchestrator.synthesis_agent, "run", AsyncMock(return_value="Paris [1]. Rome [7].")):  # fmt: skip
        resp = await orchestrator.run("capital of France?")
    assert resp.run_id and resp.cache_hits == 1
    assert any(f.kind == "invalid_citation" for f in resp.findings)
    run = await records.get_run(resp.run_id)
    assert run["status"] == "ok" and run["subquestions"] == ["capital of France?"]
    assert "search" in run["stages"]


def test_feedback_endpoint_and_eval_requires_redis():
    from backend.main import app

    client = TestClient(app)
    assert client.post("/feedback", json={"run_id": "missing"}).status_code == 404
    assert client.post("/eval/run").status_code == 503  # Redis disabled in tests
    assert client.get("/eval/run").status_code == 405  # no longer a GET side effect


async def test_retired_model_falls_through_and_is_skipped(fake_redis, monkeypatch):
    monkeypatch.setattr(_gemini, "GEMINI_MODELS", ["old", "new"])
    call, _ = _fake_call({"old": errors.ClientError(404, {"error": {"message": "retired"}}), "new": "ok"})
    monkeypatch.setattr(_gemini, "_call", call)
    assert await _gemini.generate("p") == "ok"
    assert await redis_layer.is_exhausted("old")


async def test_query_plan_is_cached_so_repeat_queries_reuse_searches(fake_redis):
    from backend.agents import orchestrator

    raw = [RawResult(url="https://a.com", title="A", snippet="p", evidence_status="grounded")]
    classify = AsyncMock(return_value="comparison")
    decompose = AsyncMock(return_value=["q1", "q2"])
    with patch.object(orchestrator, "_classify", classify), \
         patch.object(orchestrator, "_decompose", decompose), \
         patch.object(orchestrator.search_agent, "run", AsyncMock(return_value=(raw, False))), \
         patch.object(orchestrator.synthesis_agent, "run", AsyncMock(return_value="A [1].")):  # fmt: skip
        await orchestrator.run("Compare X and Y")
        await orchestrator.run("compare  x and y")
    assert classify.await_count == 1 and decompose.await_count == 1
