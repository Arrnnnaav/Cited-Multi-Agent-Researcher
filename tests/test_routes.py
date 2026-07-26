from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from backend.schemas import CitedSource, ResearchResponse


def _make_response() -> ResearchResponse:
    return ResearchResponse(
        answer="Paris is the capital [1].",
        sources=[
            CitedSource(
                id=1,
                url="https://wiki.org",
                title="France",
                snippet="Paris...",
                credibility_score=0.85,
            )
        ],
        query_type="fact",
        subagents_used=1,
        latency_ms=200,
    )


def test_health():
    from backend.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_research_returns_sse_content_type():
    from backend.main import app

    client = TestClient(app)
    with patch(
        "backend.api.routes.orchestrator.run",
        new_callable=AsyncMock,
        return_value=_make_response(),
    ):
        with client.stream(
            "POST", "/research", json={"query": "What is the capital of France?"}
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


def test_research_streams_done_event():
    from backend.main import app

    client = TestClient(app)
    with patch(
        "backend.api.routes.orchestrator.run",
        new_callable=AsyncMock,
        return_value=_make_response(),
    ):
        with client.stream("POST", "/research", json={"query": "test"}) as resp:
            content = b"".join(resp.iter_bytes()).decode()
    assert '"type": "done"' in content


def test_research_streams_sources_event():
    from backend.main import app

    client = TestClient(app)
    with patch(
        "backend.api.routes.orchestrator.run",
        new_callable=AsyncMock,
        return_value=_make_response(),
    ):
        with client.stream("POST", "/research", json={"query": "test"}) as resp:
            content = b"".join(resp.iter_bytes()).decode()
    assert '"type": "sources"' in content
    assert "wiki.org" in content
