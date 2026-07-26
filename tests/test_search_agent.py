from unittest.mock import MagicMock, patch
import pytest


def _mock_response(text: str, urls: list[str] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    if urls:
        chunks = []
        for url in urls:
            chunk = MagicMock()
            chunk.web.uri = url
            chunk.web.title = f"Page: {url}"
            chunks.append(chunk)
        resp.candidates[0].grounding_metadata.grounding_chunks = chunks
    else:
        resp.candidates[0].grounding_metadata.grounding_chunks = []
    return resp


@pytest.fixture(autouse=True)
def patch_genai_types():
    with patch(
        "google.generativeai.types.GoogleSearchRetrieval",
        MagicMock(),
        create=True,
    ):
        with patch("google.generativeai.types.Tool", MagicMock(), create=True):
            yield


from backend.agents.search_agent import run  # noqa: E402


async def test_run_returns_results_from_grounding():
    mock_resp = _mock_response("Paris is the capital.", ["https://example.com/france"])
    with patch("backend.agents.search_agent.genai.GenerativeModel") as MockModel:
        MockModel.return_value.generate_content.return_value = mock_resp
        results = await run("What is the capital of France?")
    assert len(results) == 1
    assert results[0].url == "https://example.com/france"
    assert results[0].title == "Page: https://example.com/france"


async def test_run_falls_back_when_no_grounding_chunks():
    mock_resp = _mock_response("Some knowledge about X.")
    mock_resp.candidates[0].grounding_metadata.grounding_chunks = []
    with patch("backend.agents.search_agent.genai.GenerativeModel") as MockModel:
        MockModel.return_value.generate_content.return_value = mock_resp
        results = await run("What is X?")
    assert len(results) == 1
    assert results[0].url == ""
    assert "Some knowledge" in results[0].snippet


async def test_run_snippet_is_truncated_to_300_chars():
    long_text = "A" * 500
    mock_resp = _mock_response(long_text, ["https://a.com"])
    with patch("backend.agents.search_agent.genai.GenerativeModel") as MockModel:
        MockModel.return_value.generate_content.return_value = mock_resp
        results = await run("Q?")
    assert len(results[0].snippet) <= 300
