from unittest.mock import AsyncMock, MagicMock, patch
from backend.agents.synthesis_agent import run
from backend.schemas import CitedSource


def _make_source(id: int, url: str, title: str) -> CitedSource:
    return CitedSource(
        id=id, url=url, title=title, snippet="snippet", credibility_score=0.8
    )


async def test_run_returns_model_text():
    mock_resp = MagicMock()
    mock_resp.text = "Paris is the capital of France [1]."
    sources = [_make_source(1, "https://wiki.org", "France Wikipedia")]
    with patch("backend.agents._gemini._call", new_callable=AsyncMock) as MockModel:
        MockModel.return_value = mock_resp
        result = await run("What is the capital of France?", sources)
    assert result == "Paris is the capital of France [1]."


async def test_run_prompt_includes_source_ids_and_titles():
    mock_resp = MagicMock()
    mock_resp.text = "Answer [1] and [2]."
    sources = [
        _make_source(1, "https://a.com", "Source A"),
        _make_source(2, "https://b.com", "Source B"),
    ]
    with patch("backend.agents._gemini._call", new_callable=AsyncMock) as MockModel:
        MockModel.return_value = mock_resp
        await run("Query about A and B", sources)
        prompt = MockModel.call_args[0][1]
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Source A" in prompt
    assert "Source B" in prompt
