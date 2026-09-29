from unittest.mock import AsyncMock, MagicMock, patch
from backend.agents.orchestrator import _classify, _decompose, _get_subagent_count


def _mock_gemini(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


# --- _get_subagent_count ---


def test_get_subagent_count_fact_always_one():
    assert _get_subagent_count("fact", 0) == 1
    assert _get_subagent_count("fact", 10) == 1


def test_get_subagent_count_comparison_under_cap():
    assert _get_subagent_count("comparison", 3) == 3


def test_get_subagent_count_comparison_at_cap():
    assert _get_subagent_count("comparison", 5) == 5


def test_get_subagent_count_comparison_over_cap():
    assert _get_subagent_count("comparison", 10) == 5


# --- _classify ---


async def test_classify_returns_fact():
    with patch("backend.agents._gemini._call", new_callable=AsyncMock) as MockModel:
        MockModel.return_value = _mock_gemini("fact")
        result = await _classify("What is the speed of light?")
    assert result == "fact"


async def test_classify_returns_comparison():
    with patch("backend.agents._gemini._call", new_callable=AsyncMock) as MockModel:
        MockModel.return_value = _mock_gemini(
            "comparison"
        )
        result = await _classify("Compare Python vs JavaScript")
    assert result == "comparison"


async def test_classify_defaults_to_fact_for_unknown_output():
    with patch("backend.agents._gemini._call", new_callable=AsyncMock) as MockModel:
        MockModel.return_value = _mock_gemini(
            "unclear text"
        )
        result = await _classify("Random query")
    assert result == "fact"


# --- _decompose ---


async def test_decompose_fact_returns_original_query():
    result = await _decompose("What is X?", "fact")
    assert result == ["What is X?"]


async def test_decompose_comparison_parses_json_array():
    with patch("backend.agents._gemini._call", new_callable=AsyncMock) as MockModel:
        MockModel.return_value = _mock_gemini(
            '["Q1", "Q2", "Q3"]'
        )
        result = await _decompose("Compare A vs B vs C", "comparison")
    assert result == ["Q1", "Q2", "Q3"]


async def test_decompose_handles_json_in_markdown_fence():
    with patch("backend.agents._gemini._call", new_callable=AsyncMock) as MockModel:
        MockModel.return_value = _mock_gemini(
            '```json\n["Q1", "Q2"]\n```'
        )
        result = await _decompose("Compare A vs B", "comparison")
    assert result == ["Q1", "Q2"]


async def test_decompose_falls_back_to_original_on_bad_json():
    with patch("backend.agents._gemini._call", new_callable=AsyncMock) as MockModel:
        MockModel.return_value = _mock_gemini(
            "not json at all"
        )
        result = await _decompose("Compare A vs B", "comparison")
    assert result == ["Compare A vs B"]
