from unittest.mock import AsyncMock, patch
from backend.schemas import CitedSource, EvalResult, JudgeScore, ResearchResponse


def _make_research_response() -> ResearchResponse:
    return ResearchResponse(
        answer="Answer with [1].",
        sources=[
            CitedSource(
                id=1,
                url="https://test.com",
                title="Test",
                snippet="snip",
                credibility_score=0.8,
            )
        ],
        query_type="fact",
        subagents_used=1,
        latency_ms=100,
    )


def _make_judge_score() -> JudgeScore:
    return JudgeScore(factuality=0.9, citation_coverage=0.85, reasoning="Good answer.")


async def test_run_produces_one_result_per_query():
    with patch(
        "backend.eval.runner.orchestrator.run",
        new_callable=AsyncMock,
        return_value=_make_research_response(),
    ):
        with patch(
            "backend.eval.runner.judge_run",
            new_callable=AsyncMock,
            return_value=_make_judge_score(),
        ):
            from backend.eval.runner import run

            results = await run()
    assert len(results) == 10


async def test_run_all_results_are_eval_result():
    with patch(
        "backend.eval.runner.orchestrator.run",
        new_callable=AsyncMock,
        return_value=_make_research_response(),
    ):
        with patch(
            "backend.eval.runner.judge_run",
            new_callable=AsyncMock,
            return_value=_make_judge_score(),
        ):
            from backend.eval.runner import run

            results = await run()
    assert all(isinstance(r, EvalResult) for r in results)


async def test_run_scores_match_judge_output():
    with patch(
        "backend.eval.runner.orchestrator.run",
        new_callable=AsyncMock,
        return_value=_make_research_response(),
    ):
        with patch(
            "backend.eval.runner.judge_run",
            new_callable=AsyncMock,
            return_value=_make_judge_score(),
        ):
            from backend.eval.runner import run

            results = await run()
    assert results[0].scores.factuality == 0.9
    assert results[0].scores.citation_coverage == 0.85
