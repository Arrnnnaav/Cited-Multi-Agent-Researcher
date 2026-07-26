import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from backend.agents import orchestrator
from backend.eval import runner as eval_runner
from backend.schemas import ResearchRequest

router = APIRouter()


async def _research_stream(query: str):
    try:
        response = await orchestrator.run(query)
        words = response.answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.02)
        sources_payload = {
            "type": "sources",
            "sources": [s.model_dump() for s in response.sources],
            "query_type": response.query_type,
            "subagents_used": response.subagents_used,
            "latency_ms": response.latency_ms,
        }
        yield f"data: {json.dumps(sources_payload)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@router.post("/research")
async def research(request: ResearchRequest):
    return StreamingResponse(
        _research_stream(request.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/eval/run")
async def run_eval():
    results = await eval_runner.run()
    return {"status": "complete", "num_queries": len(results)}


@router.get("/eval/report")
async def get_report():
    report_path = eval_runner.latest_report_path()
    if report_path is None or not report_path.exists():
        return HTMLResponse("<p>No report yet. Hit /eval/run first.</p>")
    return HTMLResponse(report_path.read_text())


@router.get("/health")
async def health():
    return {"status": "ok"}
