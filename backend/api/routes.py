import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from backend import jobs, records, redis_layer
from backend.agents import orchestrator
from backend.eval import runner as eval_runner
from backend.schemas import FeedbackRequest, ResearchRequest

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
            "run_id": response.run_id,
            "findings": [f.model_dump() for f in response.findings],
            "cache_hits": response.cache_hits,
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


@router.post("/feedback")
async def feedback(fb: FeedbackRequest):
    if not await records.add_feedback(fb.model_dump()):
        raise HTTPException(404, "unknown run_id")
    return {"status": "recorded"}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = await records.get_run(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")
    return run


@router.post("/eval/run", status_code=202)
async def run_eval():
    r = await redis_layer.get_redis()
    if r is None:
        raise HTTPException(
            503, "eval queue needs Redis; run `python -m backend.eval.runner` instead"
        )
    job_id = await jobs.enqueue(r)
    return {"job_id": job_id, "status": "queued"}


@router.get("/eval/jobs/{job_id}")
async def eval_job(job_id: str):
    r = await redis_layer.get_redis()
    if r is None:
        raise HTTPException(503, "Redis unavailable")
    st = await jobs.status(r, job_id)
    if st is None:
        raise HTTPException(404, "unknown job")
    return st


@router.get("/eval/report")
async def get_report():
    report_path = eval_runner.latest_report_path()
    if report_path is None or not report_path.exists():
        return HTMLResponse("<p>No report yet. POST /eval/run first.</p>")
    return HTMLResponse(report_path.read_text())


@router.get("/metrics")
async def metrics():
    return JSONResponse(await redis_layer.metrics())


@router.get("/health")
async def health():
    return {"status": "ok"}
