"""Offline eval jobs on a Redis Stream, off the request path.

POST /eval/run used to run ten live research queries plus judge calls inside
the HTTP request (minutes, and a GET that spent quota). Now it enqueues a job
and returns 202; `python -m backend.worker` consumes it. Failed jobs are
retried up to MAX_ATTEMPTS, then moved to a dead-letter stream.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Awaitable, Callable

from redis.asyncio import Redis

STREAM = "research:eval_jobs"
DLQ = "research:eval_jobs:dlq"
GROUP = "eval-workers"
MAX_ATTEMPTS = 3


def _job_key(job_id: str) -> str:
    return f"research:job:{job_id}"


async def ensure_group(r: Redis) -> None:
    try:
        await r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise


async def enqueue(r: Redis, kind: str = "eval") -> str:
    job_id = uuid.uuid4().hex[:12]
    await r.hset(_job_key(job_id), mapping={"status": "queued", "kind": kind, "attempts": 0,
                                            "created": int(time.time())})  # fmt: skip
    await r.expire(_job_key(job_id), 7 * 24 * 3600)
    await r.xadd(STREAM, {"job_id": job_id, "kind": kind})
    return job_id


async def status(r: Redis, job_id: str) -> dict | None:
    data = await r.hgetall(_job_key(job_id))
    return data or None


async def process_once(
    r: Redis,
    consumer: str,
    handler: Callable[[str], Awaitable[dict]],
    block_ms: int = 1000,
) -> int:
    """Read and handle available jobs; returns how many were handled."""
    await ensure_group(r)
    resp = await r.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=1, block=block_ms)
    handled = 0
    for _, entries in resp or []:
        for entry_id, fields in entries:
            job_id, kind = fields["job_id"], fields.get("kind", "eval")
            attempts = await r.hincrby(_job_key(job_id), "attempts", 1)
            await r.hset(_job_key(job_id), "status", "running")
            try:
                result = await handler(kind)
                await r.hset(_job_key(job_id), mapping={"status": "done",
                                                        "result": json.dumps(result)})  # fmt: skip
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:500]
                if attempts >= MAX_ATTEMPTS:
                    await r.hset(
                        _job_key(job_id), mapping={"status": "dead", "error": err}
                    )
                    await r.xadd(DLQ, {"job_id": job_id, "error": err})
                else:
                    await r.hset(
                        _job_key(job_id), mapping={"status": "retrying", "error": err}
                    )
                    await r.xadd(STREAM, {"job_id": job_id, "kind": kind})
            await r.xack(STREAM, GROUP, entry_id)
            await r.xdel(STREAM, entry_id)
            handled += 1
    return handled
