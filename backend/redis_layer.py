"""Redis: search-result cache, shared model-quota state, per-model rate limit.

Redis is optional. If REDIS_URL is unset or unreachable every helper degrades
to a no-op, so the app still runs (just without sharing state across workers).

Why Redis rather than in-process dicts: with several uvicorn workers or eval
jobs running, each process would otherwise re-discover the same exhausted
Gemini model (burning a call and a 429 each time), keep its own rate count,
and repeat identical grounded searches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

from redis.asyncio import Redis

from backend.config import GEMINI_RPM, REDIS_URL, SEARCH_CACHE_TTL_S

log = logging.getLogger(__name__)

_client: Redis | None = None
_disabled = False


async def get_redis() -> Redis | None:
    global _client, _disabled
    if _disabled or not REDIS_URL:
        return None
    if _client is None:
        client = Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        try:
            await client.ping()
        except Exception as e:
            log.warning("Redis unavailable (%s); running without cache/rate limit", e)
            _disabled = True
            return None
        _client = client
    return _client


def set_client(client: Redis | None) -> None:
    """For tests: inject a (fake) client or disable Redis."""
    global _client, _disabled
    _client, _disabled = client, client is None


def _cache_key(sub_question: str, version: str) -> str:
    norm = " ".join(sub_question.lower().split())
    return (
        "research:search:"
        + hashlib.sha256(f"{version}|{norm}".encode()).hexdigest()[:32]
    )


async def cache_get(sub_question: str, version: str) -> list[dict] | None:
    r = await get_redis()
    if r is None:
        return None
    raw = await r.get(_cache_key(sub_question, version))
    await r.hincrby("research:metrics", "cache_hit" if raw else "cache_miss", 1)
    return json.loads(raw) if raw else None


async def cache_set(sub_question: str, version: str, results: list[dict]) -> None:
    r = await get_redis()
    if r is not None:
        await r.set(
            _cache_key(sub_question, version),
            json.dumps(results),
            ex=SEARCH_CACHE_TTL_S,
        )


async def mark_exhausted(model: str, ttl_s: int = 3600) -> None:
    r = await get_redis()
    if r is not None:
        await r.set(f"research:quota_exhausted:{model}", 1, ex=ttl_s)
        await r.hincrby("research:metrics", "quota_exhausted", 1)


async def is_exhausted(model: str) -> bool:
    r = await get_redis()
    return bool(r is not None and await r.exists(f"research:quota_exhausted:{model}"))


async def acquire_rate_slot(
    model: str, rpm: int | None = None, max_wait_s: float = 30
) -> None:
    """Fixed one-minute window shared by all processes: wait for the next
    window instead of sending a request the provider will reject with 429."""
    limit = rpm or GEMINI_RPM
    r = await get_redis()
    if r is None or limit <= 0:
        return
    deadline = time.monotonic() + max_wait_s
    while True:
        window = int(time.time() // 60)
        key = f"research:rate:{model}:{window}"
        n = await r.incr(key)
        if n == 1:
            await r.expire(key, 65)
        if n <= limit:
            return
        await r.hincrby("research:metrics", "rate_limited_wait", 1)
        if time.monotonic() > deadline:
            return  # let the call go; the provider's 429 handling still applies
        await asyncio.sleep(60 - time.time() % 60 + 0.05)


async def metrics() -> dict[str, int]:
    r = await get_redis()
    if r is None:
        return {}
    return {k: int(v) for k, v in (await r.hgetall("research:metrics")).items()}


def _plan_key(query: str, version: str) -> str:
    norm = " ".join(query.lower().split())
    return "research:plan:" + hashlib.sha256(f"{version}|{norm}".encode()).hexdigest()[:32]


async def plan_get(query: str, version: str) -> dict | None:
    r = await get_redis()
    if r is None:
        return None
    raw = await r.get(_plan_key(query, version))
    await r.hincrby("research:metrics", "plan_hit" if raw else "plan_miss", 1)
    return json.loads(raw) if raw else None


async def plan_set(query: str, version: str, query_type: str, sub_questions: list[str]) -> None:
    r = await get_redis()
    if r is not None:
        await r.set(
            _plan_key(query, version),
            json.dumps({"query_type": query_type, "sub_questions": sub_questions}),
            ex=SEARCH_CACHE_TTL_S,
        )
