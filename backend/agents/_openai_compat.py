"""OpenAI-compatible chat completions (OpenRouter, NVIDIA NIM, ...) with the
same failover semantics as the Gemini helper.

Free-tier models on these gateways are rate limited per minute, so a 429
parks the model in Redis for a minute (not a day) and the next model in
OPENAI_COMPAT_MODELS is tried. A 404/400 naming the model (retired or not
available on this key) parks it for a day. 5xx retries with backoff.
"""

import asyncio
import time

import httpx

from backend import redis_layer
from backend.agents._gemini import _record
from backend.config import (
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_MODELS,
)

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=90)
    return _client


class CompatError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


async def _call(model: str, prompt: str) -> str:
    r = await client().post(
        f"{OPENAI_COMPAT_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_COMPAT_API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
    )
    if r.status_code >= 400:
        raise CompatError(r.status_code, r.text)
    choices = r.json().get("choices") or []
    if not choices:
        raise CompatError(502, "no choices in response")
    return choices[0]["message"].get("content") or ""


async def generate_text(
    prompt: str,
    *,
    stage: str = "",
    models: list[str] | None = None,
    retries_per_model: int = 2,
    base_delay: float = 2.0,
) -> str:
    last_err: Exception | None = None
    for model in models or OPENAI_COMPAT_MODELS:
        if await redis_layer.is_exhausted(model):
            continue
        delay = base_delay
        for attempt in range(retries_per_model):
            await redis_layer.acquire_rate_slot(model)
            started = time.perf_counter()
            try:
                text = await _call(model, prompt)
                _record(model, stage, started, "ok")
                return text
            except (CompatError, httpx.TransportError) as e:
                last_err = e
                status = getattr(e, "status", 0)
                if status == 429:
                    _record(model, stage, started, "rate_limited")
                    await redis_layer.mark_exhausted(model, ttl_s=60)
                    break
                if status in (400, 404):
                    _record(model, stage, started, "model_not_found")
                    await redis_layer.mark_exhausted(model, ttl_s=24 * 3600)
                    break
                if status in (401, 403):
                    _record(model, stage, started, "auth_error")
                    raise
                _record(model, stage, started, "unavailable")
                if attempt == retries_per_model - 1:
                    break
                await asyncio.sleep(delay)
                delay *= 2
    raise (
        last_err if last_err else RuntimeError("No OpenAI-compatible models available")
    )
