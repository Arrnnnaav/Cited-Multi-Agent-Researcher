"""OpenAI-compatible chat completions (OpenRouter, NVIDIA NIM, ...) with the
same failover semantics as the Gemini helper.

Routes are tried in order: NVIDIA NIM models first (if NVIDIA_API_KEY is set),
then every model on the OPENAI_COMPAT_* endpoint (OpenRouter free models)
as the fallback. Free-tier models on these gateways are rate limited
per minute, so a 429 parks the model in Redis for a minute (not a day) and
the next route is tried. A 404/400 naming the model (retired or not
available on this key) parks it for a day. An auth error skips the rest of
that endpoint. 5xx retries with backoff.
"""

import asyncio
import time

import httpx

from backend import redis_layer
from backend.agents._gemini import _record
from backend.config import (
    COMPAT_RPM,
    NIM_BASE_URL,
    NIM_MODELS,
    NVIDIA_API_KEY,
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


def routes(models: list[str] | None = None) -> list[tuple[str, str, str]]:
    """(base_url, api_key, model) in failover order."""
    fallback = [
        (OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_API_KEY, m)
        for m in (models or OPENAI_COMPAT_MODELS)
    ]
    if models is None and NVIDIA_API_KEY:
        return [(NIM_BASE_URL, NVIDIA_API_KEY, m) for m in NIM_MODELS] + fallback
    return fallback


async def _call(base_url: str, api_key: str, model: str, prompt: str) -> str:
    r = await client().post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
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
    bad_endpoints: set[str] = set()
    for base_url, api_key, model in routes(models):
        if base_url in bad_endpoints or await redis_layer.is_exhausted(model):
            continue
        delay = base_delay
        for attempt in range(retries_per_model):
            await redis_layer.acquire_rate_slot(model, rpm=COMPAT_RPM)
            started = time.perf_counter()
            try:
                text = await _call(base_url, api_key, model, prompt)
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
                    bad_endpoints.add(base_url)  # bad key: skip this endpoint entirely
                    break
                _record(model, stage, started, "unavailable")
                if attempt == retries_per_model - 1:
                    break
                await asyncio.sleep(delay)
                delay *= 2
    raise (
        last_err if last_err else RuntimeError("No OpenAI-compatible models available")
    )
