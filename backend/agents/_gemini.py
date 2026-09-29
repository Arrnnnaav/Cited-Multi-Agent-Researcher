"""Shared Gemini call helper on the `google-genai` SDK: async, failover, backoff.

v1 used the deprecated `google-generativeai` SDK, whose `generate_content` is
blocking; calls were offloaded with `asyncio.to_thread` so concurrent search
agents actually overlapped (152s -> 26s for five workers). `google-genai` has a
native async client (`client.aio`), so no thread offload is needed.

Failover: free-tier quota is per model, so `generate()` walks `GEMINI_MODELS`
in order. A 429 (quota) or 404 (model retired / not enabled for this key)
marks the model unavailable in Redis so every worker skips it; a 5xx retries
the same model with exponential backoff. Calls wait for a shared per-model
rate window, and every call is appended to the current run's call log (model
actually used, latency, outcome) for tracing.
"""

import asyncio
import contextvars
import time

from google import genai
from google.genai import errors, types

from backend import redis_layer
from backend.config import GEMINI_MODELS, GOOGLE_API_KEY

# Per-run call log; the orchestrator sets a fresh list for each request.
call_log: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "call_log", default=None
)

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def search_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )


async def _call(model: str, prompt: str, config: types.GenerateContentConfig | None):
    """Single SDK call; the seam unit tests patch."""
    return await client().aio.models.generate_content(
        model=model, contents=prompt, config=config
    )


def _record(model: str, stage: str, started: float, outcome: str) -> None:
    log = call_log.get()
    if log is not None:
        log.append(
            {
                "model": model,
                "stage": stage,
                "ms": int((time.perf_counter() - started) * 1000),
                "outcome": outcome,
            }
        )


async def generate(
    prompt: str,
    *,
    stage: str = "",
    search: bool = False,
    retries_per_model: int = 2,
    base_delay: float = 2.0,
):
    """Generate against the first model in GEMINI_MODELS that is available.
    Re-raises the last error if every model fails."""
    config = search_config() if search else None
    last_err: Exception | None = None
    for model_name in GEMINI_MODELS:
        if await redis_layer.is_exhausted(model_name):
            continue
        delay = base_delay
        for attempt in range(retries_per_model):
            await redis_layer.acquire_rate_slot(model_name)
            started = time.perf_counter()
            try:
                resp = await _call(model_name, prompt, config)
                _record(model_name, stage, started, "ok")
                return resp
            except errors.ClientError as e:
                last_err = e
                if e.code == 429:
                    _record(model_name, stage, started, "quota_exhausted")
                    await redis_layer.mark_exhausted(model_name)
                elif e.code == 404:
                    # Model retired or not enabled for this key: skip it everywhere.
                    _record(model_name, stage, started, "model_not_found")
                    await redis_layer.mark_exhausted(model_name, ttl_s=24 * 3600)
                else:
                    _record(model_name, stage, started, f"client_error_{e.code}")
                    raise
                break  # try the next model
            except errors.ServerError as e:
                last_err = e
                _record(model_name, stage, started, "unavailable")
                if attempt == retries_per_model - 1:
                    break
                await asyncio.sleep(delay)
                delay *= 2
    raise last_err if last_err else RuntimeError("No Gemini models available")
