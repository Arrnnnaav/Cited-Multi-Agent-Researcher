"""Shared Gemini call helper: async, model failover, backoff.

Two problems this solves:

1. The `google-generativeai` SDK's `generate_content` is blocking. Calling it
   directly inside `async def` freezes the event loop, so `asyncio.gather` over
   several SearchAgents would run sequentially. We offload to a worker thread
   via `asyncio.to_thread` so concurrent calls actually overlap.

2. Free-tier Gemini gives only ~20 requests/day *per model*, and one query
   fans out to several calls. `generate()` walks `GEMINI_MODELS` in order: a
   daily-quota 429 (ResourceExhausted) on one model immediately falls through
   to the next (its quota is separate); a transient 503 retries the same model
   with exponential backoff. Callers pass a `make_model(name)` factory so each
   agent keeps control of its own model config (e.g. SearchAgent's search tool)
   and unit tests can still patch `genai.GenerativeModel` per module.
"""

import asyncio
from typing import Callable

from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from backend.config import GEMINI_MODELS


async def generate(
    make_model: Callable[[str], object],
    prompt: str,
    retries_per_model: int = 2,
    base_delay: float = 2.0,
):
    """Generate against the first model in GEMINI_MODELS that has quota.

    `make_model` builds a configured model given a model name. On a daily-quota
    429 we skip straight to the next model; on a transient 503 we back off and
    retry the same one. Re-raises the last error if every model is exhausted.
    """
    last_err: Exception | None = None
    for model_name in GEMINI_MODELS:
        model = make_model(model_name)
        delay = base_delay
        for attempt in range(retries_per_model):
            try:
                return await asyncio.to_thread(model.generate_content, prompt)
            except ResourceExhausted as e:
                last_err = e
                break  # daily quota won't recover soon — try the next model
            except ServiceUnavailable as e:
                last_err = e
                if attempt == retries_per_model - 1:
                    break
                await asyncio.sleep(delay)
                delay *= 2
    raise last_err if last_err else RuntimeError("No Gemini models configured")
