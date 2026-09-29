import os
from dotenv import load_dotenv

load_dotenv()

SUBAGENT_CAP: int = int(os.getenv("SUBAGENT_CAP", "5"))
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Primary model, plus a failover chain. Free-tier quota is per-model and resets
# daily, so when one model returns a quota 429 we fall through to the next.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
_FALLBACKS = "gemini-flash-lite-latest,gemini-flash-latest,gemini-2.5-flash"
GEMINI_MODELS: list[str] = [GEMINI_MODEL] + [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", _FALLBACKS).split(",")
    if m.strip() and m.strip() != GEMINI_MODEL
]

# LLM provider for classify/decompose/synthesis/judge:
#   gemini         - google-genai SDK, GEMINI_MODELS chain
#   openai_compat  - any OpenAI-compatible /chat/completions API
#                    (OpenRouter: https://openrouter.ai/api/v1,
#                     NVIDIA NIM: https://integrate.api.nvidia.com/v1)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
OPENAI_COMPAT_BASE_URL: str = os.getenv(
    "OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENAI_COMPAT_API_KEY: str = os.getenv("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_MODELS: list[str] = [
    m.strip()
    for m in os.getenv(
        "OPENAI_COMPAT_MODELS",
        "qwen/qwen3.8-27b:free,nvidia/nemotron-3-super-120b-a12b:free,google/gemma-4-31b-it:free",
    ).split(",")
    if m.strip()
]

# NVIDIA NIM: primary OpenAI-compatible endpoint (free models); OpenRouter free
# models above are the fallback.
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL: str = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODELS: list[str] = [
    m.strip()
    for m in os.getenv(
        "NIM_MODELS",
        "nvidia/nemotron-3-super-120b-a12b,openai/gpt-oss-20b,deepseek-ai/deepseek-v4.1-flash",
    ).split(",")
    if m.strip()
]

# Web search provider:
#   tavily - Tavily Search API; returns extracted page text per URL (real
#            per-source evidence). Falls back to Gemini grounding on failure
#            when GOOGLE_API_KEY is set.
#   gemini - Gemini google_search grounding.
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
SEARCH_PROVIDER: str = os.getenv(
    "SEARCH_PROVIDER", "tavily" if TAVILY_API_KEY else "gemini"
)
TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# Redis (optional): search cache, shared quota state, rate limit, eval job queue.
REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/1")
SEARCH_CACHE_TTL_S: int = int(os.getenv("SEARCH_CACHE_TTL_S", str(24 * 3600)))
GEMINI_RPM: int = int(os.getenv("GEMINI_RPM", "15"))
# NIM's free tier allows ~40 requests/minute per model; OpenRouter free ~20.
COMPAT_RPM: int = int(os.getenv("COMPAT_RPM", "35"))

# Run traces + feedback (SQLite).
TRACE_DB: str = os.getenv("TRACE_DB", "data/runs.db")

# Bump when search/extraction logic changes so stale cache entries are ignored.
PIPELINE_VERSION: str = "v3-evidence"
