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

# Redis (optional): search cache, shared quota state, rate limit, eval job queue.
REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/1")
SEARCH_CACHE_TTL_S: int = int(os.getenv("SEARCH_CACHE_TTL_S", str(24 * 3600)))
GEMINI_RPM: int = int(os.getenv("GEMINI_RPM", "15"))

# Run traces + feedback (SQLite).
TRACE_DB: str = os.getenv("TRACE_DB", "data/runs.db")

# Bump when search/extraction logic changes so stale cache entries are ignored.
PIPELINE_VERSION: str = "v2-evidence"
