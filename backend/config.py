import os
from dotenv import load_dotenv

load_dotenv()

SUBAGENT_CAP: int = int(os.getenv("SUBAGENT_CAP", "5"))
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Primary model, plus a failover chain. Free-tier quota is per-model and resets
# daily, so when one model returns a quota 429 we fall through to the next.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
_FALLBACKS = "gemini-flash-latest,gemini-2.5-flash,gemini-2.0-flash,gemini-2.5-pro"
GEMINI_MODELS: list[str] = [GEMINI_MODEL] + [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", _FALLBACKS).split(",")
    if m.strip() and m.strip() != GEMINI_MODEL
]
