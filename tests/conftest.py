import os
import tempfile

import pytest

# Pin providers before backend.config loads .env (load_dotenv never
# overrides variables that are already set), so unit tests never depend on,
# or spend, the developer's real keys.
os.environ["GOOGLE_API_KEY"] = "test-key"
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["SEARCH_PROVIDER"] = "gemini"
os.environ["TAVILY_API_KEY"] = ""
os.environ["OPENAI_COMPAT_API_KEY"] = ""
os.environ["NVIDIA_API_KEY"] = ""
os.environ["TRACE_DB"] = os.path.join(tempfile.mkdtemp(), "runs.db")
os.environ["CONFIG_ROOT"] = tempfile.mkdtemp()  # versions/active pointer


@pytest.fixture(autouse=True)
def _no_real_redis():
    """Unit tests never touch a real Redis; tests that need one inject fakeredis."""
    from backend import redis_layer

    redis_layer.set_client(None)
    yield
    redis_layer.set_client(None)
