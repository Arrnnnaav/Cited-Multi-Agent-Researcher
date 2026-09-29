import os
import tempfile

import pytest

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ["TRACE_DB"] = os.path.join(tempfile.mkdtemp(), "runs.db")


@pytest.fixture(autouse=True)
def _no_real_redis():
    """Unit tests never touch a real Redis; tests that need one inject fakeredis."""
    from backend import redis_layer

    redis_layer.set_client(None)
    yield
    redis_layer.set_client(None)
