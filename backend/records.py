"""Run traces and user feedback in SQLite.

One row per research run (query, classification, sub-questions, per-stage
timings, every model call with the model that actually answered, sources with
evidence status, citation-check findings, errors) plus feedback rows joined
by run_id. This is the raw material for measuring and later improving
citation quality; nothing here calls an LLM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from backend.config import TRACE_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    config_version TEXT NOT NULL,
    query         TEXT NOT NULL,
    query_type    TEXT,
    subquestions  TEXT,
    stages        TEXT,
    model_calls   TEXT,
    sources       TEXT,
    answer        TEXT,
    findings      TEXT,
    cache_hits    INTEGER DEFAULT 0,
    latency_ms    INTEGER,
    status        TEXT NOT NULL,
    error         TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    created_at   TEXT NOT NULL,
    rating       INTEGER,
    issue        TEXT,
    comment      TEXT,
    citation_id  INTEGER
);
CREATE INDEX IF NOT EXISTS feedback_run ON feedback(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def _db(path: str | None = None) -> aiosqlite.Connection:
    p = path or TRACE_DB
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(p)
    await conn.executescript(SCHEMA)
    conn.row_factory = aiosqlite.Row
    return conn


async def save_run(rec: dict[str, Any], path: str | None = None) -> None:
    conn = await _db(path)
    try:
        await conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec["run_id"], _now(), rec["config_version"], rec["query"],
                rec.get("query_type"), json.dumps(rec.get("subquestions", [])),
                json.dumps(rec.get("stages", {})), json.dumps(rec.get("model_calls", [])),
                json.dumps(rec.get("sources", [])), rec.get("answer"),
                json.dumps(rec.get("findings", [])), rec.get("cache_hits", 0),
                rec.get("latency_ms"), rec["status"], rec.get("error"),
            ),
        )  # fmt: skip
        await conn.commit()
    finally:
        await conn.close()


async def get_run(run_id: str, path: str | None = None) -> dict[str, Any] | None:
    conn = await _db(path)
    try:
        cur = await conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = await cur.fetchone()
        if row is None:
            return None
        out = dict(row)
        for k in ("subquestions", "stages", "model_calls", "sources", "findings"):
            out[k] = json.loads(out[k]) if out[k] else None
        cur = await conn.execute(
            "SELECT rating, issue, comment, citation_id, created_at FROM feedback WHERE run_id = ?",
            (run_id,),
        )
        out["feedback"] = [dict(r) for r in await cur.fetchall()]
        return out
    finally:
        await conn.close()


async def add_feedback(fb: dict[str, Any], path: str | None = None) -> bool:
    """Returns False if the run does not exist."""
    conn = await _db(path)
    try:
        cur = await conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (fb["run_id"],))
        if await cur.fetchone() is None:
            return False
        await conn.execute(
            "INSERT INTO feedback (run_id, created_at, rating, issue, comment, citation_id) "
            "VALUES (?,?,?,?,?,?)",
            (fb["run_id"], _now(), fb.get("rating"), fb.get("issue"), fb.get("comment", ""),
             fb.get("citation_id")),
        )  # fmt: skip
        await conn.commit()
        return True
    finally:
        await conn.close()
