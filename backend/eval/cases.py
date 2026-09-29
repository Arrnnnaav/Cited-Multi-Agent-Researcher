"""Versioned evaluation cases with frozen evidence.

Splits (a proposal may only be drafted from `dev`):
  dev         cases used to find failures and draft a change
  promotion   held-out cases a candidate is judged on
  regression  previously good cases that must not get worse
  challenge   missing evidence, prompt injection in source text

A case freezes the query plan (query_type + sub_questions) and the search
results for each sub-question, so a baseline and a candidate see identical
inputs. Scoring uses deterministic checks, never a single "desired score":
  expected_facts    any-of groups, e.g. [["December 25, 2021", "25 December 2021"]]
  must_abstain      the evidence cannot answer this; the answer should say so
  must_not_contain  strings that must not appear (e.g. an injected payload)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from backend.schemas import RawResult

CASES_PATH = Path(__file__).resolve().parents[2] / "eval" / "cases" / "cases.jsonl"
Split = Literal["dev", "promotion", "regression", "challenge"]


class EvalCase(BaseModel):
    case_id: str
    split: Split
    query: str
    query_type: Literal["fact", "comparison"]
    sub_questions: list[str]
    evidence: dict[str, list[RawResult]]
    expected_facts: list[list[str]] = Field(default_factory=list)
    must_abstain: bool = False
    must_not_contain: list[str] = Field(default_factory=list)
    notes: str = ""
    labelled_by: str = "claude (review before relying on)"
    captured_at: str = ""


def load(path: Path | None = None, split: str | None = None) -> list[EvalCase]:
    path = path or CASES_PATH  # resolved at call time so tests can redirect it
    if not path.exists():
        return []
    cases = [
        EvalCase.model_validate_json(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    return [c for c in cases if split is None or c.split == split]


def save(cases: list[EvalCase], path: Path | None = None) -> None:
    path = path or CASES_PATH
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8"
    )


def dataset_version(path: Path | None = None) -> str:
    path = path or CASES_PATH
    return (
        hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "empty"
    )


def check_split_leakage(cases: list[EvalCase]) -> list[str]:
    """Same query in two splits would let dev-derived changes be scored on seen data."""
    seen: dict[str, str] = {}
    problems = []
    for c in cases:
        key = " ".join(c.query.lower().split())
        if (
            key in seen
            and seen[key] != c.split
            and c.split != "challenge"
            and seen[key] != "challenge"
        ):
            problems.append(f"{c.case_id}: query also in split {seen[key]}")
        seen.setdefault(key, c.split)
    return problems


def replay_search(case: EvalCase):
    """search_fn for orchestrator.run that serves the frozen evidence."""

    async def search(sub_question: str):
        return [r.model_copy() for r in case.evidence.get(sub_question, [])], False

    return search
