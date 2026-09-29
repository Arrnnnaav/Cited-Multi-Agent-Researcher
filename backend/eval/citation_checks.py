"""Deterministic structural checks on a cited answer. No LLM involved.

These catch errors a judge model can miss or reward: citing an ID that does
not exist, citing a "source" that is really the model's own text, citing a
source with no attributed passage, and factual-looking sentences with no
citation at all (a heuristic, reported as a warning for review).
"""

from __future__ import annotations

import re

from backend.evidence import normalize_url
from backend.schemas import CitedSource, Finding

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Sentences worth a citation: contain a number, a year, or a proper noun
# after the first word. Short connective sentences are ignored.
_FACTY = re.compile(r"\d|(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+")


def cited_ids(answer: str) -> list[int]:
    return [int(x) for m in CITE.findall(answer) for x in m.split(",")]


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*(?:#+\s*|\*\*[^*]+\*\*\s*:?\s*$)")


def sentences(answer: str) -> list[str]:
    """Split into claim-sized units. Lines first (each bullet is its own
    claim; markdown headings are dropped), then sentences within a line.
    Flattening newlines first merged whole bullet lists into one "sentence",
    which hid unsupported bullets from the claim-support review."""
    out = []
    for line in answer.splitlines():
        if not line.strip() or _HEADING.match(line):
            continue
        line = _BULLET.sub("", line).replace("**", "").strip()
        out += [s.strip() for s in _SENTENCE.split(line) if s.strip()]
    return out


def check_citations(answer: str, sources: list[CitedSource]) -> list[Finding]:
    findings: list[Finding] = []
    by_id = {s.id: s for s in sources}
    used = set(cited_ids(answer))

    for i in sorted(used - by_id.keys()):
        findings.append(Finding(kind="invalid_citation", severity="error",
                                detail=f"[{i}] cited but no source has that id"))  # fmt: skip
    for i in sorted(used & by_id.keys()):
        s = by_id[i]
        if s.evidence_status == "model_only" or not s.url:
            findings.append(Finding(kind="cites_model_only", severity="error",
                                    detail=f"[{i}] is model-generated text, not a web source"))  # fmt: skip
        elif s.evidence_status == "metadata_only":
            findings.append(Finding(kind="cites_unverified", severity="warning",
                                    detail=f"[{i}] has no passage attributed to it ({s.url})"))  # fmt: skip

    seen: dict[str, int] = {}
    for s in sources:
        key = normalize_url(s.url)
        if key and key in seen:
            findings.append(Finding(kind="duplicate_source", severity="warning",
                                    detail=f"[{s.id}] duplicates [{seen[key]}] ({key})"))  # fmt: skip
        elif key:
            seen[key] = s.id
    for s in sources:
        if s.id not in used:
            findings.append(Finding(kind="unused_source", severity="info",
                                    detail=f"[{s.id}] listed but never cited"))  # fmt: skip

    for sent in sentences(answer):
        if len(sent.split()) >= 6 and not CITE.search(sent) and _FACTY.search(sent):
            findings.append(Finding(kind="uncited_sentence", severity="warning",
                                    detail=sent[:160]))  # fmt: skip
    return findings


def summarize(findings: list[Finding]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        out[f.kind] = out.get(f.kind, 0) + 1
    return out
