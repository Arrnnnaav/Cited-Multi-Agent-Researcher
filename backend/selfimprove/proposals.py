"""Failure analysis on dev cases and the proposal contract.

A proposal is one versioned change to one surface, justified only by dev
evidence. It is evaluated on held-out splits, then a human promotes or
rejects it. The candidate text is reviewed as a diff; nothing here lets a
model write and apply executable changes.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from backend import versioning
from backend.eval import cases as C
from backend.eval.compare import RESULTS, run_case

PROPOSALS = versioning.ROOT / "proposals"
FailureCategory = Literal[
    "unsupported_citation", "citation_error", "missed_fact", "abstention", "injection"
]
ALLOWED_SURFACES = {"synthesis_instructions"}


class Proposal(BaseModel):
    proposal_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    target_failure: FailureCategory
    dev_case_ids: list[str]
    change_surface: str = "synthesis_instructions"
    from_version: str
    candidate_version: str
    rationale: str
    predicted_benefit: str
    predicted_risk: str
    status: Literal["draft", "evaluated", "promoted", "rejected"] = "draft"
    evaluation_id: str | None = None
    evaluated_candidate_hash: str | None = None
    decision: dict | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


async def analyze_failures(version: versioning.ConfigVersion | None = None) -> dict:
    """Runs the active (or given) version on dev cases only and groups failures."""
    version = version or versioning.active()
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {}
    for case in C.load(split="dev"):
        row = await run_case(case, version)
        found = []
        for r in row["reviews"]:
            if r["label"] in ("insufficient", "contradicted"):
                found.append(("unsupported_citation", r["claim"]))
        if row["citation_errors"]:
            found.append(("citation_error", f"{row['citation_errors']} errors"))
        if row["fact_recall"] is not None and row["fact_recall"] < 1:
            found.append(("missed_fact", f"recall {row['fact_recall']}"))
        for cat, detail in found:
            counts[cat] += 1
            examples.setdefault(cat, []).append(
                {"case_id": case.case_id, "detail": detail[:200]}
            )
    return {"version": version.label, "dataset_version": C.dataset_version(),
            "counts": dict(counts), "examples": examples}  # fmt: skip


def save(p: Proposal) -> Proposal:
    if p.change_surface not in ALLOWED_SURFACES:
        raise ValueError(f"surface {p.change_surface} is not candidate-editable")
    dev_ids = {c.case_id for c in C.load(split="dev")}
    if not set(p.dev_case_ids) <= dev_ids:
        raise ValueError("proposals may only cite dev cases")
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    (PROPOSALS / f"{p.proposal_id}.json").write_text(
        p.model_dump_json(indent=2), encoding="utf-8"
    )
    return p


def get(proposal_id: str) -> Proposal:
    path = PROPOSALS / f"{proposal_id}.json"
    if not path.exists():
        raise KeyError(f"unknown proposal {proposal_id}")
    return Proposal.model_validate_json(path.read_text(encoding="utf-8"))


def diff(p: Proposal) -> str:
    old = versioning.get(p.from_version).synthesis_instructions.splitlines()
    new = versioning.get(p.candidate_version).synthesis_instructions.splitlines()
    return "\n".join(
        difflib.unified_diff(old, new, p.from_version, p.candidate_version, lineterm="")
    )


def record_evaluation(p: Proposal, report: dict) -> Proposal:
    p.status = "evaluated"
    p.evaluation_id = report["evaluation_id"]
    p.evaluated_candidate_hash = report["candidate_hash"]
    return save(p)


def decide(p: Proposal, *, promote: bool, actor: str, rationale: str) -> dict:
    """Human decision. Promotion is refused unless the evaluated candidate
    hash still matches and the active version is the one evaluated against."""
    if p.status != "evaluated" or not p.evaluation_id:
        raise ValueError("proposal has not been evaluated")
    report_path = RESULTS / f"{p.evaluation_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = versioning.get(p.candidate_version)
    if candidate.content_hash != p.evaluated_candidate_hash:
        raise versioning.StalePromotion(
            "candidate changed after evaluation; re-evaluate"
        )
    evidence = {
        "evaluation_id": p.evaluation_id,
        "gate": report["gate"],
        "verdicts": report["verdicts"],
    }
    if promote:
        event = versioning.set_active(p.candidate_version, expected_active=report["expected_active"],
                                      actor=actor, reason=rationale, evidence=evidence)  # fmt: skip
        p.status = "promoted"
    else:
        event = {"action": "reject", "actor": actor, "reason": rationale, "evidence": evidence,
                 "at": datetime.now(timezone.utc).isoformat()}  # fmt: skip
        p.status = "rejected"
    p.decision = event
    save(p)
    return event
