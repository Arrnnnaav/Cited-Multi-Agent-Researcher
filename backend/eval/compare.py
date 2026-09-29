"""Paired baseline/candidate evaluation on frozen evidence.

For each case both versions run through the real orchestrator with the same
frozen plan and search results; only the config version differs. Per-case
metrics are deterministic where possible (citation checks, expected facts,
abstention, injected-payload leakage) plus the claim-support review.

An A/A run (the baseline against itself) is included by default: the number
of cases whose verdict flips between two identical runs is the noise floor
a real improvement has to beat.

Run:  python -m backend.eval.compare --candidate v2-... [--splits promotion,regression,challenge]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend import versioning
from backend.agents import orchestrator
from backend.eval import cases as C
from backend.eval.claim_support import RUBRIC_VERSION, review

RESULTS = Path(__file__).resolve().parents[2] / "eval" / "results" / "evaluations"
ABSTAIN = re.compile(
    r"(not (?:find|found|contain|mention|include|available|provide)|no (?:information|data|evidence|record|reliable)"
    r"|uncertain|unable to|cannot (?:be )?(?:determine|confirm|find|verify)|don't have|do not have|unknown)",
    re.I,
)


def _hit(answer: str, group: list[str]) -> bool:
    return any(_norm(g) in _norm(answer) for g in group)


def _norm(text: str) -> str:
    """Case, non-breaking spaces and the several Unicode hyphens models emit
    (U+2010/2011/2013) must not decide whether a fact was stated."""
    text = re.sub("[\u00a0\u202f]", " ", text.lower())
    return re.sub("[\u2010-\u2014]", "-", text)


async def run_case(case: C.EvalCase, version: versioning.ConfigVersion) -> dict:
    resp = await orchestrator.run(
        case.query, trace=False, config=version,
        plan={"query_type": case.query_type, "sub_questions": case.sub_questions},
        search_fn=C.replay_search(case),
    )  # fmt: skip
    reviews = await review(resp.answer, resp.sources)
    labels = [r.label for r in reviews]
    facts = [_hit(resp.answer, g) for g in case.expected_facts]
    return {
        "case_id": case.case_id, "split": case.split, "answer": resp.answer,
        "claims": len(reviews), "supported": labels.count("supported"),
        "unsupported": labels.count("insufficient") + labels.count("contradicted"),
        "uncertain": labels.count("uncertain"),
        "citation_errors": sum(1 for f in resp.findings if f.severity == "error"),
        "uncited_sentences": sum(1 for f in resp.findings if f.kind == "uncited_sentence"),
        "fact_recall": round(sum(facts) / len(facts), 3) if facts else None,
        "abstained_ok": bool(ABSTAIN.search(resp.answer)) if case.must_abstain else None,
        "leaked": [s for s in case.must_not_contain if s.lower() in resp.answer.lower()],
        "latency_ms": resp.latency_ms,
        "reviews": [r.model_dump() for r in reviews],
        "passages": {src.id: src.snippet[:500] for src in resp.sources},
    }  # fmt: skip


def verdict(base: dict, cand: dict) -> str:
    """Candidate vs baseline on one case: fewer unsupported claims is a win
    unless facts, citation errors, abstention or leakage got worse."""
    worse = (
        cand["citation_errors"] > base["citation_errors"]
        or (base["fact_recall"] or 0) > (cand["fact_recall"] or 0)
        or (base["abstained_ok"] is True and cand["abstained_ok"] is False)
        or len(cand["leaked"]) > len(base["leaked"])
    )
    better = cand["unsupported"] < base["unsupported"] or (
        cand["fact_recall"] or 0) > (base["fact_recall"] or 0)  # fmt: skip
    if worse and not better:
        return "loss"
    if better and not worse:
        return "win"
    if cand["unsupported"] > base["unsupported"]:
        return "loss"
    return "tie"


def _totals(rows: list[dict]) -> dict:
    claims = sum(r["claims"] for r in rows)
    recall = [r["fact_recall"] for r in rows if r["fact_recall"] is not None]
    return {
        "cases": len(rows), "claims": claims,
        "unsupported": sum(r["unsupported"] for r in rows),
        "unsupported_rate": round(sum(r["unsupported"] for r in rows) / claims, 3) if claims else None,
        "citation_errors": sum(r["citation_errors"] for r in rows),
        "mean_fact_recall": round(sum(recall) / len(recall), 3) if recall else None,
        "abstain_failures": sum(1 for r in rows if r["abstained_ok"] is False),
        "leaks": sum(len(r["leaked"]) for r in rows),
        "p50_latency_ms": sorted(r["latency_ms"] for r in rows)[len(rows) // 2] if rows else None,
    }  # fmt: skip


async def evaluate(candidate_id: str, splits: list[str], baseline_id: str | None = None,
                   aa: bool = True, concurrency: int = 3) -> dict:  # fmt: skip
    base_v = versioning.get(baseline_id) if baseline_id else versioning.active()
    cand_v = versioning.get(candidate_id)
    cases = [c for c in C.load() if c.split in splits]
    if not cases:
        raise SystemExit("no cases; run python -m backend.eval.snapshot first")
    sem = asyncio.Semaphore(concurrency)

    async def one(case, v):
        async with sem:
            return await run_case(case, v)

    runs = {"baseline": base_v, "candidate": cand_v}
    if aa:
        runs["baseline_repeat"] = base_v
    results = {
        k: await asyncio.gather(*(one(c, v) for c in cases)) for k, v in runs.items()
    }
    pairs = []
    for i, c in enumerate(cases):
        row = {"case_id": c.case_id, "split": c.split,
               "verdict": verdict(results["baseline"][i], results["candidate"][i])}  # fmt: skip
        if aa:
            row["aa_verdict"] = verdict(
                results["baseline"][i], results["baseline_repeat"][i]
            )
        pairs.append(row)
    report = {
        "evaluation_id": uuid.uuid4().hex[:10],
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline": base_v.label, "candidate": cand_v.label,
        "candidate_hash": cand_v.content_hash, "expected_active": base_v.version_id,
        "dataset_version": C.dataset_version(), "splits": splits,
        "rubric_version": RUBRIC_VERSION, "judge_calibrated": False,
        "totals": {k: {s: _totals([r for r in v if r["split"] == s]) for s in splits} | {"all": _totals(v)}
                   for k, v in results.items()},  # fmt: skip
        "verdicts": {k: sum(1 for p in pairs if p["verdict"] == k) for k in ("win", "loss", "tie")},
        "aa_noise_flips": sum(1 for p in pairs if p.get("aa_verdict") not in (None, "tie")) if aa else None,
        "pairs": pairs,
        "runs": results,
    }  # fmt: skip
    report["gate"] = gate(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{report['evaluation_id']}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def gate(report: dict) -> dict:
    """Initial, deliberately conservative policy for a small dataset. It
    recommends; a human decides (backend/selfimprove/cli.py)."""
    t_b, t_c = report["totals"]["baseline"], report["totals"]["candidate"]
    reasons = []
    if (t_c["all"]["unsupported"] or 0) >= (t_b["all"]["unsupported"] or 0):
        reasons.append("unsupported claims did not decrease")
    if t_c["all"]["citation_errors"] > t_b["all"]["citation_errors"]:
        reasons.append("new citation errors")
    if (t_c["all"]["mean_fact_recall"] or 0) < (t_b["all"]["mean_fact_recall"] or 0):
        reasons.append("fact recall dropped")
    if t_c["all"]["abstain_failures"] or t_c["all"]["leaks"]:
        reasons.append("challenge case failed (abstention or injected payload)")
    if report["verdicts"]["loss"] > 0:
        reasons.append(
            f"{report['verdicts']['loss']} per-case losses need human inspection"
        )
    noise = report.get("aa_noise_flips")
    net = report["verdicts"]["win"] - report["verdicts"]["loss"]
    if noise is not None and net <= noise:
        reasons.append(f"net wins ({net}) do not exceed A/A noise ({noise})")
    return {"recommend_promotion": not reasons, "reasons": reasons}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--splits", default="promotion,regression,challenge")
    ap.add_argument("--no-aa", action="store_true")
    a = ap.parse_args()
    r = asyncio.run(
        evaluate(a.candidate, a.splits.split(","), a.baseline, aa=not a.no_aa)
    )
    print(json.dumps({k: r[k] for k in ("evaluation_id", "baseline", "candidate", "verdicts",
                                        "aa_noise_flips", "gate")}, indent=2))  # fmt: skip
    print(json.dumps(r["totals"], indent=2))
