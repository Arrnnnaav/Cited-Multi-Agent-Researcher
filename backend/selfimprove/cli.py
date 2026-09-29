"""Local operations for the improvement loop. No HTTP admin routes on purpose.

  python -m backend.selfimprove.cli status
  python -m backend.selfimprove.cli failures                 # dev split only
  python -m backend.selfimprove.cli propose --id P --from V --candidate V2 \\
        --instructions-file f.txt --target unsupported_citation \\
        --dev-cases a,b --rationale "..." --benefit "..." --risk "..."
  python -m backend.selfimprove.cli diff --proposal P
  python -m backend.selfimprove.cli evaluate --proposal P
  python -m backend.selfimprove.cli decide --proposal P (--promote|--reject) --by NAME --why "..."
  python -m backend.selfimprove.cli rollback --to V --by NAME --why "..."
  python -m backend.selfimprove.cli history
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend import versioning
from backend.eval import cases as C
from backend.eval.compare import evaluate
from backend.selfimprove import proposals as P


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="selfimprove")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("failures")
    sub.add_parser("history")
    pr = sub.add_parser("propose")
    for f in (
        "id",
        "from",
        "candidate",
        "instructions-file",
        "target",
        "dev-cases",
        "rationale",
        "benefit",
        "risk",
    ):
        pr.add_argument(f"--{f}", required=True)
    for name in ("diff", "evaluate"):
        sub.add_parser(name).add_argument("--proposal", required=True)
    de = sub.add_parser("decide")
    de.add_argument("--proposal", required=True)
    g = de.add_mutually_exclusive_group(required=True)
    g.add_argument("--promote", action="store_true")
    g.add_argument("--reject", action="store_true")
    de.add_argument("--by", required=True)
    de.add_argument("--why", required=True)
    rb = sub.add_parser("rollback")
    rb.add_argument("--to", required=True)
    rb.add_argument("--by", required=True)
    rb.add_argument("--why", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "status":
        versioning.ensure_baseline()
        return {"active": versioning.active().label, "versions": [v.label for v in versioning.list_versions()],
                "dataset_version": C.dataset_version(),
                "cases": {s: len(C.load(split=s)) for s in ("dev", "promotion", "regression", "challenge")}}  # fmt: skip
    if a.cmd == "failures":
        return asyncio.run(P.analyze_failures())
    if a.cmd == "history":
        return {"history": versioning.history()}
    if a.cmd == "propose":
        text = Path(a.instructions_file).read_text(encoding="utf-8").strip()
        cand = versioning.save(versioning.ConfigVersion(
            version_id=a.candidate, synthesis_instructions=text, parent=getattr(a, "from"),
            notes=f"candidate for {a.id}"))  # fmt: skip
        prop = P.save(P.Proposal(
            proposal_id=a.id, target_failure=a.target, dev_case_ids=a.dev_cases.split(","),
            from_version=getattr(a, "from"), candidate_version=cand.version_id, rationale=a.rationale,
            predicted_benefit=a.benefit, predicted_risk=a.risk))  # fmt: skip
        return {
            "proposal": prop.proposal_id,
            "candidate": cand.label,
            "diff": P.diff(prop),
        }
    if a.cmd == "diff":
        return {"diff": P.diff(P.get(a.proposal))}
    if a.cmd == "evaluate":
        prop = P.get(a.proposal)
        report = asyncio.run(evaluate(prop.candidate_version, ["promotion", "regression", "challenge"],
                                      baseline_id=prop.from_version))  # fmt: skip
        P.record_evaluation(prop, report)
        return {k: report[k] for k in ("evaluation_id", "baseline", "candidate", "verdicts", "aa_noise_flips", "gate")} | {
            "totals": {k: v["all"] for k, v in report["totals"].items()}}  # fmt: skip
    if a.cmd == "decide":
        return P.decide(
            P.get(a.proposal), promote=a.promote, actor=a.by, rationale=a.why
        )
    if a.cmd == "rollback":
        current = versioning.active()
        return versioning.set_active(a.to, expected_active=current.version_id, actor=a.by,
                                     reason=a.why, action="rollback")  # fmt: skip
    raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
