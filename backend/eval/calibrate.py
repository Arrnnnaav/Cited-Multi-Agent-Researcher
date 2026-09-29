"""Calibrates the claim-support reviewer against human labels.

  python -m backend.eval.calibrate export   # writes claims to label
  (edit eval/cases/claim_labels.jsonl: set "human_label" on each line)
  python -m backend.eval.calibrate score    # agreement + Cohen's kappa

Until enough human labels exist, evaluation reports mark the reviewer
"uncalibrated" and its aggregate is not used as a sole gate.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from backend import versioning
from backend.eval import cases as C
from backend.eval.compare import run_case

LABELS = Path(__file__).resolve().parents[2] / "eval" / "cases" / "claim_labels.jsonl"
VALID = {"supported", "contradicted", "insufficient", "uncertain"}


async def export(limit: int = 25) -> int:
    rows = []
    for case in C.load(split="dev") + C.load(split="regression"):
        r = await run_case(case, versioning.active())
        for rev in r["reviews"]:
            rows.append({"case_id": case.case_id, "claim": rev["claim"], "cited": rev["cited"],
                         "passages": {str(n): r["passages"].get(n, "") for n in rev["cited"]},
                         "model_label": rev["label"], "human_label": ""})  # fmt: skip
    rows = rows[:limit]
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    return len(rows)


def score() -> dict:
    rows = [
        json.loads(l)
        for l in LABELS.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    pairs = [
        (r["model_label"], r["human_label"])
        for r in rows
        if r.get("human_label") in VALID
    ]
    if not pairs:
        return {"labelled": 0, "calibrated": False}
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    pm, ph = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chance = sum(pm[k] * ph[k] for k in VALID) / (n * n)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else 1.0
    return {"labelled": n, "agreement": round(agree, 3), "cohen_kappa": round(kappa, 3),
            "calibrated": n >= 20 and kappa >= 0.6}  # fmt: skip


if __name__ == "__main__":
    if sys.argv[1:] == ["export"]:
        print(f"wrote {asyncio.run(export())} claims to {LABELS}")
    else:
        print(json.dumps(score(), indent=2))
