"""Versioning, replay, comparison gate, proposals and promotion/rollback (M2-M4)."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend import versioning
from backend.eval import cases as C
from backend.eval import compare
from backend.eval.claim_support import _parse, extract_claims
from backend.schemas import RawResult
from backend.selfimprove import proposals as P


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolated config root, cases file and results dir."""
    root = tmp_path / "configs"
    monkeypatch.setattr(versioning, "ROOT", root)
    monkeypatch.setattr(versioning, "VERSIONS", root / "versions")
    monkeypatch.setattr(versioning, "ACTIVE", root / "active.json")
    monkeypatch.setattr(versioning, "HISTORY", root / "promotions.jsonl")
    monkeypatch.setattr(P, "PROPOSALS", root / "proposals")
    monkeypatch.setattr(C, "CASES_PATH", tmp_path / "cases.jsonl")
    monkeypatch.setattr(compare, "RESULTS", tmp_path / "evals")
    monkeypatch.setattr(P, "RESULTS", tmp_path / "evals")
    versioning.ensure_baseline()
    return tmp_path


def _case(cid, split, facts=(("Paris",),), **kw):
    ev = {"q": [RawResult(url="https://a.com", title="A", snippet="Paris is the capital of France.",
                          evidence_status="grounded")]}  # fmt: skip
    return C.EvalCase(case_id=cid, split=split, query=f"query {cid}", query_type="fact",
                      sub_questions=["q"], evidence=ev, expected_facts=[list(f) for f in facts], **kw)  # fmt: skip


# ---- versioning -------------------------------------------------------------


def test_versions_are_immutable_and_hashed(iso):
    v = versioning.save(
        versioning.ConfigVersion(
            version_id="v2", synthesis_instructions="Be strict about citing."
        )
    )
    assert versioning.get("v2").content_hash == v.content_hash
    with pytest.raises(ValueError):
        versioning.save(
            versioning.ConfigVersion(
                version_id="v2", synthesis_instructions="Something else entirely"
            )
        )


def test_stale_promotion_is_refused_and_history_recorded(iso):
    versioning.save(
        versioning.ConfigVersion(
            version_id="v2", synthesis_instructions="Be strict about citing."
        )
    )
    with pytest.raises(versioning.StalePromotion):
        versioning.set_active(
            "v2", expected_active="not-the-active-one", actor="t", reason="r"
        )
    versioning.set_active("v2", expected_active="v1-baseline", actor="t", reason="r")
    assert versioning.active().version_id == "v2"
    versioning.set_active(
        "v1-baseline", expected_active="v2", actor="t", reason="rb", action="rollback"
    )
    assert versioning.active().version_id == "v1-baseline"
    assert [h["action"] for h in versioning.history()] == ["promote", "rollback"]


def test_tampered_version_file_is_detected(iso):
    path = versioning.VERSIONS / "v1-baseline.json"
    data = json.loads(path.read_text())
    data["synthesis_instructions"] = "tampered instructions here"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        versioning.active()


async def test_run_reads_config_once_and_records_it(iso):
    from backend.agents import orchestrator

    v2 = versioning.save(
        versioning.ConfigVersion(
            version_id="v2", synthesis_instructions="Be strict about citing."
        )
    )
    seen = {}

    async def synth(query, sources, instructions=None):
        seen["instructions"] = instructions
        # a promotion landing mid-run must not change this run's config
        versioning.set_active(
            "v2", expected_active="v1-baseline", actor="t", reason="r"
        )
        return "Paris [1]."

    case = _case("c1", "promotion")
    with patch.object(orchestrator.synthesis_agent, "run", synth):
        resp = await orchestrator.run(case.query, trace=False,
                                      plan={"query_type": "fact", "sub_questions": ["q"]},
                                      search_fn=C.replay_search(case))  # fmt: skip
    assert seen["instructions"] == versioning.BASELINE_INSTRUCTIONS
    assert resp.config_version.startswith("v1-baseline@")
    assert versioning.active().version_id == v2.version_id


# ---- cases, claims, gate -------------------------------------------------------


def test_split_leakage_detected():
    a, b = _case("a", "dev"), _case("b", "promotion")
    b.query = a.query
    assert C.check_split_leakage([a, b])


def test_claim_extraction_and_judge_parsing():
    claims = extract_claims(
        "Paris is the capital [1]. It is large. It has the Louvre [1, 2]."
    )
    assert [c[1] for c in claims] == [[1], [1, 2]]
    parsed = _parse(
        'noise [{"i": 1, "label": "supported"}, {"i": 2, "label": "bogus"}]', 2
    )
    assert [p["label"] for p in parsed] == ["supported", "uncertain"]


def _row(**kw):
    base = dict(
        unsupported=0, citation_errors=0, fact_recall=1.0, abstained_ok=None, leaked=[]
    )
    return base | kw


def test_verdicts():
    assert compare.verdict(_row(unsupported=2), _row(unsupported=0)) == "win"
    assert compare.verdict(_row(), _row(fact_recall=0.5)) == "loss"
    assert compare.verdict(_row(), _row(leaked=["1999"])) == "loss"
    assert compare.verdict(_row(), _row()) == "tie"


async def test_evaluate_replays_frozen_evidence_and_worse_candidate_fails_gate(iso):
    C.save(
        [
            _case("p1", "promotion"),
            _case("p2", "promotion"),
            _case("c1", "challenge", facts=(), must_abstain=True),
        ]
    )
    versioning.save(
        versioning.ConfigVersion(
            version_id="v-bad", synthesis_instructions="Cite whatever you like."
        )
    )
    search_calls = []

    async def synth(query, sources, instructions=None):
        search_calls.append(query)
        if "whatever" in (instructions or ""):
            return "Berlin is the capital [1]."  # wrong + unsupported
        return (
            "Paris is the capital [1]."
            if "c1" not in query
            else "The sources do not contain this."
        )

    async def judge(answer, sources):
        from backend.eval.claim_support import ClaimReview

        label = "supported" if "Paris" in answer else "contradicted"
        return [
            ClaimReview(claim=c, cited=ids, label=label)
            for c, ids in extract_claims(answer)
        ]

    with patch("backend.agents.orchestrator.synthesis_agent.run", synth), \
         patch("backend.eval.compare.review", judge), \
         patch("backend.agents.search_agent.run", AsyncMock(side_effect=AssertionError("live search"))):  # fmt: skip
        report = await compare.evaluate("v-bad", ["promotion", "challenge"])
    assert report["verdicts"]["loss"] >= 2
    assert report["gate"]["recommend_promotion"] is False
    assert report["aa_noise_flips"] == 0


async def test_proposal_lifecycle_promote_then_rollback(iso):
    C.save([_case("d1", "dev"), _case("p1", "promotion")])
    versioning.save(
        versioning.ConfigVersion(
            version_id="v2", synthesis_instructions="Be strict about citing."
        )
    )
    with pytest.raises(ValueError):  # proposals may only cite dev cases
        P.save(P.Proposal(proposal_id="x", target_failure="unsupported_citation", dev_case_ids=["p1"],
                          from_version="v1-baseline", candidate_version="v2", rationale="r",
                          predicted_benefit="b", predicted_risk="r"))  # fmt: skip
    prop = P.save(P.Proposal(proposal_id="p-1", target_failure="unsupported_citation", dev_case_ids=["d1"],
                             from_version="v1-baseline", candidate_version="v2", rationale="r",
                             predicted_benefit="b", predicted_risk="r"))  # fmt: skip
    with pytest.raises(ValueError):
        P.decide(prop, promote=True, actor="t", rationale="not evaluated yet")
    report = {"evaluation_id": "e1", "candidate_hash": versioning.get("v2").content_hash,
              "expected_active": "v1-baseline", "gate": {"recommend_promotion": True, "reasons": []},
              "verdicts": {"win": 1, "loss": 0, "tie": 0}}  # fmt: skip
    (compare.RESULTS).mkdir(parents=True, exist_ok=True)
    (compare.RESULTS / "e1.json").write_text(json.dumps(report))
    prop = P.record_evaluation(prop, report)
    P.decide(prop, promote=True, actor="reviewer", rationale="held-out wins")
    assert versioning.active().version_id == "v2"
    versioning.set_active(
        "v1-baseline",
        expected_active="v2",
        actor="reviewer",
        reason="drill",
        action="rollback",
    )
    assert versioning.active().version_id == "v1-baseline"
    assert P.get("p-1").status == "promoted"


def test_bullets_are_separate_claims_and_headings_dropped():
    answer = ("TCP is reliable.\n\n**Reliability**\n- TCP retransmits lost segments [1][2].\n"
              "- UDP has no acknowledgments [3].\n1. Both run over IP [4].")  # fmt: skip
    claims = extract_claims(answer)
    assert [c[1] for c in claims] == [[1, 2], [3], [4]]


def test_fact_matching_normalizes_unicode_hyphens():
    assert compare._hit("no connection\u2011setup handshake", ["connection-setup"])


def test_iso_fixture_really_redirects_case_file(iso):
    real = C.Path(__file__).resolve().parents[1] / "eval" / "cases" / "cases.jsonl"
    before = real.read_bytes() if real.exists() else None
    C.save([_case("zz", "dev")])
    assert (real.read_bytes() if real.exists() else None) == before
