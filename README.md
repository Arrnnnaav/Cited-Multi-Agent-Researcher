# Cited Multi-Agent Researcher

Multi-agent research system that answers queries with grounded, cited responses. Dynamically scales search subagents based on query type, deduplicates sources, and evaluates output quality via an LLM-as-judge pipeline.

Inspired by [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

## Architecture

```
Query → OrchestratorAgent
  ├─ classify: "fact" | "comparison"
  ├─ decompose: list[SubQuestion]
  ├─ [fact]       → 1× SearchAgent → CitationAgent → SynthesisAgent
  └─ [comparison] → N× SearchAgent (N = min(subtopics, 5), parallel) → CitationAgent → SynthesisAgent
```

| Agent | Role |
|---|---|
| `OrchestratorAgent` | Classify, decompose, spawn subagents, enforce cap |
| `SearchAgent` | Gemini Flash + grounding → raw results + URLs |
| `CitationAgent` | Dedup sources, assign `[N]` IDs, score credibility |
| `SynthesisAgent` | Write cited answer with inline `[N]` refs |
| `JudgeAgent` | Score factuality (0–1) + citation coverage (0–1) |

## v2: evidence-checked citations, run traces, Redis

**Evidence per source (M0).** v1 gave every grounding URL the same first 300 characters of the model's answer as its "snippet", so `[2]` appeared to support whatever `[1]` said. v2 inverts Gemini's `grounding_supports` (answer segment → chunk indices), so each source carries only the passages attributed to it. Each source is labelled `grounded`, `metadata_only` (retrieved, but no passage attributed) or `model_only` (no web source; never presented as a citation). Synthesis is told to cite only passages that support a claim.

**Deterministic citation checks** (`backend/eval/citation_checks.py`, no LLM): invalid `[N]` IDs, citing model-only text, citing a source with no passage, duplicate sources after URL normalization (case, fragments, `utm_*`/tracking params, trailing slash), unused sources, and factual-looking sentences with no citation (a warning for review). Every response returns its findings, and the eval runner records them next to the judge scores.

**Run traces + feedback (M1).** Each request gets a `run_id`. SQLite stores the query, classification, sub-questions, per-stage timings, every model call (the model that actually answered, latency, outcome), sources with evidence status, findings, cache hits and errors. `POST /feedback` joins ratings, issue categories and corrections to the run, and `GET /runs/{id}` returns the full trace. No judge call runs on the request path.

**Redis (optional; the app degrades to no-ops without it):**
- **Search cache:** keyed on normalized sub-question + pipeline version. A repeated sub-question skips the grounded search call. Model-only fallbacks are never cached.
- **Shared quota state:** a model that returns a daily-quota 429 is marked exhausted in Redis, so every worker skips it instead of each one burning a call to rediscover it.
- **Shared per-model rate window** (`GEMINI_RPM`): calls wait for the next window instead of collecting 429s.
- **Eval job queue on a Redis Stream:** `POST /eval/run` returns 202 with a job ID (it used to be a `GET` that ran ten live queries inside the request). `python -m backend.worker` consumes jobs with retries and a dead-letter stream. Poll `GET /eval/jobs/{id}`.

**Pluggable providers.**
- **LLM:** `LLM_PROVIDER=gemini` (google-genai SDK) or `openai_compat`. With `openai_compat`, NVIDIA NIM free models are tried first and OpenRouter `:free` models are the fallback. The failover rules: a rate-limited model is parked in Redis for 60s, a missing or retired model for a day, and an auth error skips that whole endpoint.
- **Search:** `SEARCH_PROVIDER=tavily` returns the extracted page text for each URL. Every citation is then checked against text that really came from that source, and the search step needs no LLM call. If Tavily fails, it falls back to Gemini grounding when a Google key is set. Cache keys include the provider, so results from one provider are never served as the other's.

`GET /metrics` returns cache hit/miss, quota-exhaustion and rate-limit-wait counters.

**Live check (29 Sep 2026).**
- **Grounding.** With the current Gemini models, `google_search` grounding through the deprecated `google-generativeai` SDK returned no web chunks for any of 3 test queries. v1 would have shown the model's own text as a citation. v2 labelled every source `model_only`, and the checker flagged all 4 completed answers as `cites_model_only`. The Gemini calls now use the supported `google-genai` SDK: a native async client, `google_search` grounding and typed errors, with 429 and 404 both falling through the model chain. A live re-check of grounding is pending until the key's daily free-tier quota resets; the 29 Sep runs used it up.
- **Retired models.** The run also showed that a retired model (404 `NotFound`) did not fall through the failover chain. It now does, and the model is marked unavailable in Redis for every worker.

**Live check (30 Sep 2026): Tavily + NIM, free tiers only** (`eval/live_v3_run.json`). Each of 3 queries was run twice.
- **Grounding:** 66 of 66 sources carry real page text, and there were 0 error-level citation findings.
- **Repeat queries:** all 7 searches were served from the Redis cache, cutting total time from 67.0s to 29.5s (56%). A repeated single-fact query went from 17.4s to 1.2s. The query-plan cache is what makes this work: LLM decomposition otherwise rewords the sub-questions on every run.
- **Failover:** NIM's primary model failed once during the run and `gpt-oss-20b` answered instead.

## v3: evaluation and one improvement cycle (plan M2–M4)

**Frozen, versioned cases** (`eval/cases/cases.jsonl`, built once by `python -m backend.eval.snapshot`). 13 cases, each storing its query plan and the Tavily evidence for every sub-question:

| Split | Cases | Purpose |
|---|---|---|
| dev | 3 | The only split a proposal may cite |
| promotion | 5 | Held out; a candidate is judged on these |
| regression | 3 | Previously good behaviour that must not get worse |
| challenge | 2 | Unanswerable query; prompt injection planted in a source passage |

Expected facts are any-of string groups, checked deterministically. A leakage check rejects the same query appearing in two splits.

**Replay.** `orchestrator.run(..., config=, plan=, search_fn=)` runs the real pipeline on frozen evidence, so a baseline and a candidate differ only in their config version.

**Claim-support review** (`backend/eval/claim_support.py`, rubric `support-v1`). Every cited sentence or bullet is a claim. One LLM call per answer labels each claim supported, contradicted, insufficient or uncertain, judged only against the passages it cites. It is **uncalibrated**: `python -m backend.eval.calibrate export` writes claims for a person to label, and `score` then reports agreement and Cohen's κ. Until that is done, the reviewer is one signal, not a gate on its own.

**Versioning** (`backend/versioning.py`). Versions are immutable and content-hashed (`configs/versions/`). The active pointer (`configs/active.json`) is a compare-and-swap that is replaced atomically. Each run reads its config once, so a promotion landing mid-request cannot change that request. Promotions and rollbacks are appended to `configs/promotions.jsonl`. Only the synthesis instructions are candidate-editable; the rubric is not.

**Loop CLI** (`python -m backend.selfimprove.cli`): `status`, `failures` (dev only), `propose`, `diff`, `evaluate` (held-out splits plus an A/A baseline repeat), `decide --promote|--reject --by NAME`, `rollback`, `history`. Promotion is refused if the candidate changed since it was evaluated, or if the active version is no longer the one it was evaluated against. There are no HTTP admin routes.

**First cycle (30 Sep 2026), result: no promotion.**
1. **Failures on dev with the baseline:** 2 unsupported citations, both "bundled" sentences that combine several facts and stack citations such as [2][6][7], where each passage supports only part of the sentence.
2. **Proposal `p1-atomic-citations`:** one claim per sentence, cite only sources whose passage states that claim, and give conflicting values separately.
3. **Held-out evaluation** (`eval/results/evaluations/93c61d360e.json`):

| | Claims | Unsupported | Rate | Fact recall | Challenge failures |
|---|---|---|---|---|---|
| baseline `v1` | 32 | 3 | 9.4% | 1.00 | 0 |
| candidate `v2` | 76 | 7 | 9.2% | 1.00 | 0 |
| baseline repeated (A/A) | 34 | 5 | 14.7% | 1.00 | 0 |

The candidate splits answers into 2.4× more claims, but its unsupported rate is unchanged. The baseline repeated against itself moved from 9.4% to 14.7% and flipped 5 of 10 per-case verdicts. On 10 held-out cases, run-to-run noise is larger than the effect being tested, so the gate recommends rejecting the candidate.

Two changes the evaluation forced along the way:
- The claim extractor used to merge bullet lists into a single claim, which hid unsupported bullets from review. It now splits per line and per sentence.
- One hand-written expected fact was too literal and was revised.

Known limit: per-case verdicts compare unsupported *counts*, which penalizes candidates that write more, smaller claims. It stays as is for this cycle, since changing a metric after seeing the result would be moving the goalposts. The next steps are more cases and human-labelled calibration.

## Setup

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Frontend:
```bash
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173` — backend runs on `http://localhost:8000`.

## Scaling Heuristics

```python
SUBAGENT_CAP = 5  # configurable via env

def get_subagent_count(query_type, num_subtopics):
    if query_type == "fact":
        return 1
    return min(num_subtopics, SUBAGENT_CAP)
```

Parallel via `asyncio.gather` — latency = max(agents), not sum.

## Eval Pipeline

```bash
# Run LLM-as-judge eval on 10 test queries (5 fact + 5 comparison)
curl http://localhost:8000/eval/run

# View HTML report with per-query scores
curl http://localhost:8000/eval/report
# or open in browser: http://localhost:8000/eval/report
```

Report scores each answer on:
- **Factuality** (0–1): Are claims accurate?
- **Citation Coverage** (0–1): Are claims backed by `[N]` refs?

## Tests

```bash
pytest -v                              # all 33 unit tests
pytest tests/test_orchestrator.py      # orchestrator logic + cap
pytest tests/test_citation_agent.py    # dedup + credibility scoring
pytest tests/test_eval_runner.py       # eval pipeline
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/research` | SSE stream: token events + sources + done |
| `GET` | `/eval/run` | Run LLM-as-judge eval on 10 queries |
| `GET` | `/eval/report` | Serve latest HTML eval report |
| `GET` | `/health` | Health check |

## Key Design Decisions

- **Dynamic scaling** — fact → 1 agent, comparison → N agents (capped at 5)
- **CitationAgent decoupled from SynthesisAgent** — deduplicates across all SearchAgent outputs independently; each agent independently testable
- **LLM-as-judge** — two orthogonal eval axes: factuality and citation coverage; structured JSON output enforced
- **Gemini Flash grounding** — native web search in the model, no separate retrieval infra needed
- **asyncio.gather** — subagents run in parallel, latency bounded by slowest not sum of all
- **SSE streaming** — user sees answer tokens as they arrive, sources sidebar populates on completion
