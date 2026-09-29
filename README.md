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

`GET /metrics` returns cache hit/miss, quota-exhaustion and rate-limit-wait counters.

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
