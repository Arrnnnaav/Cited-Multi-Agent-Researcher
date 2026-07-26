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
