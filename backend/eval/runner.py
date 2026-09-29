from datetime import datetime
from pathlib import Path

from jinja2 import Template

from backend.agents import orchestrator
from backend.agents.judge_agent import run as judge_run
from backend.eval.test_queries import TEST_QUERIES
from backend.eval.citation_checks import check_citations
from backend.schemas import EvalResult

REPORTS_DIR = Path("eval/reports")


async def run() -> list[EvalResult]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[EvalResult] = []
    for query in TEST_QUERIES:
        response = await orchestrator.run(query)
        scores = await judge_run(query, response.answer, response.sources)
        results.append(
            EvalResult(
                query=query,
                answer=response.answer,
                sources=response.sources,
                scores=scores,
                findings=check_citations(response.answer, response.sources),
            )
        )
    _write_report(results)
    return results


def latest_report_path() -> Path | None:
    reports = sorted(REPORTS_DIR.glob("*.html"))
    return reports[-1] if reports else None


def _write_report(results: list[EvalResult]) -> None:
    template_path = Path(__file__).parent / "report_template.html"
    template = Template(template_path.read_text())
    avg_factuality = sum(r.scores.factuality for r in results) / len(results)
    avg_citation = sum(r.scores.citation_coverage for r in results) / len(results)
    html = template.render(
        results=results,
        avg_factuality=round(avg_factuality, 3),
        avg_citation=round(avg_citation, 3),
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    (REPORTS_DIR / f"{timestamp}.html").write_text(html, encoding="utf-8")
