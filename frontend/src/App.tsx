import QueryInput from "./components/QueryInput";
import AnswerView from "./components/AnswerView";
import CitationPanel from "./components/CitationPanel";
import { useResearch } from "./useResearch";

export default function App() {
  const { phase, answer, sources, meta, error, query, run } = useResearch();
  const busy = phase === "streaming";
  const showResults = phase !== "idle";

  const scrollToSource = (id: number) => {
    document
      .getElementById(`source-${id}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <div style={{ maxWidth: 1080, margin: "0 auto", padding: "56px 24px 80px" }}>
      {/* Header */}
      <header style={{ marginBottom: 36 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 11, marginBottom: 10 }}>
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 9,
              background: "linear-gradient(135deg, var(--accent), var(--accent-bright))",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 17,
              boxShadow: "0 4px 16px var(--accent-glow)",
            }}
          >
            🔎
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>
            Cited Researcher
          </h1>
          <span
            style={{
              fontSize: 11,
              fontFamily: "var(--mono)",
              color: "var(--text-faint)",
              border: "1px solid var(--border)",
              padding: "2px 8px",
              borderRadius: 99,
            }}
          >
            multi-agent
          </span>
        </div>
        <p style={{ color: "var(--text-dim)", fontSize: 14.5, lineHeight: 1.5, maxWidth: 620 }}>
          An orchestrator classifies your query, fans out to parallel search
          subagents (1 for facts, N for comparisons), then deduplicates and
          grounds every claim with inline citations.
        </p>
      </header>

      <QueryInput onSubmit={run} busy={busy} />

      {showResults && (
        <div style={{ marginTop: 30 }}>
          {phase === "error" ? (
            <div
              style={{
                padding: "18px 20px",
                background: "rgba(248,113,113,0.08)",
                border: "1px solid rgba(248,113,113,0.3)",
                borderRadius: "var(--radius)",
                color: "var(--red)",
                fontSize: 14,
              }}
            >
              <strong>Request failed.</strong> {error}
              <div style={{ color: "var(--text-dim)", marginTop: 6, fontSize: 13 }}>
                Is the backend running on <code>localhost:8000</code>? Free-tier
                rate limits can also cause this — wait a moment and retry.
              </div>
            </div>
          ) : (
            <>
              <div
                style={{
                  fontSize: 13,
                  color: "var(--text-faint)",
                  marginBottom: 14,
                  fontFamily: "var(--mono)",
                }}
              >
                ❯ {query}
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: sources.length ? "1.7fr 1fr" : "1fr",
                  gap: 26,
                  alignItems: "start",
                }}
              >
                <AnswerView
                  answer={answer}
                  meta={meta}
                  phase={phase}
                  onCitationClick={scrollToSource}
                />
                {sources.length > 0 && <CitationPanel sources={sources} />}
              </div>
            </>
          )}
        </div>
      )}

      <footer
        style={{
          marginTop: 64,
          paddingTop: 20,
          borderTop: "1px solid var(--border)",
          color: "var(--text-faint)",
          fontSize: 12,
          display: "flex",
          gap: 16,
          flexWrap: "wrap",
        }}
      >
        <span>Gemini 2.5 Flash</span>
        <span>·</span>
        <span>FastAPI · SSE streaming</span>
        <span>·</span>
        <span>LLM-as-judge eval at /eval/report</span>
      </footer>
    </div>
  );
}
