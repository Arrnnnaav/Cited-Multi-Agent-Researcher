import type { Phase, ResearchMeta } from "../types";

interface Props {
  answer: string;
  meta: ResearchMeta | null;
  phase: Phase;
  onCitationClick: (id: number) => void;
}

// Split answer on [N] markers and render each marker as a clickable chip.
function renderWithCitations(text: string, onClick: (id: number) => void) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) return <span key={i}>{part}</span>;
    const id = Number(m[1]);
    return (
      <sup
        key={i}
        onClick={() => onClick(id)}
        title={`Jump to source ${id}`}
        style={{
          cursor: "pointer",
          color: "var(--accent-bright)",
          background: "var(--accent-glow)",
          padding: "1px 5px",
          borderRadius: 5,
          fontSize: "0.7em",
          fontWeight: 700,
          margin: "0 1px",
          fontFamily: "var(--mono)",
        }}
      >
        {id}
      </sup>
    );
  });
}

const TYPE_LABEL: Record<string, string> = {
  fact: "◆ Fact lookup · 1 agent",
  comparison: "◆◆ Comparison · multi-agent",
};

export default function AnswerView({ answer, meta, phase, onCitationClick }: Props) {
  const streaming = phase === "streaming";

  return (
    <div
      style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "26px 28px",
        animation: "fadeUp 0.4s ease both",
        minHeight: 120,
      }}
    >
      {meta && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 18 }}>
          <Badge
            text={TYPE_LABEL[meta.query_type] ?? meta.query_type}
            color="var(--accent-bright)"
          />
          <Badge text={`${meta.subagents_used} subagent${meta.subagents_used > 1 ? "s" : ""}`} />
          <Badge text={`${(meta.latency_ms / 1000).toFixed(1)}s`} />
        </div>
      )}

      <div style={{ fontSize: 16, lineHeight: 1.8, color: "var(--text)" }}>
        {answer ? (
          renderWithCitations(answer, onCitationClick)
        ) : streaming ? (
          <span style={{ color: "var(--text-dim)", display: "inline-flex", gap: 10, alignItems: "center" }}>
            <Dots /> orchestrating agents…
          </span>
        ) : null}
        {streaming && answer && (
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 18,
              marginLeft: 2,
              background: "var(--accent-bright)",
              verticalAlign: "text-bottom",
              animation: "blink 1s step-start infinite",
            }}
          />
        )}
      </div>
    </div>
  );
}

function Badge({ text, color }: { text: string; color?: string }) {
  return (
    <span
      style={{
        fontSize: 11.5,
        fontWeight: 600,
        padding: "4px 10px",
        borderRadius: 99,
        background: "var(--bg-elev)",
        border: "1px solid var(--border-bright)",
        color: color ?? "var(--text-dim)",
        fontFamily: "var(--mono)",
        letterSpacing: "0.02em",
      }}
    >
      {text}
    </span>
  );
}

function Dots() {
  return (
    <span style={{ display: "inline-flex", gap: 4 }}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: "var(--accent)",
            animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
    </span>
  );
}
