import type { CitedSource } from "../types";

interface Props {
  sources: CitedSource[];
}

function credColor(score: number): string {
  if (score >= 0.85) return "var(--green)";
  if (score >= 0.6) return "var(--amber)";
  return "var(--red)";
}

function hostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "model knowledge";
  }
}

export default function CitationPanel({ sources }: Props) {
  return (
    <aside style={{ position: "sticky", top: 24, alignSelf: "start" }}>
      <h3
        style={{
          fontSize: 12,
          color: "var(--text-faint)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: 14,
          fontWeight: 700,
        }}
      >
        Sources · {sources.length}
      </h3>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {sources.map((s, i) => (
          <div
            key={s.id}
            id={`source-${s.id}`}
            style={{
              padding: "13px 15px",
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: 11,
              animation: `fadeUp 0.4s ease ${i * 0.05}s both`,
              scrollMarginTop: 24,
            }}
          >
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 6 }}>
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 11,
                  fontWeight: 700,
                  color: "var(--accent-bright)",
                  background: "var(--accent-glow)",
                  padding: "1px 7px",
                  borderRadius: 5,
                  flexShrink: 0,
                }}
              >
                {s.id}
              </span>
              <span style={{ fontSize: 13.5, fontWeight: 600, lineHeight: 1.35 }}>
                {s.url ? (
                  <a
                    href={s.url}
                    target="_blank"
                    rel="noreferrer"
                    style={{ color: "var(--text)", textDecoration: "none" }}
                  >
                    {s.title}
                  </a>
                ) : (
                  s.title
                )}
              </span>
            </div>

            <div
              style={{
                fontSize: 11,
                color: "var(--text-faint)",
                fontFamily: "var(--mono)",
                marginBottom: 9,
                marginLeft: 2,
              }}
            >
              {hostname(s.url)}
            </div>

            <p
              style={{
                fontSize: 12.5,
                color: "var(--text-dim)",
                lineHeight: 1.55,
                marginBottom: 11,
              }}
            >
              {s.snippet.slice(0, 130)}
              {s.snippet.length > 130 ? "…" : ""}
            </p>

            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  flex: 1,
                  height: 5,
                  borderRadius: 99,
                  background: "var(--border)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${Math.round(s.credibility_score * 100)}%`,
                    height: "100%",
                    borderRadius: 99,
                    background: credColor(s.credibility_score),
                    transition: "width 0.5s ease",
                  }}
                />
              </div>
              <span
                style={{
                  fontSize: 10.5,
                  fontFamily: "var(--mono)",
                  color: credColor(s.credibility_score),
                  fontWeight: 600,
                }}
              >
                {Math.round(s.credibility_score * 100)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
