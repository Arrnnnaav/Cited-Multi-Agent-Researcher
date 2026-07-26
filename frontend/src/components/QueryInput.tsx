import { useState } from "react";

interface Props {
  onSubmit: (query: string) => void;
  busy: boolean;
}

const EXAMPLES = [
  "What is the speed of light?",
  "Compare React and Vue for SPAs",
  "Compare SQL vs NoSQL at scale",
  "Who invented the World Wide Web?",
];

export default function QueryInput({ onSubmit, busy }: Props) {
  const [value, setValue] = useState("");

  const submit = (q: string) => {
    const text = q.trim();
    if (text && !busy) onSubmit(text);
  };

  return (
    <div style={{ animation: "fadeUp 0.5s ease both" }}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(value);
        }}
        style={{
          display: "flex",
          gap: 10,
          padding: 6,
          background: "var(--bg-elev)",
          border: "1px solid var(--border-bright)",
          borderRadius: 16,
          boxShadow: "0 8px 30px rgba(0,0,0,0.4)",
        }}
      >
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask a research question…"
          disabled={busy}
          autoFocus
          style={{
            flex: 1,
            padding: "14px 16px",
            fontSize: 16,
            background: "transparent",
            border: "none",
            outline: "none",
            color: "var(--text)",
            fontFamily: "inherit",
          }}
        />
        <button
          type="submit"
          disabled={busy || !value.trim()}
          style={{
            padding: "0 24px",
            fontSize: 15,
            fontWeight: 600,
            borderRadius: 11,
            border: "none",
            cursor: busy || !value.trim() ? "default" : "pointer",
            color: "#fff",
            background:
              busy || !value.trim()
                ? "var(--border-bright)"
                : "linear-gradient(135deg, var(--accent), var(--accent-bright))",
            boxShadow: busy || !value.trim() ? "none" : "0 4px 16px var(--accent-glow)",
            transition: "transform 0.1s, box-shadow 0.2s",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          {busy ? (
            <>
              <span
                style={{
                  width: 13,
                  height: 13,
                  border: "2px solid rgba(255,255,255,0.4)",
                  borderTopColor: "#fff",
                  borderRadius: "50%",
                  display: "inline-block",
                  animation: "spin 0.7s linear infinite",
                }}
              />
              Researching
            </>
          ) : (
            "Research →"
          )}
        </button>
      </form>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 14 }}>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => {
              setValue(ex);
              submit(ex);
            }}
            disabled={busy}
            style={{
              padding: "7px 13px",
              fontSize: 12.5,
              borderRadius: 99,
              border: "1px solid var(--border)",
              background: "var(--bg-card)",
              color: "var(--text-dim)",
              cursor: busy ? "default" : "pointer",
              fontFamily: "inherit",
              transition: "border-color 0.15s, color 0.15s",
            }}
            onMouseEnter={(e) => {
              if (busy) return;
              e.currentTarget.style.borderColor = "var(--accent)";
              e.currentTarget.style.color = "var(--text)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--border)";
              e.currentTarget.style.color = "var(--text-dim)";
            }}
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
