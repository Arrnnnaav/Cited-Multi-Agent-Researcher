import { useCallback, useRef, useState } from "react";
import type { CitedSource, Phase, ResearchMeta, SSEEvent } from "./types";

const API = "http://localhost:8000";

interface ResearchState {
  phase: Phase;
  answer: string;
  sources: CitedSource[];
  meta: ResearchMeta | null;
  error: string | null;
  query: string;
}

const initial: ResearchState = {
  phase: "idle",
  answer: "",
  sources: [],
  meta: null,
  error: null,
  query: "",
};

export function useResearch() {
  const [state, setState] = useState<ResearchState>(initial);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async (query: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState({ ...initial, phase: "streaming", query });

    try {
      const resp = await fetch(`${API}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        throw new Error(`Backend responded ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          const event = JSON.parse(part.slice(6)) as SSEEvent;

          if (event.type === "token") {
            setState((s) => ({ ...s, answer: s.answer + event.content }));
          } else if (event.type === "sources") {
            setState((s) => ({
              ...s,
              sources: event.sources,
              meta: {
                query_type: event.query_type as "fact" | "comparison",
                subagents_used: event.subagents_used,
                latency_ms: event.latency_ms,
              },
            }));
          } else if (event.type === "done") {
            setState((s) => ({ ...s, phase: "done" }));
          } else if (event.type === "error") {
            setState((s) => ({ ...s, phase: "error", error: event.message }));
          }
        }
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      setState((s) => ({
        ...s,
        phase: "error",
        error: err instanceof Error ? err.message : "Request failed",
      }));
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(initial);
  }, []);

  return { ...state, run, reset };
}
