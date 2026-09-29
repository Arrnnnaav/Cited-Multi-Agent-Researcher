export interface CitedSource {
  id: number;
  url: string;
  title: string;
  snippet: string;
  credibility_score: number;
  evidence_status?: "grounded" | "metadata_only" | "model_only";
}

export interface Finding {
  kind: string;
  severity: "error" | "warning" | "info";
  detail: string;
}

export interface ResearchMeta {
  query_type: "fact" | "comparison";
  subagents_used: number;
  latency_ms: number;
}

export type SSEEvent =
  | { type: "token"; content: string }
  | {
      type: "sources";
      sources: CitedSource[];
      query_type: string;
      subagents_used: number;
      latency_ms: number;
      run_id?: string;
      findings?: Finding[];
      cache_hits?: number;
    }
  | { type: "done" }
  | { type: "error"; message: string };

export type Phase = "idle" | "streaming" | "done" | "error";
