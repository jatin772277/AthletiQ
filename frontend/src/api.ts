import type {
  InteractionGraph,
  MatchBundle,
  MatchListItem,
  MatchSummary,
  PlayerCard,
  Possession,
  RankingItem,
  TacticalAnswer,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit, asText = false): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("Cannot reach the AthletiQ API. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return (asText ? res.text() : res.json()) as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; version: string; default_match: string }>("/api/health"),
  // Only folders with a RAG package are fully processed matches.
  matches: async () =>
    (await request<MatchListItem[]>("/api/matches")).filter((m) => m.has_rag_package),
  report: (id: string) =>
    request<{ report_content: string }>(`/api/matches/${id}/report`).then((r) => r.report_content),
  ask: (id: string, query: string) =>
    request<TacticalAnswer>(`/api/matches/${id}/tactical-query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }),
  bundle: async (id: string): Promise<MatchBundle> => {
    const m = `/api/matches/${id}`;
    const [summary, players, rankings, possession, graph, manifest] = await Promise.all([
      request<MatchSummary>(`${m}/summary`),
      request<PlayerCard[]>(`${m}/players`),
      request<RankingItem[]>(`${m}/rankings`),
      request<Possession>(`${m}/possession`),
      request<InteractionGraph>(`${m}/interaction-graph`),
      request<string>(`${m}/manifest`, undefined, true),
    ]);
    return { summary, players, rankings, possession, graph, manifest };
  },
};
