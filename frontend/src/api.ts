import type {
  AuthResponse,
  InteractionGraph,
  Job,
  MatchBundle,
  MatchListItem,
  MatchSummary,
  PlayerCard,
  Possession,
  RankingItem,
  TacticalAnswer,
  User,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";
const TOKEN_KEY = "athletiq.token";

export const UNREACHABLE = "Cannot reach the AthletiQ API — the backend isn't responding.";

export class ApiError extends Error {
  /** `unreachable` means the backend itself could not be reached, as opposed to it rejecting the request. */
  constructor(message: string, public status?: number, public unreachable = false) {
    super(message);
  }
}

// The Vite dev proxy answers 502/503/504 with an empty body when the backend is down.
const GATEWAY = new Set([502, 503, 504]);

// ---- Session token (localStorage can throw in private mode, so guard it) ----
let token: string | null = (() => {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
})();
let onUnauthorized: () => void = () => {};

export const session = {
  get token() { return token; },
  set(t: string | null) {
    token = t;
    try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
  },
  onUnauthorized(fn: () => void) { onUnauthorized = fn; },
};

const authHeaders = (): Record<string, string> => (token ? { Authorization: `Bearer ${token}` } : {});

function detailOf(body: unknown): string | null {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d: { msg: string }) => d.msg).join("; ");
  return null;
}

async function errorFor(res: Response): Promise<ApiError> {
  let detail: string | null = null;
  try { detail = detailOf(await res.json()); } catch { /* non-JSON error body */ }
  if (!detail && GATEWAY.has(res.status)) return new ApiError(UNREACHABLE, res.status, true);
  return new ApiError(detail ?? res.statusText, res.status);
}

async function request<T>(path: string, init: RequestInit = {}, asText = false): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers: { ...authHeaders(), ...init.headers } });
  } catch {
    throw new ApiError(UNREACHABLE, undefined, true);
  }
  if (res.status === 401 && token) onUnauthorized();
  if (!res.ok) throw await errorFor(res);
  if (res.status === 204) return undefined as T;
  return (asText ? res.text() : res.json()) as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export interface UploadOptions {
  title: string;
  startSeconds: number;
  durationSeconds?: number;
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
}

export const api = {
  health: () => request<{ status: string; version: string; default_match: string; database: string }>("/api/health"),

  register: (name: string, email: string, password: string) => request<AuthResponse>("/api/auth/register", json({ name, email, password })),
  login: (email: string, password: string) => request<AuthResponse>("/api/auth/login", json({ email, password })),
  requestPasswordReset: (email: string) => request<{ message: string; reset_token?: string }>("/api/auth/password-reset/request", json({ email })),
  confirmPasswordReset: (token: string, password: string) => request<void>("/api/auth/password-reset/confirm", json({ token, password })),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  me: () => request<User>("/api/auth/me"),

  matches: () => request<MatchListItem[]>("/api/matches"),
  jobs: () => request<Job[]>("/api/jobs"),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  deleteJob: (id: string) => request<void>(`/api/jobs/${id}`, { method: "DELETE" }),

  /** Multipart upload via XHR, since fetch cannot report upload progress. */
  upload: (file: File, opts: UploadOptions) =>
    new Promise<Job>((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      form.append("title", opts.title);
      form.append("start_seconds", String(opts.startSeconds));
      if (opts.durationSeconds) form.append("duration_seconds", String(opts.durationSeconds));
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/api/jobs`);
      if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      xhr.upload.onprogress = (e) => e.lengthComputable && opts.onProgress?.(e.loaded / e.total);
      xhr.onload = () => {
        let body: { detail?: unknown } = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* empty */ }
        if (xhr.status >= 200 && xhr.status < 300) return resolve(body as Job);
        if (xhr.status === 401) onUnauthorized();
        const detail = detailOf(body);
        if (!detail && GATEWAY.has(xhr.status)) return reject(new ApiError(UNREACHABLE, xhr.status, true));
        reject(new ApiError(detail ?? `Upload failed (${xhr.status})`, xhr.status));
      };
      xhr.onerror = () => reject(new ApiError(UNREACHABLE, undefined, true));
      xhr.onabort = () => reject(new ApiError("Upload cancelled"));
      opts.signal?.addEventListener("abort", () => xhr.abort());
      xhr.send(form);
    }),

  report: (id: string) => request<{ report_content: string }>(`/api/matches/${id}/report`).then((r) => r.report_content),
  ask: (id: string, query: string) => request<TacticalAnswer>(`/api/matches/${id}/tactical-query`, json({ query })),
  bundle: async (id: string): Promise<MatchBundle> => {
    const m = `/api/matches/${id}`;
    const [summary, players, rankings, possession, graph, manifest] = await Promise.all([
      request<MatchSummary>(`${m}/summary`),
      request<PlayerCard[]>(`${m}/players`),
      request<RankingItem[]>(`${m}/rankings`),
      request<Possession>(`${m}/possession`),
      request<InteractionGraph>(`${m}/interaction-graph`),
      request<string>(`${m}/manifest`, {}, true),
    ]);
    return { summary, players, rankings, possession, graph, manifest };
  },
};
