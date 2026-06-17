// Typed API client for the EthicLens service. Requests go through Vite's /api
// proxy in dev and the nginx reverse-proxy in production.

const TOKEN_KEY = "ethiclens_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (!(init.body instanceof FormData) && init.body) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`/api${path}`, { ...init, headers });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

// --- Types (mirror the API schemas) ---------------------------------------

export type Role = "compliance_officer" | "ml_engineer" | "governance_approver" | "admin";

export interface Token {
  access_token: string;
  token_type: string;
  role: Role;
}
export interface SessionOut {
  id: string;
  name: string;
  status: string;
  dataset: string;
  composite_score: number | null;
  composite_band: string | null;
  min_di: number | null;
  parent_session_id: string | null;
  locked: boolean;
  created_at: string;
  completed_at: string | null;
}
export interface MetricOut {
  group_label: string;
  metric_type: string;
  value: number | null;
  ci_low: number | null;
  ci_high: number | null;
  p_value: number | null;
  n: number | null;
  classification: string | null;
}
export interface MetricsResponse {
  session_id: string;
  composite_score: number | null;
  composite_band: string | null;
  min_di: number | null;
  has_labels: boolean;
  metrics: MetricOut[];
}
export interface Recommendation {
  rank: number;
  strategy: string;
  strategy_name: string;
  description: string;
  estimated_di_improvement: number;
  stage: string;
  measured: boolean;
}
export interface RecommendationsResponse {
  flagged: boolean;
  recommendations: Record<string, Recommendation[]>;
}

// --- Endpoints ------------------------------------------------------------

export const api = {
  async login(email: string, password: string): Promise<Token> {
    const body = new URLSearchParams({ username: email, password });
    const res = await fetch("/api/auth/login", { method: "POST", body });
    if (!res.ok) throw new Error("Invalid email or password");
    return (await res.json()) as Token;
  },
  register: (email: string, password: string, role: Role) =>
    request("/auth/register", { method: "POST", body: JSON.stringify({ email, password, role }) }),
  listSessions: () => request<SessionOut[]>("/sessions"),
  getSession: (id: string) => request<SessionOut>(`/sessions/${id}`),
  createSession: (payload: Record<string, unknown>) =>
    request<SessionOut>("/sessions/create", { method: "POST", body: JSON.stringify(payload) }),
  runSession: (id: string) => request(`/sessions/${id}/run`, { method: "POST" }),
  metrics: (id: string) => request<MetricsResponse>(`/sessions/${id}/metrics`),
  recommendations: (id: string) => request<RecommendationsResponse>(`/sessions/${id}/recommendations`),
  mitigate: (id: string, strategy: string) =>
    request<{ result_session_id: string | null }>(`/sessions/${id}/mitigate`, {
      method: "POST",
      body: JSON.stringify({ strategy }),
    }),
  reportUrl: (id: string) => `/api/sessions/${id}/report`,
};
