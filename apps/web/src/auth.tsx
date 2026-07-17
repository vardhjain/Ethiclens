import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import {
  api,
  clearRole,
  clearToken,
  getRole,
  getToken,
  setRole as persistRole,
  setToken,
  type Role,
} from "./api/client";

interface AuthState {
  token: string | null;
  role: Role | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTok] = useState<string | null>(getToken());
  // Restored from localStorage, not just held in memory — otherwise the role
  // silently reverts to null on every page refresh even though the token (and thus
  // the session) survives it.
  const [role, setRole] = useState<Role | null>(getRole());

  const value = useMemo<AuthState>(
    () => ({
      token,
      role,
      async login(email, password) {
        const t = await api.login(email, password);
        setToken(t.access_token);
        setTok(t.access_token);
        persistRole(t.role);
        setRole(t.role);
      },
      logout() {
        clearToken();
        clearRole();
        setTok(null);
        setRole(null);
      },
    }),
    [token, role],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
