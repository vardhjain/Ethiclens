import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { api, clearToken, getToken, setToken, type Role } from "./api/client";

interface AuthState {
  token: string | null;
  role: Role | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTok] = useState<string | null>(getToken());
  const [role, setRole] = useState<Role | null>(null);

  const value = useMemo<AuthState>(
    () => ({
      token,
      role,
      async login(email, password) {
        const t = await api.login(email, password);
        setToken(t.access_token);
        setTok(t.access_token);
        setRole(t.role);
      },
      logout() {
        clearToken();
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
