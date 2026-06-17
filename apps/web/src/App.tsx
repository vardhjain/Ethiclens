import { AppShell, Button, Group, Title } from "@mantine/core";
import { Navigate, Link, Route, Routes, useNavigate } from "react-router-dom";
import { useAuth } from "./auth";
import { LoginPage } from "./pages/Login";
import { NewAuditPage } from "./pages/NewAudit";
import { SessionDetailPage } from "./pages/SessionDetail";
import { SessionsPage } from "./pages/Sessions";

function Shell({ children }: { children: React.ReactNode }) {
  const { logout } = useAuth();
  const nav = useNavigate();
  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group>
            <Title order={4}>⚖️ EthicLens</Title>
            <Button variant="subtle" component={Link} to="/">
              Sessions
            </Button>
            <Button variant="subtle" component={Link} to="/new">
              New audit
            </Button>
          </Group>
          <Button
            variant="light"
            color="gray"
            onClick={() => {
              logout();
              nav("/login");
            }}
          >
            Sign out
          </Button>
        </Group>
      </AppShell.Header>
      <AppShell.Main>{children}</AppShell.Main>
    </AppShell>
  );
}

export function App() {
  const { token } = useAuth();
  if (!token) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<SessionsPage />} />
        <Route path="/new" element={<NewAuditPage />} />
        <Route path="/sessions/:id" element={<SessionDetailPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  );
}
