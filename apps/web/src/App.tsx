import {
  ActionIcon,
  AppShell,
  Badge,
  Burger,
  Button,
  Center,
  Group,
  Loader,
  Paper,
  Stack,
  Title,
  useComputedColorScheme,
  useMantineColorScheme,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { lazy, Suspense } from "react";
import { Navigate, Link, Route, Routes, useNavigate } from "react-router-dom";
import { useAuth } from "./auth";
import { AgentNewPage } from "./pages/AgentNew";
import { LoginPage } from "./pages/Login";
import { NewAuditPage } from "./pages/NewAudit";
import { SessionsPage } from "./pages/Sessions";

// These two pages pull in recharts, which is heavy enough to noticeably bloat the
// main bundle. Split them into their own chunks, loaded only when visited.
const AgentRecordPage = lazy(() =>
  import("./pages/AgentRecord").then((m) => ({ default: m.AgentRecordPage })),
);
const SessionDetailPage = lazy(() =>
  import("./pages/SessionDetail").then((m) => ({
    default: m.SessionDetailPage,
  })),
);

function PageFallback() {
  return (
    <Center py="xl">
      <Loader />
    </Center>
  );
}

function ColorSchemeToggle() {
  const { setColorScheme } = useMantineColorScheme();
  const computedColorScheme = useComputedColorScheme("light");

  return (
    <ActionIcon
      variant="light"
      color="gray"
      size="lg"
      aria-label="Toggle color scheme"
      onClick={() =>
        setColorScheme(computedColorScheme === "dark" ? "light" : "dark")
      }
    >
      {computedColorScheme === "dark" ? "☀️" : "🌙"}
    </ActionIcon>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  const { logout, role } = useAuth();
  const nav = useNavigate();
  // Below the `sm` breakpoint the three nav links don't fit alongside the title and
  // Sign out button in the fixed-height header — they used to wrap onto extra lines
  // that spilled out of the header and over the page content. Collapse them into a
  // burger-triggered dropdown instead of letting that happen. Plain conditional
  // rendering rather than AppShell.Navbar's collapse mechanism or a Drawer/Modal:
  // both route the content through a CSS-variable-driven breakpoint system / portal
  // that never reflected the opened state changing in this Mantine version, even
  // though the state itself was confirmed toggling correctly on every click.
  const [opened, { toggle, close }] = useDisclosure();

  const navLinks = (
    <>
      <Button
        variant="subtle"
        component={Link}
        to="/"
        onClick={close}
        justify="start"
        fullWidth
      >
        Sessions
      </Button>
      <Button
        variant="subtle"
        component={Link}
        to="/new"
        onClick={close}
        justify="start"
        fullWidth
      >
        New audit
      </Button>
      <Button
        variant="subtle"
        component={Link}
        to="/agent/new"
        onClick={close}
        justify="start"
        fullWidth
      >
        Agent audit
      </Button>
    </>
  );

  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header style={{ position: "relative" }}>
        <Group h="100%" px="md" justify="space-between">
          <Group>
            <Burger
              opened={opened}
              onClick={toggle}
              hiddenFrom="sm"
              size="sm"
            />
            <Title order={4}>⚖️ EthicLens</Title>
            <Group visibleFrom="sm">{navLinks}</Group>
          </Group>
          <Group>
            {role && (
              <Badge variant="light" tt="capitalize" visibleFrom="sm">
                {role.replace(/_/g, " ")}
              </Badge>
            )}
            <ColorSchemeToggle />
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
        </Group>
        {opened && (
          <Paper
            hiddenFrom="sm"
            withBorder
            shadow="md"
            p="sm"
            style={{
              position: "absolute",
              top: "100%",
              left: 0,
              right: 0,
              zIndex: 199,
            }}
          >
            <Stack gap={4}>{navLinks}</Stack>
          </Paper>
        )}
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
        <Route
          path="/sessions/:id"
          element={
            <Suspense fallback={<PageFallback />}>
              <SessionDetailPage />
            </Suspense>
          }
        />
        <Route path="/agent/new" element={<AgentNewPage />} />
        <Route
          path="/agent/records/:id"
          element={
            <Suspense fallback={<PageFallback />}>
              <AgentRecordPage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  );
}
