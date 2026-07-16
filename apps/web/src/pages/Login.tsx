import {
  Alert,
  Button,
  Card,
  Center,
  PasswordInput,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { useAuth } from "../auth";

const DEMO_EMAIL = "eng@example.com";
const DEMO_PASSWORD = "password123";

export function LoginPage() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState(DEMO_EMAIL);
  const [password, setPassword] = useState(DEMO_PASSWORD);
  const [busy, setBusy] = useState(false);

  async function submit(register: boolean) {
    setBusy(true);
    try {
      if (register) await api.register(email, password, "ml_engineer");
      await login(email, password);
      nav("/");
    } catch (e) {
      notifications.show({ color: "red", message: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Center h="100vh">
      <Card withBorder shadow="sm" w={380} padding="lg">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(false);
          }}
        >
          <Stack>
            <Title order={3}>⚖️ EthicLens</Title>
            <Text size="sm" c="dimmed">
              Sign in to audit models for bias and apply measured mitigations.
            </Text>
            <Alert color="blue" variant="light" title="This is a public demo">
              The fields below are pre-filled with a shared demo account — just click{" "}
              <strong>Sign in</strong> to explore. No registration needed.
            </Alert>
            <TextInput
              label="Email"
              type="email"
              autoComplete="username"
              value={email}
              onChange={(e) => setEmail(e.currentTarget.value)}
            />
            <PasswordInput
              label="Password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.currentTarget.value)}
            />
            <Button type="submit" loading={busy}>
              Sign in
            </Button>
            <Button
              type="button"
              variant="subtle"
              loading={busy}
              onClick={() => submit(true)}
            >
              Register a new account
            </Button>
          </Stack>
        </form>
      </Card>
    </Center>
  );
}
