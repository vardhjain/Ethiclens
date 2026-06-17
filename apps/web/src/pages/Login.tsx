import { Button, Card, Center, PasswordInput, Stack, Text, TextInput, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { useAuth } from "../auth";

export function LoginPage() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState("eng@example.com");
  const [password, setPassword] = useState("password123");
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
        <Stack>
          <Title order={3}>⚖️ EthicLens</Title>
          <Text size="sm" c="dimmed">
            Sign in to audit models for bias and apply measured mitigations.
          </Text>
          <TextInput label="Email" value={email} onChange={(e) => setEmail(e.currentTarget.value)} />
          <PasswordInput
            label="Password"
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
          />
          <Button loading={busy} onClick={() => submit(false)}>
            Sign in
          </Button>
          <Button variant="subtle" loading={busy} onClick={() => submit(true)}>
            Register a new account
          </Button>
        </Stack>
      </Card>
    </Center>
  );
}
