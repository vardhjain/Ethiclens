import { Button, Card, Select, Stack, TextInput, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";

export function NewAuditPage() {
  const nav = useNavigate();
  const [name, setName] = useState("Lending model audit");
  const [dataset, setDataset] = useState("synthetic");
  const [attribute, setAttribute] = useState("race");
  const [busy, setBusy] = useState(false);

  async function create() {
    setBusy(true);
    try {
      const session = await api.createSession({
        name,
        dataset,
        target: "approved",
        protected_attributes: [{ name: attribute }],
      });
      await api.runSession(session.id);
      notifications.show({ color: "green", message: "Audit started." });
      nav(`/sessions/${session.id}`);
    } catch (e) {
      notifications.show({ color: "red", message: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card withBorder maw={520} padding="lg">
      <Stack>
        <Title order={3}>New audit session</Title>
        <TextInput label="Name" value={name} onChange={(e) => setName(e.currentTarget.value)} />
        <Select
          label="Dataset"
          data={[
            { value: "synthetic", label: "Synthetic lending (built-in)" },
            { value: "golden", label: "Golden reference (DI ≈ 0.55)" },
          ]}
          value={dataset}
          onChange={(v) => setDataset(v ?? "synthetic")}
        />
        <Select
          label="Protected attribute"
          data={["race", "gender"]}
          value={attribute}
          onChange={(v) => setAttribute(v ?? "race")}
        />
        <Button loading={busy} onClick={create}>
          Create &amp; run audit
        </Button>
      </Stack>
    </Card>
  );
}
