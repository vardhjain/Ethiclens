import { Badge, Card, Group, Loader, Stack, Table, Text, Title } from "@mantine/core";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, type SessionOut } from "../api/client";

const STATUS_COLOR: Record<string, string> = {
  FLAGGED: "red",
  COMPLETED: "green",
  SIGNED_OFF: "blue",
  RUNNING: "yellow",
  FAILED: "gray",
};

export function SessionsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["sessions"],
    queryFn: api.listSessions,
    refetchInterval: 4000,
  });

  if (isLoading) return <Loader />;
  const sessions = data ?? [];

  return (
    <Stack>
      <Title order={3}>Audit sessions</Title>
      {sessions.length === 0 && (
        <Card withBorder>
          <Text c="dimmed">No audits yet. Create one from “New audit”.</Text>
        </Card>
      )}
      {sessions.length > 0 && (
        <Table highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Name</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Composite</Table.Th>
              <Table.Th>Worst DI</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sessions.map((s: SessionOut) => (
              <Table.Tr key={s.id}>
                <Table.Td>
                  <Link to={`/sessions/${s.id}`}>{s.name}</Link>
                </Table.Td>
                <Table.Td>
                  <Group gap="xs">
                    <Badge color={STATUS_COLOR[s.status] ?? "gray"}>{s.status}</Badge>
                    {s.locked && <Badge color="blue" variant="outline">locked</Badge>}
                  </Group>
                </Table.Td>
                <Table.Td>{s.composite_score?.toFixed(3) ?? "—"}</Table.Td>
                <Table.Td>{s.min_di?.toFixed(3) ?? "—"}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      )}
    </Stack>
  );
}
